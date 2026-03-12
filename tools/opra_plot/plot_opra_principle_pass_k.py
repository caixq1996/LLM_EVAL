#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OPRA principal-rank (k) ablation pass@k curves.

Modes:
  - rank (plot_mode=k): x = principal rank, y = pass@k (one figure per pass@k + a combined figure)
  - step (plot_mode=step): x = global step, y = pass@k (one figure per pass@k, lines per principal rank)

Outputs are grouped by model under plots_visualization.
"""
import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from plot_config import setup_plot_style, add_font_size_args, get_font_sizes, get_plot_visibility, create_legend

# ---------------------- 配置 ----------------------
sns.set_theme(style="whitegrid")

PLOT_STYLE_DEFAULTS = {
    "xlabel": 20,
    "ylabel": 20,
    "legend": 16,
    "tick": 12,
    "xtick": 16,
    "ytick": 16,
    "title": 16,
    "colorbar": 12,
    "show_title": False,
    "show_xlabel": True,
    "show_ylabel": True,
    "show_legend": True,
}

def save_figure(output_path):
    plt.savefig(output_path, dpi=300)
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300)
    print(f"[Plot] Saved: {output_path}")
    print(f"[Plot] Saved: {pdf_path}")


def _default_eval_root():
    return Path(__file__).resolve().parents[2] / "eval_results" / "OPRA-K-ABLATION_think-boxed" / "json"

def _default_opra_lora_root():
    return Path(__file__).resolve().parents[2] / "eval_results" / "OPRA-LoRA_think-boxed" / "json"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot OPRA principal-rank pass@k results")
    parser.add_argument('--eval_root', type=str, default=str(_default_eval_root()), help='Path to results root')
    parser.add_argument('--opra_lora_root', type=str, default=str(_default_opra_lora_root()),
                        help='Path to OPRA-LoRA json root (used for r_k=16 max merge)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--target_ks', type=str, default="1,8,16,32", help='Comma separated list of pass@k')
    parser.add_argument('--plot_mode', type=str, default="k", choices=["step", "k", "all"],
                        help='Plot mode: "k" (Rank vs Pass@k), "step" (Pass@k vs Step), or "all"')
    parser.add_argument('--rank_k', type=str, default="0,1,2,4,8,16",
                        help='Comma separated list of r_k to plot (e.g., "0,1,2"). Default: all.')
    parser.add_argument('--line_only', action='store_true',
                        help='Only draw line plots (no bars).')
    parser.add_argument('--show_values', action='store_true',
                        help='Show value annotations on each point/bar.')
    add_font_size_args(parser, defaults=PLOT_STYLE_DEFAULTS)
    return parser.parse_args()


def extract_info_from_dirname(dirname):
    if dirname.startswith("base__"):
        return True, "Base Model", -1
    match = re.match(r"(.+)__global_step_(\d+)$", dirname)
    if match:
        return False, match.group(1), int(match.group(2))
    return None, None, None


def extract_model_from_algo(algo_name):
    if algo_name == "Base Model":
        return "Base Model"
    match = re.match(r"(.+)_opra_r\d+_k\d+$", algo_name)
    if match:
        return match.group(1)
    match = re.match(r"(.+)_opra_opra$", algo_name)
    if match:
        return match.group(1)
    match = re.match(r"(.+)_opra$", algo_name)
    if match:
        return match.group(1)
    if "_" in algo_name:
        return algo_name.rsplit("_", 1)[0]
    return algo_name


def extract_principle_rank(algo_name):
    match = re.search(r"_k(\d+)$", algo_name)
    if match:
        return int(match.group(1))
    return None


def is_opra_algo(algo_name):
    return re.search(r"(^|[_-])opra($|[_-])", algo_name) is not None


def sanitize_name(name):
    return re.sub(r"[\\/]", "_", str(name))


def iter_run_dirs(eval_path):
    for run_dir in eval_path.iterdir():
        if not run_dir.is_dir() or run_dir.name in {"plots", "plots_visualization"}:
            continue
        is_base, algo_name, step = extract_info_from_dirname(run_dir.name)
        if algo_name is not None:
            yield run_dir, is_base, algo_name, step
            continue
        for sub_dir in run_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            is_base, algo_name, step = extract_info_from_dirname(sub_dir.name)
            if algo_name is None:
                continue
            yield sub_dir, is_base, algo_name, step


def load_data(eval_root, target_ks):
    data = defaultdict(lambda: defaultdict(list))
    eval_path = Path(eval_root)
    if not eval_path.exists():
        return data

    for run_dir, is_base, algo_name, step in iter_run_dirs(eval_path):
        rank = extract_principle_rank(algo_name)
        model = extract_model_from_algo(algo_name)
        if rank is None:
            continue

        for group_dir in run_dir.iterdir():
            if not group_dir.is_dir() or not group_dir.name.startswith('g'):
                continue
            for dataset_dir in group_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                metrics_files = list(dataset_dir.glob("*_metrics.json"))
                if not metrics_files:
                    continue
                metrics_file = sorted(metrics_files, key=lambda x: x.stat().st_mtime)[-1]

                try:
                    with open(metrics_file, 'r') as f:
                        res = json.load(f)
                    pass_at_k = res.get('pass_at_k_percent', {})
                    if not pass_at_k and 'acc' in res:
                        pass_at_k = {'1': res['acc']}

                    for k in target_ks:
                        k_str = str(k)
                        if k_str in pass_at_k and pass_at_k[k_str] is not None:
                            data[dataset_dir.name][k].append({
                                'model': model,
                                'rank': rank,
                                'step': step,
                                'score': float(pass_at_k[k_str]),
                                'json_path': str(metrics_file),
                            })
                except Exception:
                    pass
    return data


def merge_opra_lora_rank16(data, opra_lora_root, target_ks, rank_override=16):
    eval_path = Path(opra_lora_root)
    if not eval_path.exists():
        return data
    for run_dir, is_base, algo_name, step in iter_run_dirs(eval_path):
        if algo_name is None or not is_opra_algo(algo_name):
            continue
        model = extract_model_from_algo(algo_name)
        if not model:
            continue
        rank = rank_override
        for group_dir in run_dir.iterdir():
            if not group_dir.is_dir() or not group_dir.name.startswith('g'):
                continue
            for dataset_dir in group_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                metrics_files = list(dataset_dir.glob("*_metrics.json"))
                if not metrics_files:
                    continue
                metrics_file = sorted(metrics_files, key=lambda x: x.stat().st_mtime)[-1]
                try:
                    with open(metrics_file, 'r') as f:
                        res = json.load(f)
                    pass_at_k = res.get('pass_at_k_percent', {})
                    if not pass_at_k and 'acc' in res:
                        pass_at_k = {'1': res['acc']}
                    for k in target_ks:
                        k_str = str(k)
                        if k_str in pass_at_k and pass_at_k[k_str] is not None:
                            data[dataset_dir.name][k].append({
                                'model': model,
                                'rank': rank,
                                'step': step,
                                'score': float(pass_at_k[k_str]),
                                'json_path': str(metrics_file),
                            })
                except Exception:
                    pass
    return data


def compute_average_dataset(data):
    grouped_scores = defaultdict(lambda: defaultdict(list))
    grouped_paths = defaultdict(lambda: defaultdict(list))
    grouped_meta = {}

    for ds_name, k_data in data.items():
        for k, records in k_data.items():
            for rec in records:
                key = (rec['model'], rec['rank'], rec['step'])
                grouped_scores[k][key].append(rec['score'])
                grouped_paths[k][key].append(rec.get('json_path'))
                grouped_meta[key] = {
                    'model': rec.get('model'),
                    'rank': rec.get('rank'),
                    'step': rec.get('step'),
                }

    avg_data = defaultdict(list)
    for k, entries in grouped_scores.items():
        for key, scores in entries.items():
            paths = [p for p in grouped_paths[k][key] if p]
            unique_paths = sorted(set(paths))
            meta = grouped_meta[key]
            avg_data[k].append({
                'model': meta['model'],
                'rank': meta['rank'],
                'step': meta['step'],
                'score': np.mean(scores),
                'json_path': ";".join(unique_paths),
            })

    data["Average"] = avg_data
    return data


def collect_models(data):
    models = set()
    for k_data in data.values():
        for records in k_data.values():
            for rec in records:
                model = rec.get('model')
                if model:
                    models.add(model)
    return sorted(models)


def filter_data_by_ranks(data, ranks):
    if not ranks:
        return data
    ranks_set = set(ranks)
    filtered = defaultdict(lambda: defaultdict(list))
    for ds_name, k_data in data.items():
        for k, records in k_data.items():
            filtered_records = [r for r in records if r.get('rank') in ranks_set]
            if filtered_records:
                filtered[ds_name][k].extend(filtered_records)
    return filtered


def select_max_step_by_rank(records):
    best = {}
    for rec in records:
        rank = rec['rank']
        prev = best.get(rank)
        if prev is None or rec['step'] > prev['step'] or (
            rec['step'] == prev['step'] and rec['score'] > prev['score']
        ):
            best[rank] = rec
    return best


def select_rank_scores_varying_strategy(records):
    if not records:
        return {}
    best_by_rank = {}
    max_step_by_rank = select_max_step_by_rank(records)
    for rank, rec in max_step_by_rank.items():
        if rank == 0:
            best_by_rank[rank] = rec
        else:
            best_by_rank[rank] = rec
    for rec in records:
        rank = rec['rank']
        if rank == 0:
            continue
        prev = best_by_rank.get(rank)
        if prev is None or rec['score'] > prev['score'] or (
            rec['score'] == prev['score'] and rec['step'] > prev['step']
        ):
            best_by_rank[rank] = rec
    return best_by_rank


def plot_rank_scaling(dataset_name, pass_k, rank_scores, output_path, font_sizes, plot_visibility, line_only=False, show_values=False):
    if not rank_scores:
        return
    ranks = sorted(rank_scores.keys())
    scores = [rank_scores[r] for r in ranks]

    fontfamily = font_sizes.get('fontfamily', 'Times New Roman')
    x_indices = np.arange(len(ranks))

    fig, ax = plt.subplots(figsize=(10, 6))
    if line_only:
        ax.plot(x_indices, scores, marker='o', linewidth=2, label=f"Pass@{pass_k}")
    else:
        ax.bar(x_indices, scores, width=0.6, color='#4C72B0', alpha=0.8, edgecolor='white', label=f"Pass@{pass_k}")
        ax.plot(x_indices, scores, marker='o', linewidth=2, color='#1f77b4')

    if show_values:
        for x, y in zip(x_indices, scores):
            if y > 0:
                ax.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom',
                        fontsize=max(6, font_sizes['tick'] - 4), fontfamily=fontfamily)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(r) for r in ranks])
    if plot_visibility['show_xlabel']:
        ax.set_xlabel(r"$r_k$", fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if plot_visibility['show_ylabel']:
        ax.set_ylabel(f"Pass@{pass_k} (%)", fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if plot_visibility['show_title']:
        ax.set_title(f"{dataset_name} - Pass@{pass_k} vs r_k", fontsize=font_sizes['title'], fontfamily=fontfamily)
    if plot_visibility['show_legend']:
        create_legend(ax, font_sizes)
    ax.tick_params(axis='x', labelsize=font_sizes['xtick'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.1)

    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def plot_rank_combined(dataset_name, passk_to_rank_scores, output_path, font_sizes, plot_visibility, line_only=False, show_values=False):
    if not passk_to_rank_scores:
        return

    all_ranks = set()
    for scores in passk_to_rank_scores.values():
        all_ranks.update(scores.keys())
    ranks = sorted(all_ranks)
    if not ranks:
        return

    fontfamily = font_sizes.get('fontfamily', 'Times New Roman')
    x_indices = np.arange(len(ranks))
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.get_cmap('tab10', len(passk_to_rank_scores))
    for idx, (pass_k, rank_scores) in enumerate(sorted(passk_to_rank_scores.items(), key=lambda x: x[0])):
        scores = [rank_scores.get(r, 0.0) for r in ranks]
        label = f"Pass@{pass_k}"
        if line_only:
            ax.plot(x_indices, scores, marker='o', linewidth=2, color=colors(idx), label=label)
        else:
            width = 0.8 / max(1, len(passk_to_rank_scores))
            offset = (idx - len(passk_to_rank_scores) / 2 + 0.5) * width
            x_pos = x_indices + offset
            ax.bar(x_pos, scores, width=width, color=colors(idx), alpha=0.75, edgecolor='white', label=label)
            ax.plot(x_pos, scores, marker='o', linewidth=1.5, color=colors(idx))

        if show_values:
            for x, y in zip(x_indices if line_only else x_pos, scores):
                if y > 0:
                    ax.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom',
                            fontsize=max(6, font_sizes['tick'] - 4), fontfamily=fontfamily)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(r) for r in ranks])
    if plot_visibility['show_xlabel']:
        ax.set_xlabel(r"$r_k$", fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if plot_visibility['show_ylabel']:
        ax.set_ylabel("Pass (%)", fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if plot_visibility['show_title']:
        ax.set_title(f"{dataset_name} - Pass@k vs r_k", fontsize=font_sizes['title'], fontfamily=fontfamily)
    if plot_visibility['show_legend']:
        create_legend(ax, font_sizes)
    ax.tick_params(axis='x', labelsize=font_sizes['xtick'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.1)

    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def plot_step_scaling(dataset_name, pass_k, records, output_path, font_sizes, plot_visibility, line_only=False, show_values=False):
    if not records:
        return
    df = pd.DataFrame(records)
    steps = sorted(df[df['step'] != -1]['step'].unique())
    if not steps:
        return

    ranks = sorted(df['rank'].unique().tolist())
    fontfamily = font_sizes.get('fontfamily', 'Times New Roman')
    x_indices = np.arange(len(steps))
    width = 0.8 / max(1, len(ranks))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(ranks))

    for i, rank in enumerate(ranks):
        sub = df[df['rank'] == rank]
        y_values = []
        for step in steps:
            row = sub[sub['step'] == step]
            y_values.append(row.iloc[0]['score'] if not row.empty else 0.0)

        label = f"$r_k$={rank}"
        if line_only:
            ax.plot(x_indices, y_values, marker='o', markersize=4, color=colors(i), linewidth=2, label=label)
        else:
            offset = (i - len(ranks) / 2 + 0.5) * width
            x_pos = x_indices + offset
            ax.bar(x_pos, y_values, width, label=label, color=colors(i), alpha=0.7, edgecolor='white')
            ax.plot(x_pos, y_values, marker='o', markersize=4, color=colors(i), linewidth=2)

        if show_values:
            x_pos_values = x_indices if line_only else x_pos
            for x, y in zip(x_pos_values, y_values):
                if y > 0:
                    ax.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom',
                            fontsize=max(6, font_sizes['tick'] - 4), fontfamily=fontfamily)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in steps])
    if plot_visibility['show_xlabel']:
        ax.set_xlabel("Global Step", fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if plot_visibility['show_ylabel']:
        ax.set_ylabel(f"Pass@{pass_k} (%)", fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if plot_visibility['show_title']:
        ax.set_title(f"{dataset_name} - Pass@{pass_k} vs Steps", fontsize=font_sizes['title'], fontfamily=fontfamily)
    if plot_visibility['show_legend']:
        create_legend(ax, font_sizes)
    ax.tick_params(axis='x', labelsize=font_sizes['xtick'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.1)

    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def main():
    args = parse_args()
    setup_plot_style()
    font_sizes = get_font_sizes(args, defaults=PLOT_STYLE_DEFAULTS)
    plot_visibility = get_plot_visibility(args, defaults=PLOT_STYLE_DEFAULTS)

    if not args.output_dir:
        args.output_dir = os.path.join(args.eval_root, "plots_visualization")
    os.makedirs(args.output_dir, exist_ok=True)

    pass_ks = sorted([int(k) for k in args.target_ks.split(',')])
    ranks = None
    if args.rank_k:
        ranks = [int(r.strip()) for r in args.rank_k.split(",") if r.strip()]

    print(f"Loading data from: {args.eval_root}")
    data = load_data(args.eval_root, pass_ks)
    if not data:
        print("[Error] No data found.")
        return
    data = merge_opra_lora_rank16(data, args.opra_lora_root, pass_ks, rank_override=16)
    data = filter_data_by_ranks(data, ranks)
    data = compute_average_dataset(data)

    models = collect_models(data)

    # Mode 1: Rank-wise plots (k = principle rank)
    if args.plot_mode in ["k", "all"]:
        rank_dir = os.path.join(args.output_dir, "vs_principle_rank")
        os.makedirs(rank_dir, exist_ok=True)
        for model in models:
            model_dir = os.path.join(rank_dir, sanitize_name(model))
            os.makedirs(model_dir, exist_ok=True)
            for ds_name, k_data in data.items():
                passk_to_rank_scores = {}
                for pass_k in pass_ks:
                    records = [r for r in k_data.get(pass_k, []) if r.get('model') == model]
                    best_by_rank = select_rank_scores_varying_strategy(records)
                    rank_scores = {rank: rec['score'] for rank, rec in best_by_rank.items()}
                    passk_to_rank_scores[pass_k] = rank_scores

                combined_bar = os.path.join(model_dir, f"{ds_name}_passAt_all_by_rank_bar.png".replace("/", "_"))
                combined_line = os.path.join(model_dir, f"{ds_name}_passAt_all_by_rank_line.png".replace("/", "_"))
                if not args.line_only:
                    plot_rank_combined(ds_name, passk_to_rank_scores, combined_bar, font_sizes, plot_visibility,
                                       line_only=False, show_values=args.show_values)
                plot_rank_combined(ds_name, passk_to_rank_scores, combined_line, font_sizes, plot_visibility,
                                   line_only=True, show_values=args.show_values)

    # Mode 2: Step-wise plots (x = global step, lines per principle rank)
    if args.plot_mode in ["step", "all"]:
        step_dir = os.path.join(args.output_dir, "vs_steps")
        os.makedirs(step_dir, exist_ok=True)
        for model in models:
            model_dir = os.path.join(step_dir, sanitize_name(model))
            os.makedirs(model_dir, exist_ok=True)
            for ds_name, k_data in data.items():
                for pass_k in pass_ks:
                    records = [r for r in k_data.get(pass_k, []) if r.get('model') == model]
                    if not records:
                        continue

                    bar_path = os.path.join(model_dir, f"{ds_name}_passAt{pass_k}_by_steps_bar.png".replace("/", "_"))
                    line_path = os.path.join(model_dir, f"{ds_name}_passAt{pass_k}_by_steps_line.png".replace("/", "_"))

                    if not args.line_only:
                        plot_step_scaling(ds_name, pass_k, records, bar_path, font_sizes, plot_visibility,
                                          line_only=False, show_values=args.show_values)
                    plot_step_scaling(ds_name, pass_k, records, line_path, font_sizes, plot_visibility,
                                      line_only=True, show_values=args.show_values)


if __name__ == "__main__":
    main()
