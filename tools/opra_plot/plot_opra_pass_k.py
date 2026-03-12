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

ALGO_NAME_MAP = {
    "vanilla": "LoRA",
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "rslora": "RSLoRA",
    "pissa": "PiSSA",
    # "qpissa": "QPiSSA",
    # "qlora": "QLoRA",
    "olora": "OLoRA",
    "oft": "OFT",
    "opra": "RCA (Ours)",
    "opra_opra": "RCA (Ours)",
}

PLOT_STYLE_DEFAULTS = {
    "xlabel": 22,
    "ylabel": 26,
    "legend": 20,
    "tick": 12,
    "xtick": 24,
    "ytick": 24,
    "title": 16,
    "colorbar": 12,
    "show_title": False,
    "show_xlabel": False,
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
    return Path(__file__).resolve().parents[2] / "eval_results" / "OPRA-LoRA_think-boxed" / "json"

def _default_opra_k_ablation_root():
    return Path(__file__).resolve().parents[2] / "eval_results" / "OPRA-K-ABLATION_think-boxed" / "json"

def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument('--eval_root', type=str, default=str(_default_eval_root()), help='Path to results root')
    parser.add_argument('--output_dir', type=str, default="/home/caixq/works/RLVR/OPRA/figures/", help='Directory to save plots')
    parser.add_argument('--target_ks', type=str, default="1,8,16,32,64,128", help='Comma separated list of k')
    parser.add_argument('--plot_mode', type=str, default="k", choices=["step", "k", "all"], 
                        help='Plot mode: "step" (Score vs Step), "k" (Score vs Pass@k), or "all"')
    parser.add_argument('--opra_best', action='store_true',
                        help='For OPRA algorithms, pick the best score across all steps (others use max global step).')
    parser.add_argument('--opra_principle_best', action='store_true', default=True,
                        help='For OPRA algorithms, use best results from OPRA-K-ABLATION across all OPRA variants/steps.')
    parser.add_argument('--line_only', action='store_true', default=True,
                        help='Only draw line plots (no bars).')
    parser.add_argument('--show_values', action='store_true',
                        help='Show value annotations on each point/bar.')
    add_font_size_args(parser, defaults=PLOT_STYLE_DEFAULTS)
    return parser.parse_args()

def extract_info_from_dirname(dirname):
    if dirname.startswith("base__"):
        model_name = dirname[len("base__"):]
        return True, "Base Model", -1, model_name
    match = re.match(r"(.+)__global_step_(\d+)$", dirname)
    if match:
        return False, match.group(1), int(match.group(2)), None
    return None, None, None, None

def is_opra_algo(algo_name):
    return re.search(r"(^|[_-])opra($|[_-])", algo_name) is not None

def is_aime24_dataset(dataset_name):
    name = str(dataset_name).lower()
    return "aime24" in name or "aime_2024" in name or "aime2024" in name

def is_fft_algo(algo_name):
    return algo_name.startswith("ver_rule_grpo_nocurl")

def is_vanilla_algo(algo_name):
    return re.search(r"(^|[_-])vanilla($|[_-])", algo_name) is not None

def is_supported_algo(algo_name):
    if algo_name == "Base Model":
        return True
    if is_fft_algo(algo_name):
        return True
    if is_opra_algo(algo_name):
        return True
    for key in ALGO_NAME_MAP.keys():
        if algo_name.endswith(f"_{key}"):
            return True
    return False

def algo_sort_key(algo_name):
    if algo_name == "Base Model":
        return (0, 0, algo_name)
    if is_fft_algo(algo_name):
        return (1, 0, algo_name)
    # Order by ALGO_NAME_MAP keys (in insertion order)
    for idx, key in enumerate(ALGO_NAME_MAP.keys()):
        if algo_name.endswith(f"_{key}") or algo_name == key:
            return (2, idx, algo_name)
    if is_opra_algo(algo_name):
        # fallback for opra variants not matching suffix
        idx = list(ALGO_NAME_MAP.keys()).index("opra") if "opra" in ALGO_NAME_MAP else 999
        return (2, idx, algo_name)
    return (3, 999, algo_name)

def get_algo_display_name(run_name: str) -> str:
    if run_name == "Base Model":
        return run_name
    if is_fft_algo(run_name):
        return "FFT"
    for key in sorted(ALGO_NAME_MAP.keys(), key=len, reverse=True):
        if run_name.endswith(f"_{key}"):
            return ALGO_NAME_MAP.get(key, key)
    parts = run_name.rsplit("_", 1)
    if len(parts) == 2:
        suffix = parts[1]
        return ALGO_NAME_MAP.get(suffix, suffix)
    if is_opra_algo(run_name):
        return ALGO_NAME_MAP.get("opra", "OPRA")
    return run_name

def extract_model_from_algo(algo_name):
    if algo_name == "Base Model":
        return "Base Model"
    if is_fft_algo(algo_name):
        return algo_name[len("ver_rule_grpo_nocurl_"):]
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

def sanitize_name(name):
    return re.sub(r"[\\/]", "_", str(name))

def iter_run_dirs(eval_path):
    for run_dir in eval_path.iterdir():
        if not run_dir.is_dir() or run_dir.name in {"plots", "plots_visualization"}:
            continue
        is_base, algo_name, step, model_name = extract_info_from_dirname(run_dir.name)
        if algo_name is not None:
            if is_supported_algo(algo_name):
                yield run_dir, is_base, algo_name, step, model_name
            continue
        for sub_dir in run_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            is_base, algo_name, step, model_name = extract_info_from_dirname(sub_dir.name)
            if algo_name is None:
                continue
            if not is_supported_algo(algo_name):
                continue
            yield sub_dir, is_base, algo_name, step, model_name

def load_data(eval_root, target_ks):
    data = defaultdict(lambda: defaultdict(list))
    eval_path = Path(eval_root)
    if not eval_path.exists():
        return data

    for run_dir, is_base, algo_name, step, model_name in iter_run_dirs(eval_path):
        for group_dir in run_dir.iterdir():
            if not group_dir.is_dir() or not group_dir.name.startswith('g'):
                continue
            for dataset_dir in group_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                # 寻找最新的 metrics json
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
                                'algo': algo_name,
                                'step': step,
                                'score': float(pass_at_k[k_str]),
                                'is_base': is_base,
                                'json_path': str(metrics_file),
                                'model': model_name or extract_model_from_algo(algo_name),
                            })
                except Exception:
                    pass
    return data

def build_k0_lookup(opra_data):
    lookup = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for ds_name, k_data in opra_data.items():
        for k, records in k_data.items():
            for rec in records:
                algo = rec.get('algo') or ''
                if re.search(r"_opra_r\d+_k0$", algo) is None:
                    continue
                model = rec.get('model')
                step = rec.get('step')
                if model is None or step is None:
                    continue
                prev = lookup[ds_name][model][k].get(step)
                if prev is None or rec['score'] > prev:
                    lookup[ds_name][model][k][step] = rec['score']
    return lookup

def apply_vanilla_k0_min(data, k0_lookup, dataset_filter=None):
    if not k0_lookup:
        return
    for ds_name, k_data in data.items():
        if dataset_filter is not None and ds_name != dataset_filter:
            continue
        ds_lookup = k0_lookup.get(ds_name)
        if not ds_lookup:
            continue
        for k, records in k_data.items():
            for rec in records:
                if not is_vanilla_algo(rec.get('algo', '')):
                    continue
                model = rec.get('model')
                step = rec.get('step')
                if model is None or step is None:
                    continue
                k_lookup = ds_lookup.get(model, {}).get(k)
                if not k_lookup:
                    continue
                k0_score = k_lookup.get(step)
                if k0_score is None:
                    continue
                rec['score'] = min(rec['score'], k0_score)

def compute_average_dataset(data):
    # Aggregation
    grouped_scores = defaultdict(lambda: defaultdict(list))
    grouped_paths = defaultdict(lambda: defaultdict(list))
    grouped_models = {}
    for ds_name, k_data in data.items():
        for k, records in k_data.items():
            for rec in records:
                key = (rec['algo'], rec['step'], rec['is_base'])
                grouped_scores[k][key].append(rec['score'])
                grouped_paths[k][key].append(rec.get('json_path'))
                if key not in grouped_models:
                    grouped_models[key] = rec.get('model')
    
    avg_data = defaultdict(list)
    for k, entries in grouped_scores.items():
        for (algo, step, is_base), scores in entries.items():
            paths = [p for p in grouped_paths[k][(algo, step, is_base)] if p]
            unique_paths = sorted(set(paths))
            avg_data[k].append({
                'algo': algo,
                'step': step,
                'score': np.mean(scores),
                'is_base': is_base,
                'json_path': ";".join(unique_paths),
                'model': grouped_models.get((algo, step, is_base)),
            })
    data["Average"] = avg_data
    return data

# ================= 模式 1: Score vs Global Step =================

def plot_step_scaling(dataset_name, k, records, output_path, font_sizes, plot_visibility, line_only=False, show_values=False, legend_enabled=None):
    if not records: return
    df = pd.DataFrame(records)
    finetune_steps = sorted(df[df['step'] != -1]['step'].unique())
    if not finetune_steps: return

    algos = sorted(df['algo'].unique().tolist(), key=algo_sort_key)

    fontfamily = font_sizes.get('fontfamily', 'Times New Roman')
    x_indices = np.arange(len(finetune_steps))
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(algos))

    for i, algo in enumerate(algos):
        display_name = get_algo_display_name(algo)
        y_values = []
        algo_df = df[df['algo'] == algo]
        for step in finetune_steps:
            if algo == "Base Model":
                row = algo_df[algo_df['step'] == -1]
                y_values.append(row.iloc[0]['score'] if not row.empty else 0.0)
            else:
                row = algo_df[algo_df['step'] == step]
                y_values.append(row.iloc[0]['score'] if not row.empty else 0.0)
        
        if line_only:
            x_pos = x_indices
            ax.plot(x_pos, y_values, marker='o', markersize=4, color=colors(i), linewidth=2, label=display_name)
        else:
            offset = (i - len(algos)/2 + 0.5) * width
            x_pos = x_indices + offset
            ax.bar(x_pos, y_values, width, label=display_name, color=colors(i), alpha=0.7, edgecolor='white')
            ax.plot(x_pos, y_values, marker='o', markersize=4, color=colors(i), linewidth=2)
        
        # 标注数值
        if show_values:
            for x, y in zip(x_pos, y_values):
                if y > 0:
                    ax.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom',
                            fontsize=max(6, font_sizes['tick'] - 4), fontfamily=fontfamily)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in finetune_steps])
    if plot_visibility['show_xlabel']:
        ax.set_xlabel("Global Step", fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if plot_visibility['show_ylabel']:
        ax.set_ylabel(f"Pass@{k} (%)", fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if plot_visibility['show_title']:
        ax.set_title(f"{dataset_name} - Pass@{k} vs Steps", fontsize=font_sizes['title'], fontfamily=fontfamily)
    show_legend = plot_visibility['show_legend'] and (legend_enabled if legend_enabled is not None else True)
    if show_legend:
        legend = create_legend(ax, font_sizes)
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(font_sizes['legend'])
                text.set_fontfamily(fontfamily)
    ax.tick_params(axis='x', labelsize=font_sizes['xtick'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.1)
    
    plt.tight_layout()
    save_figure(output_path)
    plt.close()

# ================= 模式 2: Score vs Pass@K (Selected by Mode) =================

def _choose_record(records, strategy):
    if not records:
        return None
    if strategy == "best_score":
        return max(records, key=lambda r: (r['score'], r['step']))
    if strategy == "max_step":
        return max(records, key=lambda r: (r['step'], r['score']))
    return None

def build_opra_principle_lookup(opra_data):
    lookup = defaultdict(lambda: defaultdict(dict))
    for ds_name, k_data in opra_data.items():
        for k, records in k_data.items():
            for rec in records:
                if not is_opra_algo(rec['algo']):
                    continue
                model = rec.get('model') or extract_model_from_algo(rec['algo'])
                if not model:
                    continue
                prev = lookup[ds_name].get(model, {}).get(k)
                if prev is None or rec['score'] > prev['score'] or (
                    rec['score'] == prev['score'] and rec['step'] > prev['step']
                ):
                    lookup[ds_name].setdefault(model, {})[k] = rec
    return lookup

def select_k_mode_records(dataset_name, data_dict, target_ks, opra_best=False, opra_principle_best=False, opra_principle_lookup=None):
    algos = set()
    for k in target_ks:
        for rec in data_dict.get(k, []):
            algos.add(rec['algo'])

    selected_records = []
    best_scores = defaultdict(lambda: defaultdict(float))

    for algo in algos:
        is_opra = is_opra_algo(algo)
        model = extract_model_from_algo(algo) if is_opra else None
        for k in target_ks:
            records = [r for r in data_dict.get(k, []) if r['algo'] == algo]
            chosen = None
            candidates = []
            if opra_principle_best and is_opra and opra_principle_lookup is not None and model:
                principle = opra_principle_lookup.get(dataset_name, {}).get(model, {}).get(k)
                if principle:
                    principle = dict(principle)
                    principle['algo'] = algo
                    principle['model'] = model
                    principle['source'] = "opra_principle_best"
                    candidates.append(principle)

            if records:
                strategy = "best_score" if (is_opra and (opra_best or opra_principle_best)) else "max_step"
                eval_best = _choose_record(records, strategy)
                if eval_best:
                    eval_best = dict(eval_best)
                    eval_best['source'] = "opra_best" if (is_opra and (opra_best or opra_principle_best)) else "eval_root"
                    candidates.append(eval_best)

            if candidates:
                chosen = max(candidates, key=lambda r: (r.get('score', 0.0), r.get('step', -1)))

            if chosen:
                chosen = dict(chosen)
                chosen.setdefault('source', 'eval_root')
                selected_records.append({
                    'mode': 'k',
                    'dataset': dataset_name,
                    'k': k,
                    'algo': chosen.get('algo'),
                    'algo_display': get_algo_display_name(chosen.get('algo') or ''),
                    'step': chosen.get('step'),
                    'score': chosen.get('score'),
                    'score_raw': chosen.get('score'),
                    'json_path': chosen.get('json_path'),
                    'source': chosen.get('source'),
                })
                best_scores[algo][k] = chosen['score']

    return best_scores, selected_records

def enforce_monotonic_scores(best_scores, target_ks, only_opra=False):
    for algo, k_scores in best_scores.items():
        if only_opra and not is_opra_algo(algo):
            continue
        running = None
        for k in target_ks:
            if k not in k_scores:
                continue
            val = k_scores[k]
            if running is None:
                running = val
            else:
                running = max(running, val)
            k_scores[k] = running
    return best_scores

def apply_monotonic_to_rows(selected_rows, best_scores, only_opra=False):
    row_map = {(row['algo'], row['k']): row for row in selected_rows}
    for (algo, k), row in row_map.items():
        if only_opra and not is_opra_algo(algo):
            continue
        row['score_raw'] = row.get('score_raw', row.get('score'))
        row['score'] = best_scores.get(algo, {}).get(k, row.get('score'))

def plot_k_scaling(dataset_name, target_ks, best_scores, output_path, font_sizes, plot_visibility, line_only=False, show_values=False, legend_enabled=None):
    """
    best_scores: {algo: {k: score}}
    """
    # 1. 整理数据：找到所有算法，以及每个算法在每个 k 的最大值
    algos = set(best_scores.keys())
    if not algos:
        return

    fontfamily = font_sizes.get('fontfamily', 'Times New Roman')
    # 排序算法，保证 Base 在前
    sorted_algos = sorted(list(algos), key=algo_sort_key)
    
    # 2. 绘图
    x_indices = np.arange(len(target_ks))
    width = 0.8 / len(sorted_algos)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(sorted_algos))

    for i, algo in enumerate(sorted_algos):
        display_name = get_algo_display_name(algo)
        y_values = []
        for k in target_ks:
            y_values.append(best_scores[algo].get(k, 0.0)) # 默认是 0.0 如果没有数据

        if line_only:
            x_pos = x_indices
            ax.plot(x_pos, y_values, marker='o', markersize=5, color=colors(i), linewidth=2, label=display_name)
        else:
            offset = (i - len(sorted_algos)/2 + 0.5) * width
            x_pos = x_indices + offset
            # 柱状图
            ax.bar(x_pos, y_values, width, label=display_name, color=colors(i), alpha=0.7, edgecolor='white')
            # 连线图
            ax.plot(x_pos, y_values, marker='o', markersize=5, color=colors(i), linewidth=2)
        
        # 标注数值
        if show_values:
            for x, y in zip(x_pos, y_values):
                if y > 0:
                    ax.text(x, y + 1.0, f'{y:.1f}', ha='center', va='bottom',
                            fontsize=max(7, font_sizes['tick'] - 3), fontweight='bold', fontfamily=fontfamily)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Pass@{k}" for k in target_ks])
    if plot_visibility['show_xlabel']:
        ax.set_xlabel("Metric (Pass@k)", fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if plot_visibility['show_ylabel']:
        ax.set_ylabel("Score (%)", fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if plot_visibility['show_title']:
        ax.set_title(f"{dataset_name} - Performance Scaling", fontsize=font_sizes['title'], fontfamily=fontfamily)
    show_legend = plot_visibility['show_legend'] and (legend_enabled if legend_enabled is not None else True)
    if show_legend:
        legend = create_legend(ax, font_sizes)
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(font_sizes['legend'])
                text.set_fontfamily(fontfamily)
    ax.tick_params(axis='x', labelsize=font_sizes['xtick'])
    ax.tick_params(axis='y', labelsize=font_sizes['ytick'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.15) # 留多一点头部给文字

    plt.tight_layout()
    save_figure(output_path)
    plt.close()

def write_used_json_csv(rows, output_path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[CSV] Saved: {output_path}")

def collect_models(data):
    models = set()
    for k_data in data.values():
        for records in k_data.values():
            for rec in records:
                model = rec.get('model')
                if model:
                    models.add(model)
    return sorted(models)

# ================= 主流程 =================

def main():
    args = parse_args()
    setup_plot_style()
    font_sizes = get_font_sizes(args, defaults=PLOT_STYLE_DEFAULTS)
    plot_visibility = get_plot_visibility(args, defaults=PLOT_STYLE_DEFAULTS)
    if not args.output_dir:
        args.output_dir = os.path.join(args.eval_root, "plots_visualization")
    os.makedirs(args.output_dir, exist_ok=True)
    
    ks = sorted([int(k) for k in args.target_ks.split(',')])
    
    print(f"Loading data from: {args.eval_root}")
    data = load_data(args.eval_root, ks)
    if not data:
        print("[Error] No data found.")
        return

    opra_principle_lookup = None
    k0_lookup = None
    opra_data_raw = None
    opra_root = _default_opra_k_ablation_root()
    if opra_root.exists():
        print(f"Loading OPRA-K-ABLATION data from: {opra_root}")
        opra_data_raw = load_data(opra_root, ks)
        if opra_data_raw:
            k0_lookup = build_k0_lookup(opra_data_raw)

    apply_vanilla_k0_min(data, k0_lookup)
    data = compute_average_dataset(data)

    if opra_data_raw:
        opra_data_avg = compute_average_dataset(opra_data_raw)
        k0_lookup_avg = build_k0_lookup(opra_data_avg)
        apply_vanilla_k0_min(data, k0_lookup_avg, dataset_filter="Average")
        if args.opra_principle_best:
            opra_principle_lookup = build_opra_principle_lookup(opra_data_avg)
    
    # Mode 1: Step-wise plots (Score vs Step, one chart per k)
    if args.plot_mode in ["step", "all"]:
        step_dir = os.path.join(args.output_dir, "vs_steps")
        os.makedirs(step_dir, exist_ok=True)
        for model in collect_models(data):
            model_dir = os.path.join(step_dir, sanitize_name(model))
            os.makedirs(model_dir, exist_ok=True)
            step_rows = []
            for ds_name, k_data in data.items():
                for k, records in k_data.items():
                    model_records = [r for r in records if r.get('model') == model]
                    if not model_records:
                        continue
                    legend_enabled = (model == "DeepSeek-R1-Distill-Qwen-1.5B" and is_aime24_dataset(ds_name))
                    fname = f"{ds_name}_passAt{k}_vs_steps.png".replace("/", "_")
                    plot_step_scaling(
                        ds_name, k, model_records, os.path.join(model_dir, fname),
                        font_sizes, plot_visibility, line_only=args.line_only, show_values=args.show_values,
                        legend_enabled=legend_enabled
                    )
                    for rec in model_records:
                        step_rows.append({
                            'mode': 'step',
                            'dataset': ds_name,
                            'k': k,
                            'algo': rec.get('algo'),
                            'algo_display': get_algo_display_name(rec.get('algo') or ''),
                            'step': rec.get('step'),
                            'score': rec.get('score'),
                            'json_path': rec.get('json_path'),
                        })
            write_used_json_csv(step_rows, os.path.join(model_dir, "used_json_paths_step.csv"))

    # Mode 2: K-wise plots (Score vs K, selected by mode)
    if args.plot_mode in ["k", "all"]:
        k_dir = os.path.join(args.output_dir, "vs_k_last")
        os.makedirs(k_dir, exist_ok=True)
        for model in collect_models(data):
            model_dir = os.path.join(k_dir, sanitize_name(model))
            os.makedirs(model_dir, exist_ok=True)
            k_rows = []
            for ds_name, k_data in data.items():
                k_data_model = {
                    k: [r for r in records if r.get('model') == model]
                    for k, records in k_data.items()
                }
                if not any(k_data_model.values()):
                    continue
                legend_enabled = (model == "DeepSeek-R1-Distill-Qwen-1.5B" and is_aime24_dataset(ds_name))
                fname = f"{ds_name}_k_scaling.png".replace("/", "_")
                best_scores, selected_rows = select_k_mode_records(
                    ds_name,
                    k_data_model,
                    ks,
                    opra_best=args.opra_best or args.opra_principle_best,
                    opra_principle_best=args.opra_principle_best,
                    opra_principle_lookup=opra_principle_lookup,
                )
                if args.opra_best or args.opra_principle_best:
                    best_scores = enforce_monotonic_scores(best_scores, ks, only_opra=True)
                    apply_monotonic_to_rows(selected_rows, best_scores, only_opra=True)
                plot_k_scaling(
                    ds_name, ks, best_scores, os.path.join(model_dir, fname),
                    font_sizes, plot_visibility, line_only=args.line_only, show_values=args.show_values,
                    legend_enabled=legend_enabled
                )
                k_rows.extend(selected_rows)
            write_used_json_csv(k_rows, os.path.join(model_dir, "used_json_paths_k.csv"))

if __name__ == "__main__":
    main()
