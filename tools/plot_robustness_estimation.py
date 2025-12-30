#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Plot robustness bar+line charts for different estimation settings.

This script expects evaluation outputs arranged similarly to run_qwen_eval_all.
Each run directory should follow the naming convention:
  B_rb_manual_{algo}_est_r00.10_r10.20_est{est1}_{est2}_Qwen2.5-math-1.5B__global_step_{step}
where `algo` is one of {algo1, algo2} and est1/est2 encode \hat{rho_0}/\hat{rho_1}.

For each (algo, est1, est2, dataset), the script keeps the best metric value
across global steps and aggregates average scores across six datasets to build
four bar+line charts:
  1. algo1 with est1 fixed at 0.10, varying est2.
  2. algo1 with est2 fixed at 0.20, varying est1.
  3. algo2 with est1 fixed at 0.10, varying est2.
  4. algo2 with est2 fixed at 0.20, varying est1.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_NAME_RE = re.compile(
    r'^B_rb_manual_(?P<algo>[^_]+)_est_r00\.10_r10\.20_est'
    r'(?P<est1>[0-9.]+)_(?P<est2>[0-9.]+)_Qwen2\.5-math-1\.5B__global_step_(?P<step>\d+)$'
)

DATASET_DISPLAY_MAP: Dict[str, str] = {
    'aime24x8': 'AIME2024',
    'aime25x8': 'AIME2025',
    'amc23x8': 'AMC2023',
    'math500': 'Math500',
    'minerva_math': 'Minerva MATH',
    'olympiadbench': 'Olympiad Bench',
    'olypiadbench': 'Olympiad Bench',
}

ALGO_DISPLAY_MAP: Dict[str, str] = {
    'algo1': 'Backward Correction',
    'algo2': 'Forward Correction',
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', type=Path, required=True,
                    help='Root directory containing robustness evaluation outputs.')
    ap.add_argument('--save_dir', type=Path, default=None,
                    help='Output directory for charts/tables (default: <out_root>/_summary_robustness).')
    ap.add_argument('--metric', type=str, default='pass@1',
                    help='Metric key to plot (e.g., acc, total_acc, pass@1).')
    ap.add_argument('--dpi', type=int, default=200,
                    help='Figure DPI.')
    ap.add_argument('--show_title_xlabel', action='store_true', default=False,
                    help='Render both figure titles and x-axis labels (default: disabled).')
    ap.add_argument('--show_title', action='store_true', default=False,
                    help='Render figure titles (default: disabled).')
    ap.add_argument('--show_xlabel', action='store_true', default=False,
                    help='Render figure x-axis labels (default: disabled).')
    ap.add_argument('--tick_fontsize', type=float, default=10.0,
                    help='Font size for axis tick labels.')
    ap.add_argument('--axis_label_fontsize', type=float, default=11.0,
                    help='Font size for axis labels.')
    ap.add_argument('--title_fontsize', type=float, default=13.0,
                    help='Font size for plot titles.')
    ap.add_argument('--bar_value_fontsize', type=float, default=9.0,
                    help='Font size for numeric annotations above bars.')
    ap.add_argument('--y_axis_padding', type=float, default=5.0,
                    help='Extra padding added to the y-axis upper bound in plot units (default: 5).')
    return ap.parse_args()


def format_float(v: float) -> str:
    return f"{v:.3f}".rstrip('0').rstrip('.') if np.isfinite(v) else 'nan'


def _find_latest_metrics_json(dataset_dir: Path) -> Path | None:
    candidates = sorted(dataset_dir.glob('*metrics.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _extract_metric(payload: Dict, metric: str) -> float | None:
    metric = metric.lower()
    if metric.startswith('pass@'):
        k = metric.split('@', 1)[1]
        pass_map = payload.get('pass_at_k_percent') or {}
        if k in pass_map and pass_map[k] is not None:
            return float(pass_map[k])
        return None
    value = payload.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_run(run_name: str):
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return None
    algo = m.group('algo').lower()
    est1 = float(m.group('est1'))
    est2 = float(m.group('est2'))
    step = int(m.group('step'))
    return {'algo': algo, 'est1': est1, 'est2': est2, 'step': step}


def load_best_scores(out_root: Path, metric: str, target_algos: Iterable[str]) -> Dict[Tuple[str, float, float, str], float]:
    target_algos = {a.lower() for a in target_algos}
    records: Dict[Tuple[str, float, float, str], float] = {}
    for run_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        info = parse_run(run_dir.name)
        if not info or info['algo'] not in target_algos:
            continue
        for group_dir in (run_dir / 'g1', run_dir / 'g2'):
            if not group_dir.exists():
                continue
            for dataset_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                metrics_path = _find_latest_metrics_json(dataset_dir)
                if not metrics_path:
                    continue
                try:
                    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    continue
                value = _extract_metric(payload, metric)
                if value is None or not np.isfinite(value):
                    continue
                key = (info['algo'], info['est1'], info['est2'], dataset_dir.name)
                prev = records.get(key)
                if prev is None or value > prev:
                    records[key] = value
    return records


def aggregate_macro(records: Dict[Tuple[str, float, float, str], float],
                    algo: str,
                    fixed_key: str,
                    fixed_value: float,
                    variable_key: str) -> Dict[float, float]:
    result: Dict[float, List[float]] = {}
    tol = 1e-6
    for (algo_name, est1, est2, dataset), value in records.items():
        if algo_name != algo:
            continue
        if fixed_key == 'est1' and abs(est1 - fixed_value) > tol:
            continue
        if fixed_key == 'est2' and abs(est2 - fixed_value) > tol:
            continue
        var_value = est2 if variable_key == 'est2' else est1
        result.setdefault(var_value, []).append(value)
    macro = {}
    for var_value, values in result.items():
        arr = np.asarray(values, dtype=float)
        macro[var_value] = float(np.nanmean(arr)) if values else np.nan
    return dict(sorted(macro.items(), key=lambda item: item[0]))


def prepare_dataframe(series: Dict[float, float], index_name: str, value_name: str) -> pd.DataFrame:
    df = pd.DataFrame({index_name: list(series.keys()), value_name: list(series.values())})
    df.set_index(index_name, inplace=True)
    return df


def bar_line_plot(series: Dict[float, float],
                  title: str,
                  xlabel: str,
                  ylabel: str,
                  out_prefix: Path,
                  dpi: int,
                  true_value: float | None = None,
                  *,
                  show_title: bool,
                  show_xlabel: bool,
                  tick_fontsize: float,
                  axis_label_fontsize: float,
                  title_fontsize: float,
                  bar_value_fontsize: float,
                  y_axis_padding: float) -> None:
    if not series:
        return
    x_vals = list(series.keys())
    y_vals = list(series.values())
    x_pos = np.arange(len(x_vals))

    plt.rcParams.update({
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'font.size': 11,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.35,
        'figure.autolayout': True,
    })

    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(x_vals)), 4.5))
    bars = ax.bar(x_pos, y_vals, color='#7fb3d5', edgecolor='black', linewidth=0.6, alpha=0.8)
    ax.plot(x_pos, y_vals, color='#1f618d', marker='o', linewidth=2.0)

    finite_vals = [v for v in y_vals if np.isfinite(v)]

    for xpos, bar, val in zip(x_pos, bars, y_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=bar_value_fontsize)

    if finite_vals:
        pad = max(0.0, y_axis_padding)
        ymin = min(finite_vals + [0.0])
        ymax = max(finite_vals) + pad
        if ymax <= ymin:
            ymax = ymin + max(pad, 1.0)
        ax.set_ylim(ymin, ymax)

    tol = 1e-6
    labels = []
    for v in x_vals:
        label = format_float(v)
        if true_value is not None and abs(v - true_value) <= tol:
            label = f"{label} (True)"
        labels.append(label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    if show_title:
        ax.set_title(title, fontsize=title_fontsize)
    else:
        ax.set_title('')
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    else:
        ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir or (args.out_root / '_summary_robustness')
    save_dir.mkdir(parents=True, exist_ok=True)

    records = load_best_scores(args.out_root, args.metric, target_algos=['algo1', 'algo2'])
    if not records:
        raise SystemExit(f'[ERR] No matching runs found under: {args.out_root}')

    show_title = args.show_title or args.show_title_xlabel
    show_xlabel = args.show_xlabel or args.show_title_xlabel

    rho0_true = 0.10
    rho1_true = 0.20
    scenarios = [
        ('algo1', 'est1', rho0_true, 'est2'),
        ('algo1', 'est2', rho1_true, 'est1'),
        ('algo2', 'est1', rho0_true, 'est2'),
        ('algo2', 'est2', rho1_true, 'est1'),
    ]

    for algo, fixed_key, fixed_value, variable_key in scenarios:
        series = aggregate_macro(records, algo, fixed_key, fixed_value, variable_key)
        display_algo = ALGO_DISPLAY_MAP.get(algo, algo)
        fixed_label = '0' if fixed_key == 'est1' else '1'
        variable_label = '1' if variable_key == 'est2' else '0'
        true_variable = rho1_true if variable_key == 'est2' else rho0_true
        xlabel = fr'$\hat{{\rho}}_{variable_label}$'
        title = (
            fr'{display_algo} — $\hat{{\rho}}_{fixed_label} = {fixed_value:.2f}$ '
            fr'(true $\rho_0 = {rho0_true:.2f},\; \rho_1 = {rho1_true:.2f}$)'
        )
        df = prepare_dataframe(series, variable_key, f'{args.metric}_macro')
        out_tbl = save_dir / 'tables' / f'{display_algo}_{fixed_key}_{format_float(fixed_value)}__vary_{variable_key}'
        out_tbl.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{out_tbl}.csv')
        try:
            df.to_latex(f'{out_tbl}.tex', float_format='%.2f')
        except Exception:
            pass
        out_fig = save_dir / 'figs' / f'{display_algo}_{fixed_key}_{format_float(fixed_value)}__vary_{variable_key}'
        bar_line_plot(
            series,
            title,
            xlabel,
            f'{args.metric} (macro %)',
            out_fig,
            args.dpi,
            true_value=true_variable,
            show_title=show_title,
            show_xlabel=show_xlabel,
            tick_fontsize=args.tick_fontsize,
            axis_label_fontsize=args.axis_label_fontsize,
            title_fontsize=args.title_fontsize,
            bar_value_fontsize=args.bar_value_fontsize,
            y_axis_padding=args.y_axis_padding,
        )

    readme_path = save_dir / 'README.txt'
    readme_path.write_text(
        "This folder contains robustness bar+line charts aggregated across estimation settings.\n"
        "Tables reside in tables/*.csv (and *.tex when available); figures are saved as PNG/PDF pairs.\n",
        encoding='utf-8'
    )

    print(f'[OK] Robustness summary saved to: {save_dir.resolve()}')


if __name__ == '__main__':
    main()
