#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize Qwen-eval outputs **by metric** and **by base model**, with clean charts/tables.

Differences vs summarize_qwen_eval.py:
- For each base model and for **each metric** (e.g., pass@1, pass@8, acc, total_acc),
  generate an individual table and figure that compare **all algorithms** for that base.
- Provide two extra comparison figures per base with fuzzy patterns:
  (G1) base, *oracle*, *manual_grpo*, *algo1*, *manual_algo2
  (G2) base, *llm_verifier*, *online*, *rule
  where `*` means fnmatch-style wildcard.
- Step selection policy:
  • By default, if an algorithm has multiple global_steps, **take the worst (min)** result per dataset.
  • For algorithms matching patterns in `--prefer_best`, **take the best (max)** result per dataset.

Input directory layout (same as run_qwen_eval_all.sge outputs):
  out_root/
    base__<base_key>/{g1,g2}/<dataset>/*metrics.json
    <run_name>__global_step_<N>/{g1,g2}/<dataset>/*metrics.json

Each *metrics.json contains (subset):
  {
    "acc": <float>,
    "total_acc": <float>,
    "pass_at_k_percent": {"1": 42.0, "8": 65.5},
    ...
  }

Usage example:
  python summarize_by_metric.py \
    --out_root /path/to/eval_qwen_test_v7 \
    --save_dir  /path/to/eval_qwen_test_v7/_summary_by_metric \
    --metrics pass@1,pass@8 \
    --prefer_best "*algo1*" "*algo2*"

The script intentionally avoids over-engineering and version-compat layers.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Pre-defined hatch patterns to differentiate bars while keeping color palette untouched.
_HATCH_PATTERNS = ['///', '\\\\', 'xxx', '---', '...', '+++', '***', 'ooo']

_ALGO_DISPLAY_MAP: List[Tuple[str, str]] = [
    ('base', 'Base'),
    ('*oracle*', 'Oracle'),
    ('*manual*grpo*', 'Noise'),
    ('*manual*algo1*', 'PGBC (Ours)'),
    ('*manual*algo2*', 'PGFC (Ours)'),
    ('*llm*verifier*', 'LV'),
    ('*rule*', 'Adds on'),
    ('*online*algo2*', 'PGFC (Ours)'),
]

_DATASET_DISPLAY_MAP: Dict[str, str] = {
    'aime24x8': 'AIME2024',
    'aime25x8': 'AIME2025',
    'amc23x8': 'AMC2023',
    'math500': 'Math500',
    'minerva_math': 'Minerva MATH',
    'olypiadbench': 'Olympiad Bench',
    'olympiadbench': 'Olympiad Bench',
}

# ------------------------------- CLI ---------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', type=Path, nargs='+', required=True,
                    help='One or more root dirs of eval outputs (from run_qwen_eval_all.sge)')
    ap.add_argument('--save_dir', type=Path, default=None,
                    help='Where to save summary outputs (default: <out_root>/_summary_by_metric)')
    ap.add_argument('--metrics', type=str, default='pass@1,pass@8',
                    help='Comma-separated metrics to summarize: e.g. pass@1,pass@8,acc,total_acc')
    ap.add_argument('--group1', nargs='*', default=['base', '*oracle*', '*manual*grpo*', '*algo1*', '*manual*algo2*'],
                    help='Patterns for the first comparison figure (fnmatch wildcards).')
    ap.add_argument('--group2', nargs='*', default=['base', '*llm*verifier*', '*online*', '*rule*', '*oracle*'],
                    help='Patterns for the second comparison figure (fnmatch wildcards).')
    ap.add_argument('--prefer_best', nargs='*', default=['*algo1*', '*algo2*', '*online*', '*oracle*', '*rule*'],
                    help='Patterns of algorithms whose best (max) result across steps is kept; others use worst (min).')
    ap.add_argument('--prefer_mean', nargs='*', default=[],
                    help='Patterns of algorithms aggregated by mean across steps.')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--show_bar_values', dest='show_bar_values', action='store_true', default=False,
                    help='Display numeric labels on each bar (default: disabled).')
    ap.add_argument('--bar_value_fontsize', type=float, default=8.0,
                    help='Font size for numeric labels drawn on bars.')
    ap.add_argument('--algo_order', nargs='*', default=['*base*', '*oracle*', '*manual*grpo*', '*manual*algo1*', '*manual*algo2*', '*llm*verifier*', '*rule*', '*online*',],
                    help='Explicit algorithm name ordering (case-insensitive). Unlisted algos follow the default sort order.')
    return ap.parse_args()

# -------------------------- Utilities --------------------------------

# Patterns to infer base model segment from run name (reused idea, simplified)
_BASE_PATTERNS = [
    r'qwen3(?:[.\-]\d+)?(?:-math)?(?:-[\d.]+b)?', r'qwen2[.\-]?5(?:-math)?(?:-[\d.]+b)?', r'qwen1[.\-]?5(?:-math)?(?:-[\d.]+b)?',
    r'llama-?3(?:[.\-]\d+)?(?:-[\d.]+b)?', r'deepseek(?:-r1(?:-distill)?(?:-qwen)?(?:-[\d.]+b)?)?', r'internlm(?:-math)?(?:-[\d.]+b)?',
    r'mathstral(?:-[\d.]+b)?', r'mistral(?:-[\d.]+b)?', r'gemma(?:-[\d.]+b)?', r'phi-?\d(?:-[\d.]+b)?'
]
_FALLBACK_KEYS = ['qwen3', 'qwen2.5', 'qwen25', 'qwen1.5', 'llama3', 'llama', 'deepseek', 'internlm', 'mathstral', 'mistral', 'gemma', 'phi']


_STEP_RE = re.compile(r'__global_step_(\d+)$', re.IGNORECASE)


def _strip_seps(s: str) -> str:
    s = re.sub(r'[_\-+./()\[\]{}]+', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

_STEP_RE = re.compile(r'__global_step_(\d+)$', re.IGNORECASE)

def infer_base_and_algo(run_name: str) -> Tuple[str, str]:
    """Heuristic split: [base token] + [the rest] => (base_display, algo_name).
    Example: 'qwen25-math-1_5b-noise_rlvr__global_step_5000' -> ('qwen25-math-1_5b', 'noise_rlvr__global_step_5000')
    """
    s = run_name
    s_lower = s.lower()
    matches = []
    for pat in _BASE_PATTERNS:
        for m in re.finditer(pat, s_lower):
            matches.append(m.span())
    if matches:
        # choose longest span
        (i, j) = max(matches, key=lambda x: x[1] - x[0])
        base_display = s[i:j]
    else:
        key = next((k for k in _FALLBACK_KEYS if k in s_lower), None)
        base_display = s[s_lower.find(key): s_lower.find(key) + len(key)] if key else 'unknown'
    algo_name = re.sub(re.escape(base_display), '', s, flags=re.IGNORECASE)
    algo_name = _strip_seps(algo_name)
    if not algo_name:
        algo_name = 'base'
    algo_raw = re.sub(re.escape(base_display), '', s, flags=re.IGNORECASE)
    # ⚠️ 先在“原始未清洗”的串上剥 step
    m = _STEP_RE.search(algo_raw)
    if m:
        algo_raw = _STEP_RE.sub('', algo_raw)
    # 再做归一化清洗
    algo_name = _strip_seps(algo_raw) or 'base'
    return (base_display, algo_name)


def split_algo_and_step(algo_name: str) -> Tuple[str, int | None]:
    """Return (algo_label_without_step, step_id or None)."""
    m = _STEP_RE.search(algo_name)
    if m:
        step = int(m.group(1))
        base_algo = _STEP_RE.sub('', algo_name)
        return (base_algo.strip() or 'base', step)
    return (algo_name or 'base', None)


# ------------------------ Load results --------------------------------

def _find_latest_metrics_json(dataset_dir: Path) -> Path | None:
    cand = sorted(dataset_dir.glob('*metrics.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


def load_results(out_roots: Sequence[Path]) -> pd.DataFrame:
    """Return tidy DataFrame with columns:
        base, algo, step, dataset, metric, value
    """
    rows: List[Dict] = []
    for out_root in out_roots:
        out_root = Path(out_root)
        if not out_root.exists():
            print(f'[WARN] out_root not found, skipped: {out_root}')
            continue
        for run_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
            run = run_dir.name
            for g in ('g1', 'g2'):
                gdir = run_dir / g
                if not gdir.exists():
                    continue
                for ds_dir in sorted([p for p in gdir.iterdir() if p.is_dir()]):
                    dataset = ds_dir.name
                    mpath = _find_latest_metrics_json(ds_dir)
                    if not mpath:
                        continue
                    try:
                        m = json.loads(mpath.read_text(encoding='utf-8'))
                    except Exception:
                        continue
                    (base_display, algo_raw) = infer_base_and_algo(run)
                    (algo_label, step) = split_algo_and_step(algo_raw)
                    step = None  # 如果真要保留 step，可在 infer 里一并返回
                    # extract metrics of interest
                    # normalize: allowed metrics are acc, total_acc, and pass@K
                    pass_at = m.get('pass_at_k_percent') or {}
                    # flat set of metrics present
                    metrics_found = {
                        'acc': m.get('acc', np.nan),
                        'total_acc': m.get('total_acc', np.nan),
                    }
                    for k_str, v in pass_at.items():
                        if v is None:
                            continue
                        metrics_found[f'pass@{k_str}'] = float(v)
                    for (metric, value) in metrics_found.items():
                        try:
                            val = float(value)
                        except Exception:
                            val = np.nan
                        rows.append({
                            'base': base_display,
                            'algo': algo_label.lower(),  # use lower for stable matching
                            'step': step if step is not None else -1,
                            'dataset': dataset,
                            'metric': metric.lower(),
                            'value': val,
                        })
    df = pd.DataFrame(rows)
    return df

# ---------------------- Selection & Pivot ------------------------------

def _matches_any(name: str, patterns: List[str]) -> bool:
    name = name.lower()
    for pat in patterns:
        if fnmatch(name, pat.lower()):
            return True
    return False

def reduce_steps(
    df_raw: pd.DataFrame,
    prefer_best: List[str] | None = None,
    prefer_mean: List[str] | None = None,
) -> pd.DataFrame:
    """
    折叠同一 (base, algo, dataset, metric) 的多 step：
      - 命中 prefer_mean -> 取 mean
      - 否则命中 prefer_best -> 取 max
      - 其他 -> 取 min
    返回不含 step 的 DataFrame。
    """
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    df["__algo_lc"] = df["algo"].str.lower()
    pats_mean = [p.lower() for p in (prefer_mean or [])]
    pats_best = [p.lower() for p in (prefer_best or [])]

    def _tag(algo_lc: str) -> str:
        if any(fnmatch(algo_lc, pat) for pat in pats_mean):
            return "mean"
        if any(fnmatch(algo_lc, pat) for pat in pats_best):
            return "max"
        return "min"

    df["__agg"] = df["__algo_lc"].apply(_tag)
    group_keys = [k for k in ("base", "algo", "dataset", "metric") if k in df.columns]

    part_mean = (
        df[df["__agg"] == "mean"].groupby(group_keys, as_index=False)["value"].mean()
    )
    part_max = (
        df[df["__agg"] == "max"].groupby(group_keys, as_index=False)["value"].max()
    )
    part_min = (
        df[df["__agg"] == "min"].groupby(group_keys, as_index=False)["value"].min()
    )
    out = pd.concat([part_mean, part_max, part_min], ignore_index=True)
    return out


def _canonical_algo(name: str) -> str:
    """Normalize algorithm names for matching (ignore case and punctuation)."""
    return re.sub(r'[^a-z0-9]+', '', (name or '').lower())


def reorder_algos(pvt: pd.DataFrame, algo_order: List[str] | None) -> pd.DataFrame:
    """Reorder algorithm rows according to an explicit list (case-insensitive)."""
    if not algo_order:
        return pvt
    order_terms = [a for a in algo_order if a]
    if not order_terms:
        return pvt

    algos = list(pvt.index)
    base_idx = {name: idx for idx, name in enumerate(algos)}

    def _order_rank(name: str) -> int:
        canon = _canonical_algo(name)
        for idx, pattern in enumerate(order_terms):
            canon_pat = _canonical_algo(pattern)
            if canon and canon == canon_pat:
                return idx
            if fnmatch(name.lower(), pattern.lower()):
                return idx
        return len(order_terms)

    algos_sorted = sorted(
        algos,
        key=lambda name: (_order_rank(name), base_idx[name])
    )
    return pvt.loc[algos_sorted]


def map_algo_display(name: str) -> str:
    lname = (name or '').lower()
    for (pattern, display) in _ALGO_DISPLAY_MAP:
        if fnmatch(lname, pattern.lower()):
            return display
    return name


def apply_display_labels(pvt: pd.DataFrame) -> pd.DataFrame:
    if pvt.empty:
        return pvt
    labels = [map_algo_display(idx) for idx in pvt.index]
    seen: Dict[str, int] = {}
    final_labels: List[str] = []
    for disp, original in zip(labels, pvt.index):
        count = seen.get(disp, 0)
        if count == 0:
            final_labels.append(disp)
        else:
            final_labels.append(f"{disp} ({original})")
        seen[disp] = count + 1
    out = pvt.copy()
    out.index = final_labels
    return out


def map_dataset_display(name: str) -> str:
    key = (name or '').lower() if isinstance(name, str) else ''
    return _DATASET_DISPLAY_MAP.get(key, name)


def apply_dataset_labels(pvt: pd.DataFrame) -> pd.DataFrame:
    if pvt.empty:
        return pvt
    rename: Dict[str, str] = {}
    for col in pvt.columns:
        if isinstance(col, str):
            mapped = map_dataset_display(col)
            if mapped != col:
                rename[col] = mapped
    if not rename:
        return pvt
    return pvt.rename(columns=rename)


def pivot_for_base_metric(df: pd.DataFrame, base: str, metric: str) -> pd.DataFrame:
    sub = df[(df['base'].str.lower() == base.lower()) & (df['metric'].str.lower() == metric.lower())]
    if sub.empty:
        return pd.DataFrame()
    pvt = sub.pivot_table(index='algo', columns='dataset', values='value', aggfunc='first')
    # sort algos by macro avg desc
    pvt['macro_avg'] = pvt.mean(axis=1, skipna=True)
    pvt.sort_values('macro_avg', ascending=False, inplace=True)
    return pvt

# ---------------------------- Plotting --------------------------------

def _paper_rc(dpi: int = 200):
    plt.rcParams.update({
        'figure.dpi': dpi, 'savefig.dpi': dpi,
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.35,
        'figure.autolayout': True,
    })


def plot_grouped_bars(
    pvt: pd.DataFrame,
    title: str,
    out_prefix: Path,
    show_bar_values: bool,
    bar_value_fontsize: float,
):
    if pvt.empty:
        return
    _paper_rc()
    avg_label = 'Average'
    if 'macro_avg' in pvt.columns:
        data = pd.concat(
            [pvt.drop(columns=['macro_avg'], errors='ignore'),
             pvt[['macro_avg']].rename(columns={'macro_avg': avg_label})],
            axis=1,
        )
    else:
        data = pvt.copy()
    algos = list(data.index)
    ds = [str(c) for c in data.columns]
    if not ds:
        return
    x = np.arange(len(ds))
    n_algo = max(1, len(algos))
    width = min(0.8 / max(n_algo, 1), 0.18)
    fig_width = max(8.0, 1.2 * len(ds))
    fig_height = max(3.8, 0.35 * n_algo + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    arr = data.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size:
        span = finite_vals.max() - finite_vals.min()
        if not np.isfinite(span) or span <= 0:
            span = abs(finite_vals.max()) or 1.0
        value_offset = max(0.01 * span, 0.2)
    else:
        value_offset = 0.2
    prop_cycle = plt.rcParams.get('axes.prop_cycle')
    color_cycle = []
    if prop_cycle is not None:
        color_cycle = prop_cycle.by_key().get('color', [])
    if not color_cycle:
        color_cycle = [f'C{i}' for i in range(len(algos) or 10)]

    for (i, algo) in enumerate(algos):
        y = data.loc[algo].astype(float).values
        hatch = _HATCH_PATTERNS[i % len(_HATCH_PATTERNS)]
        color = color_cycle[i % len(color_cycle)]
        ax.bar(
            x + i * width,
            y,
            width,
            label=algo,
            hatch=hatch,
            color='white',
            edgecolor=color,
            linewidth=0.8,
        )
        if not show_bar_values:
            continue
        for (j, v) in enumerate(y):
            if not np.isfinite(v):
                continue
            text_y = v + value_offset if v >= 0 else v - value_offset
            va = 'bottom' if v >= 0 else 'top'
            ax.text(
                x[j] + i * width,
                text_y,
                f'{v:.1f}',
                ha='center',
                va=va,
                fontsize=bar_value_fontsize,
            )
    ax.set_xticks(x + width * (n_algo - 1) / 2)
    ax.set_xticklabels(ds, rotation=35, ha='right')
    ax.set_ylabel('Score (%)')
    # ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncols=2, frameon=False, fontsize=9)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)

# ----------------------- Group comparisons ----------------------------

def filter_algos_by_patterns(pvt: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    if pvt.empty:
        return pvt
    keep = [a for a in pvt.index if _matches_any(a, patterns)]
    sub = pvt.loc[keep]
    # ensure 'base' is always included even if not matched by wildcard
    if 'base' in pvt.index and 'base' not in sub.index and _matches_any('base', patterns):
        sub = pd.concat([pvt.loc[['base']], sub])
    return sub

# ------------------------------ I/O -----------------------------------

def save_table(df: pd.DataFrame, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Always save raw CSV (no styling in CSV)
    df.to_csv(f"{out_prefix}.csv")

    # Build LaTeX with bolded column-wise maxima, and a vertical bar before 'macro_avg'
    try:
        cols = list(df.columns)

        # Compute per-column numeric maxima (ignoring NaN). Non-numeric columns will be skipped.
        max_by_col: Dict[str, float] = {}
        for c in cols:
            col_vals = pd.to_numeric(df[c], errors='coerce')
            if np.isfinite(col_vals).any():
                max_by_col[c] = float(np.nanmax(col_vals))
            else:
                max_by_col[c] = np.nan  # no numeric data -> no bolding

        # Per-column formatter: 1 decimal; bold if equals the (finite) max for that column (ties allowed)
        def make_formatter(cname: str):
            cmax = max_by_col.get(cname, np.nan)
            def _fmt(x):
                # Render as '--' for NaN / non-finite; otherwise 1-decimal
                try:
                    vx = float(x)
                except Exception:
                    return '--' if (isinstance(x, float) and np.isnan(x)) else str(x)
                if not np.isfinite(vx):
                    return '--'
                txt = f"{vx:.1f}"
                if np.isfinite(cmax) and np.isclose(vx, cmax, rtol=0.0, atol=1e-9):
                    return r"\textbf{" + txt + "}"
                return txt
            return _fmt

        formatters = {c: make_formatter(c) for c in cols}

        # Column format: left for index, right-aligned for data; add a vertical bar before 'macro_avg'
        colfmt = 'l'
        if 'macro_avg' in cols:
            avg_idx = cols.index('macro_avg')
            for j in range(len(cols)):
                if j == avg_idx:
                    colfmt += '|'
                colfmt += 'r'
        else:
            colfmt += 'r' * len(cols)

        tex = df.to_latex(na_rep='--', escape=False, formatters=formatters, column_format=colfmt)
        Path(f"{out_prefix}.tex").write_text(tex, encoding='utf-8')
    except Exception:
        # Fallback (no bolding / no custom colfmt)
        try:
            tex = df.to_latex(float_format='%.1f', na_rep='--', escape=False)
            Path(f"{out_prefix}.tex").write_text(tex, encoding='utf-8')
        except Exception:
            pass

# ----------------------------- Main -----------------------------------

def main():
    args = parse_args()
    out_roots = list(args.out_root)
    if not out_roots:
        raise SystemExit('[ERR] No out_root provided.')
    first_root = out_roots[0]
    save_dir = args.save_dir or (first_root / '_summary_by_metric')
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip().lower() for m in args.metrics.split(',') if m.strip()]

    df_raw = load_results(out_roots)
    if df_raw.empty:
        raise SystemExit(f'[ERR] No metrics found under: {args.out_root}')

    df = reduce_steps(df_raw, prefer_best=args.prefer_best, prefer_mean=args.prefer_mean)

    bases = sorted(df['base'].unique(), key=lambda s: s.lower())
    avg_records: List[Dict[str, float]] = []

    for base in bases:
        for metric in metrics:
            pvt = pivot_for_base_metric(df, base=base, metric=metric)
            if pvt.empty:
                continue
            pvt = reorder_algos(pvt, args.algo_order)
            if 'macro_avg' in pvt.columns:
                for algo, val in pvt['macro_avg'].items():
                    avg_records.append({
                        'metric': metric,
                        'base': base,
                        'algo': algo,
                        'avg': float(val) if np.isfinite(val) else np.nan,
                    })
            pvt_display = apply_display_labels(pvt)
            pvt_display = apply_dataset_labels(pvt_display)
            # save table
            base_key = re.sub(r'[^a-zA-Z0-9]+', '-', base).strip('-').lower()
            out_tbl = save_dir / 'tables' / 'by_base_metric' / f'{base_key}__{metric}'
            save_table(pvt_display, out_tbl)
            # plot all algos for this base & metric
            title = f"{base} — {metric} (all algos)"
            out_fig = save_dir / 'figs' / 'by_base_metric' / f'{base_key}__{metric}'
            plot_grouped_bars(pvt_display, title, out_fig, args.show_bar_values, args.bar_value_fontsize)
            # group comparisons
            for (gi, patterns) in enumerate([args.group1, args.group2], start=1):
                sub = filter_algos_by_patterns(pvt, patterns)
                if sub.empty:
                    continue
                sub = reorder_algos(sub, args.algo_order)
                sub_display = apply_display_labels(sub)
                sub_display = apply_dataset_labels(sub_display)
                gkey = f'g{gi}'
                out_tbl_g = save_dir / 'tables' / 'compare_groups' / gkey / f'{base_key}__{metric}'
                save_table(sub_display, out_tbl_g)
                title_g = f"{base} — {metric}"
                out_fig_g = save_dir / 'figs' / 'compare_groups' / gkey / f'{base_key}__{metric}'
                plot_grouped_bars(sub_display, title_g, out_fig_g, args.show_bar_values, args.bar_value_fontsize)

    if avg_records:
        avg_df = pd.DataFrame(avg_records)
        for metric in metrics:
            sub = avg_df[avg_df['metric'] == metric]
            if sub.empty:
                continue
            pvt_avg = sub.pivot_table(index='algo', columns='base', values='avg', aggfunc='first')
            if pvt_avg.empty:
                continue
            base_cols = sorted(pvt_avg.columns, key=lambda s: s.lower())
            pvt_avg = pvt_avg[base_cols]
            macro_series = pvt_avg.mean(axis=1, skipna=True)
            pvt_avg = pvt_avg.assign(__macro=macro_series)
            pvt_avg.sort_values('__macro', ascending=False, inplace=True)
            pvt_avg.drop(columns='__macro', inplace=True)
            pvt_avg = reorder_algos(pvt_avg, args.algo_order)
            pvt_avg_display = apply_display_labels(pvt_avg)
            metric_key = re.sub(r'[^a-zA-Z0-9]+', '-', metric).strip('-').lower()
            out_tbl_avg = save_dir / 'tables' / 'cross_base_avg' / f'{metric_key}'
            save_table(pvt_avg_display, out_tbl_avg)
            title_avg = f"{metric} — macro avg"
            out_fig_avg = save_dir / 'figs' / 'cross_base_avg' / f'{metric_key}'
            plot_grouped_bars(pvt_avg_display, title_avg, out_fig_avg, args.show_bar_values, args.bar_value_fontsize)

    (save_dir / 'README.txt').write_text(
        "Files:\n"
        "- tables/by_base_metric/*.csv, *.tex : per-base & per-metric tables (all algos)\n"
        "- figs/by_base_metric/*.png, *.pdf  : per-base & per-metric grouped bars (all algos)\n"
        "- tables/compare_groups/g{1,2}/*.csv, *.tex : per-base & per-metric tables for pattern groups\n"
        "- figs/compare_groups/g{1,2}/*.png, *.pdf  : per-base & per-metric grouped bars for pattern groups\n"
        "- tables/cross_base_avg/*.csv, *.tex : per-metric macro results of algorithms across bases\n"
        "- figs/cross_base_avg/*.png, *.pdf  : per-metric cross-base comparison of algorithms\n"
        "\nNotes:\n"
        "  • Default step aggregation = worst(min). Patterns in --prefer_best use best(max).\n"
        "  • pass@K is taken from 'pass_at_k_percent' if present; fallback to NaN.\n",
        encoding='utf-8')

    print(f'[OK] Summary saved to: {save_dir.resolve()}')


if __name__ == '__main__':
    main()
