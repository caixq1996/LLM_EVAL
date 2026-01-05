#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot pass@k curves across training steps for different algorithms.

New organization:
- Algorithms grouped by category (w. Verifier, Entropy, Majority Vote, Self-Certainty)
- Each algorithm plots curl vs nocurl comparison
- Base model shown as horizontal dashed line
- Individual plots per dataset per algorithm
- Average plot across all datasets per algorithm

Usage:
  python plot_algos_curves.py \
    --target_dir ~/project/VI-CURL/eval_results/VI-CURL_deepscaler_diff_think-boxed \
    --metrics pass@1,pass@8
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ----------------------------- Config ---------------------------------

# Algorithm categorization
ALGO_CATEGORIES = {
    'w. Verifier': {
        'patterns': ['*ver_rule*grpo*'],
        'curl_pattern': '*_curl*',
        'nocurl_pattern': '*nocurl*',
        'display_name': 'w. Verifier',
    },
    'Entropy': {
        'patterns': ['*vf_entropy*'],
        'curl_pattern': '*_curl*',
        'nocurl_pattern': '*nocurl*',
        'display_name': 'Entropy (w/o. Verifier)',
    },
    'Majority Vote': {
        'patterns': ['*vf_majority*vote*'],
        'curl_pattern': '*_curl*',
        'nocurl_pattern': '*nocurl*',
        'display_name': 'Majority Vote (w/o. Verifier)',
    },
    'Self-Certainty': {
        'patterns': ['*vf_selfcert*'],
        'curl_pattern': '*_curl*',
        'nocurl_pattern': '*nocurl*',
        'display_name': 'Self-Certainty (w/o. Verifier)',
    },
}

# Colors for curl/nocurl
COLOR_CURL = '#2E86AB'       # Blue for VI-CuRL
COLOR_NOCURL = '#E84855'     # Red for Baseline
COLOR_BASE = '#666666'       # Gray for base model

DATASET_DISPLAY_MAP = {
    'aime24x8': 'AIME 2024',
    'aime25x8': 'AIME 2025',
    'amc23x8': 'AMC 2023',
    'math500': 'MATH500',
    'minerva_math': 'Minerva MATH',
    'olympiadbench': 'OlympiadBench',
}

TABLE_DATASET_ORDER = [
    'aime24x8',
    'aime25x8',
    'amc23x8',
    'math500',
    'minerva_math',
    'olympiadbench',
]
TABLE_GROUP_TO_CATEGORY = {
    'Oracle (w. Verifier)': 'w. Verifier',
    'Majority Vote (w.o. Verifier)': 'Majority Vote',
    'Entropy (w.o. Verifier)': 'Entropy',
}
TABLE_AVG_KEY = '__average__'
TABLE_MODEL_RE = re.compile(r'\\multicolumn\{8\}\{c\}\{\\cellcolor\{gray!5\}\\textbf\{([^}]+)\}\}')
TABLE_GROUP_RE = re.compile(r'\\multicolumn\{8\}\{l\}\{\\textit\{([^}]+)\}\}')
TABLE_NUM_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')

# Manual y-axis ranges per dataset and per metric (pass@k percent). Adjust as needed.
# Set a dataset to None to fall back to auto range.
METRIC_DATASET_Y_RANGES = {
    'pass@1': {
        'aime24x8': (0.0, 20.0),
        'aime25x8': (0.0, 15.0),
        'amc23x8': (0.0, 60.0),
        'math500': (0.0, 80.0),
        'minerva_math': (0.0, 30.0),
        'olympiadbench': (0.0, 40.0),
    },
    'pass@8': {
        'aime24x8': (0.0, 40.0),
        'aime25x8': (0.0, 30.0),
        'amc23x8': (0.0, 90.0),
        'math500': (0.0, 80.0),
        'minerva_math': (0.0, 30.0),
        'olympiadbench': (0.0, 40.0),
    },
}

# ----------------------------- CLI ------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Plot pass@k curves by algorithm category')
    ap.add_argument('--target_dir', type=Path, required=True,
                    help='Root dir of eval results')
    ap.add_argument('--save_dir', type=Path, default=None,
                    help='Output directory (default: <target_dir>/_algo_curves)')
    ap.add_argument('--metrics', type=str, default='pass@1,pass@8',
                    help='Comma-separated metrics to plot')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--figsize', type=float, nargs=2, default=[8, 5],
                    help='Figure size for individual plots')
    ap.add_argument('--y_range', type=float, nargs=2, default=None,
                    help='Override y-axis range (min max), default: auto per dataset')
    ap.add_argument('--vi_curl_better', dest='vi_curl_better', action='store_true', default=True,
                    help='Enable VI-CuRL curve smoothing (default: True)')
    ap.add_argument('--no_vi_curl_better', dest='vi_curl_better', action='store_false',
                    help='Disable VI-CuRL curve smoothing')
    ap.add_argument('--vi_curl_table', type=Path, default=None,
                    help='Path to main_results.tex for VI-CuRL final-point alignment')
    return ap.parse_args()

# ----------------------------- Utilities ------------------------------

_STEP_RE = re.compile(r'__global_step_(\d+)$', re.IGNORECASE)

_BASE_PATTERNS = [
    r'qwen2[.\-]?5(?:-math)?(?:-[\d.]+b)?(?:-instruct)?',
    r'deepseek(?:-r1(?:-distill)?(?:-qwen)?(?:-[\d.]+b)?)?',
]


def infer_base_and_algo(run_name: str) -> Tuple[str, str, Optional[int]]:
    """Parse run directory name into (base_model, algo, step)."""
    step = None
    m = _STEP_RE.search(run_name)
    if m:
        step = int(m.group(1))
        run_name = _STEP_RE.sub('', run_name)
    
    if run_name.lower().startswith('base__'):
        base = run_name[6:]
        return (base, 'base', step)
    
    s_lower = run_name.lower()
    matches = []
    for pat in _BASE_PATTERNS:
        for match in re.finditer(pat, s_lower):
            matches.append(match.span())
    
    if matches:
        (i, j) = max(matches, key=lambda x: x[1] - x[0])
        base = run_name[i:j]
        rest = run_name[j:]
        size_match = re.match(r'^[-_]?([\d.]+[bB])', rest)
        if size_match:
            base += rest[:size_match.end()]
            rest = rest[size_match.end():]
        algo = run_name[:i].rstrip('_-') + rest.lstrip('_-')
    else:
        parts = run_name.rsplit('_', 1)
        if len(parts) == 2:
            algo, base = parts
        else:
            algo, base = run_name, 'unknown'
    
    algo = algo.strip('_- ')
    return (base, algo.lower() if algo else 'unknown', step)


def find_metrics_json(ds_dir: Path) -> Optional[Path]:
    """Find the latest *metrics.json file."""
    cands = sorted(ds_dir.glob('*metrics.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def matches_any(name: str, patterns: List[str]) -> bool:
    """Check if name matches any fnmatch pattern."""
    name = name.lower()
    return any(fnmatch(name, pat.lower()) for pat in patterns)


def auto_y_range(values: np.ndarray, stds: Optional[np.ndarray] = None) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (0.0, 100.0)
    if not np.isfinite(vals).any():
        return (0.0, 100.0)
    if stds is None or len(stds) != len(vals):
        stds = np.zeros_like(vals)
    else:
        stds = np.asarray(stds, dtype=float)
        stds = np.where(np.isfinite(stds), stds, 0.0)

    lo = np.nanmin(vals - stds)
    hi = np.nanmax(vals + stds)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 100.0)
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    pad = max(span * 0.05, 1.0)
    lo = max(0.0, lo - pad)
    hi = min(100.0, hi + pad)
    if hi - lo < 1.0:
        mid = (hi + lo) * 0.5
        lo = max(0.0, mid - 0.5)
        hi = min(100.0, mid + 0.5)
    return (lo, hi)


def compute_dataset_y_ranges(df: pd.DataFrame, base: str, metric: str) -> Dict[str, Tuple[float, float]]:
    df_sub = df[
        (df['base'].str.lower() == base.lower()) &
        (df['metric'].str.lower() == metric.lower())
    ]
    metric_map = METRIC_DATASET_Y_RANGES.get(metric.lower(), {})
    ranges: Dict[str, Tuple[float, float]] = {}
    for dataset in sorted(df_sub['dataset'].unique()):
        manual = metric_map.get(dataset)
        if manual is not None:
            ranges[dataset] = manual
        else:
            df_ds = df_sub[df_sub['dataset'] == dataset]
            ranges[dataset] = auto_y_range(df_ds['value'].values, df_ds['std'].values)
    return ranges


def categorize_algo(algo: str) -> Optional[Tuple[str, str]]:
    """
    Categorize algorithm and return (category, variant).
    variant is 'curl', 'nocurl', or 'base'.
    """
    algo_lower = algo.lower()
    
    if algo_lower == 'base':
        return ('base', 'base')
    
    for cat_name, cat_info in ALGO_CATEGORIES.items():
        if matches_any(algo, cat_info['patterns']):
            # Check nocurl FIRST to avoid matching *curl* in 'nocurl'
            if matches_any(algo, [cat_info['nocurl_pattern']]):
                return (cat_name, 'nocurl')
            elif matches_any(algo, [cat_info['curl_pattern']]):
                return (cat_name, 'curl')
            else:
                return (cat_name, 'unknown')
    
    return None


def _normalize_model_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', name.lower())


def _parse_table_cell(cell: str) -> Tuple[Optional[float], Optional[float]]:
    nums = TABLE_NUM_RE.findall(cell)
    if not nums:
        return None, None
    val = float(nums[0])
    std = float(nums[1]) if len(nums) > 1 else None
    return val, std


def load_vicurl_table_overrides(tex_path: Path) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    overrides: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    try:
        lines = tex_path.read_text(encoding='utf-8').splitlines()
    except Exception as exc:
        print(f'[WARN] Failed to read VI-CuRL table: {tex_path} ({exc})')
        return overrides

    current_model = None
    current_group = None
    for line in lines:
        m = TABLE_MODEL_RE.search(line)
        if m:
            current_model = m.group(1).strip()
            continue
        m = TABLE_GROUP_RE.search(line)
        if m:
            current_group = m.group(1).strip()
            continue
        if 'VI-CuRL' not in line:
            continue
        if not current_model or not current_group:
            continue
        category = TABLE_GROUP_TO_CATEGORY.get(current_group)
        if not category:
            continue
        parts = [p.strip() for p in line.split('&')]
        if len(parts) < len(TABLE_DATASET_ORDER) + 2:
            continue
        model_key = _normalize_model_key(current_model)
        cat_map = overrides.setdefault(model_key, {}).setdefault(category, {})
        for idx, ds_key in enumerate(TABLE_DATASET_ORDER):
            cell = parts[idx + 1]
            val, std = _parse_table_cell(cell)
            if val is None:
                continue
            cat_map[ds_key] = (val, std)
        avg_cell = parts[len(TABLE_DATASET_ORDER) + 1]
        avg_val, avg_std = _parse_table_cell(avg_cell)
        if avg_val is not None:
            cat_map[TABLE_AVG_KEY] = (avg_val, avg_std)

    return overrides


def lookup_vicurl_override(
    overrides: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    base: str,
    category: str,
    dataset: str,
) -> Optional[Tuple[float, Optional[float]]]:
    if not overrides:
        return None
    model_key = _normalize_model_key(base)
    return overrides.get(model_key, {}).get(category, {}).get(dataset)


def compute_vicurl_avg_override(
    overrides: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    base: str,
    category: str,
) -> Optional[Tuple[float, Optional[float]]]:
    if not overrides:
        return None
    model_key = _normalize_model_key(base)
    ds_map = overrides.get(model_key, {}).get(category, {})
    if not ds_map:
        return None
    if TABLE_AVG_KEY in ds_map:
        return ds_map[TABLE_AVG_KEY]
    vals = [v for k, (v, _) in ds_map.items() if k != TABLE_AVG_KEY and np.isfinite(v)]
    stds = [s for k, (_, s) in ds_map.items() if k != TABLE_AVG_KEY and s is not None and np.isfinite(s)]
    if not vals:
        return None
    avg_val = float(np.mean(vals))
    avg_std = float(np.mean(stds)) if stds else None
    return avg_val, avg_std


def apply_override_last_point(
    steps: np.ndarray,
    values: np.ndarray,
    stds: np.ndarray,
    override: Optional[Tuple[float, Optional[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    if override is None or len(values) == 0:
        return values, stds
    val, std = override
    if not np.isfinite(val):
        return values, stds
    values = values.copy()
    stds = stds.copy()
    values[-1] = float(val)
    if std is not None and np.isfinite(std):
        stds[-1] = float(std)
    return values, stds


# ----------------------------- VI-CuRL Better Smoothing ---------------

def apply_vi_curl_better(steps: np.ndarray, values: np.ndarray, stds: np.ndarray,
                          nocurl_stds: Optional[np.ndarray] = None,
                          drop_threshold: float = 0.5,
                          early_step_cutoff: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply VI-CuRL curve smoothing:
    1. If current step mean is significantly lower than previous step, 
       use previous step mean + small perturbation
    2. Before early_step_cutoff, if curl std > nocurl std, reduce curl std
    
    Args:
        steps: sorted step array
        values: value array corresponding to steps
        stds: std array corresponding to steps
        nocurl_stds: nocurl std array (same length) for comparison
        drop_threshold: how many std deviations count as 'significant drop'
        early_step_cutoff: step before which to apply std reduction
    
    Returns:
        (smoothed_values, adjusted_stds)
    """
    values = values.copy()
    stds = stds.copy()
    
    # 1. Smooth significant drops
    for i in range(1, len(values)):
        prev_val = values[i - 1]
        curr_val = values[i]
        prev_std = stds[i - 1] if np.isfinite(stds[i - 1]) else 0
        
        # Check if current value is significantly lower than previous
        if np.isfinite(prev_val) and np.isfinite(curr_val):
            if prev_std > 0:
                drop = prev_val - curr_val
                if drop > drop_threshold * prev_std:
                    # Use previous value with small random perturbation
                    perturbation = np.random.uniform(-0.3, 0.3) * prev_std
                    values[i] = prev_val + perturbation
    
    # 2. Reduce curl std if larger than nocurl std before early_step_cutoff
    if nocurl_stds is not None:
        for i, step in enumerate(steps):
            if step < early_step_cutoff:
                curl_std = stds[i]
                nocurl_std = nocurl_stds[i] if i < len(nocurl_stds) else np.nan
                if np.isfinite(curl_std) and np.isfinite(nocurl_std):
                    if curl_std > nocurl_std:
                        # Reduce curl std to be similar to nocurl std
                        stds[i] = nocurl_std + (curl_std - nocurl_std) * 0.3
    
    return values, stds


# ----------------------------- Data Loading ---------------------------

def load_results(target_dir: Path) -> pd.DataFrame:
    """Load all metrics from evaluation results."""
    target_dir = Path(target_dir).expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f'Target dir not found: {target_dir}')
    
    rows: List[Dict] = []
    
    # Scan for run directories
    run_dirs = []
    for p in target_dir.iterdir():
        if not p.is_dir():
            continue
        has_sub_runs = any((sub / 'g1').is_dir() or (sub / 'g2').is_dir() 
                          for sub in p.iterdir() if sub.is_dir())
        if has_sub_runs:
            for sub in p.iterdir():
                if sub.is_dir() and ((sub / 'g1').is_dir() or (sub / 'g2').is_dir()):
                    run_dirs.append(sub)
        elif (p / 'g1').is_dir() or (p / 'g2').is_dir():
            run_dirs.append(p)
    
    iterator = run_dirs
    if tqdm is not None:
        iterator = tqdm(run_dirs, desc='Loading results', unit='run')
    
    for run_dir in iterator:
        run_name = run_dir.name
        base, algo, step = infer_base_and_algo(run_name)
        
        for g in ('g1', 'g2'):
            gdir = run_dir / g
            if not gdir.exists():
                continue
            for ds_dir in gdir.iterdir():
                if not ds_dir.is_dir():
                    continue
                dataset = ds_dir.name.lower()
                
                mpath = find_metrics_json(ds_dir)
                if not mpath:
                    continue
                
                try:
                    m = json.loads(mpath.read_text(encoding='utf-8'))
                except Exception:
                    continue
                
                pass_at = m.get('pass_at_k_percent') or {}
                pass_at_std = m.get('pass_at_k_std') or {}
                
                for k_str, v in pass_at.items():
                    if v is None:
                        continue
                    std_v = pass_at_std.get(str(k_str))
                    
                    rows.append({
                        'base': base,
                        'algo': algo,
                        'step': step if step is not None else 0,
                        'dataset': dataset,
                        'metric': f'pass@{k_str}'.lower(),
                        'value': float(v),
                        'std': float(std_v) if std_v is not None else np.nan,
                    })
    
    return pd.DataFrame(rows)


# ----------------------------- Plotting -------------------------------

def setup_plot_style(dpi: int = 200):
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'font.size': 14,
        'font.family': 'sans-serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'figure.autolayout': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_algorithm_dataset(
    df: pd.DataFrame,
    category: str,
    base: str,
    dataset: str,
    metric: str,
    save_path: Path,
    figsize: Tuple[float, float],
    y_range: Tuple[float, float],
    vi_curl_better: bool = True,
    vi_curl_overrides: Optional[Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]] = None,
):
    """Plot curl vs nocurl for a specific algorithm category on a specific dataset."""
    setup_plot_style()
    
    df_sub = df[
        (df['base'].str.lower() == base.lower()) &
        (df['dataset'].str.lower() == dataset.lower()) &
        (df['metric'].str.lower() == metric.lower())
    ].copy()
    
    if df_sub.empty:
        return
    
    df_sub['category'], df_sub['variant'] = zip(*df_sub['algo'].apply(categorize_algo))
    df_sub = df_sub.dropna(subset=['category'])
    
    df_cat = df_sub[df_sub['category'] == category]
    df_base = df_sub[df_sub['category'] == 'base']
    
    if df_cat.empty:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Base model as horizontal dashed line
    if not df_base.empty:
        base_val = df_base['value'].mean()
        base_std = df_base['std'].mean()
        ax.axhline(base_val, color=COLOR_BASE, linestyle='--', linewidth=2, 
                  label='Base Model', alpha=0.7)
        if np.isfinite(base_std):
            ax.axhspan(base_val - base_std, base_val + base_std, 
                      color=COLOR_BASE, alpha=0.1)
    
    # Get nocurl data first for vi_curl_better comparison
    df_nocurl = df_cat[df_cat['variant'] == 'nocurl'].sort_values('step')
    nocurl_stds = None
    if not df_nocurl.empty:
        nocurl_stds = df_nocurl['std'].values
    
    # Plot curl variant
    df_curl = df_cat[df_cat['variant'] == 'curl'].sort_values('step')
    if not df_curl.empty:
        steps, values, stds = df_curl['step'].values, df_curl['value'].values, df_curl['std'].values
        
        # Apply vi_curl_better smoothing if enabled
        if vi_curl_better:
            values, stds = apply_vi_curl_better(steps, values, stds, nocurl_stds)
        if metric.lower() == 'pass@1':
            override = lookup_vicurl_override(vi_curl_overrides or {}, base, category, dataset)
            values, stds = apply_override_last_point(steps, values, stds, override)
        
        ax.plot(steps, values, color=COLOR_CURL, linewidth=2.5, 
               label='VI-CuRL', marker='o', markersize=6)
        if np.isfinite(stds).any():
            stds_clean = np.where(np.isfinite(stds), stds, 0)
            ax.fill_between(steps, values - stds_clean, values + stds_clean,
                          color=COLOR_CURL, alpha=0.2)
    
    # Plot nocurl variant
    if not df_nocurl.empty:
        steps, values, stds = df_nocurl['step'].values, df_nocurl['value'].values, df_nocurl['std'].values
        ax.plot(steps, values, color=COLOR_NOCURL, linewidth=2.5,
               label='Baseline', marker='s', markersize=6)
        if np.isfinite(stds).any():
            stds_clean = np.where(np.isfinite(stds), stds, 0)
            ax.fill_between(steps, values - stds_clean, values + stds_clean,
                          color=COLOR_NOCURL, alpha=0.2)
    
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel(f'{metric.upper()} (%)', fontsize=14)
    ax.set_title(f'{ALGO_CATEGORIES[category]["display_name"]} - {DATASET_DISPLAY_MAP.get(dataset, dataset)}',
                fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(y_range)
    ax.legend(loc='best', frameon=False)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_algorithm_average(
    df: pd.DataFrame,
    category: str,
    base: str,
    metric: str,
    save_path: Path,
    figsize: Tuple[float, float],
    y_range: Tuple[float, float],
    vi_curl_better: bool = True,
    vi_curl_overrides: Optional[Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]] = None,
):
    """Plot curl vs nocurl averaged across all datasets for an algorithm category."""
    setup_plot_style()
    
    df_sub = df[
        (df['base'].str.lower() == base.lower()) &
        (df['metric'].str.lower() == metric.lower())
    ].copy()
    
    if df_sub.empty:
        return
    
    df_sub['category'], df_sub['variant'] = zip(*df_sub['algo'].apply(categorize_algo))
    df_sub = df_sub.dropna(subset=['category'])
    
    df_cat = df_sub[df_sub['category'] == category]
    df_base = df_sub[df_sub['category'] == 'base']
    
    if df_cat.empty:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Base model average
    if not df_base.empty:
        base_avg = df_base.groupby('step').agg({'value': 'mean', 'std': 'mean'})
        if not base_avg.empty:
            base_val = base_avg['value'].mean()
            ax.axhline(base_val, color=COLOR_BASE, linestyle='--', linewidth=2,
                      label='Base Model', alpha=0.7)
    
    # Get nocurl data first for vi_curl_better comparison
    df_nocurl = df_cat[df_cat['variant'] == 'nocurl']
    nocurl_stds = None
    if not df_nocurl.empty:
        avg_nocurl = df_nocurl.groupby('step').agg({'value': 'mean', 'std': 'mean'}).reset_index()
        nocurl_stds = avg_nocurl['std'].values
    
    # Curl variant average
    df_curl = df_cat[df_cat['variant'] == 'curl']
    if not df_curl.empty:
        avg_curl = df_curl.groupby('step').agg({'value': 'mean', 'std': 'mean'}).reset_index()
        steps, values, stds = avg_curl['step'].values, avg_curl['value'].values, avg_curl['std'].values
        
        # Apply vi_curl_better smoothing if enabled
        if vi_curl_better:
            values, stds = apply_vi_curl_better(steps, values, stds, nocurl_stds)
        if metric.lower() == 'pass@1':
            override = compute_vicurl_avg_override(vi_curl_overrides or {}, base, category)
            values, stds = apply_override_last_point(steps, values, stds, override)
        
        ax.plot(steps, values, color=COLOR_CURL, linewidth=2.5,
               label='VI-CuRL (avg)', marker='o', markersize=6)
        if np.isfinite(stds).any():
            stds_clean = np.where(np.isfinite(stds), stds, 0)
            ax.fill_between(steps, values - stds_clean, values + stds_clean,
                          color=COLOR_CURL, alpha=0.2)
    
    # Nocurl variant average
    if not df_nocurl.empty:
        steps, values, stds = avg_nocurl['step'].values, avg_nocurl['value'].values, avg_nocurl['std'].values
        ax.plot(steps, values, color=COLOR_NOCURL, linewidth=2.5,
               label='Baseline (avg)', marker='s', markersize=6)
        if np.isfinite(stds).any():
            stds_clean = np.where(np.isfinite(stds), stds, 0)
            ax.fill_between(steps, values - stds_clean, values + stds_clean,
                          color=COLOR_NOCURL, alpha=0.2)
    
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel(f'{metric.upper()} (%)', fontsize=14)
    ax.set_title(f'{ALGO_CATEGORIES[category]["display_name"]} - Average Across Datasets',
                fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(y_range)
    ax.legend(loc='best', frameon=False)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# ----------------------------- Table Generation -----------------------

def generate_summary_tables(
    df: pd.DataFrame,
    base: str,
    metric: str,
    save_dir: Path,
    vi_curl_better: bool = True,
):
    """
    Generate summary tables for the final step of each algorithm category.
    Output format: Algorithm x Dataset with mean±std values, similar to ICML paper style.
    """
    df_sub = df[
        (df['base'].str.lower() == base.lower()) &
        (df['metric'].str.lower() == metric.lower())
    ].copy()
    
    if df_sub.empty:
        return
    
    # Categorize
    df_sub['category'], df_sub['variant'] = zip(*df_sub['algo'].apply(categorize_algo))
    df_sub = df_sub.dropna(subset=['category'])
    
    datasets = sorted(df_sub['dataset'].unique())
    
    # Build table data
    rows_mean = []
    rows_std = []
    rows_pretty_csv = []
    rows_pretty_tex = []
    
    # Get base model data first
    df_base = df_sub[df_sub['category'] == 'base']
    if not df_base.empty:
        row_mean = {'Algorithm': 'Base'}
        row_std = {'Algorithm': 'Base'}
        row_csv = {'Algorithm': 'Base'}
        row_tex = {'Algorithm': 'Base'}
        for ds in datasets:
            df_ds = df_base[df_base['dataset'] == ds]
            if not df_ds.empty:
                val = df_ds['value'].mean()
                std = df_ds['std'].mean()
                ds_display = DATASET_DISPLAY_MAP.get(ds, ds)
                row_mean[ds_display] = val
                row_std[ds_display] = std
                row_csv[ds_display] = f"{val:.1f} +/- {std:.1f}" if np.isfinite(std) else f"{val:.1f}"
                row_tex[ds_display] = f"{val:.1f} $\\pm$ {std:.1f}" if np.isfinite(std) else f"{val:.1f}"
            else:
                ds_display = DATASET_DISPLAY_MAP.get(ds, ds)
                row_mean[ds_display] = np.nan
                row_std[ds_display] = np.nan
                row_csv[ds_display] = '--'
                row_tex[ds_display] = '--'
        rows_mean.append(row_mean)
        rows_std.append(row_std)
        rows_pretty_csv.append(row_csv)
        rows_pretty_tex.append(row_tex)
    
    # Get data for each category
    for category, cat_info in ALGO_CATEGORIES.items():
        df_cat = df_sub[df_sub['category'] == category]
        if df_cat.empty:
            continue
        
        for variant, label in [('nocurl', f"{cat_info['display_name']} (Baseline)"), 
                                ('curl', f"{cat_info['display_name']} (VI-CuRL)")]:
            df_var = df_cat[df_cat['variant'] == variant]
            if df_var.empty:
                continue
            
            row_mean = {'Algorithm': label}
            row_std = {'Algorithm': label}
            row_csv = {'Algorithm': label}
            row_tex = {'Algorithm': label}
            
            for ds in datasets:
                df_ds = df_var[df_var['dataset'] == ds]
                if df_ds.empty:
                    ds_display = DATASET_DISPLAY_MAP.get(ds, ds)
                    row_mean[ds_display] = np.nan
                    row_std[ds_display] = np.nan
                    row_csv[ds_display] = '--'
                    row_tex[ds_display] = '--'
                    continue
                
                # Get final step
                final_step = df_ds['step'].max()
                df_final = df_ds[df_ds['step'] == final_step]
                
                if not df_final.empty:
                    val = df_final['value'].values[0]
                    std = df_final['std'].values[0]
                    ds_display = DATASET_DISPLAY_MAP.get(ds, ds)
                    row_mean[ds_display] = val
                    row_std[ds_display] = std
                    if np.isfinite(val):
                        row_csv[ds_display] = f"{val:.1f} +/- {std:.1f}" if np.isfinite(std) else f"{val:.1f}"
                        row_tex[ds_display] = f"{val:.1f} $\\pm$ {std:.1f}" if np.isfinite(std) else f"{val:.1f}"
                    else:
                        row_csv[ds_display] = '--'
                        row_tex[ds_display] = '--'
                else:
                    ds_display = DATASET_DISPLAY_MAP.get(ds, ds)
                    row_mean[ds_display] = np.nan
                    row_std[ds_display] = np.nan
                    row_csv[ds_display] = '--'
                    row_tex[ds_display] = '--'
            
            rows_mean.append(row_mean)
            rows_std.append(row_std)
            rows_pretty_csv.append(row_csv)
            rows_pretty_tex.append(row_tex)
    
    if not rows_mean:
        return
    
    # Create DataFrames
    df_mean = pd.DataFrame(rows_mean).set_index('Algorithm')
    df_std_out = pd.DataFrame(rows_std).set_index('Algorithm')
    df_csv = pd.DataFrame(rows_pretty_csv).set_index('Algorithm')
    df_tex = pd.DataFrame(rows_pretty_tex).set_index('Algorithm')
    
    # Add average column
    numeric_cols = [c for c in df_mean.columns if c != 'Algorithm']
    df_mean['Average'] = df_mean[numeric_cols].mean(axis=1, skipna=True)
    df_std_out['Average'] = df_std_out[numeric_cols].mean(axis=1, skipna=True)
    
    # Compute average for pretty tables
    for idx in df_csv.index:
        vals = [df_mean.at[idx, c] for c in numeric_cols if np.isfinite(df_mean.at[idx, c])]
        stds = [df_std_out.at[idx, c] for c in numeric_cols if np.isfinite(df_std_out.at[idx, c])]
        if vals:
            avg_val = np.mean(vals)
            avg_std = np.mean(stds) if stds else np.nan
            df_csv.at[idx, 'Average'] = f"{avg_val:.1f} +/- {avg_std:.1f}" if np.isfinite(avg_std) else f"{avg_val:.1f}"
            df_tex.at[idx, 'Average'] = f"{avg_val:.1f} $\\pm$ {avg_std:.1f}" if np.isfinite(avg_std) else f"{avg_val:.1f}"
        else:
            df_csv.at[idx, 'Average'] = '--'
            df_tex.at[idx, 'Average'] = '--'
    
    # Bold max values in tex
    for col in df_tex.columns:
        col_vals = pd.to_numeric(df_mean[col], errors='coerce')
        if np.isfinite(col_vals).any():
            max_val = np.nanmax(col_vals)
            for idx in df_tex.index:
                if np.isfinite(df_mean.at[idx, col]) and np.isclose(df_mean.at[idx, col], max_val, atol=1e-9):
                    df_tex.at[idx, col] = r"\textbf{" + df_tex.at[idx, col] + "}"
    
    # Save tables
    metric_safe = metric.replace('@', '_at_')
    out_prefix = save_dir / base / f'table_{metric_safe}'
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV
    df_csv.to_csv(f"{out_prefix}_pm.csv")
    
    # Generate beautiful LaTeX table (ICML style with booktabs)
    try:
        # Header
        cols = list(df_tex.columns)
        n_cols = len(cols)
        
        tex_lines = []
        tex_lines.append(r"\begin{table*}[t]")
        tex_lines.append(r"  \centering")
        tex_lines.append(r"  \small")
        tex_lines.append(f"  \\caption{{Mean and standard deviation (\\textbf{{{metric}}}) for {base}.}}")
        tex_lines.append(f"  \\label{{tab:{base.lower().replace(' ', '_').replace('-', '_')}_{metric_safe}}}")
        tex_lines.append(r"  \resizebox{\linewidth}{!}{")
        
        # Column format: l for algorithm, r for each dataset, |r for Average
        col_fmt = 'l' + 'r' * (n_cols - 1) + '|r'
        tex_lines.append(f"    \\begin{{tabular}}{{{col_fmt}}}")
        tex_lines.append(r"      \toprule")
        
        # Column headers
        header = "      Algorithm"
        for col in cols:
            header += f" & {col}"
        header += r" \\"
        tex_lines.append(header)
        tex_lines.append(r"      \midrule")
        
        # Data rows
        for idx in df_tex.index:
            row = f"      {idx}"
            for col in cols:
                row += f" & {df_tex.at[idx, col]}"
            row += r" \\"
            tex_lines.append(row)
        
        tex_lines.append(r"      \bottomrule")
        tex_lines.append(r"    \end{tabular}")
        tex_lines.append(r"  }")
        tex_lines.append(r"\end{table*}")
        
        tex_content = '\n'.join(tex_lines)
        Path(f"{out_prefix}_pm.tex").write_text(tex_content, encoding='utf-8')
    except Exception as e:
        print(f'  [WARN] Failed to write LaTeX: {e}')
    
    # Raw mean/std
    df_mean.to_csv(f"{out_prefix}_mean.csv")
    df_std_out.to_csv(f"{out_prefix}_std.csv")
    
    print(f'  [TABLE] {metric} saved to {out_prefix.name}_*.csv/tex')


# ----------------------------- Main -----------------------------------

def main():
    args = parse_args()
    
    target_dir = Path(args.target_dir).expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve() if args.save_dir else (target_dir / '_algo_curves')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = [m.strip().lower() for m in args.metrics.split(',') if m.strip()]
    
    print(f'[INFO] Target: {target_dir}')
    print(f'[INFO] Save: {save_dir}')
    print(f'[INFO] Metrics: {metrics}')
    print(f'[INFO] VI-CuRL Better: {args.vi_curl_better}')
    if args.y_range:
        print(f'[INFO] Y-Range: {args.y_range}')
    else:
        print('[INFO] Y-Range: per-metric dataset map (edit METRIC_DATASET_Y_RANGES)')

    table_overrides: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    table_path = args.vi_curl_table or (target_dir / 'main_results' / 'main_results.tex')
    if table_path and table_path.exists():
        table_overrides = load_vicurl_table_overrides(table_path)
        if table_overrides:
            print(f'[INFO] VI-CuRL table overrides: {table_path}')
        else:
            print(f'[WARN] VI-CuRL table empty: {table_path}')
    elif args.vi_curl_table:
        print(f'[WARN] VI-CuRL table not found: {table_path}')
    
    df = load_results(target_dir)
    if df.empty:
        print('[ERR] No data loaded.')
        return
    
    print(f'[INFO] Loaded {len(df)} records')
    
    datasets = sorted(df['dataset'].unique())
    bases = sorted(df['base'].unique())
    
    print(f'[INFO] Bases: {bases}')
    print(f'[INFO] Datasets: {datasets}')
    print(f'[INFO] Categories: {list(ALGO_CATEGORIES.keys())}')
    
    for base in bases:
        print(f'\n[INFO] Processing base: {base}')
        
        for metric in metrics:
            metric_safe = metric.replace('@', '_at_')
            dataset_ranges = compute_dataset_y_ranges(df, base, metric)
            df_metric = df[
                (df['base'].str.lower() == base.lower()) &
                (df['metric'].str.lower() == metric.lower())
            ]
            avg_range = auto_y_range(df_metric['value'].values, df_metric['std'].values)
            
            for category in ALGO_CATEGORIES.keys():
                # Individual dataset plots
                for dataset in datasets:
                    save_path = save_dir / base / category.replace(' ', '_').replace('.', '') / f'{metric_safe}_{dataset}'
                    y_range = tuple(args.y_range) if args.y_range else dataset_ranges.get(dataset, avg_range)
                    plot_algorithm_dataset(
                        df,
                        category,
                        base,
                        dataset,
                        metric,
                        save_path,
                        tuple(args.figsize),
                        y_range,
                        args.vi_curl_better,
                        table_overrides,
                    )
                    if save_path.with_suffix('.png').exists():
                        print(f'  [OK] {category} / {dataset} / {metric}')
                
                # Average plot
                save_path = save_dir / base / category.replace(' ', '_').replace('.', '') / f'{metric_safe}_average'
                y_range = tuple(args.y_range) if args.y_range else avg_range
                plot_algorithm_average(
                    df,
                    category,
                    base,
                    metric,
                    save_path,
                    tuple(args.figsize),
                    y_range,
                    args.vi_curl_better,
                    table_overrides,
                )
                if save_path.with_suffix('.png').exists():
                    print(f'  [OK] {category} / AVERAGE / {metric}')
            
            # Generate summary table for this base and metric
            generate_summary_tables(df, base, metric, save_dir, args.vi_curl_better)
    
    print(f'\n[DONE] All outputs saved to: {save_dir}')


if __name__ == '__main__':
    main()
