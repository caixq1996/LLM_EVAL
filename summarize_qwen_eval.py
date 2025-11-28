import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import TABLEAU_COLORS

DATA_GROUPS = {
    'g1': ['aime25x8', 'amc23x8', 'aime24x8'],
    'g2': ['minerva_math', 'olympiadbench', 'math500']
}
ALL_DATASETS = DATA_GROUPS['g1'] + DATA_GROUPS['g2']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', type=Path, required=True, help='Root dir of eval outputs')
    ap.add_argument('--prompt_type', type=str, default='qwen25-math-cot')
    ap.add_argument('--save_dir', type=Path, default=None, help='Where to save summary outputs')
    ap.add_argument('--topk_radar', type=int, default=5, help='Number of top models to show in radar chart')
    return ap.parse_args()


def _find_latest_metrics_json(dataset_dir):
    cand = sorted(dataset_dir.glob('*metrics.json'),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True)
    return cand[0] if cand else None


def load_all_results(out_root):
    results = {}
    for run_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        run_name = run_dir.name
        for group in ('g1', 'g2'):
            gdir = run_dir / group
            if not gdir.exists():
                continue
            for ds_dir in sorted([p for p in gdir.iterdir() if p.is_dir()]):
                dataset = ds_dir.name
                mpath = _find_latest_metrics_json(ds_dir)
                if not mpath:
                    print(f'[WARN] No metrics.json under: {ds_dir}')
                    continue
                try:
                    metrics = json.loads(mpath.read_text(encoding='utf-8'))
                    results[run_name, group, dataset] = metrics
                except Exception as e:
                    print(f'[WARN] Failed to load {mpath}: {e}')
    return results


def build_frames(results):
    runs = sorted({k[0] for k in results.keys()})
    df_acc = pd.DataFrame(index=runs, columns=ALL_DATASETS, dtype=float)
    df_total_acc = pd.DataFrame(index=runs, columns=ALL_DATASETS, dtype=float)
    df_time_sec = pd.DataFrame(index=runs, columns=ALL_DATASETS, dtype=float)
    df_meta_nsamp = pd.DataFrame(index=runs, columns=ALL_DATASETS, dtype=float)

    for (run, group, ds), m in results.items():
        if ds not in df_acc.columns:
            continue
        df_acc.at[run, ds] = float(m.get('acc', np.nan))
        df_total_acc.at[run, ds] = float(m.get('total_acc', np.nan))
        df_time_sec.at[run, ds] = float(m.get('time_use_in_second', np.nan))
        df_meta_nsamp.at[run, ds] = float(m.get('num_samples', np.nan))
    return df_acc, df_total_acc, df_time_sec, df_meta_nsamp


def ensure_dirs(save_dir):
    (save_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (save_dir / 'figs').mkdir(parents=True, exist_ok=True)


def _paper_rc():
    plt.rcParams.update({
        'figure.dpi': 200, 'savefig.dpi': 300,
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.4,
        'figure.autolayout': True
    })


def plot_heatmap(df, title, out_prefix, vmin=None, vmax=None):
    _paper_rc()
    fig, ax = plt.subplots(
        figsize=(max(6, 0.5 * df.shape[1] + 2), max(4, 0.4 * df.shape[0] + 1.5))
    )
    data = df.values
    im = ax.imshow(data, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[0]))
    ax.set_xticklabels(df.columns, rotation=35, ha='right')
    ax.set_yticklabels(df.index)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = '–'
                color = 'black'
            else:
                text = f'{val:.1f}'
                color = 'white' if im.norm(val) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=270, labelpad=12)
    ax.set_title(title)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Model')
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


def plot_model_mean_bar(df, title, out_prefix):
    _paper_rc()
    means = df.mean(axis=1, skipna=True).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(means) + 2), 4))
    ax.bar(means.index, means.values)
    ax.set_title(title)
    ax.set_ylabel('Macro Avg Acc (%)')
    ax.set_xticklabels(means.index, rotation=35, ha='right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


def plot_grouped_bars(df, title, out_prefix):
    _paper_rc()
    fig, ax = plt.subplots(
        figsize=(max(8, 1.2 * df.shape[1]), max(4, 0.35 * df.shape[0] + 2))
    )
    x = np.arange(len(df.columns))
    width = min(0.8 / max(1, len(df.index)), 0.18)
    for i, model in enumerate(df.index):
        ax.bar(x + i * width, df.loc[model].values, width, label=model)
    ax.set_xticks(x + width * (len(df.index) - 1) / 2)
    ax.set_xticklabels(df.columns, rotation=35, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend(ncols=2, fontsize=9, frameon=False)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


def plot_box_by_dataset(df, title, out_prefix):
    _paper_rc()
    fig, ax = plt.subplots(figsize=(max(6, 0.65 * df.shape[1]), 4))
    ax.boxplot([df[c].dropna().values for c in df.columns], showmeans=True)
    ax.set_xticklabels(df.columns, rotation=35, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


def plot_acc_vs_time(df_acc, df_time, title, out_prefix):
    _paper_rc()
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in df_acc.columns:
        x = df_time[ds]
        y = df_acc[ds]
        mask = ~(x.isna() | y.isna())
        if mask.sum() == 0:
            continue
        ax.scatter(x[mask], y[mask], label=ds, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


def plot_radar(df, title, out_prefix, topk=5):
    _paper_rc()
    if df.shape[0] == 0:
        return
    means = df.mean(axis=1, skipna=True).sort_values(ascending=False)
    models = list(means.index[:max(1, min(topk, len(means)))])
    theta = np.linspace(0, 2 * np.pi, len(df.columns), endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    vmin = df.min(axis=0, skipna=True)
    vmax = df.max(axis=0, skipna=True)
    span = (vmax - vmin).replace(0, np.nan)

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    for m in models:
        vals = df.loc[m, :]
        norm = ((vals - vmin) / span).clip(lower=0, upper=1)
        y = np.concatenate([norm.values, [norm.values[0]]])
        ax.plot(theta, y, linewidth=1.5)
        ax.fill(theta, y, alpha=0.15, label=m)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=False)
    fig.savefig(f'{out_prefix}.png')
    fig.savefig(f'{out_prefix}.pdf')
    plt.close(fig)


# =========================
# New: base model grouping
# =========================

# 常见 base model 关键字/模式（小写匹配）
_BASE_PATTERNS = [
    r'qwen3(?:[\.\-]\d+)?(?:-math)?(?:-[\d\.]+b)?',
    r'qwen2[\.\-]?5(?:-math)?(?:-[\d\.]+b)?',
    r'qwen1[\.\-]?5(?:-math)?(?:-[\d\.]+b)?',
    r'llama-?3(?:[\.\-]\d+)?(?:-[\d\.]+b)?',
    r'deepseek(?:-r1(?:-distill)?(?:-qwen)?(?:-[\d\.]+b)?)?',
    r'internlm(?:-math)?(?:-[\d\.]+b)?',
    r'mathstral(?:-[\d\.]+b)?',
    r'mistral(?:-[\d\.]+b)?',
    r'gemma(?:-[\d\.]+b)?',
    r'phi-?\d(?:-[\d\.]+b)?'
]
_BASE_FALLBACK_KEYS = [
    'qwen3', 'qwen2.5', 'qwen25', 'qwen1.5', 'llama3', 'llama',
    'deepseek', 'internlm', 'mathstral', 'mistral', 'gemma', 'phi'
]


def _norm_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '-', s).strip('-').lower()


def _strip_seps(s: str) -> str:
    # 清理多余分隔符，漂亮一点
    s = re.sub(r'[_\-\+\./\(\)\[\]\{\}]+', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s


def infer_base_and_algo(run_name: str) -> Tuple[str, str]:
    """
    从 run_name 中解析 base model，并返回 (base_model, algo_name_去掉base).
    """
    s = run_name.lower()
    matches = []
    for pat in _BASE_PATTERNS:
        for m in re.finditer(pat, s):
            matches.append(m.group(0))
    if matches:
        base_raw = max(matches, key=len)
        # 尽量用原始大小写切片
        i = s.find(base_raw)
        base_display = run_name[i:i + len(base_raw)] if i >= 0 else base_raw
    else:
        # 关键词后备策略
        key = next((k for k in _BASE_FALLBACK_KEYS if k in s), None)
        if key:
            i = s.find(key)
            base_display = run_name[i:i + len(key)]
        else:
            base_display = 'unknown'

    # 从算法名里剔除 base 片段（忽略大小写），再做美化
    algo_name = re.sub(re.escape(base_display), '', run_name, flags=re.IGNORECASE)
    algo_name = _strip_seps(algo_name)
    if not algo_name:
        algo_name = run_name  # 冗余保护

    return base_display, algo_name


def group_runs_by_base(run_names: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    返回：
      - base -> [run_name...]
      - algo_labels: run_name -> algo_name_without_base
    """
    base_to_runs: Dict[str, List[str]] = {}
    algo_labels: Dict[str, str] = {}
    for rn in run_names:
        base, algo = infer_base_and_algo(rn)
        base_to_runs.setdefault(base, []).append(rn)
        algo_labels[rn] = algo
    return base_to_runs, algo_labels


def _nice_colors(n: int) -> List:
    """
    优先使用 Tableau 颜色（舒适、区分度好），不足时补 tab20。
    参考 Matplotlib 对定性配色的建议。  # 文献见回复说明
    """
    tableau = list(TABLEAU_COLORS.values())  # 10色
    if n <= len(tableau):
        return tableau[:n]
    # 扩展色板
    extra = list(plt.cm.tab20.colors)
    colors = tableau + extra
    if n <= len(colors):
        return colors[:n]
    # 仍不足则循环（极端情况）
    times = (n + len(colors) - 1) // len(colors)
    return (colors * times)[:n]


def plot_bars_by_base(df_acc: pd.DataFrame, save_dir: Path):
    """
    每个 base model 出一张“分组柱状图”：
      - x 轴：6 个数据集
      - y 轴：acc (%)
      - 同一 base 下，不同算法各一根 bar（颜色各不相同）
    """
    _paper_rc()
    out_dir = save_dir / 'figs' / 'bars_by_base'
    out_dir.mkdir(parents=True, exist_ok=True)

    base_to_runs, algo_labels = group_runs_by_base(list(df_acc.index))

    for base, runs in sorted(base_to_runs.items(), key=lambda kv: kv[0].lower()):
        sub = df_acc.loc[runs, ALL_DATASETS].copy()

        # 全 NaN 的行先去掉，避免空图
        sub = sub.loc[~sub.isna().all(axis=1)]
        if sub.empty:
            continue

        # 图例显示“算法名（去掉 base）”，并按平均准确率降序
        display_index = [algo_labels[r] for r in sub.index]
        sub.index = display_index
        order = sub.mean(axis=1, skipna=True).sort_values(ascending=False).index
        sub = sub.loc[order]

        n_algo = sub.shape[0]
        x = np.arange(len(ALL_DATASETS))
        width = min(0.8 / max(1, n_algo), 0.18)
        colors = _nice_colors(n_algo)

        fig, ax = plt.subplots(
            figsize=(max(8, 1.2 * len(ALL_DATASETS)), max(4, 0.35 * n_algo + 2))
        )
        for i, (algo, color) in enumerate(zip(sub.index, colors)):
            y = sub.loc[algo].values
            ax.bar(x + i * width, y, width, label=algo, color=color)

        ax.set_xticks(x + width * (n_algo - 1) / 2)
        ax.set_xticklabels(ALL_DATASETS, rotation=35, ha='right')
        yvals = sub.values.flatten()
        yvals = yvals[np.isfinite(yvals)]
        if yvals.size:
            y_min = float(np.min(yvals))
            y_max = float(np.max(yvals))
            if y_min == y_max:
                pad = max(1.0, 0.05 * max(1.0, abs(y_max)))
            else:
                pad = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy by Dataset — Base: {base}')
        ax.legend(ncols=2, frameon=False, fontsize=9, title='Algorithm')

        prefix = out_dir / f'bars_base_{_norm_filename(base)}'
        fig.savefig(f'{prefix}.png')
        fig.savefig(f'{prefix}.pdf')
        plt.close(fig)


def main():
    args = parse_args()
    save_dir = args.save_dir or args.out_root / '_summary'
    ensure_dirs(save_dir)

    results = load_all_results(args.out_root)
    if not results:
        raise SystemExit(f'No metrics found under {args.out_root}')

    df_acc, df_total_acc, df_time_sec, df_nsamp = build_frames(results)

    def save_table(df, name):
        df = df.copy()
        df = df.loc[sorted(df.index), ALL_DATASETS]
        (save_dir / 'tables' / f'{name}.csv').write_text(
            df.to_csv(), encoding='utf-8'
        )
        try:
            latex_str = df.to_latex(float_format='%.1f', na_rep='--', escape=False)
            (save_dir / 'tables' / f'{name}.tex').write_text(latex_str, encoding='utf-8')
        except Exception as e:
            print(f'[WARN] to_latex failed for {name}: {e}')

    save_table(df_acc, 'acc_percent')
    save_table(df_total_acc, 'total_acc_percent')
    save_table(df_time_sec, 'time_seconds')
    save_table(df_nsamp, 'num_samples')

    df_group_avg = pd.DataFrame(index=df_acc.index,
                                columns=['g1_avg', 'g2_avg', 'macro_avg'])
    df_group_avg['g1_avg'] = df_acc[DATA_GROUPS['g1']].mean(axis=1, skipna=True)
    df_group_avg['g2_avg'] = df_acc[DATA_GROUPS['g2']].mean(axis=1, skipna=True)
    df_group_avg['macro_avg'] = df_acc.mean(axis=1, skipna=True)
    df_group_avg.sort_values('macro_avg', ascending=False, inplace=True)
    (save_dir / 'tables' / 'group_macro_avg.csv').write_text(
        df_group_avg.to_csv(), encoding='utf-8'
    )
    try:
        (save_dir / 'tables' / 'group_macro_avg.tex').write_text(
            df_group_avg.to_latex(float_format='%.1f', na_rep='--', escape=False),
            encoding='utf-8'
        )
    except Exception as e:
        print(f'[WARN] to_latex failed for group_macro_avg: {e}')

    figs = save_dir / 'figs'
    plot_heatmap(
        df=df_acc.loc[df_group_avg.index],
        title='Accuracy (%) on Six Datasets',
        out_prefix=figs / 'heatmap_overall_acc',
        vmin=max(0, np.nanmin(df_acc.values) if np.isfinite(df_acc.values).any() else None),
        vmax=min(100, np.nanmax(df_acc.values) if np.isfinite(df_acc.values).any() else None)
    )
    plot_heatmap(
        df=df_acc[DATA_GROUPS['g1']].loc[df_group_avg.index],
        title='Accuracy (%) — Group g1',
        out_prefix=figs / 'heatmap_g1'
    )
    plot_heatmap(
        df=df_acc[DATA_GROUPS['g2']].loc[df_group_avg.index],
        title='Accuracy (%) — Group g2',
        out_prefix=figs / 'heatmap_g2'
    )
    plot_model_mean_bar(
        df=df_acc,
        title='Macro Average Accuracy (%) per Model',
        out_prefix=figs / 'bar_model_macro_avg'
    )
    plot_grouped_bars(
        df=df_acc.loc[df_group_avg.index[:min(12, len(df_acc))]],
        title='Accuracy by Dataset (Top Models Shown)',
        out_prefix=figs / 'grouped_bars_by_dataset'
    )
    plot_box_by_dataset(
        df=df_acc,
        title='Accuracy Distribution across Models (per Dataset)',
        out_prefix=figs / 'box_by_dataset'
    )
    plot_acc_vs_time(
        df_acc=df_acc, df_time=df_time_sec,
        title='Accuracy vs. Time (per Dataset)',
        out_prefix=figs / 'scatter_acc_vs_time'
    )
    plot_radar(
        df=df_acc[ALL_DATASETS].loc[df_group_avg.index],
        title=f'Radar (Normalized per Dataset) — Top {args.topk_radar}',
        out_prefix=figs / 'radar_topk',
        topk=args.topk_radar
    )

    # === 新增：按 base model 的分组柱状图 ===
    plot_bars_by_base(df_acc=df_acc, save_dir=save_dir)

    (save_dir / 'README.txt').write_text(
        "Files:\n"
        "- tables/*.csv, *.tex : per-dataset & (g1,g2,macro) tables\n"
        "- figs/*.png, *.pdf   : heatmaps, bars, radar, box, scatter\n"
        "- figs/bars_by_base/*.png, *.pdf : grouped bars by base model\n\n"
        "Notes:\n  Accuracies are as reported in '*metrics.json' "
        "(keys: 'acc', 'total_acc'). Missing results are shown as NaN/-- and skipped in averages.\n",
        encoding='utf-8'
    )
    print(f'[OK] Summary saved to: {save_dir.resolve()}')


if __name__ == '__main__':
    main()
