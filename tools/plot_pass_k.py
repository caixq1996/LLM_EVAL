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

# ---------------------- 配置 ----------------------
sns.set_theme(style="whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument('--eval_root', type=str, required=True, help='Path to results root')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--target_ks', type=str, default="1,8,64", help='Comma separated list of k')
    parser.add_argument('--plot_mode', type=str, default="all", choices=["step", "k", "all"], 
                        help='Plot mode: "step" (Score vs Step), "k" (Score vs Pass@k), or "all"')
    return parser.parse_args()

def extract_info_from_dirname(dirname):
    if dirname.startswith("base__"):
        return True, "Base Model", -1
    match = re.match(r"(.+)__global_step_(\d+)$", dirname)
    if match:
        return False, match.group(1), int(match.group(2))
    return None, None, None

def load_data(eval_root, target_ks):
    data = defaultdict(lambda: defaultdict(list))
    eval_path = Path(eval_root)
    
    for run_dir in eval_path.iterdir():
        if not run_dir.is_dir() or run_dir.name == "plots" or run_dir.name == "plots_visualization":
            continue
        is_base, algo_name, step = extract_info_from_dirname(run_dir.name)
        if algo_name is None: continue

        for group_dir in run_dir.iterdir():
            if not group_dir.is_dir() or not group_dir.name.startswith('g'): continue
            for dataset_dir in group_dir.iterdir():
                if not dataset_dir.is_dir(): continue
                
                # 寻找最新的 metrics json
                metrics_files = list(dataset_dir.glob("*_metrics.json"))
                if not metrics_files: continue
                metrics_file = sorted(metrics_files, key=lambda x: x.stat().st_mtime)[-1]
                
                try:
                    with open(metrics_file, 'r') as f: res = json.load(f)
                    pass_at_k = res.get('pass_at_k_percent', {})
                    if not pass_at_k and 'acc' in res: pass_at_k = {'1': res['acc']}

                    for k in target_ks:
                        k_str = str(k)
                        if k_str in pass_at_k and pass_at_k[k_str] is not None:
                            data[dataset_dir.name][k].append({
                                'algo': algo_name, 'step': step, 'score': float(pass_at_k[k_str]), 'is_base': is_base
                            })
                except Exception: pass
    return data

def compute_average_dataset(data):
    # Aggregation
    grouped_scores = defaultdict(lambda: defaultdict(list))
    for ds_name, k_data in data.items():
        for k, records in k_data.items():
            for rec in records:
                key = (rec['algo'], rec['step'], rec['is_base'])
                grouped_scores[k][key].append(rec['score'])
    
    avg_data = defaultdict(list)
    for k, entries in grouped_scores.items():
        for (algo, step, is_base), scores in entries.items():
            avg_data[k].append({
                'algo': algo, 'step': step, 'score': np.mean(scores), 'is_base': is_base
            })
    data["Average"] = avg_data
    return data

# ================= 模式 1: Score vs Global Step =================

def plot_step_scaling(dataset_name, k, records, output_path):
    if not records: return
    df = pd.DataFrame(records)
    finetune_steps = sorted(df[df['step'] != -1]['step'].unique())
    if not finetune_steps: return

    algos = sorted(df['algo'].unique().tolist())
    if "Base Model" in algos:
        algos.remove("Base Model")
        algos.insert(0, "Base Model")

    x_indices = np.arange(len(finetune_steps))
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(algos))

    for i, algo in enumerate(algos):
        y_values = []
        algo_df = df[df['algo'] == algo]
        for step in finetune_steps:
            if algo == "Base Model":
                row = algo_df[algo_df['step'] == -1]
                y_values.append(row.iloc[0]['score'] if not row.empty else 0.0)
            else:
                row = algo_df[algo_df['step'] == step]
                y_values.append(row.iloc[0]['score'] if not row.empty else 0.0)
        
        offset = (i - len(algos)/2 + 0.5) * width
        x_pos = x_indices + offset
        ax.bar(x_pos, y_values, width, label=algo, color=colors(i), alpha=0.7, edgecolor='white')
        ax.plot(x_pos, y_values, marker='o', markersize=4, color=colors(i), linewidth=2)
        
        # 标注数值
        for x, y in zip(x_pos, y_values):
            if y > 0: ax.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in finetune_steps])
    ax.set_xlabel("Global Step")
    ax.set_ylabel(f"Pass@{k} (%)")
    ax.set_title(f"{dataset_name} - Pass@{k} vs Steps")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Step] Saved: {output_path}")

# ================= 模式 2: Score vs Pass@K (Best of Steps) =================

def plot_k_scaling(dataset_name, data_dict, target_ks, output_path):
    """
    data_dict: {k: [records...]}
    需要提取每个算法在每个 k 下的 best score
    """
    # 1. 整理数据：找到所有算法，以及每个算法在每个 k 的最大值
    algos = set()
    best_scores = defaultdict(lambda: defaultdict(float)) # algo -> k -> score

    for k in target_ks:
        records = data_dict.get(k, [])
        for rec in records:
            algo = rec['algo']
            algos.add(algo)
            # 取最大值
            if rec['score'] > best_scores[algo][k]:
                best_scores[algo][k] = rec['score']

    if not algos: return

    # 排序算法，保证 Base 在前
    sorted_algos = sorted(list(algos))
    if "Base Model" in sorted_algos:
        sorted_algos.remove("Base Model")
        sorted_algos.insert(0, "Base Model")
    
    # 2. 绘图
    x_indices = np.arange(len(target_ks))
    width = 0.8 / len(sorted_algos)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(sorted_algos))

    for i, algo in enumerate(sorted_algos):
        y_values = []
        for k in target_ks:
            y_values.append(best_scores[algo][k]) # 默认是 0.0 如果没有数据

        offset = (i - len(sorted_algos)/2 + 0.5) * width
        x_pos = x_indices + offset
        
        # 柱状图
        ax.bar(x_pos, y_values, width, label=algo, color=colors(i), alpha=0.7, edgecolor='white')
        # 连线图
        ax.plot(x_pos, y_values, marker='o', markersize=5, color=colors(i), linewidth=2)
        
        # 标注数值
        for x, y in zip(x_pos, y_values):
            if y > 0: ax.text(x, y + 1.0, f'{y:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Pass@{k}" for k in target_ks])
    ax.set_xlabel("Metric (Pass@k)")
    ax.set_ylabel("Best Score (%) (Max across steps)")
    ax.set_title(f"{dataset_name} - Performance Scaling (Best of Steps)")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.15) # 留多一点头部给文字

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot K] Saved: {output_path}")

# ================= 主流程 =================

def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = os.path.join(args.eval_root, "plots_visualization")
    os.makedirs(args.output_dir, exist_ok=True)
    
    ks = sorted([int(k) for k in args.target_ks.split(',')])
    
    print(f"Loading data from: {args.eval_root}")
    data = load_data(args.eval_root, ks)
    if not data:
        print("[Error] No data found.")
        return

    data = compute_average_dataset(data)
    
    # Mode 1: Step-wise plots (Score vs Step, one chart per k)
    if args.plot_mode in ["step", "all"]:
        step_dir = os.path.join(args.output_dir, "vs_steps")
        os.makedirs(step_dir, exist_ok=True)
        for ds_name, k_data in data.items():
            for k, records in k_data.items():
                fname = f"{ds_name}_passAt{k}_vs_steps.png".replace("/", "_")
                plot_step_scaling(ds_name, k, records, os.path.join(step_dir, fname))

    # Mode 2: K-wise plots (Score vs K, best of steps)
    if args.plot_mode in ["k", "all"]:
        k_dir = os.path.join(args.output_dir, "vs_k_best")
        os.makedirs(k_dir, exist_ok=True)
        for ds_name, k_data in data.items():
            # 这里需要传入整个 dataset 的数据字典，因为要聚合不同 k
            fname = f"{ds_name}_k_scaling.png".replace("/", "_")
            plot_k_scaling(ds_name, k_data, ks, os.path.join(k_dir, fname))

if __name__ == "__main__":
    main()