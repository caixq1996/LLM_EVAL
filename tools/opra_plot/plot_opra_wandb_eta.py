#!/usr/bin/env python3
"""
Plot grad_eta and delta_w_eta from WandB logs for OPRA experiments.

This script extracts eta values from WandB output.log files and plots
the evolution of gradient alignment (grad_eta) and parameter alignment (delta_w_eta)
across training steps for different PEFT algorithms.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Algorithm name mapping for legend
ALGO_NAME_MAP = {
    "vanilla": "LoRA",
    "opra": "OPRA",
    "opra_opra": "OPRA-OPRA",
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "rslora": "RSLoRA",
    "pissa": "PiSSA",
    "qpissa": "QPiSSA",
    "qlora": "QLoRA",
    "olora": "OLoRA",
    "oft": "OFT",
}

# Color palette for algorithms
ALGO_COLORS = {
    "LoRA": "#1f77b4",
    "OPRA": "#d62728",
    "OPRA-OPRA": "#ff7f0e",
    "AdaLoRA": "#2ca02c",
    "DoRA": "#9467bd",
    "RSLoRA": "#8c564b",
    "PiSSA": "#e377c2",
    "QPiSSA": "#7f7f7f",
    "QLoRA": "#bcbd22",
    "OLoRA": "#17becf",
    "OFT": "#ff9896",
}


def extract_algo_suffix(exp_name: str) -> str:
    """Extract algorithm suffix from experiment name like 'Qwen2.5-math-1.5B_vanilla'."""
    parts = exp_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return exp_name


def get_algo_display_name(exp_name: str) -> str:
    """Get display name for legend."""
    suffix = extract_algo_suffix(exp_name)
    return ALGO_NAME_MAP.get(suffix, suffix)


def parse_wandb_output_log(log_path: Path) -> List[Dict]:
    """Parse WandB output.log to extract step, grad_eta, delta_w_eta."""
    records = []
    if not log_path.exists():
        return records
    
    pattern = re.compile(
        r"step:(\d+).*?actor/grad_eta:([+-]?[\d.eE+-]+).*?actor/delta_w_eta:([+-]?[\d.eE+-]+)"
    )
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                grad_eta = float(match.group(2))
                delta_w_eta = float(match.group(3))
                records.append({
                    "step": step,
                    "grad_eta": grad_eta,
                    "delta_w_eta": delta_w_eta,
                })
    return records


def get_experiment_name(run_dir: Path) -> Optional[str]:
    """Extract experiment_name from WandB config.yaml."""
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        trainer = cfg.get("trainer", {})
        if isinstance(trainer, dict):
            trainer_val = trainer.get("value", trainer)
            if isinstance(trainer_val, dict):
                exp = trainer_val.get("experiment_name")
                if isinstance(exp, dict):
                    exp = exp.get("value")
                return exp
    except Exception:
        pass
    return None


def find_wandb_runs(wandb_root: Path, model_filter: str) -> Dict[str, Path]:
    """Find the latest WandB run for each experiment matching model_filter."""
    runs_by_exp: Dict[str, List[Tuple[float, Path]]] = defaultdict(list)
    
    for run_dir in wandb_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
            continue
        exp_name = get_experiment_name(run_dir)
        if not exp_name or model_filter not in exp_name:
            continue
        
        # Get run start time from metadata or mtime
        meta_path = run_dir / "files" / "wandb-metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                ts = meta.get("startedAt", "")
                from datetime import datetime
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                start_time = dt.timestamp()
            except Exception:
                start_time = run_dir.stat().st_mtime
        else:
            start_time = run_dir.stat().st_mtime
        
        runs_by_exp[exp_name].append((start_time, run_dir))
    
    # Select latest run for each experiment
    latest_runs = {}
    for exp, runs in runs_by_exp.items():
        if runs:
            _, latest_dir = max(runs, key=lambda x: x[0])
            latest_runs[exp] = latest_dir
    
    return latest_runs


def plot_eta_curves(
    data: Dict[str, pd.DataFrame],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
):
    """Plot eta curves for all algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, df in sorted(data.items()):
        if df.empty or metric not in df.columns:
            continue
        algo_name = get_algo_display_name(exp_name)
        color = ALGO_COLORS.get(algo_name, None)
        ax.plot(df["step"], df[metric], label=algo_name, color=color, linewidth=2)
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log") if metric == "delta_w_eta" else None
    
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    print(f"[INFO] Saved: {out_path.with_suffix('.png')} and .pdf")


def main():
    parser = argparse.ArgumentParser(description="Plot grad_eta and delta_w_eta from WandB logs")
    parser.add_argument("--wandb_root", type=str, default="/home/caixq/project/OPRA/wandb",
                        help="Path to WandB root directory")
    parser.add_argument("--model_filter", type=str, default="Qwen2.5-math-1.5B",
                        help="Filter for model name in experiment_name")
    parser.add_argument("--out_dir", type=str, default="/home/caixq/project/LLM_EVAL/eval_log/opra/wandb_eta",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    wandb_root = Path(args.wandb_root)
    out_dir = Path(args.out_dir) / args.model_filter
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Scanning WandB runs in {wandb_root}")
    print(f"[INFO] Model filter: {args.model_filter}")
    
    runs = find_wandb_runs(wandb_root, args.model_filter)
    print(f"[INFO] Found {len(runs)} experiments:")
    for exp, path in sorted(runs.items()):
        print(f"  - {exp} -> {path.name}")
    
    # Extract eta data from each run
    all_data: Dict[str, pd.DataFrame] = {}
    for exp_name, run_dir in runs.items():
        log_path = run_dir / "files" / "output.log"
        records = parse_wandb_output_log(log_path)
        if records:
            df = pd.DataFrame(records)
            all_data[exp_name] = df
            print(f"[INFO] {exp_name}: {len(records)} steps, grad_eta range [{df['grad_eta'].min():.4f}, {df['grad_eta'].max():.4f}]")
    
    if not all_data:
        print("[WARN] No eta data found in any WandB run!")
        return
    
    # Save combined CSV
    csv_path = out_dir / "wandb_eta_data.csv"
    all_rows = []
    for exp_name, df in all_data.items():
        df = df.copy()
        df["experiment"] = exp_name
        df["algorithm"] = get_algo_display_name(exp_name)
        all_rows.append(df)
    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved data to {csv_path}")
    
    # Plot grad_eta
    plot_eta_curves(
        all_data,
        metric="grad_eta",
        title=f"Gradient Alignment η (grad_eta) - {args.model_filter}",
        ylabel="grad_eta (gradient's desire to modify principal space)",
        out_path=out_dir / "grad_eta_curve",
    )
    
    # Plot delta_w_eta
    plot_eta_curves(
        all_data,
        metric="delta_w_eta",
        title=f"Parameter Alignment η (delta_w_eta) - {args.model_filter}",
        ylabel="delta_w_eta (actual modification to principal space)",
        out_path=out_dir / "delta_w_eta_curve",
    )
    
    print(f"[INFO] All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
