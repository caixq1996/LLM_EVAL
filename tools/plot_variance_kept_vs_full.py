#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot variance decomposition: kept vs full within the same model.

This script validates Theorem 3.1 by showing that the curriculum-selected
"kept" subset has lower variance than the full dataset for the same model.

Plots:
1. Absolute variance values: sigma_kept vs sigma_full, vprob_kept vs vprob_full
2. Variance ratios: kept/full (should be < 1 when β < 1)

Usage:
    python tools/plot_variance_kept_vs_full.py \
        --grad_variance_dir eval_log/vi_curl/grad_variance \
        --out_dir eval_log/vi_curl/variance_kept_vs_full
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass
class RunData:
    """Data extracted from a grad_variance JSON file."""
    run_name: str
    steps: List[int]
    sigma_kept: np.ndarray
    sigma_full: np.ndarray
    vprob_kept: np.ndarray
    vprob_full: np.ndarray
    betas: np.ndarray
    kept_counts: np.ndarray
    total_counts: np.ndarray


def load_run_data(json_path: Path) -> Optional[RunData]:
    """Load a grad_variance JSON file and extract relevant data."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None
    
    rows = data.get("rows", [])
    if not rows:
        return None
    
    steps = data.get("steps", [int(r.get("step", 0)) for r in rows])
    
    def extract_array(key: str) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)
    
    config = data.get("config", {})
    run_dir = config.get("run_dir", "")
    run_name = Path(run_dir).name if run_dir else json_path.stem
    
    sigma_kept = extract_array("sigma_kept")
    sigma_full = extract_array("sigma_full")
    vprob_kept = extract_array("vprob_kept")
    vprob_full = extract_array("vprob_full")
    betas = extract_array("beta_target")
    betas = np.clip(np.nan_to_num(betas, nan=1.0), 1e-6, 1.0)
    kept_counts = extract_array("kept")
    total_counts = extract_array("num_prompts")
    
    return RunData(
        run_name=run_name,
        steps=list(steps),
        sigma_kept=sigma_kept,
        sigma_full=sigma_full,
        vprob_kept=vprob_kept,
        vprob_full=vprob_full,
        betas=betas,
        kept_counts=kept_counts,
        total_counts=total_counts,
    )


def find_runs(grad_variance_dir: Path, filter_pattern: str = "") -> List[RunData]:
    """Find all runs in the grad_variance directory."""
    json_files = list(grad_variance_dir.glob("vf_curl_grad_variance__*.json"))
    # Filter out part files
    json_files = [f for f in json_files if "__part" not in f.name and ".partial" not in f.name]
    
    runs = []
    for json_path in json_files:
        run_data = load_run_data(json_path)
        if run_data is not None:
            if filter_pattern and filter_pattern not in run_data.run_name:
                continue
            runs.append(run_data)
    
    return runs


def extract_model_name(run_name: str) -> str:
    """Extract a short model name from the run name."""
    if "Qwen2.5-math-1.5B" in run_name or "Qwen2.5-math-1" in run_name:
        return "Qwen2.5-Math-1.5B"
    if "DeepSeek-R1-Distill-Qwen-1.5B" in run_name or "DeepSeek-R1-Distill-Qwen-1" in run_name:
        return "DeepSeek-R1-1.5B"
    patterns = [
        r"(Qwen[\d\.]+-[^\s_]+)",
        r"(DeepSeek[^\s_]+)",
        r"(Llama[^\s_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, run_name)
        if match:
            return match.group(1)
    return run_name


def extract_reward_type(run_name: str) -> str:
    """Extract the reward type from run name."""
    if "ver_rule" in run_name:
        return "Oracle"
    elif "majority_vote" in run_name:
        return "Majority Vote"
    elif "entropy" in run_name:
        return "Entropy"
    elif "selfcert" in run_name:
        return "Self-Cert"
    return "Unknown"


def extract_curl_type(run_name: str) -> str:
    """Extract curl or nocurl type."""
    if "_nocurl_" in run_name:
        return "No-CuRL"
    elif "_curl_" in run_name:
        return "VI-CuRL"
    return "Unknown"


def plot_absolute_variance(
    run: RunData,
    out_dir: Path,
) -> List[Path]:
    """
    Plot absolute variance values: kept vs full for the same model.
    """
    saved: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = extract_model_name(run.run_name)
    reward_type = extract_reward_type(run.run_name)
    curl_type = extract_curl_type(run.run_name)
    
    xs = np.array(run.steps)
    
    # Style settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })
    
    kept_color = "#2166ac"  # Blue (curriculum selected)
    full_color = "#b2182b"  # Red (full dataset)
    
    # Sanitize filename
    safe_name = run.run_name.replace("/", "-").replace(".", "_")[:60]
    
    # ================== Action Variance Plot ==================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(xs, run.sigma_kept, marker="o", markersize=5, linewidth=2,
            color=kept_color, label=r"$\sigma_{g,t}^2$ (kept)")
    ax.plot(xs, run.sigma_full, marker="s", markersize=5, linewidth=2,
            color=full_color, label=r"$\sigma_{g,t}^2$ (full)")
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Action Variance $\sigma_{g,t}^2$")
    ax.set_title(f"{model_name} ({reward_type}, {curl_type})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    fig.tight_layout()
    out_path = out_dir / f"action_variance_abs__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    # ================== Problem Variance Plot ==================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(xs, run.vprob_kept, marker="o", markersize=5, linewidth=2,
            color=kept_color, label=r"$V_{\mathrm{prob},t}$ (kept)")
    ax.plot(xs, run.vprob_full, marker="s", markersize=5, linewidth=2,
            color=full_color, label=r"$V_{\mathrm{prob},t}$ (full)")
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Problem Variance $V_{\mathrm{prob},t}$")
    ax.set_title(f"{model_name} ({reward_type}, {curl_type})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    fig.tight_layout()
    out_path = out_dir / f"problem_variance_abs__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    # ================== Combined Plot ==================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action Variance
    ax = axes[0]
    ax.plot(xs, run.sigma_kept, marker="o", markersize=5, linewidth=2,
            color=kept_color, label="kept")
    ax.plot(xs, run.sigma_full, marker="s", markersize=5, linewidth=2,
            color=full_color, label="full")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Action Variance $\sigma_{g,t}^2$")
    ax.set_title("Action Variance")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    # Problem Variance
    ax = axes[1]
    ax.plot(xs, run.vprob_kept, marker="o", markersize=5, linewidth=2,
            color=kept_color, label="kept")
    ax.plot(xs, run.vprob_full, marker="s", markersize=5, linewidth=2,
            color=full_color, label="full")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Problem Variance $V_{\mathrm{prob},t}$")
    ax.set_title("Problem Variance")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    fig.suptitle(f"{model_name} ({reward_type}, {curl_type}) - Kept vs Full", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"variance_abs_combined__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    return saved


def plot_variance_ratio(
    run: RunData,
    out_dir: Path,
) -> List[Path]:
    """
    Plot variance ratio: kept/full (validates Theorem 3.1).
    """
    saved: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = extract_model_name(run.run_name)
    reward_type = extract_reward_type(run.run_name)
    curl_type = extract_curl_type(run.run_name)
    
    xs = np.array(run.steps)
    
    # Compute ratios
    sigma_ratio = run.sigma_kept / np.clip(run.sigma_full, 1e-12, None)
    vprob_ratio = run.vprob_kept / np.clip(run.vprob_full, 1e-12, None)
    
    # Style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })
    
    sigma_color = "#1f77b4"  # Blue
    vprob_color = "#ff7f0e"  # Orange
    beta_color = "#2ca02c"   # Green
    
    # Sanitize filename
    safe_name = run.run_name.replace("/", "-").replace(".", "_")[:60]
    
    # ================== Combined Ratio Plot with Beta ==================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(xs, sigma_ratio, marker="o", markersize=5, linewidth=2,
            color=sigma_color, label=r"$\sigma_{g,t}^2$ ratio (kept/full)")
    ax.plot(xs, vprob_ratio, marker="s", markersize=5, linewidth=2,
            color=vprob_color, label=r"$V_{\mathrm{prob},t}$ ratio (kept/full)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="ratio = 1")
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Variance Ratio (kept / full)")
    ax.set_title(f"{model_name} ({reward_type}, {curl_type}) - Variance Reduction")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_ylim(0.0, 1.15)
    
    # Add beta on secondary axis
    ax2 = ax.twinx()
    ax2.plot(xs, run.betas, linestyle=":", linewidth=2, color=beta_color, alpha=0.7, label=r"$\beta_t$")
    ax2.set_ylabel(r"$\beta_t$ (retention rate)", color=beta_color)
    ax2.tick_params(axis='y', labelcolor=beta_color)
    ax2.set_ylim(0.0, 1.15)
    ax2.legend(loc="upper right")
    
    fig.tight_layout()
    out_path = out_dir / f"variance_ratio__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    # ================== Separate Ratio Plots ==================
    # Action Variance Ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, sigma_ratio, marker="o", markersize=5, linewidth=2, color=sigma_color)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$\sigma_{g,t}^2$ Ratio (kept / full)")
    ax.set_title(f"{model_name} ({reward_type}, {curl_type})")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_ylim(0.0, 1.15)
    fig.tight_layout()
    out_path = out_dir / f"sigma_ratio__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    # Problem Variance Ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, vprob_ratio, marker="o", markersize=5, linewidth=2, color=vprob_color)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$V_{\mathrm{prob},t}$ Ratio (kept / full)")
    ax.set_title(f"{model_name} ({reward_type}, {curl_type})")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_ylim(0.0, 1.15)
    fig.tight_layout()
    out_path = out_dir / f"vprob_ratio__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    return saved


def plot_grid_ratio(
    runs: List[RunData],
    out_dir: Path,
    variance_type: str = "sigma",  # "sigma" or "vprob"
) -> Optional[Path]:
    """
    Create a grid plot showing variance ratio for all runs.
    """
    if not runs:
        return None
    
    # Group by model and reward type
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], List[RunData]] = defaultdict(list)
    for run in runs:
        model = extract_model_name(run.run_name)
        reward = extract_reward_type(run.run_name)
        grouped[(model, reward)].append(run)
    
    models = sorted(set(k[0] for k in grouped.keys()))
    rewards = ["Oracle", "Majority Vote", "Entropy", "Self-Cert"]
    rewards = [r for r in rewards if any(k[1] == r for k in grouped.keys())]
    
    if not models or not rewards:
        return None
    
    n_rows = len(models)
    n_cols = len(rewards)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    
    curl_color = "#2166ac"
    nocurl_color = "#b2182b"
    
    var_symbol = r"$\sigma_{g,t}^2$" if variance_type == "sigma" else r"$V_{\mathrm{prob},t}$"
    var_name = "Action Variance" if variance_type == "sigma" else "Problem Variance"
    
    for i, model in enumerate(models):
        for j, reward in enumerate(rewards):
            ax = axes[i, j]
            
            key = (model, reward)
            if key not in grouped:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(reward, fontsize=12)
                if j == 0:
                    ax.set_ylabel(model, fontsize=11)
                continue
            
            for run in grouped[key]:
                curl_type = extract_curl_type(run.run_name)
                color = curl_color if curl_type == "VI-CuRL" else nocurl_color
                linestyle = "-" if curl_type == "VI-CuRL" else "--"
                marker = "o" if curl_type == "VI-CuRL" else "s"
                
                xs = np.array(run.steps)
                if variance_type == "sigma":
                    ratio = run.sigma_kept / np.clip(run.sigma_full, 1e-12, None)
                else:
                    ratio = run.vprob_kept / np.clip(run.vprob_full, 1e-12, None)
                
                ax.plot(xs, ratio, marker=marker, markersize=4, linewidth=1.5,
                        linestyle=linestyle, color=color, label=curl_type)
            
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.tick_params(labelsize=9)
            ax.set_ylim(0.0, 1.15)
            
            if i == 0:
                ax.set_title(reward, fontsize=12)
            if j == 0:
                ax.set_ylabel(model, fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel("Steps", fontsize=10)
            
            # Legend in upper right plot
            if i == 0 and j == n_cols - 1:
                ax.legend(loc="lower right", fontsize=9)
    
    var_file = "action_variance" if variance_type == "sigma" else "problem_variance"
    fig.suptitle(f"{var_name} Ratio (kept / full)", fontsize=14, y=1.01)
    fig.tight_layout()
    
    out_path = out_dir / f"grid__{var_file}_ratio.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot variance decomposition: kept vs full within same model.")
    parser.add_argument(
        "--grad_variance_dir",
        type=Path,
        default=Path("eval_log/vi_curl/grad_variance"),
        help="Directory containing grad_variance JSON files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("eval_log/vi_curl/variance_kept_vs_full"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter runs by pattern (e.g., 'ver_rule' for Oracle only).",
    )
    parser.add_argument(
        "--no_grid",
        action="store_true",
        help="Skip generating grid plots.",
    )
    parser.add_argument(
        "--ratio_only",
        action="store_true",
        help="Only generate ratio plots (skip absolute value plots).",
    )
    args = parser.parse_args()
    
    grad_variance_dir = args.grad_variance_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    
    if not grad_variance_dir.exists():
        print(f"[ERROR] Directory not found: {grad_variance_dir}")
        sys.exit(1)
    
    print(f"[INFO] Scanning {grad_variance_dir} for grad_variance JSON files...")
    runs = find_runs(grad_variance_dir, args.filter)
    
    if not runs:
        print("[WARN] No runs found.")
        sys.exit(0)
    
    print(f"[INFO] Found {len(runs)} run(s):")
    for run in runs:
        print(f"       - {run.run_name}")
    
    all_saved: List[Path] = []
    
    for run in runs:
        print(f"\n[INFO] Processing: {run.run_name}")
        
        # Ratio plots (main validation of Theorem 3.1)
        saved = plot_variance_ratio(run, out_dir)
        all_saved.extend(saved)
        for p in saved:
            print(f"       Saved: {p.name}")
        
        # Absolute value plots
        if not args.ratio_only:
            saved = plot_absolute_variance(run, out_dir)
            all_saved.extend(saved)
            for p in saved:
                print(f"       Saved: {p.name}")
    
    # Generate grid plots
    if not args.no_grid:
        print(f"\n[INFO] Generating grid plots...")
        for var_type in ["sigma", "vprob"]:
            grid_path = plot_grid_ratio(runs, out_dir, variance_type=var_type)
            if grid_path:
                all_saved.append(grid_path)
                print(f"       Saved: {grid_path.name}")
    
    print(f"\n[DONE] Generated {len(all_saved)} plots in {out_dir}")


if __name__ == "__main__":
    main()
