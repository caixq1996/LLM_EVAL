#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot bias-variance analysis for VI-CuRL paper.

This script generates:
1. Combined variance+bias+beta plots (for paper Figure 3)
2. Standalone bias plots (bias vs training steps, bias vs beta)
3. Scatter plots validating Bias ∝ (1-β_t)

Usage:
    python tools/plot_bias_variance.py \
        --grad_variance_dir eval_log/vi_curl/grad_variance \
        --out_dir eval_log/vi_curl/bias_variance_plots

Outputs:
    - bias_variance_combined_<model>.pdf: Combined plot for paper
    - bias_vs_beta_<model>.pdf: Bias metrics vs retention rate
    - variance_ratio_<model>.pdf: Existing variance ratio plot with bias overlay
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
    # Variance metrics
    sigma_kept: np.ndarray
    sigma_full: np.ndarray
    vprob_kept: np.ndarray
    vprob_full: np.ndarray
    sigma_ratio: np.ndarray
    vprob_ratio: np.ndarray
    # Bias metrics
    bias_l2: np.ndarray
    bias_relative: np.ndarray
    cosine_similarity: np.ndarray
    # Retention rate
    betas: np.ndarray
    # Pass@k (optional)
    passk_kept: Optional[Dict[int, np.ndarray]] = None
    passk_full: Optional[Dict[int, np.ndarray]] = None
    passk_dropped: Optional[Dict[int, np.ndarray]] = None


def load_run_data(json_path: Path) -> Optional[RunData]:
    """Load a grad_variance JSON file and extract relevant data."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None

    rows = data.get("rows", [])
    if not rows:
        return None

    run_name = data.get("title", json_path.stem)

    def extract_array(key: str, default: float = float("nan")) -> np.ndarray:
        return np.array([r.get(key, default) for r in rows], dtype=float)

    steps = [r.get("step", i) for i, r in enumerate(rows)]
    
    # Check if bias metrics exist
    if not any("bias_l2" in r for r in rows):
        print(f"[WARN] No bias metrics in {json_path.name}, skipping")
        return None

    # Extract pass@k if available
    passk_kept = None
    passk_full = None
    passk_dropped = None
    if "pass_at_k_kept" in rows[0] and rows[0]["pass_at_k_kept"]:
        passk_ks = rows[0].get("passk_ks", [1, 8])
        passk_kept = {}
        passk_full = {}
        passk_dropped = {}
        for k in passk_ks:
            passk_kept[k] = np.array([r.get("pass_at_k_kept", {}).get(str(k), float("nan")) for r in rows])
            passk_full[k] = np.array([r.get("pass_at_k_full", {}).get(str(k), float("nan")) for r in rows])
            passk_dropped[k] = np.array([r.get("pass_at_k_dropped", {}).get(str(k), float("nan")) for r in rows])

    return RunData(
        run_name=run_name,
        steps=steps,
        sigma_kept=extract_array("sigma_kept"),
        sigma_full=extract_array("sigma_full"),
        vprob_kept=extract_array("vprob_kept"),
        vprob_full=extract_array("vprob_full"),
        sigma_ratio=extract_array("sigma_ratio"),
        vprob_ratio=extract_array("vprob_ratio"),
        bias_l2=extract_array("bias_l2"),
        bias_relative=extract_array("bias_relative"),
        cosine_similarity=extract_array("cosine_similarity"),
        betas=extract_array("beta_actual"),
        passk_kept=passk_kept,
        passk_full=passk_full,
        passk_dropped=passk_dropped,
    )


def find_runs(grad_variance_dir: Path, filter_pattern: str = "") -> List[RunData]:
    """Find all runs in the grad_variance directory."""
    runs = []
    for json_path in sorted(grad_variance_dir.glob("*.json")):
        if ".partial" in json_path.name:
            continue
        if filter_pattern and filter_pattern not in json_path.name:
            continue
        run_data = load_run_data(json_path)
        if run_data:
            runs.append(run_data)
    return runs


def extract_model_name(run_name: str) -> str:
    """Extract a short model name from the run name."""
    if "Qwen2.5-math-1.5B" in run_name.lower() or "qwen2.5-math-1" in run_name.lower():
        return "Qwen2.5-Math-1.5B"
    elif "deepseek" in run_name.lower() and "1.5" in run_name:
        return "DeepSeek-R1-Distill-Qwen-1.5B"
    elif "qwen" in run_name.lower() and "7b" in run_name.lower():
        return "Qwen2.5-Math-7B"
    elif "llama" in run_name.lower():
        return "Llama-3.2-3B"
    return run_name[:40]


def extract_reward_type(run_name: str) -> str:
    """Extract the reward type from run name."""
    if "ver_rule" in run_name or "oracle" in run_name.lower():
        return "Oracle"
    elif "majority" in run_name.lower() or "mv" in run_name.lower():
        return "Majority Vote"
    elif "entropy" in run_name.lower() or "vf_entropy" in run_name:
        return "Entropy"
    elif "selfcert" in run_name.lower():
        return "Self-Cert"
    return "Unknown"


def extract_curl_type(run_name: str) -> str:
    """Extract curl or nocurl type."""
    if "_nocurl_" in run_name or "nocurl" in run_name.lower():
        return "No-CuRL"
    return "VI-CuRL"


def plot_combined_bias_variance(
    run: RunData,
    out_dir: Path,
) -> List[Path]:
    """
    Plot combined variance ratios + bias metrics + beta schedule.
    This creates a figure suitable for the paper.
    """
    saved: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = extract_model_name(run.run_name)
    reward_type = extract_reward_type(run.run_name)
    curl_type = extract_curl_type(run.run_name)
    
    xs = np.array(run.steps)
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    
    safe_name = run.run_name.replace("/", "-").replace(".", "_")[:60]
    
    # ================== Combined Plot: Variance + Bias + Beta ==================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left panel: Variance ratios (same as existing Figure 3)
    ax1 = axes[0]
    ax1.plot(xs, run.sigma_ratio, marker="o", markersize=4, linewidth=2,
             color="#2166ac", label=r"Action Var. ratio $r_{\sigma,t}$")
    ax1.plot(xs, run.vprob_ratio, marker="s", markersize=4, linewidth=2,
             color="#d95f02", label=r"Problem Var. ratio $r_{V,t}$")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Variance Ratio (kept / full)")
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    
    # Add beta on secondary axis
    ax1_beta = ax1.twinx()
    ax1_beta.fill_between(xs, 0, run.betas, alpha=0.15, color="#1b9e77")
    ax1_beta.plot(xs, run.betas, linestyle="--", linewidth=1.5, color="#1b9e77", label=r"$\beta_t$")
    ax1_beta.set_ylabel(r"Retention Rate $\beta_t$", color="#1b9e77")
    ax1_beta.set_ylim(0, 1.1)
    ax1_beta.tick_params(axis='y', labelcolor="#1b9e77")
    
    ax1.set_title("(a) Variance Reduction")
    
    # Right panel: Bias metrics
    ax2 = axes[1]
    ax2.plot(xs, run.bias_relative, marker="^", markersize=4, linewidth=2,
             color="#7570b3", label=r"Relative Bias $\|\bar{g}_{\mathrm{kept}} - \bar{g}_{\mathrm{full}}\| / \|\bar{g}_{\mathrm{full}}\|$")
    ax2.plot(xs, 1 - run.betas, linestyle=":", linewidth=2, color="#e7298a",
             alpha=0.8, label=r"$(1 - \beta_t)$ (theoretical bound)")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Relative Bias")
    ax2.set_ylim(0, None)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    
    # Add cosine similarity on secondary axis
    ax2_cos = ax2.twinx()
    ax2_cos.plot(xs, run.cosine_similarity, linestyle="-.", linewidth=1.5, 
                 color="#66a61e", label=r"Cosine Sim.")
    ax2_cos.set_ylabel("Cosine Similarity", color="#66a61e")
    ax2_cos.set_ylim(0.8, 1.02)
    ax2_cos.tick_params(axis='y', labelcolor="#66a61e")
    
    ax2.set_title("(b) Bias Analysis")
    
    fig.suptitle(f"{model_name} ({reward_type}, {curl_type})", fontsize=13, y=1.02)
    fig.tight_layout()
    
    out_path = out_dir / f"bias_variance_combined__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    return saved


def plot_bias_vs_beta(
    run: RunData,
    out_dir: Path,
) -> List[Path]:
    """
    Plot scatter: Bias vs (1 - beta_t) to validate Bias ∝ (1-β).
    """
    saved: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = extract_model_name(run.run_name)
    safe_name = run.run_name.replace("/", "-").replace(".", "_")[:60]
    
    one_minus_beta = 1 - run.betas
    
    plt.rcParams.update({'font.size': 11})
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Scatter plot
    sc = ax.scatter(one_minus_beta, run.bias_relative, c=run.steps, cmap="viridis",
                    s=60, edgecolors="k", linewidths=0.5, alpha=0.8)
    
    # Linear fit
    valid_mask = np.isfinite(one_minus_beta) & np.isfinite(run.bias_relative)
    if valid_mask.sum() > 2:
        coeffs = np.polyfit(one_minus_beta[valid_mask], run.bias_relative[valid_mask], 1)
        fit_x = np.linspace(0, one_minus_beta.max() * 1.1, 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, "r--", linewidth=2, label=f"Linear fit (slope={coeffs[0]:.3f})")
    
    ax.set_xlabel(r"$(1 - \beta_t)$")
    ax.set_ylabel(r"Relative Bias")
    ax.set_title(f"{model_name}: Bias ∝ $(1-\\beta_t)$")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Training Step")
    
    fig.tight_layout()
    out_path = out_dir / f"bias_vs_beta__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    return saved


def plot_variance_ratio_with_bias(
    run: RunData,
    out_dir: Path,
) -> List[Path]:
    """
    Plot variance ratios with bias overlay (for paper Figure 3 style).
    """
    saved: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = extract_model_name(run.run_name)
    reward_type = extract_reward_type(run.run_name)
    safe_name = run.run_name.replace("/", "-").replace(".", "_")[:60]
    
    xs = np.array(run.steps)
    
    plt.rcParams.update({'font.size': 11})
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    
    # Primary axis: Variance ratios + Bias
    l1, = ax1.plot(xs, run.sigma_ratio, marker="o", markersize=5, linewidth=2,
                   color="#2166ac", label=r"$r_{\sigma,t}$ (Action Var.)")
    l2, = ax1.plot(xs, run.vprob_ratio, marker="s", markersize=5, linewidth=2,
                   color="#d95f02", label=r"$r_{V,t}$ (Problem Var.)")
    l3, = ax1.plot(xs, run.bias_relative, marker="^", markersize=5, linewidth=2,
                   color="#7570b3", label=r"Relative Bias")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Ratio / Relative Bias")
    ax1.set_ylim(0, 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    # Secondary axis: Beta
    ax2 = ax1.twinx()
    ax2.fill_between(xs, 0, run.betas, alpha=0.1, color="#1b9e77")
    l4, = ax2.plot(xs, run.betas, linestyle="--", linewidth=2, color="#1b9e77", label=r"$\beta_t$")
    ax2.set_ylabel(r"Retention Rate $\beta_t$", color="#1b9e77")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor="#1b9e77")
    
    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=9)
    
    ax1.set_title(f"{model_name} ({reward_type})")
    fig.tight_layout()
    
    out_path = out_dir / f"variance_ratio_with_bias__{safe_name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    
    return saved


def plot_paper_figure(
    runs: List[RunData],
    out_dir: Path,
    model_filter: str = "",
    reward_filter: str = "Oracle",
) -> Optional[Path]:
    """
    Generate a 2-panel figure for paper (like Figure 3).
    Left: Qwen, Right: DeepSeek
    """
    # Filter runs
    qwen_runs = [r for r in runs if "qwen" in r.run_name.lower() and "1.5" in r.run_name]
    deepseek_runs = [r for r in runs if "deepseek" in r.run_name.lower()]
    
    if reward_filter:
        qwen_runs = [r for r in qwen_runs if reward_filter.lower() in extract_reward_type(r.run_name).lower()]
        deepseek_runs = [r for r in deepseek_runs if reward_filter.lower() in extract_reward_type(r.run_name).lower()]
    
    if not qwen_runs and not deepseek_runs:
        print("[WARN] No runs found for paper figure")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    plt.rcParams.update({'font.size': 11})
    
    def plot_model_panel(ax, run: RunData, title: str):
        xs = np.array(run.steps)
        
        # Variance ratios
        ax.plot(xs, run.sigma_ratio, marker="o", markersize=4, linewidth=2,
                color="#2166ac", label=r"$r_{\sigma,t}$ (Action)")
        ax.plot(xs, run.vprob_ratio, marker="s", markersize=4, linewidth=2,
                color="#d95f02", label=r"$r_{V,t}$ (Problem)")
        ax.plot(xs, run.bias_relative, marker="^", markersize=4, linewidth=2,
                color="#7570b3", label=r"Rel. Bias")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Ratio")
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        
        # Beta on secondary axis
        ax2 = ax.twinx()
        ax2.fill_between(xs, 0, run.betas, alpha=0.1, color="#1b9e77")
        ax2.plot(xs, run.betas, linestyle="--", linewidth=1.5, color="#1b9e77")
        ax2.set_ylabel(r"$\beta_t$", color="#1b9e77")
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='y', labelcolor="#1b9e77")
        
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
    
    if qwen_runs:
        plot_model_panel(axes[0], qwen_runs[0], "(a) Qwen2.5-Math-1.5B")
    else:
        axes[0].text(0.5, 0.5, "No Qwen data", ha="center", va="center", transform=axes[0].transAxes)
    
    if deepseek_runs:
        plot_model_panel(axes[1], deepseek_runs[0], "(b) DeepSeek-R1-Distill-Qwen-1.5B")
    else:
        axes[1].text(0.5, 0.5, "No DeepSeek data", ha="center", va="center", transform=axes[1].transAxes)
    
    fig.tight_layout()
    
    out_path = out_dir / f"bias_variance_paper_figure_{reward_filter.lower()}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot bias-variance analysis for VI-CuRL.")
    parser.add_argument(
        "--grad_variance_dir",
        type=Path,
        default=Path("eval_log/vi_curl/grad_variance"),
        help="Directory containing grad_variance JSON files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("eval_log/vi_curl/bias_variance_plots"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter runs by pattern.",
    )
    parser.add_argument(
        "--reward_filter",
        type=str,
        default="",
        help="Filter by reward type (Oracle, Entropy, etc.).",
    )
    parser.add_argument(
        "--paper_figure",
        action="store_true",
        help="Generate paper-ready 2-panel figure.",
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
        print("[WARN] No runs found with bias metrics.")
        sys.exit(0)
    
    print(f"[INFO] Found {len(runs)} run(s) with bias metrics:")
    for run in runs:
        print(f"       - {run.run_name}")
    
    all_saved: List[Path] = []
    
    # Generate individual plots
    for run in runs:
        print(f"\n[INFO] Processing: {run.run_name}")
        
        saved = plot_combined_bias_variance(run, out_dir)
        all_saved.extend(saved)
        
        saved = plot_bias_vs_beta(run, out_dir)
        all_saved.extend(saved)
        
        saved = plot_variance_ratio_with_bias(run, out_dir)
        all_saved.extend(saved)
        
        for p in saved:
            print(f"       Saved: {p.name}")
    
    # Generate paper figure
    if args.paper_figure or len(runs) >= 2:
        print(f"\n[INFO] Generating paper figure...")
        for reward in ["Oracle", "Entropy"]:
            paper_path = plot_paper_figure(runs, out_dir, reward_filter=reward)
            if paper_path:
                all_saved.append(paper_path)
                print(f"       Saved: {paper_path.name}")
    
    print(f"\n[DONE] Generated {len(all_saved)} plots in {out_dir}")


if __name__ == "__main__":
    main()
