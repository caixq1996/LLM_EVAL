#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-ready variance decomposition figures for VI-CuRL paper.

This script creates the figures for Section "Does VI-CURL Really Reduce Variance?"
showing absolute Action Variance and Problem Variance for curl vs nocurl.

Usage:
    python tools/vi_curl_plot/plot_variance_for_paper.py \
        --grad_variance_dir eval_log/vi_curl/grad_variance \
        --out_dir /path/to/VI-CURL/figures
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
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Import shared plot configuration
try:
    from plot_config import (
        setup_plot_style,
        add_font_size_args,
        get_font_sizes,
        FONT_FAMILY,
    )
    HAS_PLOT_CONFIG = True
except ImportError:
    HAS_PLOT_CONFIG = False
    FONT_FAMILY = "Times New Roman"

    def setup_plot_style() -> None:
        plt.rcParams["font.family"] = FONT_FAMILY
        plt.rcParams["font.serif"] = [FONT_FAMILY, "DejaVu Serif", "serif"]
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["mathtext.fontset"] = "stix"

    def add_font_size_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fontsize_xlabel", type=int, default=14)
        parser.add_argument("--fontsize_ylabel", type=int, default=14)
        parser.add_argument("--fontsize_legend", type=int, default=12)
        parser.add_argument("--fontsize_tick", type=int, default=12)
        parser.add_argument("--fontsize_xtick", type=int, default=12)
        parser.add_argument("--fontsize_ytick", type=int, default=12)
        parser.add_argument("--fontsize_colorbar", type=int, default=12)

    def get_font_sizes(args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "xlabel": getattr(args, "fontsize_xlabel", 14),
            "ylabel": getattr(args, "fontsize_ylabel", 14),
            "legend": getattr(args, "fontsize_legend", 12),
            "tick": getattr(args, "fontsize_tick", 12),
            "xtick": getattr(args, "fontsize_xtick", getattr(args, "fontsize_tick", 12)),
            "ytick": getattr(args, "fontsize_ytick", getattr(args, "fontsize_tick", 12)),
            "colorbar": getattr(args, "fontsize_colorbar", 12),
            "fontfamily": FONT_FAMILY,
        }


@dataclass
class RunData:
    """Data extracted from a grad_variance JSON file."""
    run_name: str
    steps: List[int]
    sigma_kept: np.ndarray  # Action Variance
    vprob_kept: np.ndarray  # Problem Variance
    betas: np.ndarray
    is_curl: bool
    # Baseline data
    baseline_sigma_kept: Optional[np.ndarray] = None
    baseline_vprob_kept: Optional[np.ndarray] = None


def load_run_data(json_path: Path) -> Optional[RunData]:
    """Load a grad_variance JSON file and extract relevant data."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None
    
    rows = data.get("rows", [])
    baseline_rows = data.get("baseline_rows")
    if not rows:
        return None
    
    steps = data.get("steps", [int(r.get("step", 0)) for r in rows])
    
    def extract_array(key: str, source: List[Dict]) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in source], dtype=np.float64)
    
    config = data.get("config", {})
    run_dir = config.get("run_dir", "")
    run_name = Path(run_dir).name if run_dir else json_path.stem
    is_curl = "_curl_" in run_name and "_nocurl_" not in run_name
    
    sigma_kept = extract_array("sigma_kept", rows)
    vprob_kept = extract_array("vprob_kept", rows)
    betas = extract_array("beta_target", rows)
    betas = np.clip(np.nan_to_num(betas, nan=1.0), 1e-6, 1.0)
    
    baseline_sigma_kept = None
    baseline_vprob_kept = None
    if baseline_rows:
        baseline_sigma_kept = extract_array("sigma_kept", baseline_rows)
        baseline_vprob_kept = extract_array("vprob_kept", baseline_rows)
    
    return RunData(
        run_name=run_name,
        steps=list(steps),
        sigma_kept=sigma_kept,
        vprob_kept=vprob_kept,
        betas=betas,
        is_curl=is_curl,
        baseline_sigma_kept=baseline_sigma_kept,
        baseline_vprob_kept=baseline_vprob_kept,
    )


def extract_model_short_name(run_name: str) -> str:
    """Extract a short model name for legend/title."""
    if "Qwen2.5-math-1.5B" in run_name or "Qwen2.5-math-1" in run_name:
        return "Qwen2.5-Math-1.5B"
    if "DeepSeek-R1-Distill-Qwen-1.5B" in run_name or "DeepSeek-R1-Distill-Qwen-1" in run_name:
        return "DeepSeek-R1-1.5B"
    return run_name.split("_")[-1]


def extract_reward_type(run_name: str) -> str:
    """Extract the reward type from run name."""
    if "ver_rule" in run_name:
        return "Oracle"
    elif "majority_vote" in run_name:
        return "Majority Vote"
    elif "entropy" in run_name:
        return "Entropy"
    return "Unknown"


def find_curl_runs(grad_variance_dir: Path) -> List[RunData]:
    """Find all curl runs with baseline data in the grad_variance directory."""
    json_files = list(grad_variance_dir.glob("vf_curl_grad_variance__*.json"))
    json_files = [f for f in json_files if "__part" not in f.name and ".partial" not in f.name]
    
    runs = []
    for json_path in json_files:
        run_data = load_run_data(json_path)
        if run_data is not None and run_data.is_curl and run_data.baseline_sigma_kept is not None:
            runs.append(run_data)
    
    return runs


def create_variance_figure(
    runs: List[RunData],
    out_dir: Path,
    variance_type: str = "sigma",  # "sigma" or "vprob"
    file_prefix: str = "vf_curl_grad_variance",
) -> Path:
    """
    Create a figure showing variance comparison for all runs.
    
    Returns the path to the saved figure.
    """
    # Colors
    curl_color = "#2166ac"  # Blue
    nocurl_color = "#b2182b"  # Red
    
    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("No runs to plot")
    
    # For 2 runs, use 1x2 layout
    if n_runs <= 2:
        fig, axes = plt.subplots(1, n_runs, figsize=(3.5 * n_runs, 2.8), squeeze=False)
        axes = axes[0]
    else:
        n_cols = min(n_runs, 3)
        n_rows = (n_runs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows))
        axes = axes.flatten() if n_runs > 1 else [axes]
    
    var_symbol = r"$\sigma_{g,t}^2$" if variance_type == "sigma" else r"$V_{\mathrm{prob},t}$"
    var_name = "Action Variance" if variance_type == "sigma" else "Problem Variance"
    
    for idx, (run, ax) in enumerate(zip(runs, axes)):
        xs = np.array(run.steps)
        
        if variance_type == "sigma":
            y_curl = run.sigma_kept
            y_nocurl = run.baseline_sigma_kept
        else:
            y_curl = run.vprob_kept
            y_nocurl = run.baseline_vprob_kept
        
        # Plot
        ax.plot(xs, y_curl, marker="o", color=curl_color, label="VI-CuRL", zorder=3)
        if y_nocurl is not None:
            ax.plot(xs, y_nocurl, marker="s", color=nocurl_color, label="No Curriculum", zorder=2)
        
        # Styling
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(var_symbol)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        
        # Scientific notation for y-axis
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3), useMathText=True)
        
        # Legend only in first plot
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)
    
    # Hide unused axes
    for idx in range(len(runs), len(axes)):
        axes[idx].set_visible(False)
    
    fig.tight_layout()
    
    # Save
    suffix = "sigma" if variance_type == "sigma" else "vprob"
    out_path = out_dir / f"{file_prefix}__{suffix}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    
    return out_path


def create_combined_figure(
    runs: List[RunData],
    out_dir: Path,
    file_prefix: str = "vf_curl_grad_variance",
) -> Path:
    """
    Create a combined figure with both Action and Problem Variance.
    Each row is a model, columns are variance type.
    """
    curl_color = "#2166ac"
    nocurl_color = "#b2182b"
    
    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("No runs to plot")
    
    fig, axes = plt.subplots(n_runs, 2, figsize=(7, 2.5 * n_runs), squeeze=False)
    
    for row_idx, run in enumerate(runs):
        xs = np.array(run.steps)
        model = extract_model_short_name(run.run_name)
        reward = extract_reward_type(run.run_name)
        
        for col_idx, (var_type, var_symbol, var_name) in enumerate([
            ("sigma", r"$\sigma_{g,t}^2$", "Action Variance"),
            ("vprob", r"$V_{\mathrm{prob},t}$", "Problem Variance"),
        ]):
            ax = axes[row_idx, col_idx]
            
            if var_type == "sigma":
                y_curl = run.sigma_kept
                y_nocurl = run.baseline_sigma_kept
            else:
                y_curl = run.vprob_kept
                y_nocurl = run.baseline_vprob_kept
            
            ax.plot(xs, y_curl, marker="o", color=curl_color, label="VI-CuRL", zorder=3)
            if y_nocurl is not None:
                ax.plot(xs, y_nocurl, marker="s", color=nocurl_color, label="No Curriculum", zorder=2)
            
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(var_symbol)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3), useMathText=True)
            
            # Row label
            if col_idx == 0:
                ax.text(-0.25, 0.5, f"{model}\n({reward})", transform=ax.transAxes,
                        va="center", ha="right", rotation=0)
            
            # Legend in first cell
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right", framealpha=0.9)
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.18)
    
    out_path = out_dir / f"{file_prefix}__combined.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready variance figures for VI-CuRL paper.")
    parser.add_argument(
        "--grad_variance_dir",
        type=Path,
        default=Path("eval_log/vi_curl/grad_variance"),
        help="Directory containing grad_variance JSON files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for plots. Defaults to grad_variance_dir parent's figures/ folder.",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="vf_curl_grad_variance",
        help="Prefix for output filenames.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by pattern (e.g., 'ver_rule' for Oracle only).",
    )
    add_font_size_args(parser)
    args = parser.parse_args()

    setup_plot_style()
    font_sizes = get_font_sizes(args)
    plt.rcParams.update(
        {
            "font.size": font_sizes["tick"],
            "axes.labelsize": font_sizes["xlabel"],
            "legend.fontsize": font_sizes["legend"],
            "xtick.labelsize": font_sizes["xtick"],
            "ytick.labelsize": font_sizes["ytick"],
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    
    grad_variance_dir = args.grad_variance_dir.expanduser().resolve()
    if args.out_dir is None:
        out_dir = grad_variance_dir
    else:
        out_dir = args.out_dir.expanduser().resolve()
    
    if not grad_variance_dir.exists():
        print(f"[ERROR] Directory not found: {grad_variance_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Scanning {grad_variance_dir} for grad_variance JSON files...")
    runs = find_curl_runs(grad_variance_dir)
    
    if args.filter:
        runs = [r for r in runs if args.filter in r.run_name]
    
    if not runs:
        print("[WARN] No curl runs with baseline data found.")
        sys.exit(0)
    
    print(f"[INFO] Found {len(runs)} curl run(s) with baseline data:")
    for run in runs:
        print(f"       - {run.run_name}")
    
    # Generate figures
    saved: List[Path] = []
    
    # Separate figures for sigma and vprob
    for var_type in ["sigma", "vprob"]:
        path = create_variance_figure(runs, out_dir, variance_type=var_type, file_prefix=args.file_prefix)
        saved.append(path)
        print(f"[SAVED] {path}")
    
    # Combined figure
    path = create_combined_figure(runs, out_dir, file_prefix=args.file_prefix)
    saved.append(path)
    print(f"[SAVED] {path}")
    
    print(f"\n[DONE] Generated {len(saved)} figures in {out_dir}")


if __name__ == "__main__":
    main()
