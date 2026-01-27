#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plotting configuration for VI-CURL visualization scripts.

This module provides consistent font settings (Times New Roman) and 
customizable font sizes for all VI-CURL plotting scripts.

Usage:
    from plot_config import setup_plot_style, add_font_size_args, get_font_sizes

    # In main():
    add_font_size_args(parser)
    add_replot_args(parser)
    args = parser.parse_args()
    setup_plot_style()
    font_sizes = get_font_sizes(args)
    
    # Then use:
    ax.set_xlabel("X Label", fontsize=font_sizes['xlabel'], fontfamily=font_sizes['fontfamily'])
    ax.set_ylabel("Y Label", fontsize=font_sizes['ylabel'], fontfamily=font_sizes['fontfamily'])
    ax.legend(fontsize=font_sizes['legend'])
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Dict, Any


# Default font sizes
DEFAULT_FONT_SIZES = {
    'xlabel': 14,
    'ylabel': 14,
    'legend': 30,
    'tick': 12,
    'xtick': 28,
    'ytick': 28,
    'title': 16,  # Not used since titles are removed, but kept for reference
    'colorbar': 12,
}

SCRIPT_FONT_SIZES = {
    "plot_algos_curves.py": {
        "xlabel": 14,
        "ylabel": 14,
        "legend": 12,
        "tick": 12,
        "xtick": 12,
        "ytick": 12,
        "title": 16,
        "colorbar": 12,
    },
    "plot_vi_curl_curriculum_effects.py": {
        "xlabel": 14,
        "ylabel": 14,
        "legend": 12,
        "tick": 12,
        "xtick": 12,
        "ytick": 12,
        "title": 16,
        "colorbar": 12,
    },
    "plot_vi_curl_paper_figs.py": {
        "xlabel": 14,
        "ylabel": 14,
        "legend": 9,
        "tick": 12,
        "xtick": 12,
        "ytick": 12,
        "title": 16,
        "colorbar": 12,
    },
    "plot_bias_variance.py": {
        "xlabel": 12,
        "ylabel": 12,
        "legend": 9,
        "tick": 11,
        "xtick": 10,
        "ytick": 10,
        "title": 16,
        "colorbar": 11,
    },
    "plot_variance_absolute.py": {
        "xlabel": 22,
        "ylabel": 22,
        "legend": 20,
        "tick": 12,
        "xtick": 18,
        "ytick": 18,
        "title": 16,
        "colorbar": 12,
    },
    "plot_variance_kept_vs_full.py": {
        "xlabel": 18,
        "ylabel": 18,
        "legend": 16,
        "tick": 12,
        "xtick": 14,
        "ytick": 14,
        "title": 16,
        "colorbar": 12,
    },
    "plot_variance_for_paper.py": {
        "xlabel": 11,
        "ylabel": 11,
        "legend": 9,
        "tick": 10,
        "xtick": 9,
        "ytick": 9,
        "title": 16,
        "colorbar": 9,
    },
    "vf_curl_grad_variance.py": {
        "xlabel": 14,
        "ylabel": 14,
        "legend": 12,
        "tick": 12,
        "xtick": 12,
        "ytick": 12,
        "title": 16,
        "colorbar": 12,
    },
    "vi_curl_passk_kept_dropped.py": {
        "xlabel": 22,
        "ylabel": 22,
        "legend": 20,
        "tick": 12,
        "xtick": 18,
        "ytick": 18,
        "title": 16,
        "colorbar": 12,
    },
    "vi_curl_table.py": dict(DEFAULT_FONT_SIZES),
}

# Font family for all text elements
def _resolve_font_family(preferred: str = "Times New Roman") -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    if preferred in available:
        return preferred
    for fallback in ("Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"):
        if fallback in available:
            return fallback
    return "serif"


FONT_FAMILY = _resolve_font_family()
FONT_SERIF_FALLBACKS = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "serif",
]


def setup_plot_style() -> None:
    """
    Configure matplotlib to use Times New Roman font globally.
    Call this at the start of your script before creating any plots.
    """
    # Set default font to Times New Roman
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.serif'] = FONT_SERIF_FALLBACKS
    
    # Use Type 1 fonts for PDF output (better for publication)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Ensure mathematical text also uses serif
    plt.rcParams['mathtext.fontset'] = 'stix'


def _resolve_script_name(script_name: str | None = None) -> str:
    if script_name:
        return Path(script_name).name
    if sys.argv:
        return Path(sys.argv[0]).name
    return ""


def get_default_font_sizes(script_name: str | None = None) -> Dict[str, int]:
    defaults = dict(DEFAULT_FONT_SIZES)
    key = _resolve_script_name(script_name)
    if key in SCRIPT_FONT_SIZES:
        defaults.update(SCRIPT_FONT_SIZES[key])
    return defaults


def add_font_size_args(parser: argparse.ArgumentParser, script_name: str | None = None) -> None:
    """
    Add font size arguments to an argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    defaults = get_default_font_sizes(script_name)
    parser.add_argument(
        "--fontsize_xlabel", type=int, default=defaults['xlabel'],
        help=f"Font size for x-axis label (default: {defaults['xlabel']})"
    )
    parser.add_argument(
        "--fontsize_ylabel", type=int, default=defaults['ylabel'],
        help=f"Font size for y-axis label (default: {defaults['ylabel']})"
    )
    parser.add_argument(
        "--fontsize_legend", type=int, default=defaults['legend'],
        help=f"Font size for legend (default: {defaults['legend']})"
    )
    parser.add_argument(
        "--fontsize_tick", type=int, default=defaults['tick'],
        help=f"Font size for axis tick labels (default: {defaults['tick']})"
    )
    parser.add_argument(
        "--fontsize_xtick", type=int, default=defaults['xtick'],
        help=f"Font size for x-axis tick labels (default: {defaults['xtick']})"
    )
    parser.add_argument(
        "--fontsize_ytick", type=int, default=defaults['ytick'],
        help=f"Font size for y-axis tick labels (default: {defaults['ytick']})"
    )
    parser.add_argument(
        "--fontsize_colorbar", type=int, default=defaults['colorbar'],
        help=f"Font size for colorbar labels (default: {defaults['colorbar']})"
    )


def add_replot_args(parser: argparse.ArgumentParser) -> None:
    """
    Add replot-related arguments to an argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--replot", action="store_true", default=False,
        help="Regenerate plots from existing cached data without recomputing"
    )
    parser.add_argument(
        "--data_cache_path", type=str, default=None,
        help="Path to cached data file (CSV/NPZ) for replotting. Auto-detected if not specified."
    )


def get_font_sizes(args: argparse.Namespace, script_name: str | None = None) -> Dict[str, Any]:
    """
    Extract font size settings from parsed arguments.
    
    Args:
        args: Parsed argparse namespace with font size arguments
        
    Returns:
        Dictionary with font size settings including 'fontfamily'
    """
    defaults = get_default_font_sizes(script_name)
    return {
        'xlabel': getattr(args, 'fontsize_xlabel', defaults['xlabel']),
        'ylabel': getattr(args, 'fontsize_ylabel', defaults['ylabel']),
        'legend': getattr(args, 'fontsize_legend', defaults['legend']),
        'tick': getattr(args, 'fontsize_tick', defaults['tick']),
        'xtick': getattr(args, 'fontsize_xtick', getattr(args, 'fontsize_tick', defaults['xtick'])),
        'ytick': getattr(args, 'fontsize_ytick', getattr(args, 'fontsize_tick', defaults['ytick'])),
        'colorbar': getattr(args, 'fontsize_colorbar', defaults['colorbar']),
        'fontfamily': FONT_FAMILY,
    }


def apply_font_to_axis(
    ax: matplotlib.axes.Axes,
    font_sizes: Dict[str, Any],
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    show_legend: bool = True,
) -> None:
    """
    Apply consistent font styling to an axis.
    
    Args:
        ax: Matplotlib axis
        font_sizes: Dictionary from get_font_sizes()
        xlabel: X-axis label text (set if provided)
        ylabel: Y-axis label text (set if provided)
        zlabel: Z-axis label text for 3D plots (set if provided)
        show_legend: Whether to show legend
    """
    fontfamily = font_sizes.get('fontfamily', FONT_FAMILY)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if zlabel is not None and hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(zlabel, fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=font_sizes.get('xtick', font_sizes.get('tick')))
    ax.tick_params(axis='y', labelsize=font_sizes.get('ytick', font_sizes.get('tick')))
    
    # Apply legend with font settings
    if show_legend:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(font_sizes['legend'])
                text.set_fontfamily(fontfamily)


def create_legend(ax: matplotlib.axes.Axes, font_sizes: Dict[str, Any], **kwargs) -> None:
    """
    Create a legend with consistent font styling.
    
    Args:
        ax: Matplotlib axis
        font_sizes: Dictionary from get_font_sizes()
        **kwargs: Additional arguments passed to ax.legend()
    """
    fontfamily = font_sizes.get('fontfamily', FONT_FAMILY)
    
    # Set default legend location if not specified
    if 'loc' not in kwargs:
        kwargs['loc'] = 'best'
    
    legend = ax.legend(fontsize=font_sizes['legend'], prop={'family': fontfamily}, **kwargs)
    return legend
