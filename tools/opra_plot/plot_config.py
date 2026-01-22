#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plotting configuration for OPRA visualization scripts.

This module provides consistent font settings (Times New Roman) and 
customizable font sizes for all OPRA plotting scripts.

Usage:
    from plot_config import setup_plot_style, add_font_size_args, get_font_sizes

    # In main():
    add_font_size_args(parser)
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
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Any


# Default font sizes
DEFAULT_FONT_SIZES = {
    'xlabel': 14,
    'ylabel': 14,
    'legend': 12,
    'tick': 12,
    'title': 16,  # Not used since titles are removed, but kept for reference
    'colorbar': 12,
}

# Font family for all text elements
FONT_FAMILY = 'Times New Roman'


def setup_plot_style() -> None:
    """
    Configure matplotlib to use Times New Roman font globally.
    Call this at the start of your script before creating any plots.
    """
    # Set default font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    
    # Use Type 1 fonts for PDF output (better for publication)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Ensure mathematical text also uses serif
    plt.rcParams['mathtext.fontset'] = 'stix'


def add_font_size_args(parser: argparse.ArgumentParser) -> None:
    """
    Add font size arguments to an argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--fontsize_xlabel", type=int, default=DEFAULT_FONT_SIZES['xlabel'],
        help=f"Font size for x-axis label (default: {DEFAULT_FONT_SIZES['xlabel']})"
    )
    parser.add_argument(
        "--fontsize_ylabel", type=int, default=DEFAULT_FONT_SIZES['ylabel'],
        help=f"Font size for y-axis label (default: {DEFAULT_FONT_SIZES['ylabel']})"
    )
    parser.add_argument(
        "--fontsize_legend", type=int, default=DEFAULT_FONT_SIZES['legend'],
        help=f"Font size for legend (default: {DEFAULT_FONT_SIZES['legend']})"
    )
    parser.add_argument(
        "--fontsize_tick", type=int, default=DEFAULT_FONT_SIZES['tick'],
        help=f"Font size for axis tick labels (default: {DEFAULT_FONT_SIZES['tick']})"
    )
    parser.add_argument(
        "--fontsize_colorbar", type=int, default=DEFAULT_FONT_SIZES['colorbar'],
        help=f"Font size for colorbar labels (default: {DEFAULT_FONT_SIZES['colorbar']})"
    )


def get_font_sizes(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Extract font size settings from parsed arguments.
    
    Args:
        args: Parsed argparse namespace with font size arguments
        
    Returns:
        Dictionary with font size settings including 'fontfamily'
    """
    return {
        'xlabel': getattr(args, 'fontsize_xlabel', DEFAULT_FONT_SIZES['xlabel']),
        'ylabel': getattr(args, 'fontsize_ylabel', DEFAULT_FONT_SIZES['ylabel']),
        'legend': getattr(args, 'fontsize_legend', DEFAULT_FONT_SIZES['legend']),
        'tick': getattr(args, 'fontsize_tick', DEFAULT_FONT_SIZES['tick']),
        'colorbar': getattr(args, 'fontsize_colorbar', DEFAULT_FONT_SIZES['colorbar']),
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
    ax.tick_params(axis='both', labelsize=font_sizes['tick'])
    
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
