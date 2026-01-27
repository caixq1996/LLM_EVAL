#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plotting configuration for OPRA visualization scripts.

This module provides consistent font settings (Times New Roman) and 
customizable font sizes for all OPRA plotting scripts.

Usage:
    from plot_config import setup_plot_style, add_font_size_args, get_font_sizes, get_plot_visibility

    # In main():
    add_font_size_args(parser)
    args = parser.parse_args()
    setup_plot_style()
    font_sizes = get_font_sizes(args)
    plot_visibility = get_plot_visibility(args)
    
    # Then use:
    if plot_visibility['show_xlabel']:
        ax.set_xlabel("X Label", fontsize=font_sizes['xlabel'], fontfamily=font_sizes['fontfamily'])
    if plot_visibility['show_ylabel']:
        ax.set_ylabel("Y Label", fontsize=font_sizes['ylabel'], fontfamily=font_sizes['fontfamily'])
    if plot_visibility['show_legend']:
        ax.legend(fontsize=font_sizes['legend'])
"""
from __future__ import annotations

import argparse
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple


# Default font sizes
DEFAULT_FONT_SIZES = {
    'xlabel': 14,
    'ylabel': 14,
    'legend': 12,
    'tick': 12,
    'xtick': 12,
    'ytick': 12,
    'title': 16,
    'colorbar': 12,
}

# Default visibility options for plot elements
DEFAULT_PLOT_VISIBILITY = {
    'show_title': False,
    'show_xlabel': True,
    'show_ylabel': True,
    'show_legend': True,
}

# Font family for all text elements
FONT_FAMILY = 'Times New Roman'


def _merge_defaults(defaults: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sizes = DEFAULT_FONT_SIZES.copy()
    visibility = DEFAULT_PLOT_VISIBILITY.copy()
    if defaults:
        for key, value in defaults.items():
            if key in sizes:
                sizes[key] = value
            elif key in visibility:
                visibility[key] = value
    return sizes, visibility


def _add_visibility_arg(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    description: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--show_{name}",
        dest=f"show_{name}",
        action="store_true",
        help=f"Show {description} (default: {default})",
    )
    group.add_argument(
        f"--hide_{name}",
        dest=f"show_{name}",
        action="store_false",
        help=f"Hide {description} (default: {default})",
    )
    parser.set_defaults(**{f"show_{name}": default})


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


def add_font_size_args(parser: argparse.ArgumentParser, defaults: Optional[Dict[str, Any]] = None) -> None:
    """
    Add font size arguments to an argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
        defaults: Optional overrides for font sizes or visibility flags
    """
    sizes, visibility = _merge_defaults(defaults)
    parser.add_argument(
        "--fontsize_xlabel", type=int, default=sizes['xlabel'],
        help=f"Font size for x-axis label (default: {sizes['xlabel']})"
    )
    parser.add_argument(
        "--fontsize_ylabel", type=int, default=sizes['ylabel'],
        help=f"Font size for y-axis label (default: {sizes['ylabel']})"
    )
    parser.add_argument(
        "--fontsize_legend", type=int, default=sizes['legend'],
        help=f"Font size for legend (default: {sizes['legend']})"
    )
    parser.add_argument(
        "--fontsize_tick", type=int, default=sizes['tick'],
        help=f"Font size for axis tick labels (default: {sizes['tick']})"
    )
    parser.add_argument(
        "--fontsize_xtick", type=int, default=sizes['xtick'],
        help=f"Font size for x-axis tick labels (default: {sizes['xtick']})"
    )
    parser.add_argument(
        "--fontsize_ytick", type=int, default=sizes['ytick'],
        help=f"Font size for y-axis tick labels (default: {sizes['ytick']})"
    )
    parser.add_argument(
        "--fontsize_title", type=int, default=sizes['title'],
        help=f"Font size for titles (default: {sizes['title']})"
    )
    parser.add_argument(
        "--fontsize_colorbar", type=int, default=sizes['colorbar'],
        help=f"Font size for colorbar labels (default: {sizes['colorbar']})"
    )

    _add_visibility_arg(parser, "title", visibility['show_title'], "plot title")
    _add_visibility_arg(parser, "xlabel", visibility['show_xlabel'], "x-axis label")
    _add_visibility_arg(parser, "ylabel", visibility['show_ylabel'], "y-axis label")
    _add_visibility_arg(parser, "legend", visibility['show_legend'], "legend")


def get_font_sizes(args: argparse.Namespace, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract font size settings from parsed arguments.
    
    Args:
        args: Parsed argparse namespace with font size arguments
        defaults: Optional overrides for font sizes or visibility flags
        
    Returns:
        Dictionary with font size settings including 'fontfamily'
    """
    sizes, _ = _merge_defaults(defaults)
    tick_default = sizes['tick']
    xtick_default = sizes.get('xtick', tick_default)
    ytick_default = sizes.get('ytick', tick_default)
    tick = getattr(args, 'fontsize_tick', tick_default)
    xtick = getattr(args, 'fontsize_xtick', xtick_default)
    ytick = getattr(args, 'fontsize_ytick', ytick_default)
    if tick != tick_default:
        if xtick == xtick_default:
            xtick = tick
        if ytick == ytick_default:
            ytick = tick
    return {
        'xlabel': getattr(args, 'fontsize_xlabel', sizes['xlabel']),
        'ylabel': getattr(args, 'fontsize_ylabel', sizes['ylabel']),
        'legend': getattr(args, 'fontsize_legend', sizes['legend']),
        'tick': tick,
        'xtick': xtick,
        'ytick': ytick,
        'title': getattr(args, 'fontsize_title', sizes['title']),
        'colorbar': getattr(args, 'fontsize_colorbar', sizes['colorbar']),
        'fontfamily': FONT_FAMILY,
    }


def get_plot_visibility(args: argparse.Namespace, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    Extract plot element visibility flags from parsed arguments.

    Args:
        args: Parsed argparse namespace with visibility arguments
        defaults: Optional overrides for visibility flags

    Returns:
        Dictionary with visibility flags: show_title/show_xlabel/show_ylabel/show_legend
    """
    _, visibility = _merge_defaults(defaults)
    return {
        'show_title': getattr(args, 'show_title', visibility['show_title']),
        'show_xlabel': getattr(args, 'show_xlabel', visibility['show_xlabel']),
        'show_ylabel': getattr(args, 'show_ylabel', visibility['show_ylabel']),
        'show_legend': getattr(args, 'show_legend', visibility['show_legend']),
    }


def apply_font_to_axis(
    ax: matplotlib.axes.Axes,
    font_sizes: Dict[str, Any],
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    show_legend: bool = True,
    title: str = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_title: bool = True,
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
        title: Title text (set if provided)
        show_xlabel: Whether to show x-axis label
        show_ylabel: Whether to show y-axis label
        show_title: Whether to show title
    """
    fontfamily = font_sizes.get('fontfamily', FONT_FAMILY)
    
    if xlabel is not None and show_xlabel:
        ax.set_xlabel(xlabel, fontsize=font_sizes['xlabel'], fontfamily=fontfamily)
    if ylabel is not None and show_ylabel:
        ax.set_ylabel(ylabel, fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if zlabel is not None and hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(zlabel, fontsize=font_sizes['ylabel'], fontfamily=fontfamily)
    if title is not None and show_title:
        ax.set_title(title, fontsize=font_sizes['title'], fontfamily=fontfamily)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=font_sizes.get('xtick', font_sizes['tick']))
    ax.tick_params(axis='y', labelsize=font_sizes.get('ytick', font_sizes['tick']))
    if hasattr(ax, 'zaxis'):
        ax.tick_params(axis='z', labelsize=font_sizes.get('ytick', font_sizes['tick']))
    
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
