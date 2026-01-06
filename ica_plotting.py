"""
Plotting utilities for ICA experiments.

This module provides consolidated plotting functions used across
different ICA experiments.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tueplots import bundles

from plot_utils import add_legend_outside, plot_typography


def setup_experiment_environment():
    """Configure matplotlib and typography for experiments.

    This function consolidates the repeated setup code from all main_* functions.
    """
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography(preset="publication")


def save_figure(filename: str, output_dir: str = "figures/ica"):
    """Save figure to the specified directory.

    Args:
        filename: Name of the file to save
        output_dir: Directory to save the figure in (created if doesn't exist)
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    data: np.ndarray,
    x_labels: List,
    y_labels: List,
    x_label: str,
    y_label: str,
    filename: str,
    center: float = 0,
    output_dir: str = "figures/ica",
    annot_fontsize: int = 12,
    figsize: tuple = (10, 8),
):
    """Create and save a heatmap plot.

    Args:
        data: 2D array of data to plot
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        x_label: Label for x-axis
        y_label: Label for y-axis
        filename: Filename to save the plot
        center: Center value for color map
        output_dir: Directory to save the figure in
        annot_fontsize: Font size for cell annotations
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fontsize},
        center=center,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    save_figure(filename, output_dir)


def plot_error_bars(
    x_values: List,
    means: List[float],
    std_devs: List[float],
    xlabel: str,
    ylabel: str,
    filename: str,
    x_ticks: Optional[List] = None,
    labels: Optional[List[str]] = None,
    output_dir: str = "figures/ica",
    use_log_scale: bool = True,
    figsize: tuple = (10, 6),
):
    """Create and save an error bar plot.

    Args:
        x_values: X-axis values
        means: Mean values for each x
        std_devs: Standard deviations for each x
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        filename: Filename to save the plot
        x_ticks: Optional custom x-tick positions
        labels: Optional labels for data points
        output_dir: Directory to save the figure in
        use_log_scale: Whether to use log scale for y-axis
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    bar_positions = np.arange(len(x_values))

    plt.errorbar(bar_positions, means, yerr=std_devs, fmt="o", capsize=5)

    if x_ticks is not None:
        plt.xticks(bar_positions, x_ticks, fontsize=11)
    else:
        # Format x-ticks based on value type
        if all(isinstance(x, (int, float)) for x in x_values):
            if all(isinstance(x, int) or x == int(x) for x in x_values):
                plt.xticks(bar_positions, [f"{int(x)}" for x in x_values], fontsize=11)
            else:
                plt.xticks(bar_positions, [f"{x:.1f}" for x in x_values], fontsize=11)
        else:
            plt.xticks(bar_positions, x_values, fontsize=11)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if use_log_scale:
        plt.yscale("log")

    if labels:
        add_legend_outside(plt.gca())

    save_figure(filename, output_dir)


def plot_multiple_error_bars(
    parameter_values: List,
    series_data: dict,
    xlabel: str,
    ylabel: str,
    filename: str,
    output_dir: str = "figures/ica",
    use_log_scale: bool = True,
    figsize: tuple = (10, 6),
):
    """Create error bar plot with multiple series.

    Args:
        parameter_values: X-axis parameter values
        series_data: Dict mapping series name to (means, stds) tuples
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        filename: Filename to save the plot
        output_dir: Directory to save the figure in
        use_log_scale: Whether to use log scale for y-axis
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    bar_positions = np.arange(len(parameter_values))

    for series_name, (means, stds) in series_data.items():
        plt.errorbar(bar_positions, means, yerr=stds, fmt="o-", capsize=5, label=series_name)

    # Format x-ticks based on value type
    if all(isinstance(x, (int, float)) for x in parameter_values):
        if all(isinstance(x, int) or x == int(x) for x in parameter_values):
            plt.xticks(bar_positions, [f"{int(x)}" for x in parameter_values], fontsize=11)
        else:
            plt.xticks(bar_positions, [f"{x:.1f}" for x in parameter_values], fontsize=11)
    else:
        plt.xticks(bar_positions, parameter_values, fontsize=11)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if use_log_scale:
        plt.yscale("log")

    add_legend_outside(plt.gca())
    save_figure(filename, output_dir)


def plot_runtime_comparison(
    parameter_values: List,
    series_data: dict,
    xlabel: str,
    filename: str,
    output_dir: str = "figures/ica",
    use_log_scale: bool = False,
    figsize: tuple = (10, 6),
):
    """Create runtime comparison bar plot with multiple series.

    Args:
        parameter_values: X-axis parameter values
        series_data: Dict mapping series name to (means, stds) tuples of runtime in seconds
        xlabel: Label for x-axis
        filename: Filename to save the plot
        output_dir: Directory to save the figure in
        use_log_scale: Whether to use log scale for y-axis
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    bar_positions = np.arange(len(parameter_values))
    n_series = len(series_data)
    bar_width = 0.8 / n_series

    for i, (series_name, (means, stds)) in enumerate(series_data.items()):
        offset = (i - n_series / 2 + 0.5) * bar_width
        plt.bar(
            bar_positions + offset,
            means,
            bar_width,
            yerr=stds,
            label=series_name,
            capsize=3,
            alpha=0.8,
        )

    # Format x-ticks based on value type
    if all(isinstance(x, (int, float)) for x in parameter_values):
        if all(isinstance(x, int) or x == int(x) for x in parameter_values):
            plt.xticks(bar_positions, [f"{int(x)}" for x in parameter_values], fontsize=11)
        else:
            plt.xticks(bar_positions, [f"{x:.1f}" for x in parameter_values], fontsize=11)
    else:
        plt.xticks(bar_positions, parameter_values, fontsize=11)

    plt.xlabel(xlabel)
    plt.ylabel("Runtime (seconds)")

    if use_log_scale:
        plt.yscale("log")

    add_legend_outside(plt.gca())
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_figure(filename, output_dir)
