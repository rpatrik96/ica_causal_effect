#!/usr/bin/env python3
"""Comprehensive script to regenerate all n20 plots.

This script regenerates all n20 plots with:
1. gennorm_heavy removed (duplicate of Laplace)
2. gennorm:1 / gennorm:1.0 removed (duplicate of Laplace)
3. Consistent color scheme
4. No empty rows in heatmaps
"""

import glob
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import utilities
from ablation_utils import (
    HOML_IDX,
    ICA_BETTER_COLOR,
    ICA_BETTER_MARKER,
    ICA_IDX,
    OML_BETTER_COLOR,
    OML_BETTER_MARKER,
    get_distribution_label,
)
from plot_utils import plot_typography


def _setup_plot():
    """Set up plot typography."""
    plot_typography(preset="publication")


def _get_comparison_color(diff_val):
    """Get color based on comparison (blue if ICA better, red if OML better)."""
    return ICA_BETTER_COLOR if diff_val < 0 else OML_BETTER_COLOR


def _get_comparison_marker(diff_val):
    """Get marker based on comparison (square if ICA better, circle if OML better)."""
    return ICA_BETTER_MARKER if diff_val < 0 else OML_BETTER_MARKER


def _scatter_with_markers(ax, x_data, y_data, diff_data, alpha=0.6, s=30):
    """Plot scatter with different colors and markers for ICA vs OML better.

    Args:
        ax: Matplotlib axis
        x_data: X coordinates
        y_data: Y coordinates
        diff_data: Difference values (negative = ICA better, positive = OML better)
        alpha: Transparency
        s: Marker size
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    diff_data = np.asarray(diff_data)

    # Split by which method is better
    ica_better_mask = diff_data < 0
    oml_better_mask = ~ica_better_mask

    # Plot ICA better points (blue squares)
    if np.any(ica_better_mask):
        ax.scatter(
            x_data[ica_better_mask],
            y_data[ica_better_mask],
            c=ICA_BETTER_COLOR,
            marker=ICA_BETTER_MARKER,
            alpha=alpha,
            s=s,
            label="ICA better",
        )

    # Plot OML better points (red circles)
    if np.any(oml_better_mask):
        ax.scatter(
            x_data[oml_better_mask],
            y_data[oml_better_mask],
            c=OML_BETTER_COLOR,
            marker=OML_BETTER_MARKER,
            alpha=alpha,
            s=s,
            label="OML better",
        )


def filter_duplicate_distributions(results):
    """Remove gennorm_heavy and gennorm:1.0 as they're duplicates of Laplace.

    Args:
        results: Dictionary with distribution results

    Returns:
        Filtered results dictionary
    """
    filtered = {}
    duplicates_found = []

    for key, value in results.items():
        # Keep metadata keys
        if key in ["treatment_coef_range", "outcome_coef_range", "treatment_effect_range"]:
            filtered[key] = value
            continue

        # Remove gennorm_heavy (beta=1, same as Laplace)
        if key == "gennorm_heavy":
            duplicates_found.append(key)
            continue

        # Remove gennorm:1 and gennorm:1.0 (same as Laplace)
        if key in ("gennorm:1", "gennorm:1.0"):
            duplicates_found.append(key)
            continue

        # Keep everything else
        filtered[key] = value

    if duplicates_found:
        print(f"  Removed duplicate distributions: {', '.join(duplicates_found)}")

    return filtered


def plot_diff_heatmap_fixed(
    results: dict,
    output_dir: str,
    n_configs: int,
    treatment_coef_range: Tuple[float, float],
    outcome_coef_range: Tuple[float, float],
    treatment_effect_range: Tuple[float, float],
    n_outcome_bins: int = 10,
):
    """Plot heatmaps with fixed binning to avoid empty rows."""
    suffix = (
        f"_n{n_configs}_tc{treatment_coef_range[0]:.1f}to{treatment_coef_range[1]:.1f}"
        f"_oc{outcome_coef_range[0]:.1f}to{outcome_coef_range[1]:.1f}"
        f"_te{treatment_effect_range[0]:.1f}to{treatment_effect_range[1]:.1f}"
    )

    os.makedirs(output_dir, exist_ok=True)
    _setup_plot()

    # Collect data from all distributions and configs
    dist_data = {}
    for dist, res in results.items():
        if res.get("config_results") is None:
            continue

        # Skip Gaussian (beta=2)
        if dist in ("gennorm:2", "gennorm:2.0"):
            continue

        # Get excess kurtosis
        excess_kurtosis = res.get("eta_empirical_excess_kurtosis", res.get("eta_excess_kurtosis", np.nan))
        if np.isnan(excess_kurtosis):
            continue

        dist_data[dist] = {
            "kurtosis": excess_kurtosis,
            "label": get_distribution_label(dist),
            "outcome_coefs": [],
            "rmse_diffs": [],
            "bias_diffs": [],
            "std_diffs": [],
        }

        for cr in res["config_results"]:
            homl_rmse = cr["rmse"][HOML_IDX]
            ica_rmse = cr["rmse"][ICA_IDX] if ICA_IDX < len(cr["rmse"]) else np.nan
            homl_bias = np.abs(cr["biases"][HOML_IDX])
            ica_bias = np.abs(cr["biases"][ICA_IDX]) if ICA_IDX < len(cr["biases"]) else np.nan
            homl_std = cr["sigmas"][HOML_IDX]
            ica_std = cr["sigmas"][ICA_IDX] if ICA_IDX < len(cr["sigmas"]) else np.nan

            dist_data[dist]["outcome_coefs"].append(cr["outcome_coef_scalar"])
            dist_data[dist]["rmse_diffs"].append(ica_rmse - homl_rmse)
            dist_data[dist]["bias_diffs"].append(ica_bias - homl_bias)
            dist_data[dist]["std_diffs"].append(ica_std - homl_std)

    if len(dist_data) == 0:
        print("No data available for heatmaps")
        return

    # Sort distributions by kurtosis
    sorted_dists = sorted(dist_data.keys(), key=lambda d: dist_data[d]["kurtosis"])
    n_dists = len(sorted_dists)

    # Get all outcome coefficients and use quantile-based binning to avoid empty bins
    all_outcome_coefs = []
    for dist in sorted_dists:
        all_outcome_coefs.extend(dist_data[dist]["outcome_coefs"])
    all_outcome_coefs = np.array(all_outcome_coefs)

    # Use quantile-based bins to ensure each bin has data
    outcome_bins = np.quantile(all_outcome_coefs, np.linspace(0, 1, n_outcome_bins + 1))
    outcome_centers = (outcome_bins[:-1] + outcome_bins[1:]) / 2

    def create_heatmap_data(metric_key):
        """Create heatmap data with distributions on x-axis and outcome bins on y-axis."""
        heatmap = np.full((n_outcome_bins, n_dists), np.nan)
        counts = np.zeros((n_outcome_bins, n_dists))

        for dist_idx, dist in enumerate(sorted_dists):
            for oc, val in zip(dist_data[dist]["outcome_coefs"], dist_data[dist][metric_key]):
                if np.isnan(val) or np.isnan(oc):
                    continue
                # Find outcome bin index
                o_idx = np.searchsorted(outcome_bins[1:], oc, side="left")
                o_idx = min(o_idx, n_outcome_bins - 1)

                if np.isnan(heatmap[o_idx, dist_idx]):
                    heatmap[o_idx, dist_idx] = val
                    counts[o_idx, dist_idx] = 1
                else:
                    # Running mean
                    counts[o_idx, dist_idx] += 1
                    heatmap[o_idx, dist_idx] += (val - heatmap[o_idx, dist_idx]) / counts[o_idx, dist_idx]

        return heatmap

    def plot_single_heatmap(data, _title, filename, cbar_label):
        """Plot a single heatmap."""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Use diverging colormap centered at 0
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )

        # Add value annotations to each cell
        for i in range(n_outcome_bins):
            for j in range(n_dists):
                if not np.isnan(data[i, j]):
                    text_color = "white" if abs(data[i, j]) > vmax * 0.5 else "black"
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=11)

        # Set x-axis ticks to distribution labels with kurtosis
        ax.set_xticks(np.arange(n_dists))
        x_labels = [f"{dist_data[d]['label']}\n(κ={dist_data[d]['kurtosis']:.2f})" for d in sorted_dists]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        # Set y-axis ticks to outcome coefficient bin centers
        ax.set_yticks(np.arange(n_outcome_bins))
        ax.set_yticklabels([f"{oc:.2f}" for oc in outcome_centers])

        ax.set_xlabel("Distribution (sorted by kurtosis)")
        ax.set_ylabel(r"Outcome Coefficient $b$")

        # Add colorbar next to the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label=cbar_label)

        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")

    # Create heatmaps for each metric
    rmse_diff_heatmap = create_heatmap_data("rmse_diffs")
    bias_diff_heatmap = create_heatmap_data("bias_diffs")
    std_diff_heatmap = create_heatmap_data("std_diffs")

    plot_single_heatmap(
        rmse_diff_heatmap,
        "RMSE Difference (ICA - OML)\n(Red = OML better, Green = ICA better)",
        f"heatmap_rmse_diff{suffix}.svg",
        "RMSE Diff",
    )

    plot_single_heatmap(
        bias_diff_heatmap,
        r"$|\mathrm{Bias}|$ Difference (ICA - OML)" + "\n(Red = OML better, Green = ICA better)",
        f"heatmap_bias_diff{suffix}.svg",
        r"$|\mathrm{Bias}|$ Diff",
    )

    plot_single_heatmap(
        std_diff_heatmap,
        "Std Difference (ICA - OML)\n(Red = OML better, Green = ICA better)",
        f"heatmap_std_diff{suffix}.svg",
        "Std Diff",
    )

    # Create combined heatmap
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    metrics_data = [
        (rmse_diff_heatmap, "RMSE Diff", "RMSE Diff"),
        (bias_diff_heatmap, r"$|\mathrm{Bias}|$ Diff", "Bias Diff"),
        (std_diff_heatmap, "Std Diff", "Std Diff"),
    ]

    for ax, (data, title, cbar_label) in zip(axes, metrics_data):
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )

        # Add value annotations to each cell
        for i in range(n_outcome_bins):
            for j in range(n_dists):
                if not np.isnan(data[i, j]):
                    text_color = "white" if abs(data[i, j]) > vmax * 0.5 else "black"
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=10)

        ax.set_xticks(np.arange(n_dists))
        x_labels = [f"{dist_data[d]['label']}\n(κ={dist_data[d]['kurtosis']:.2f})" for d in sorted_dists]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        ax.set_yticks(np.arange(n_outcome_bins))
        ax.set_yticklabels([f"{oc:.2f}" for oc in outcome_centers])

        ax.set_xlabel("Distribution")
        ax.set_ylabel(r"Outcome Coeff $b$")
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label=cbar_label)
    plt.savefig(os.path.join(output_dir, f"heatmap_combined{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap_combined{suffix}.svg")


def plot_coeff_scatter(
    results: dict,
    output_dir: str,
    n_configs: int,
    treatment_coef_range: Tuple[float, float],
    outcome_coef_range: Tuple[float, float],
    treatment_effect_range: Tuple[float, float],
):
    """Plot coefficient scatter plots with filtered distributions."""
    suffix = (
        f"_n{n_configs}_tc{treatment_coef_range[0]:.1f}to{treatment_coef_range[1]:.1f}"
        f"_oc{outcome_coef_range[0]:.1f}to{outcome_coef_range[1]:.1f}"
        f"_te{treatment_effect_range[0]:.1f}to{treatment_effect_range[1]:.1f}"
    )

    os.makedirs(output_dir, exist_ok=True)
    _setup_plot()

    # Collect data from all distributions and configs
    all_data = []
    for dist, res in results.items():
        if res.get("config_results") is None:
            continue
        for cr in res["config_results"]:
            homl_rmse = cr["rmse"][HOML_IDX]
            ica_rmse = cr["rmse"][ICA_IDX] if ICA_IDX < len(cr["rmse"]) else np.nan
            homl_bias = np.abs(cr["biases"][HOML_IDX])
            ica_bias = np.abs(cr["biases"][ICA_IDX]) if ICA_IDX < len(cr["biases"]) else np.nan
            homl_std = cr["sigmas"][HOML_IDX]
            ica_std = cr["sigmas"][ICA_IDX] if ICA_IDX < len(cr["sigmas"]) else np.nan
            all_data.append(
                {
                    "distribution": dist,
                    "treatment_coef": cr["treatment_coef_scalar"],
                    "outcome_coef": cr["outcome_coef_scalar"],
                    "treatment_effect": cr["treatment_effect"],
                    "ica_var_coeff": cr["ica_var_coeff"],
                    "rmse_diff": ica_rmse - homl_rmse,
                    "bias_diff": ica_bias - homl_bias,
                    "std_diff": ica_std - homl_std,
                    "homl_rmse": homl_rmse,
                    "ica_rmse": ica_rmse,
                }
            )

    if len(all_data) == 0:
        print("No coefficient data available for scatter plots")
        return

    # Convert to arrays
    ica_var_coeffs = np.array([d["ica_var_coeff"] for d in all_data])
    rmse_diffs = np.array([d["rmse_diff"] for d in all_data])
    bias_diffs = np.array([d["bias_diff"] for d in all_data])
    treatment_effects = np.array([d["treatment_effect"] for d in all_data])
    treatment_coefs = np.array([d["treatment_coef"] for d in all_data])
    outcome_coefs = np.array([d["outcome_coef"] for d in all_data])

    # Create scatter grid for RMSE
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    _scatter_with_markers(ax, treatment_effects, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Effect $\theta$", fontsize=20)
    ax.set_ylabel("RMSE Diff", fontsize=20)
    ax.set_title(r"RMSE Diff vs $\theta$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _scatter_with_markers(ax, treatment_coefs, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Coef $a$", fontsize=20)
    ax.set_ylabel("RMSE Diff", fontsize=20)
    ax.set_title(r"RMSE Diff vs $a$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _scatter_with_markers(ax, outcome_coefs, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Outcome Coef $b$", fontsize=20)
    ax.set_ylabel("RMSE Diff", fontsize=20)
    ax.set_title(r"RMSE Diff vs $b$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    _scatter_with_markers(ax, ica_var_coeffs, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff", fontsize=20)
    ax.set_ylabel("RMSE Diff", fontsize=20)
    ax.set_title(r"RMSE Diff vs ICA Var Coeff", fontsize=24)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add legend outside the subplots (bottom center)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_rmse_grid{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved coeff_scatter_rmse_grid{suffix}.svg")

    # Create scatter grid for Bias
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    _scatter_with_markers(ax, treatment_effects, bias_diffs, bias_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Effect $\theta$", fontsize=20)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=20)
    ax.set_title(r"$|\mathrm{Bias}|$ Diff vs $\theta$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _scatter_with_markers(ax, treatment_coefs, bias_diffs, bias_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Coef $a$", fontsize=20)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=20)
    ax.set_title(r"$|\mathrm{Bias}|$ Diff vs $a$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _scatter_with_markers(ax, outcome_coefs, bias_diffs, bias_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Outcome Coef $b$", fontsize=20)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=20)
    ax.set_title(r"$|\mathrm{Bias}|$ Diff vs $b$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    _scatter_with_markers(ax, ica_var_coeffs, bias_diffs, bias_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff", fontsize=20)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=20)
    ax.set_title(r"$|\mathrm{Bias}|$ Diff vs ICA Var Coeff", fontsize=24)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add legend outside the subplots (bottom center)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_bias_grid{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved coeff_scatter_bias_grid{suffix}.svg")

    print(f"Coefficient scatter plots saved to {output_dir}")


def process_n20_files():
    """Process all n20 files in noise_ablation directory."""
    noise_ablation_dir = "figures/noise_ablation"

    # Find all n20 .npy files
    pattern = os.path.join(noise_ablation_dir, "noise_ablation_results_n20_*.npy")
    n20_files = glob.glob(pattern)

    print(f"Found {len(n20_files)} n20 results files")

    for results_file in n20_files:
        print(f"\nProcessing {os.path.basename(results_file)}")
        filename = os.path.basename(results_file)

        # Load results
        results = np.load(results_file, allow_pickle=True).item()

        # Filter duplicate distributions
        results = filter_duplicate_distributions(results)

        # Extract parameters from filename
        # Try parsing tc, oc, te pattern first
        tc_match = re.search(r"tc(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        oc_match = re.search(r"oc(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        te_match = re.search(r"te(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)

        if tc_match and oc_match:
            # Has separate tc and oc
            treatment_coef_range = (float(tc_match.group(1)), float(tc_match.group(2)))
            outcome_coef_range = (float(oc_match.group(1)), float(oc_match.group(2)))
        else:
            # Try parsing 'coef' pattern (both tc and oc use same range)
            coef_match = re.search(r"coef(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
            if coef_match:
                treatment_coef_range = (float(coef_match.group(1)), float(coef_match.group(2)))
                outcome_coef_range = treatment_coef_range
            else:
                # Default values
                treatment_coef_range = (-2.0, 2.0)
                outcome_coef_range = (-2.0, 2.0)

        if te_match:
            treatment_effect_range = (float(te_match.group(1)), float(te_match.group(2)))
        else:
            treatment_effect_range = (0.1, 5.0)

        # Count configs
        n_configs = 0
        for res in results.values():
            if isinstance(res, dict) and "config_results" in res:
                n_configs = len(res["config_results"])
                break

        print(f"  Parameters: tc={treatment_coef_range}, oc={outcome_coef_range}, te={treatment_effect_range}")
        print(f"  Number of configs: {n_configs}")
        print(
            f"  Distributions: {[k for k in results if k not in ['treatment_coef_range', 'outcome_coef_range', 'treatment_effect_range']]}"
        )

        # Regenerate heatmaps
        plot_diff_heatmap_fixed(
            results,
            noise_ablation_dir,
            n_configs,
            treatment_coef_range,
            outcome_coef_range,
            treatment_effect_range,
            n_outcome_bins=10,
        )

        # Regenerate scatter plots
        plot_coeff_scatter(
            results,
            noise_ablation_dir,
            n_configs,
            treatment_coef_range,
            outcome_coef_range,
            treatment_effect_range,
        )


if __name__ == "__main__":
    process_n20_files()
    print("\nDone!")
