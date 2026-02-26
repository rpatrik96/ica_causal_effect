"""
Plotting functions for noise distribution and coefficient ablation studies.

This module contains all visualization functions for the ablation experiments,
including heatmaps, scatter plots, and summary figures.
"""

import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tueplots import bundles

from ablation_utils import (
    HOML_IDX,
    ICA_BETTER_COLOR,
    ICA_BETTER_MARKER,
    ICA_COLOR,
    ICA_IDX,
    OML_BETTER_COLOR,
    OML_BETTER_MARKER,
    OML_COLOR,
    get_distribution_label,
)
from plot_utils import add_legend_outside, plot_typography

# =============================================================================
# Plotting Utilities
# =============================================================================


def _setup_plot():
    """Initialize matplotlib settings for publication-quality plots."""
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography(preset="publication")


def _get_comparison_color(diff: float, ica_better: bool = True) -> str:
    """Get color based on comparison result.

    Args:
        diff: Difference value (ICA - HOML)
        ica_better: If True, negative diff means ICA is better

    Returns:
        Color string
    """
    if ica_better:
        return ICA_BETTER_COLOR if diff < 0 else OML_BETTER_COLOR
    return OML_BETTER_COLOR if diff < 0 else ICA_BETTER_COLOR


def _get_comparison_marker(diff: float, ica_better: bool = True) -> str:
    """Get marker based on comparison result.

    Args:
        diff: Difference value (ICA - HOML)
        ica_better: If True, negative diff means ICA is better

    Returns:
        Marker string
    """
    if ica_better:
        return ICA_BETTER_MARKER if diff < 0 else OML_BETTER_MARKER
    return OML_BETTER_MARKER if diff < 0 else ICA_BETTER_MARKER


def _scatter_with_markers(ax, x_data, y_data, diff_data, alpha=0.6, s=30, ica_better=True):
    """Plot scatter with different colors and markers for ICA vs OML better.

    Args:
        ax: Matplotlib axis
        x_data: X coordinates
        y_data: Y coordinates
        diff_data: Difference values (negative = ICA better when ica_better=True)
        alpha: Transparency
        s: Marker size
        ica_better: If True, negative diff means ICA is better
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    diff_data = np.asarray(diff_data)

    # Split by which method is better
    if ica_better:
        ica_better_mask = diff_data < 0
    else:
        ica_better_mask = diff_data > 0
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


def plot_distribution_diff_heatmap(results: dict, output_dir: str = "figures/noise_ablation"):
    """Plot a summary heatmap showing differences across distributions and metrics.

    Creates a heatmap with distributions on x-axis (sorted by kurtosis) and metrics on y-axis,
    showing the ICA - HOML difference for each.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    _setup_plot()

    # Extract data
    dist_data = []
    for dist, res in results.items():
        if "rmse" not in res or "biases" not in res or "sigmas" not in res:
            continue

        # Skip Gaussian (beta=2) - ICA doesn't work for Gaussian
        if dist in ("gennorm:2", "gennorm:2.0"):
            continue

        # Skip duplicate distributions
        # - gennorm_heavy, gennorm:1, gennorm:1.0: Same as Laplace (beta=1)
        # - gennorm_light: Same as gennorm:4.0 (beta=4)
        if dist in ("gennorm_heavy", "gennorm:1", "gennorm:1.0", "gennorm_light"):
            continue

        kurtosis = res.get("eta_empirical_excess_kurtosis", res.get("eta_excess_kurtosis", np.nan))
        homl_rmse = res["rmse"][HOML_IDX]
        ica_rmse = res["rmse"][ICA_IDX] if ICA_IDX < len(res["rmse"]) else np.nan
        homl_bias = np.abs(res["biases"][HOML_IDX])
        ica_bias = np.abs(res["biases"][ICA_IDX]) if ICA_IDX < len(res["biases"]) else np.nan
        homl_std = res["sigmas"][HOML_IDX]
        ica_std = res["sigmas"][ICA_IDX] if ICA_IDX < len(res["sigmas"]) else np.nan

        dist_data.append(
            {
                "distribution": dist,
                "label": get_distribution_label(dist),
                "kurtosis": kurtosis,
                "rmse_diff": ica_rmse - homl_rmse,
                "bias_diff": ica_bias - homl_bias,
                "std_diff": ica_std - homl_std,
            }
        )

    if len(dist_data) == 0:
        print("No data available for distribution diff heatmap")
        return

    # Sort by kurtosis (descending)
    dist_data.sort(key=lambda x: x["kurtosis"] if not np.isnan(x["kurtosis"]) else float("-inf"), reverse=True)

    # Create heatmap data
    labels = [d["label"] for d in dist_data]
    kurtosis_vals = [d["kurtosis"] for d in dist_data]
    metrics = ["RMSE", r"$|\mathrm{Bias}|$", "Std"]
    heatmap_data = np.array([[d["rmse_diff"], d["bias_diff"], d["std_diff"]] for d in dist_data]).T

    # Create figure with extra width for colorbar
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.4), 5))

    # Diverging colormap centered at 0
    vmax = np.nanmax(np.abs(heatmap_data))
    vmin = -vmax

    im = ax.imshow(heatmap_data, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([f"{lbl}\n(\u03ba={k:.2f})" for lbl, k in zip(labels, kurtosis_vals)], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)

    # Add value annotations
    for i in range(len(metrics)):
        for j in range(len(labels)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color)

    ax.set_xlabel("Treatment noise distribution (sorted by kurtosis)")

    # Add colorbar next to the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="ICA - OML")

    plt.savefig(os.path.join(output_dir, "distribution_diff_heatmap.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nDistribution diff heatmap saved to {output_dir}")

    # Print markdown table
    print("\n## Distribution Differences (ICA - HOML)")
    print("\n| Distribution | Kurtosis | RMSE Diff | |Bias| Diff | Std Diff | Winner |")
    print("|--------------|----------|-----------|------------|----------|--------|")
    for d in dist_data:
        rmse_diff = d["rmse_diff"]
        bias_diff = d["bias_diff"]
        std_diff = d["std_diff"]
        # Determine winner based on RMSE
        winner = "ICA" if rmse_diff < 0 else "HOML"
        row = (
            f"| {d['label']:20s} | {d['kurtosis']:8.4f}"
            f" | {rmse_diff:9.4f} | {bias_diff:10.4f}"
            f" | {std_diff:8.4f} | {winner:6s} |"
        )
        print(row)

    # Save markdown to file
    md_path = os.path.join(output_dir, "distribution_diff_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Distribution Differences (ICA - HOML)\n\n")
        f.write("Negative values indicate ICA is better, positive values indicate HOML is better.\n\n")
        f.write("| Distribution | Kurtosis | RMSE Diff | |Bias| Diff | Std Diff | Winner |\n")
        f.write("|--------------|----------|-----------|------------|----------|--------|\n")
        for d in dist_data:
            rmse_diff = d["rmse_diff"]
            bias_diff = d["bias_diff"]
            std_diff = d["std_diff"]
            winner = "ICA" if rmse_diff < 0 else "HOML"
            f.write(
                f"| {d['label']:20s} | {d['kurtosis']:.4f}"
                f" | {rmse_diff:.4f} | {bias_diff:.4f}"
                f" | {std_diff:.4f} | {winner} |\n"
            )
    print(f"\nMarkdown summary saved to {md_path}")


def plot_noise_ablation_coeff_scatter(
    results: dict,
    output_dir: str = "figures/noise_ablation",
    n_configs: int = 0,
    treatment_coef_range: Tuple[float, float] = (-2.0, 2.0),
    outcome_coef_range: Tuple[float, float] = (-2.0, 2.0),
    treatment_effect_range: Tuple[float, float] = (0.1, 5.0),
):
    """Plot RMSE/bias differences vs coefficient values when using randomized coefficients.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
        n_configs: Number of random configs (for filename)
        treatment_coef_range: Range for treatment coefficients
        outcome_coef_range: Range for outcome coefficients
        treatment_effect_range: Range for treatment effect
    """
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
            all_data.append(
                {
                    "distribution": dist,
                    "treatment_coef": cr["treatment_coef_scalar"],
                    "outcome_coef": cr["outcome_coef_scalar"],
                    "treatment_effect": cr["treatment_effect"],
                    "ica_var_coeff": cr["ica_var_coeff"],
                    "rmse_diff": ica_rmse - homl_rmse,
                    "bias_diff": ica_bias - homl_bias,
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
    treatment_effects = np.array([d["treatment_effect"] for d in all_data])
    treatment_coefs = np.array([d["treatment_coef"] for d in all_data])
    outcome_coefs = np.array([d["outcome_coef"] for d in all_data])
    distributions = [d["distribution"] for d in all_data]

    unique_dists = list(results.keys())
    cmap = plt.cm.tab10
    dist_colors = {d: cmap(i % 10) for i, d in enumerate(unique_dists)}

    # Plot 1: RMSE diff vs ICA variance coefficient by distribution
    _, ax = plt.subplots(figsize=(8, 5))
    for dist in unique_dists:
        mask = [d == dist for d in distributions]
        if not any(mask):
            continue
        x_vals = ica_var_coeffs[mask]
        y_vals = rmse_diffs[mask]
        ax.scatter(x_vals, y_vals, c=[dist_colors[dist]], alpha=0.6, s=40, label=dist[:12])

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel("RMSE Diff")
    ax.set_xscale("log")
    add_legend_outside(ax, loc="right", ncol=2)
    ax.set_title("RMSE Diff vs ICA Var Coeff (by distribution)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_rmse_vs_ica_var{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: 2x2 grid showing RMSE diff vs each coefficient
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    _scatter_with_markers(ax, treatment_effects, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Effect $\theta$")
    ax.set_ylabel("RMSE Diff")
    ax.set_title(r"RMSE Diff vs $\theta$")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _scatter_with_markers(ax, treatment_coefs, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Coef $a$")
    ax.set_ylabel("RMSE Diff")
    ax.set_title(r"RMSE Diff vs $a$")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _scatter_with_markers(ax, outcome_coefs, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Outcome Coef $b$")
    ax.set_ylabel("RMSE Diff")
    ax.set_title(r"RMSE Diff vs $b$")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    _scatter_with_markers(ax, ica_var_coeffs, rmse_diffs, rmse_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff")
    ax.set_ylabel("RMSE Diff")
    ax.set_xscale("log")
    ax.set_title(r"RMSE Diff vs ICA Var Coeff")
    ax.grid(True, alpha=0.3)

    # Add legend outside the subplots (bottom center)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, ["ICA better", "OML better"], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)

    fig.suptitle("RMSE Difference (ICA - OML) vs Coefficient Values\n(Blue = ICA better, Red = OML better)")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for legend and suptitle
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_rmse_grid{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nCoefficient scatter plots saved to {output_dir}")


def plot_noise_ablation_std_scatter(
    results: dict,
    output_dir: str = "figures/noise_ablation",
    n_configs: int = 0,
    treatment_coef_range: Tuple[float, float] = (-2.0, 2.0),
    outcome_coef_range: Tuple[float, float] = (-2.0, 2.0),
    treatment_effect_range: Tuple[float, float] = (0.1, 5.0),
):
    """Plot standard deviation differences vs coefficient values.

    Creates a 2x2 scatter grid showing how std difference (ICA - HOML) varies with
    treatment effect, treatment coefficient, outcome coefficient, and ICA variance coefficient.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
        n_configs: Number of random configs (for filename)
        treatment_coef_range: Range for treatment coefficients
        outcome_coef_range: Range for outcome coefficients
        treatment_effect_range: Range for treatment effect
    """
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
            homl_std = cr["sigmas"][HOML_IDX]
            ica_std = cr["sigmas"][ICA_IDX] if ICA_IDX < len(cr["sigmas"]) else np.nan
            all_data.append(
                {
                    "distribution": dist,
                    "treatment_coef": cr["treatment_coef_scalar"],
                    "outcome_coef": cr["outcome_coef_scalar"],
                    "treatment_effect": cr["treatment_effect"],
                    "ica_var_coeff": cr["ica_var_coeff"],
                    "std_diff": ica_std - homl_std,
                    "homl_std": homl_std,
                    "ica_std": ica_std,
                }
            )

    if len(all_data) == 0:
        print("No coefficient data available for std scatter plots")
        return

    # Convert to arrays
    ica_var_coeffs = np.array([d["ica_var_coeff"] for d in all_data])
    std_diffs = np.array([d["std_diff"] for d in all_data])
    treatment_effects = np.array([d["treatment_effect"] for d in all_data])
    treatment_coefs = np.array([d["treatment_coef"] for d in all_data])
    outcome_coefs = np.array([d["outcome_coef"] for d in all_data])

    # 2x2 grid showing Std diff vs each coefficient
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    _scatter_with_markers(ax, treatment_effects, std_diffs, std_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Effect $\theta$")
    ax.set_ylabel("Std Diff")
    ax.set_title(r"Std Diff vs $\theta$")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _scatter_with_markers(ax, treatment_coefs, std_diffs, std_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Coef $a$")
    ax.set_ylabel("Std Diff")
    ax.set_title(r"Std Diff vs $a$")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _scatter_with_markers(ax, outcome_coefs, std_diffs, std_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Outcome Coef $b$")
    ax.set_ylabel("Std Diff")
    ax.set_title(r"Std Diff vs $b$")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    _scatter_with_markers(ax, ica_var_coeffs, std_diffs, std_diffs)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff")
    ax.set_ylabel("Std Diff")
    ax.set_xscale("log")
    ax.set_title(r"Std Diff vs ICA Var Coeff")
    ax.grid(True, alpha=0.3)

    # Add legend outside the subplots (bottom center)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, ["ICA better", "OML better"], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)

    fig.suptitle("Std Difference (ICA - OML) vs Coefficient Values\n(Blue = ICA better, Red = OML better)")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for legend and suptitle
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_std_grid{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nStd scatter plots saved to {output_dir}")


def plot_diff_heatmaps(
    results: dict,
    output_dir: str = "figures/noise_ablation",
    n_configs: int = 0,
    treatment_coef_range: Tuple[float, float] = (-2.0, 2.0),
    outcome_coef_range: Tuple[float, float] = (-2.0, 2.0),
    treatment_effect_range: Tuple[float, float] = (0.1, 5.0),
    n_outcome_bins: int = 10,
):
    """Plot heatmaps of RMSE/bias/std differences vs excess kurtosis and outcome coefficient.

    Creates heatmaps showing how the differences between ICA and HOML vary across
    excess kurtosis (x-axis, discrete per distribution) and outcome coefficient (y-axis, binned).

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
        n_configs: Number of random configs (for filename)
        treatment_coef_range: Range for treatment coefficients
        outcome_coef_range: Range for outcome coefficients
        treatment_effect_range: Range for treatment effect
        n_outcome_bins: Number of bins for outcome coefficient axis
    """
    suffix = (
        f"_n{n_configs}_tc{treatment_coef_range[0]:.1f}to{treatment_coef_range[1]:.1f}"
        f"_oc{outcome_coef_range[0]:.1f}to{outcome_coef_range[1]:.1f}"
        f"_te{treatment_effect_range[0]:.1f}to{treatment_effect_range[1]:.1f}"
    )

    os.makedirs(output_dir, exist_ok=True)
    _setup_plot()

    # Collect data from all distributions and configs, organized by distribution
    dist_data = {}
    for dist, res in results.items():
        if res.get("config_results") is None:
            continue

        # Skip Gaussian (beta=2) - ICA doesn't work for Gaussian
        if dist in ("gennorm:2", "gennorm:2.0"):
            continue

        # Get excess kurtosis for this distribution
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

    # Get all outcome coefficients to determine bins
    all_outcome_coefs = []
    for dist in sorted_dists:
        all_outcome_coefs.extend(dist_data[dist]["outcome_coefs"])
    all_outcome_coefs = np.array(all_outcome_coefs)
    outcome_bins = np.linspace(all_outcome_coefs.min(), all_outcome_coefs.max(), n_outcome_bins + 1)
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

    def plot_single_heatmap(data, title, filename, cbar_label):
        """Plot a single heatmap."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use diverging colormap centered at 0
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",  # Blue = ICA better (negative), Red = HOML better (positive)
            vmin=vmin,
            vmax=vmax,
        )

        # Set x-axis ticks to distribution labels with kurtosis
        ax.set_xticks(np.arange(n_dists))
        x_labels = [f"{dist_data[d]['label']}\n(\u03ba={dist_data[d]['kurtosis']:.2f})" for d in sorted_dists]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        # Set y-axis ticks to outcome coefficient bin centers
        ax.set_yticks(np.arange(n_outcome_bins))
        ax.set_yticklabels([f"{oc:.2f}" for oc in outcome_centers])

        ax.set_xlabel("Distribution (sorted by kurtosis)")
        ax.set_ylabel(r"Outcome Coefficient $b$")
        ax.set_title(title)

        # Add colorbar next to the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label=cbar_label)

        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Create and plot heatmaps
    rmse_heatmap = create_heatmap_data("rmse_diffs")
    bias_heatmap = create_heatmap_data("bias_diffs")
    std_heatmap = create_heatmap_data("std_diffs")

    # Filter out rows that are all-NaN across all heatmaps (empty bins)
    valid_rows = ~(
        np.all(np.isnan(rmse_heatmap), axis=1)
        & np.all(np.isnan(bias_heatmap), axis=1)
        & np.all(np.isnan(std_heatmap), axis=1)
    )
    rmse_heatmap = rmse_heatmap[valid_rows]
    bias_heatmap = bias_heatmap[valid_rows]
    std_heatmap = std_heatmap[valid_rows]
    outcome_centers = outcome_centers[valid_rows]
    n_outcome_bins = int(valid_rows.sum())

    plot_single_heatmap(
        rmse_heatmap,
        "RMSE Difference\n(Blue = ICA better, Red = HOML better)",
        f"heatmap_rmse_diff{suffix}.svg",
        "RMSE Diff",
    )

    plot_single_heatmap(
        bias_heatmap,
        r"$|\mathrm{Bias}|$ Difference" + "\n(Blue = ICA better, Red = HOML better)",
        f"heatmap_bias_diff{suffix}.svg",
        r"$|\mathrm{Bias}|$ Diff",
    )

    plot_single_heatmap(
        std_heatmap,
        "Std Difference\n(Blue = ICA better, Red = HOML better)",
        f"heatmap_std_diff{suffix}.svg",
        "Std Diff",
    )

    # Also create a combined 1x3 figure
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(26, 7),
            gridspec_kw={"wspace": 0.6},
        )

    for ax, data, title in zip(
        axes,
        [rmse_heatmap, bias_heatmap, std_heatmap],
        ["RMSE Diff", r"$|\mathrm{Bias}|$ Diff", "Std Diff"],
    ):
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

        ax.set_xticks(np.arange(n_dists))
        x_labels = [f"{dist_data[d]['label']}\n(\u03ba={dist_data[d]['kurtosis']:.1f})" for d in sorted_dists]
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(0, n_outcome_bins, 2))
        ax.set_yticklabels(
            [f"{outcome_centers[i]:.2f}" for i in range(0, n_outcome_bins, 2)],
            fontsize=7,
        )
        ax.set_xlabel("Distribution", fontsize=8)
        ax.set_ylabel(r"Outcome Coef $b$", fontsize=8)
        ax.set_title(title, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "ICA - HOML Differences vs Distribution and Outcome Coef\n(Blue = ICA better, Red = HOML better)",
        fontsize=10,
    )
    plt.savefig(
        os.path.join(output_dir, f"heatmap_combined{suffix}.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nHeatmaps saved to {output_dir}")


def plot_coefficient_ablation_results(results: List[dict], output_dir: str = "figures/coefficient_ablation"):
    """Plot results from coefficient ablation study.

    Args:
        results: List of result dictionaries from coefficient ablation
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    _setup_plot()

    # Extract data
    ica_var_coeffs = [r["ica_var_coeff"] for r in results]
    homl_rmse = [r["rmse"][HOML_IDX] for r in results]
    ica_rmse = [r["rmse"][ICA_IDX] if ICA_IDX < len(r["rmse"]) else np.nan for r in results]
    homl_bias = [np.abs(r["biases"][HOML_IDX]) for r in results]
    ica_bias = [np.abs(r["biases"][ICA_IDX]) if ICA_IDX < len(r["biases"]) else np.nan for r in results]

    # Plot 1: RMSE vs ICA variance coefficient
    _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ica_var_coeffs, homl_rmse, c=OML_COLOR, alpha=0.7, label="OML", s=60, marker="o")
    ax.scatter(ica_var_coeffs, ica_rmse, c=ICA_COLOR, alpha=0.7, label="ICA", s=60, marker="s")
    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel("RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    add_legend_outside(ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Bias vs ICA variance coefficient
    _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ica_var_coeffs, homl_bias, c=OML_COLOR, alpha=0.7, label="HOML", s=60, marker="o")
    ax.scatter(ica_var_coeffs, ica_bias, c=ICA_COLOR, alpha=0.7, label="ICA", s=60, marker="s")
    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel(r"$|\mathrm{Bias}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    add_legend_outside(ax)
    ax.set_title("Absolute Bias vs ICA Variance Coefficient")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: RMSE difference
    _, ax = plt.subplots(figsize=(8, 6))
    rmse_diff = np.array(ica_rmse) - np.array(homl_rmse)
    _scatter_with_markers(ax, ica_var_coeffs, rmse_diff, rmse_diff, alpha=0.7, s=60)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel("RMSE Difference (ICA - OML)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add legend
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_diff_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nCoefficient ablation plots saved to {output_dir}")


def plot_variance_ablation_heatmaps(results: dict, output_dir: str = "figures/variance_ablation"):
    """Plot heatmaps for variance ablation study.

    Creates heatmaps with beta (gennorm shape) on x-axis and variance on y-axis,
    showing bias, std, and RMSE for both HOML and ICA methods, plus their differences.

    Args:
        results: Dictionary with variance ablation results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    _setup_plot()

    beta_values = results["beta_values"]
    variance_values = results["variance_values"]
    grid_results = results["grid_results"]

    n_betas = len(beta_values)
    n_vars = len(variance_values)

    # Initialize heatmap arrays
    homl_bias_grid = np.full((n_vars, n_betas), np.nan)
    homl_std_grid = np.full((n_vars, n_betas), np.nan)
    homl_rmse_grid = np.full((n_vars, n_betas), np.nan)
    ica_bias_grid = np.full((n_vars, n_betas), np.nan)
    ica_std_grid = np.full((n_vars, n_betas), np.nan)
    ica_rmse_grid = np.full((n_vars, n_betas), np.nan)

    # Fill in the grids
    for i, var_val in enumerate(variance_values):
        for j, beta_val in enumerate(beta_values):
            res = grid_results.get((beta_val, var_val))
            if res is None:
                continue

            homl_bias_grid[i, j] = np.abs(res["biases"][HOML_IDX])
            homl_std_grid[i, j] = res["sigmas"][HOML_IDX]
            homl_rmse_grid[i, j] = res["rmse"][HOML_IDX]

            if ICA_IDX < len(res["biases"]):
                ica_bias_grid[i, j] = np.abs(res["biases"][ICA_IDX])
                ica_std_grid[i, j] = res["sigmas"][ICA_IDX]
                ica_rmse_grid[i, j] = res["rmse"][ICA_IDX]

    # Compute difference grids
    bias_diff_grid = ica_bias_grid - homl_bias_grid
    std_diff_grid = ica_std_grid - homl_std_grid
    rmse_diff_grid = ica_rmse_grid - homl_rmse_grid

    def plot_single_heatmap(data, title, filename, cbar_label, diverging=False, log_scale=False):
        """Plot a single heatmap."""
        fig, ax = plt.subplots(figsize=(10, 7))

        if diverging:
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax
            cmap = "coolwarm"  # Blue = ICA better (negative), Red = HOML better (positive)
        else:
            if log_scale:
                # For log scale, use log10 of data for display
                data_display = np.log10(data + 1e-10)
                vmin, vmax = np.nanmin(data_display), np.nanmax(data_display)
            else:
                vmin, vmax = np.nanmin(data), np.nanmax(data)
            cmap = "coolwarm"
            data_display = data if not log_scale else data_display

        im = ax.imshow(
            data_display if log_scale else data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Set axis labels
        ax.set_xticks(np.arange(n_betas))
        ax.set_xticklabels([f"{b:.1f}" for b in beta_values])
        ax.set_yticks(np.arange(n_vars))
        ax.set_yticklabels([f"{v:.2f}" for v in variance_values])

        ax.set_xlabel(r"$\beta$ (gennorm shape)")
        ax.set_ylabel(r"Variance $\sigma^2$")
        ax.set_title(title)

        # Add value annotations
        for i in range(n_vars):
            for j in range(n_betas):
                val = data[i, j]
                if not np.isnan(val):
                    if diverging:
                        color = "white" if abs(val) > vmax * 0.5 else "black"
                    else:
                        val_norm = (val - vmin) / (vmax - vmin + 1e-10) if not log_scale else 0.5
                        color = "white" if val_norm > 0.5 or val_norm < 0.3 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)

        # Add colorbar next to the plot
        cbar_label_final = cbar_label + (" (log10)" if log_scale else "")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label=cbar_label_final)

        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Plot individual method heatmaps
    plot_single_heatmap(homl_bias_grid, r"HOML $|\mathrm{Bias}|$", "homl_bias_heatmap.svg", r"$|\mathrm{Bias}|$")
    plot_single_heatmap(homl_std_grid, "HOML Standard Deviation", "homl_std_heatmap.svg", "Std")
    plot_single_heatmap(homl_rmse_grid, "HOML RMSE", "homl_rmse_heatmap.svg", "RMSE")

    plot_single_heatmap(ica_bias_grid, r"ICA $|\mathrm{Bias}|$", "ica_bias_heatmap.svg", r"$|\mathrm{Bias}|$")
    plot_single_heatmap(ica_std_grid, "ICA Standard Deviation", "ica_std_heatmap.svg", "Std")
    plot_single_heatmap(ica_rmse_grid, "ICA RMSE", "ica_rmse_heatmap.svg", "RMSE")

    # Plot difference heatmaps (diverging colormap)
    plot_single_heatmap(
        bias_diff_grid,
        r"$|\mathrm{Bias}|$ Difference" + "\n(Blue = ICA better, Red = HOML better)",
        "bias_diff_heatmap.svg",
        r"$|\mathrm{Bias}|$ Diff",
        diverging=True,
    )
    plot_single_heatmap(
        std_diff_grid,
        "Std Difference\n(Blue = ICA better, Red = HOML better)",
        "std_diff_heatmap.svg",
        "Std Diff",
        diverging=True,
    )
    plot_single_heatmap(
        rmse_diff_grid,
        "RMSE Difference\n(Blue = ICA better, Red = HOML better)",
        "rmse_diff_heatmap.svg",
        "RMSE Diff",
        diverging=True,
    )

    # Create combined 2x3 figure for main metrics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        (homl_bias_grid, r"HOML $|\mathrm{Bias}|$"),
        (homl_std_grid, "HOML Std"),
        (homl_rmse_grid, "HOML RMSE"),
        (ica_bias_grid, r"ICA $|\mathrm{Bias}|$"),
        (ica_std_grid, "ICA Std"),
        (ica_rmse_grid, "ICA RMSE"),
    ]

    for ax, (data, title) in zip(axes.flat, metrics):
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(n_betas))
        ax.set_xticklabels([f"{b:.1f}" for b in beta_values])
        ax.set_yticks(np.arange(n_vars))
        ax.set_yticklabels([f"{v:.2f}" for v in variance_values])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"Var")
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    fig.suptitle(r"Estimation Metrics vs $\beta$ and Variance")
    plt.savefig(os.path.join(output_dir, "combined_metrics_heatmap.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create combined 1x3 figure for differences
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.5)

    diff_metrics = [
        (rmse_diff_grid, "RMSE Diff"),
        (bias_diff_grid, r"$|\mathrm{Bias}|$ Diff"),
        (std_diff_grid, "Std Diff"),
    ]

    for ax, (data, title) in zip(axes, diff_metrics):
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(n_betas))
        ax.set_xticklabels([f"{b:.1f}" for b in beta_values], fontsize=7)
        ax.set_yticks(np.arange(n_vars))
        ax.set_yticklabels([f"{v:.2f}" for v in variance_values], fontsize=7)
        ax.set_xlabel(r"$\beta$", fontsize=8)
        ax.set_ylabel(r"Var", fontsize=8)
        ax.set_title(title, fontsize=9)

        # Add value annotations to each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    # Use white text on dark backgrounds, black on light
                    val_norm = abs(val) / (vmax + 1e-10)
                    color = "white" if val_norm > 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=7)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)

    plt.savefig(os.path.join(output_dir, "combined_diff_heatmap.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nVariance ablation heatmaps saved to {output_dir}")


def plot_ica_var_filtered_rmse_heatmap(
    results: List[dict],
    output_dir: str,
    axis_mode: str = "d_vs_n",
    ica_var_threshold: float = 1.5,
    filter_below: bool = True,
):
    """Plot RMSE heatmaps filtered by ICA variance coefficient.

    Creates three heatmaps:
    1. OML RMSE heatmap
    2. ICA RMSE heatmap
    3. RMSE difference heatmap (ICA - OML)

    Args:
        results: List of result dictionaries from run_sample_dimension_grid_experiments
        output_dir: Directory to save plots
        axis_mode: "d_vs_n" (dimension vs sample size) or "beta_vs_n" (beta vs sample size)
        ica_var_threshold: Threshold for ICA variance coefficient filtering
        filter_below: If True, keep results where ica_var_coeff <= threshold
    """
    _setup_plot()
    os.makedirs(output_dir, exist_ok=True)

    # Filter results by ica_var_coeff
    if filter_below:
        filtered_results = [r for r in results if r["ica_var_coeff"] <= ica_var_threshold]
        filter_desc = f"below_{ica_var_threshold}"
    else:
        filtered_results = [r for r in results if r["ica_var_coeff"] > ica_var_threshold]
        filter_desc = f"above_{ica_var_threshold}"

    op = "<=" if filter_below else ">"
    print(f"\nFiltered {len(filtered_results)}/{len(results)} results" f" with ica_var_coeff {op} {ica_var_threshold}")

    if len(filtered_results) == 0:
        print("No results after filtering. Cannot create heatmap.")
        return

    # Determine axes based on mode
    if axis_mode == "d_vs_n":
        x_key = "support_size"
        x_label = r"Covariate dimension $d$"
        filename_suffix = "dim"
    else:  # beta_vs_n
        x_key = "beta"
        x_label = r"Gen. normal param. $\beta$"
        filename_suffix = "beta"

    # Get unique values for axes
    x_values = sorted(set(r[x_key] for r in filtered_results))
    y_values = sorted(set(r["n_samples"] for r in filtered_results), reverse=True)

    # Create heatmap data matrices for OML, ICA, and difference
    homl_rmse_data = np.full((len(y_values), len(x_values)), np.nan)
    ica_rmse_data = np.full((len(y_values), len(x_values)), np.nan)
    rmse_diff_data = np.full((len(y_values), len(x_values)), np.nan)
    count_matrix = np.zeros((len(y_values), len(x_values)), dtype=int)

    for r in filtered_results:
        x_idx = x_values.index(r[x_key])
        y_idx = y_values.index(r["n_samples"])

        # Get RMSE values
        rmse_homl = r["rmse"][HOML_IDX]
        rmse_ica = r["rmse"][ICA_IDX] if ICA_IDX < len(r["rmse"]) else np.nan

        homl_rmse_data[y_idx, x_idx] = rmse_homl
        ica_rmse_data[y_idx, x_idx] = rmse_ica
        rmse_diff_data[y_idx, x_idx] = rmse_ica - rmse_homl
        count_matrix[y_idx, x_idx] = r.get("n_experiments_kept", r.get("n_experiments", 1))

    def create_rmse_heatmap(data, method_name, filename, is_difference=False):
        """Create and save a single RMSE heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Always use diverging colormap centered at 0 (white)
        # This ensures 0 is white, negative is blue, positive is red
        vmax = np.nanmax(np.abs(data))
        if np.isnan(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

        if is_difference:
            cbar_label = r"RMSE diff (ICA $-$ OML)"
        else:
            cbar_label = r"RMSE"

        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_yticklabels([str(v) for v in y_values])
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Sample size $n$")

        # Add cell annotations
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                val = data[i, j]
                if not np.isnan(val):
                    # Color text based on background brightness (use abs for symmetric colormap)
                    text_color = "white" if abs(val) > vmax * 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {method_name} RMSE heatmap to {os.path.join(output_dir, filename)}")

    # Create HOML RMSE heatmap
    create_rmse_heatmap(
        homl_rmse_data,
        "HOML",
        f"rmse_sample_size_vs_{filename_suffix}_homl_filtered_{filter_desc}.svg",
        is_difference=False,
    )

    # Create ICA RMSE heatmap
    create_rmse_heatmap(
        ica_rmse_data,
        "ICA",
        f"rmse_sample_size_vs_{filename_suffix}_ica_filtered_{filter_desc}.svg",
        is_difference=False,
    )

    # Create RMSE difference heatmap
    create_rmse_heatmap(
        rmse_diff_data,
        "difference",
        f"rmse_diff_sample_size_vs_{filename_suffix}_filtered_{filter_desc}.svg",
        is_difference=True,
    )

    # Also create a summary of the filtering
    summary_file = os.path.join(output_dir, f"filtering_summary_{filter_desc}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("ICA Variance Coefficient Filtering Summary\n")
        f.write("==========================================\n")
        f.write(f"Threshold: {ica_var_threshold}\n")
        f.write(f"Filter mode: {'<=' if filter_below else '>'} threshold\n")
        f.write(f"Results kept: {len(filtered_results)} / {len(results)}\n\n")
        f.write("ICA var coeff range in filtered data:\n")
        if filtered_results:
            ica_coeffs = [r["ica_var_coeff"] for r in filtered_results]
            f.write(f"  Min: {min(ica_coeffs):.4f}\n")
            f.write(f"  Max: {max(ica_coeffs):.4f}\n")
            f.write(f"  Mean: {np.mean(ica_coeffs):.4f}\n")

    print(f"Saved filtering summary to {summary_file}")


def plot_ica_var_filtered_bias_heatmaps(
    results: List[dict],
    output_dir: str,
    axis_mode: str = "d_vs_n",
    ica_var_threshold: float = 1.5,
    filter_below: bool = True,
):
    """Plot bias heatmaps filtered by ICA variance coefficient.

    Creates heatmaps showing HOML and ICA bias separately (not differences),
    matching the format of bias_sample_size_vs_dim_homl_mean.svg and
    bias_sample_size_vs_beta_homl_mean.svg.

    Args:
        results: List of result dictionaries from run_sample_dimension_grid_experiments
        output_dir: Directory to save plots
        axis_mode: "d_vs_n" (dimension vs sample size) or "beta_vs_n" (beta vs sample size)
        ica_var_threshold: Threshold for ICA variance coefficient filtering
        filter_below: If True, keep results where ica_var_coeff <= threshold
    """
    _setup_plot()
    os.makedirs(output_dir, exist_ok=True)

    # Filter results by ica_var_coeff
    if filter_below:
        filtered_results = [r for r in results if r["ica_var_coeff"] <= ica_var_threshold]
        filter_desc = f"below_{ica_var_threshold}"
    else:
        filtered_results = [r for r in results if r["ica_var_coeff"] > ica_var_threshold]
        filter_desc = f"above_{ica_var_threshold}"

    print(
        f"\nFiltered {len(filtered_results)}/{len(results)} results with "
        f"ica_var_coeff {'<=' if filter_below else '>'} {ica_var_threshold}"
    )

    if len(filtered_results) == 0:
        print("No results after filtering. Cannot create heatmap.")
        return

    # Determine axes based on mode
    if axis_mode == "d_vs_n":
        x_key = "support_size"
        x_label = r"Covariate dimension $d$"
        filename_suffix = "dim"
    else:  # beta_vs_n
        x_key = "beta"
        x_label = r"Gen. normal param. $\beta$"
        filename_suffix = "beta"

    # Get unique values for axes
    x_values = sorted(set(r[x_key] for r in filtered_results))
    y_values = sorted(set(r["n_samples"] for r in filtered_results), reverse=True)

    # Create heatmap data matrices for HOML and ICA
    homl_bias_data = np.full((len(y_values), len(x_values)), np.nan)
    ica_bias_data = np.full((len(y_values), len(x_values)), np.nan)
    count_matrix = np.zeros((len(y_values), len(x_values)), dtype=int)

    for r in filtered_results:
        x_idx = x_values.index(r[x_key])
        y_idx = y_values.index(r["n_samples"])

        # Get absolute biases
        homl_bias = np.abs(r["biases"][HOML_IDX])
        ica_bias = np.abs(r["biases"][ICA_IDX]) if ICA_IDX < len(r["biases"]) else np.nan

        homl_bias_data[y_idx, x_idx] = homl_bias
        ica_bias_data[y_idx, x_idx] = ica_bias
        count_matrix[y_idx, x_idx] = r.get("n_experiments_kept", r.get("n_experiments", 1))

    def create_bias_heatmap(data, method_name, filename):
        """Create and save a single bias heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use diverging colormap centered at 0 (white)
        # This ensures 0 is white, negative is blue, positive is red
        vmax = np.nanmax(np.abs(data))
        if np.isnan(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_yticklabels([str(v) for v in y_values])
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Sample size $n$")

        # Add cell annotations
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                val = data[i, j]
                if not np.isnan(val):
                    # Color text based on background brightness (use abs for symmetric colormap)
                    text_color = "white" if abs(val) > vmax * 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$|\mathrm{Bias}|$")

        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {method_name} bias heatmap to {os.path.join(output_dir, filename)}")

    # Create HOML bias heatmap
    create_bias_heatmap(
        homl_bias_data,
        "HOML",
        f"bias_sample_size_vs_{filename_suffix}_homl_mean_filtered_{filter_desc}.svg",
    )

    # Create ICA bias heatmap
    create_bias_heatmap(
        ica_bias_data,
        "ICA",
        f"bias_sample_size_vs_{filename_suffix}_ica_mean_filtered_{filter_desc}.svg",
    )

    # Create difference heatmap (ICA - OML) for reference
    bias_diff_data = ica_bias_data - homl_bias_data

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(bias_diff_data))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(bias_diff_data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([str(v) for v in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Sample size $n$")

    # Add cell annotations
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            val = bias_diff_data[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$|\mathrm{Bias}|$ diff (ICA $-$ OML)")

    diff_filename = f"bias_diff_sample_size_vs_{filename_suffix}_filtered_{filter_desc}.svg"
    plt.savefig(os.path.join(output_dir, diff_filename), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved bias difference heatmap to {os.path.join(output_dir, diff_filename)}")

    # Save filtering summary
    summary_file = os.path.join(output_dir, f"bias_filtering_summary_{filter_desc}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("ICA Variance Coefficient Filtering Summary (Bias Heatmaps)\n")
        f.write("=========================================================\n")
        f.write(f"Threshold: {ica_var_threshold}\n")
        f.write(f"Filter mode: {'<=' if filter_below else '>'} threshold\n")
        f.write(f"Results kept: {len(filtered_results)} / {len(results)}\n\n")
        f.write("ICA var coeff range in filtered data:\n")
        if filtered_results:
            ica_coeffs = [r["ica_var_coeff"] for r in filtered_results]
            f.write(f"  Min: {min(ica_coeffs):.4f}\n")
            f.write(f"  Max: {max(ica_coeffs):.4f}\n")
            f.write(f"  Mean: {np.mean(ica_coeffs):.4f}\n")
        f.write("\nHOML bias statistics:\n")
        f.write(f"  Min: {np.nanmin(homl_bias_data):.4f}\n")
        f.write(f"  Max: {np.nanmax(homl_bias_data):.4f}\n")
        f.write(f"  Mean: {np.nanmean(homl_bias_data):.4f}\n")
        f.write("\nICA bias statistics:\n")
        f.write(f"  Min: {np.nanmin(ica_bias_data):.4f}\n")
        f.write(f"  Max: {np.nanmax(ica_bias_data):.4f}\n")
        f.write(f"  Mean: {np.nanmean(ica_bias_data):.4f}\n")

    print(f"Saved filtering summary to {summary_file}")


# =============================================================================
# N20 Regeneration Utilities (consolidated from regenerate_all_n20_plots.py
# and regenerate_n20_heatmaps.py)
# =============================================================================


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
    """Plot heatmaps with fixed binning to avoid empty rows.

    Uses quantile-based bins and improved subplot spacing for combined heatmaps.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
        n_configs: Number of configs
        treatment_coef_range: Range for treatment coefficients
        outcome_coef_range: Range for outcome coefficients
        treatment_effect_range: Range for treatment effect
        n_outcome_bins: Number of bins for outcome coefficient axis
    """
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
        x_labels = [f"{dist_data[d]['label']}\n(\u03ba={dist_data[d]['kurtosis']:.2f})" for d in sorted_dists]
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

    # Create combined heatmap with increased spacing to prevent overlap
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(1, 3, figsize=(26, 7), gridspec_kw={"wspace": 0.6})

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

        ax.set_xticks(np.arange(n_dists))
        x_labels = [f"{dist_data[d]['label']}\n(\u03ba={dist_data[d]['kurtosis']:.1f})" for d in sorted_dists]
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(0, n_outcome_bins, 2))
        ax.set_yticklabels(
            [f"{outcome_centers[i]:.2f}" for i in range(0, n_outcome_bins, 2)],
            fontsize=7,
        )
        ax.set_xlabel("Distribution", fontsize=8)
        ax.set_ylabel(r"Outcome Coeff $b$", fontsize=8)
        ax.set_title(title, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08, label=cbar_label)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "ICA - OML Differences vs Distribution and Outcome Coef\n(Blue = ICA better, Red = OML better)",
        fontsize=10,
    )
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
    """Plot coefficient scatter plots with filtered distributions.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
        n_configs: Number of configs
        treatment_coef_range: Range for treatment coefficients
        outcome_coef_range: Range for outcome coefficients
        treatment_effect_range: Range for treatment effect
    """
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


def regenerate_n20_plots(scatter=True):
    """Process all n20 files in noise_ablation directory.

    Regenerates heatmaps (always) and scatter plots (when scatter=True)
    for all n20 result files, with duplicate distributions filtered out.

    Args:
        scatter: If True, also regenerate coefficient scatter plots.
    """
    import glob as _glob
    import re as _re

    noise_ablation_dir = "figures/noise_ablation"

    # Find all n20 .npy files
    pattern = os.path.join(noise_ablation_dir, "noise_ablation_results_n20_*.npy")
    n20_files = _glob.glob(pattern)

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
        tc_match = _re.search(r"tc(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        oc_match = _re.search(r"oc(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        te_match = _re.search(r"te(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)

        if tc_match and oc_match:
            treatment_coef_range = (float(tc_match.group(1)), float(tc_match.group(2)))
            outcome_coef_range = (float(oc_match.group(1)), float(oc_match.group(2)))
        else:
            coef_match = _re.search(r"coef(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
            if coef_match:
                treatment_coef_range = (float(coef_match.group(1)), float(coef_match.group(2)))
                outcome_coef_range = treatment_coef_range
            else:
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
        skip_keys = {"treatment_coef_range", "outcome_coef_range", "treatment_effect_range"}
        print(f"  Distributions: {[k for k in results if k not in skip_keys]}")

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
        if scatter:
            plot_coeff_scatter(
                results,
                noise_ablation_dir,
                n_configs,
                treatment_coef_range,
                outcome_coef_range,
                treatment_effect_range,
            )
