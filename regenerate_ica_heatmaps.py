#!/usr/bin/env python3
"""Script to regenerate all ICA experiment heatmaps with updated formatting.

This script regenerates heatmaps from existing .npy results files with:
1. Consistent color scheme (coolwarm)
2. Value annotations with 2 decimal places (fmt=".2f")
3. Consistent font size for annotations
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tueplots import bundles

from plot_utils import plot_typography


def save_figure(filename, output_dir="figures/ica"):
    """Save figure to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def calculate_mse(true_params, est_params, relative_error=False):
    """Calculate MSE between true and estimated parameters."""
    if hasattr(true_params, "numpy"):
        true_params = true_params.numpy()

    if isinstance(est_params, list):
        if len(est_params) == 0:
            return np.nan
        est_params = np.array(est_params)

    if np.isnan(est_params).any():
        return np.nan

    diff = est_params - true_params
    if relative_error:
        return np.linalg.norm(diff / true_params)
    return np.linalg.norm(diff)


def regenerate_main_nonlinear():
    """Regenerate heatmaps from main_nonlinear results."""
    results_file = "results_main_nonlinear.npy"

    if not os.path.exists(results_file):
        print(f"Skipping {results_file} - file not found")
        return

    print(f"Processing {results_file}...")
    results_dict = np.load(results_file, allow_pickle=True).item()

    # Filter the data
    filtered_indices = [
        i
        for i, nonlinearity in enumerate(results_dict["nonlinearities"])
        if (nonlinearity == "leaky_relu" and results_dict["slopes"][i] == 0.2) or (nonlinearity != "leaky_relu")
    ]
    filtered_results = {key: [results_dict[key][i] for i in filtered_indices] for key in results_dict}

    # Prepare data for heatmap
    dimensions = sorted(set(filtered_results["n_covariates"]), reverse=True)
    nonlinearities = sorted(set(filtered_results["nonlinearities"]))
    heatmap_data = np.zeros((len(dimensions), len(nonlinearities)))
    heatmap_data_std = np.zeros((len(dimensions), len(nonlinearities)))

    for i, dim in enumerate(dimensions):
        for j, nonlinearity in enumerate(nonlinearities):
            relevant_indices = [
                index
                for index, (d, n) in enumerate(
                    zip(filtered_results["n_covariates"], filtered_results["nonlinearities"])
                )
                if d == dim and n == nonlinearity
            ]
            if relevant_indices:
                heatmap_data[i, j] = np.mean(
                    [
                        calculate_mse(
                            filtered_results["true_params"][index], filtered_results["treatment_effects"][index]
                        )
                        for index in relevant_indices
                    ]
                )
                heatmap_data_std[i, j] = np.std(
                    [
                        calculate_mse(
                            filtered_results["true_params"][index], filtered_results["treatment_effects"][index]
                        )
                        for index in relevant_indices
                    ]
                )

    # Plot heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=nonlinearities,
        yticklabels=dimensions,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
    )
    plt.xlabel("Nonlinearity")
    plt.ylabel(r"Covariate dimension $d$")
    save_figure("heatmap_dimension_vs_nonlinearity.svg")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data_std,
        xticklabels=nonlinearities,
        yticklabels=dimensions,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
    )
    plt.xlabel("Nonlinearity")
    plt.ylabel(r"Covariate dimension $d$")
    save_figure("heatmap_dimension_vs_nonlinearity_std.svg")

    # Filter for leaky_relu only
    filtered_indices_leaky_relu = [
        index for index, nonlinearity in enumerate(results_dict["nonlinearities"]) if nonlinearity == "leaky_relu"
    ]
    filtered_results_leaky_relu = {
        key: [results_dict[key][i] for i in filtered_indices_leaky_relu] for key in results_dict
    }

    # Prepare data for slope heatmap
    dimensions = sorted(set(filtered_results_leaky_relu["n_covariates"]), reverse=True)
    slopes = sorted(set(filtered_results_leaky_relu["slopes"]))
    heatmap_data = np.zeros((len(dimensions), len(slopes)))

    for i, dim in enumerate(dimensions):
        for j, slope in enumerate(slopes):
            relevant_indices = [
                index
                for index, (d, s) in enumerate(
                    zip(filtered_results_leaky_relu["n_covariates"], filtered_results_leaky_relu["slopes"])
                )
                if d == dim and s == slope
            ]
            if relevant_indices:
                heatmap_data[i, j] = np.mean(
                    [
                        calculate_mse(
                            filtered_results_leaky_relu["true_params"][index],
                            filtered_results_leaky_relu["treatment_effects"][index],
                        )
                        for index in relevant_indices
                    ]
                )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=slopes,
        yticklabels=dimensions,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
    )
    plt.xlabel("Slope")
    plt.ylabel(r"Covariate dimension $d$")
    save_figure("heatmap_dimension_vs_slope_leaky_relu.svg")


def regenerate_main_multi():
    """Regenerate heatmaps from main_multi results."""
    results_file = "results_multi_treatment.npy"

    if not os.path.exists(results_file):
        print(f"Skipping {results_file} - file not found")
        return

    print(f"Processing {results_file}...")
    results_dict = np.load(results_file, allow_pickle=True).item()

    def filter_indices(results_dict, sample_size, treatment_count=None, covariate_dim=None):
        return [
            i
            for i, (s, t, d) in enumerate(
                zip(results_dict["sample_sizes"], results_dict["n_treatments"], results_dict["n_covariates"])
            )
            if s == sample_size
            and (treatment_count is None or t == treatment_count)
            and (covariate_dim is None or d == covariate_dim)
        ]

    def calculate_treatment_effect_diff(results_dict, indices):
        est_params_ica = [results_dict["treatment_effects"][i] for i in indices]
        est_params_iv = [results_dict["treatment_effects_iv"][i] for i in indices]
        return np.nanmean([np.linalg.norm(est_ica - est_iv) for est_ica, est_iv in zip(est_params_ica, est_params_iv)])

    # Prepare data for heatmap: x-axis is number of treatments, y-axis is sample size, covariate dimension is 10
    covariate_dimension = 10
    treatment_effect_diff = {}
    treatment_effect_ica = {}
    treatment_effect_ica_std = {}

    for n_samples in set(results_dict["sample_sizes"]):
        for n_treatment in set(results_dict["n_treatments"]):
            indices = filter_indices(results_dict, n_samples, n_treatment, covariate_dimension)
            if indices:
                diff = calculate_treatment_effect_diff(results_dict, indices)
                treatment_effect_diff[(n_samples, n_treatment)] = diff
                # Calculate ICA error only
                est_params_ica = [results_dict["treatment_effects"][i] for i in indices]
                true_params = [results_dict["true_params"][i].numpy() for i in indices]

                ica_error = calculate_mse(true_params, est_params_ica, relative_error=True)
                treatment_effect_ica[(n_samples, n_treatment)] = np.nanmean(ica_error)
                treatment_effect_ica_std[(n_samples, n_treatment)] = np.nanstd(ica_error)

    # Create heatmap data
    sample_sizes = sorted(set(results_dict["sample_sizes"]), reverse=True)
    num_treatments = sorted(set(results_dict["n_treatments"]))

    # Heatmap for ICA error mean
    heatmap_data_ica = np.array(
        [[treatment_effect_ica.get((s, t), np.nan) for t in num_treatments] for s in sample_sizes]
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data_ica,
        xticklabels=num_treatments,
        yticklabels=sample_sizes,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
    )
    plt.xlabel(r"Number of treatments $m$")
    plt.ylabel(r"Sample size $n$")
    save_figure("heatmap_ica_treatments_vs_samples_rel.svg")

    # Heatmap for ICA error std
    heatmap_data_ica_std = np.array(
        [[treatment_effect_ica_std.get((s, t), np.nan) for t in num_treatments] for s in sample_sizes]
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data_ica_std,
        xticklabels=num_treatments,
        yticklabels=sample_sizes,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
    )
    plt.xlabel(r"Number of treatments $m$")
    plt.ylabel(r"Sample size $n$")
    save_figure("heatmap_ica_treatments_vs_samples_rel_std.svg")


def regenerate_main_sparsity():
    """Regenerate heatmaps from main_sparsity results."""
    results_file = "results_main_sparsity.npy"

    if not os.path.exists(results_file):
        print(f"Skipping {results_file} - file not found")
        return

    print(f"Processing {results_file}...")
    # For sparsity, this is typically an error bar plot, not a heatmap
    # Skip for now
    print("  Skipping - no heatmaps in this experiment")


def main():
    """Regenerate all ICA heatmaps."""
    # Setup plotting
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography(preset="publication")

    print("Regenerating ICA experiment heatmaps...")
    print("=" * 60)

    regenerate_main_nonlinear()
    regenerate_main_multi()
    regenerate_main_sparsity()

    print("=" * 60)
    print("Done! All ICA heatmaps regenerated.")


if __name__ == "__main__":
    main()
