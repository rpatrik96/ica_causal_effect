"""
Plotting utilities for OML experiments.

This module provides consolidated plotting functions and configurations
for Orthogonal Machine Learning experiments.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from oml_utils import OMLExperimentConfig, OMLParameterGrid


class OMLPlottingConfig:
    """Configuration for OML plotting operations."""

    # Method indices for results arrays
    METHOD_ORTHO_ML = 0
    METHOD_ROBUST_ORTHO_ML = 1
    METHOD_ROBUST_ORTHO_EST_ML = 2
    METHOD_ROBUST_ORTHO_EST_SPLIT_ML = 3
    METHOD_ICA_START = 4  # ICA methods start from index 4

    # Method names for plots
    METHOD_NAMES = [
        "Ortho ML",
        "Robust Ortho ML",
        "Robust Ortho ML (Est.)",
        "Robust Ortho ML (Split)",
        "ICA",
    ]


def plot_method_comparison_both_errors(
    ortho_rec_tau: List,
    treatment_effect: float,
    output_dir: str,
    n_samples: int,
    cov_dim_max: int,
    n_experiments: int,
    support_size: int,
    sigma_outcome: float,
    covariate_pdf: str,
    beta: float,
    plot: bool = False,
    verbose: bool = False,
) -> Dict[str, tuple]:
    """Plot method comparison for both absolute and relative errors.

    This consolidates the two separate calls to plot_method_comparison.

    Args:
        ortho_rec_tau: Recovered treatment effects
        treatment_effect: True treatment effect
        output_dir: Output directory
        n_samples: Number of samples
        cov_dim_max: Maximum covariate dimension
        n_experiments: Number of experiments
        support_size: Support size
        sigma_outcome: Outcome noise standard deviation
        covariate_pdf: Covariate PDF type
        beta: Beta parameter
        plot: Whether to generate plots
        verbose: Enable verbose output

    Returns:
        Dictionary with 'absolute' and 'relative' keys containing (biases, sigmas) tuples
    """
    from plot_utils import plot_method_comparison

    biases_abs, sigmas_abs = plot_method_comparison(
        ortho_rec_tau,
        treatment_effect,
        output_dir,
        n_samples,
        cov_dim_max,
        n_experiments,
        support_size,
        sigma_outcome,
        covariate_pdf,
        beta,
        plot=plot,
        relative_error=False,
        verbose=verbose,
    )

    biases_rel, sigmas_rel = plot_method_comparison(
        ortho_rec_tau,
        treatment_effect,
        output_dir,
        n_samples,
        cov_dim_max,
        n_experiments,
        support_size,
        sigma_outcome,
        covariate_pdf,
        beta,
        plot=plot,
        relative_error=True,
        verbose=verbose,
    )

    return {"absolute": (biases_abs, sigmas_abs), "relative": (biases_rel, sigmas_rel)}


def plot_gennorm_suite(
    all_results: List[Dict],
    config: OMLExperimentConfig,
    param_grid: OMLParameterGrid,
    save_subfolder: bool = False,
) -> None:
    """Generate suite of gennorm plots with different filter configurations.

    This consolidates multiple plot_gennorm calls into a single function.

    Args:
        all_results: List of all experiment results
        config: Experiment configuration
        param_grid: Parameter grid with filter values
        save_subfolder: Whether to save in subfolder
    """
    from plot_utils import plot_gennorm

    # Define plotting configurations
    plot_configs = [
        {
            "filter_type": "beta",
            "filter_value": param_grid.beta_filter,
            "compare_method": "homl",
            "plot_type": "bias",
        },
        {
            "filter_type": "support",
            "filter_value": param_grid.support_filter,
            "compare_method": "homl",
            "plot_type": "bias",
        },
    ]

    # Execute each plotting configuration
    for plot_config in plot_configs:
        plot_gennorm(
            all_results,
            config,
            filter_type=plot_config["filter_type"],
            filter_value=plot_config["filter_value"],
            compare_method=plot_config["compare_method"],
            plot_type=plot_config["plot_type"],
            save_subfolder=save_subfolder,
        )


def plot_ica_vs_homl_comparison(all_results: List[Dict], output_dir: str) -> None:
    """Plot comparison of ICA vs HOML biases.

    Args:
        all_results: List of all experiment results
        output_dir: Output directory for plot
    """
    # Extract data
    x_values_ica_var_coeff = [res["ica_var_coeff"] for res in all_results]
    y_values_ica_biases = [res["biases"][OMLPlottingConfig.METHOD_ICA_START] for res in all_results]
    y_values_homl_biases = [res["biases"][OMLPlottingConfig.METHOD_ROBUST_ORTHO_EST_SPLIT_ML] for res in all_results]

    # Create plot
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    y_diff_biases = np.array(y_values_ica_biases) - np.array(y_values_homl_biases)
    ax.scatter(x_values_ica_var_coeff, y_diff_biases, color="purple", alpha=0.75, label="ICA - HOML")
    ax.set_xlabel(r"$1+\Vert b+a\theta\Vert_2^2$")
    ax.set_xscale("log")
    ax.set_ylabel(r"Differences $|\theta-\hat{\theta}|$")
    ax.legend()

    plt.savefig(os.path.join(output_dir, "gennorm_asymp_var_overall.svg"), dpi=300, bbox_inches="tight")
    plt.close()


def calculate_rmse(bias: float, sigma: float) -> float:
    """Calculate RMSE from bias and standard deviation.

    RMSE = sqrt(bias^2 + sigma^2)

    Args:
        bias: Mean absolute error (bias)
        sigma: Standard deviation

    Returns:
        RMSE value
    """
    return np.sqrt(bias**2 + sigma**2)


def plot_ica_vs_homl_rmse_comparison(all_results: List[Dict], output_dir: str) -> None:
    """Plot comparison of ICA vs HOML RMSE values.

    Args:
        all_results: List of all experiment results
        output_dir: Output directory for plots
    """
    # Create output directory for RMSE plots
    rmse_dir = os.path.join(output_dir, "rmse_comparison")
    os.makedirs(rmse_dir, exist_ok=True)

    # Extract data and calculate RMSE
    x_values_ica_var_coeff = [res["ica_var_coeff"] for res in all_results]

    # Calculate RMSE for ICA and HOML
    y_values_ica_rmse = [
        calculate_rmse(
            res["biases"][OMLPlottingConfig.METHOD_ICA_START],
            res["sigmas"][OMLPlottingConfig.METHOD_ICA_START],
        )
        for res in all_results
    ]
    y_values_homl_rmse = [
        calculate_rmse(
            res["biases"][OMLPlottingConfig.METHOD_ROBUST_ORTHO_EST_SPLIT_ML],
            res["sigmas"][OMLPlottingConfig.METHOD_ROBUST_ORTHO_EST_SPLIT_ML],
        )
        for res in all_results
    ]

    # Plot 1: RMSE comparison scatter plot
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.scatter(x_values_ica_var_coeff, y_values_ica_rmse, color="blue", alpha=0.75, label="ICA")
    ax.scatter(x_values_ica_var_coeff, y_values_homl_rmse, color="red", alpha=0.75, label="HOML")
    ax.set_xlabel(r"$1+\Vert b+a\theta\Vert_2^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"RMSE $\sqrt{\text{bias}^2 + \sigma^2}$")
    ax.legend()

    plt.savefig(os.path.join(rmse_dir, "rmse_ica_vs_homl.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: RMSE difference (ICA - HOML)
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    y_diff_rmse = np.array(y_values_ica_rmse) - np.array(y_values_homl_rmse)
    ax.scatter(x_values_ica_var_coeff, y_diff_rmse, color="purple", alpha=0.75, label="ICA - HOML")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$1+\Vert b+a\theta\Vert_2^2$")
    ax.set_xscale("log")
    ax.set_ylabel(r"RMSE Difference (ICA - HOML)")
    ax.legend()

    plt.savefig(os.path.join(rmse_dir, "rmse_difference_ica_vs_homl.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Relative RMSE difference (ICA - HOML) / HOML
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    y_rel_diff_rmse = y_diff_rmse / np.array(y_values_homl_rmse)
    ax.scatter(x_values_ica_var_coeff, y_rel_diff_rmse, color="green", alpha=0.75, label="(ICA - HOML) / HOML")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$1+\Vert b+a\theta\Vert_2^2$")
    ax.set_xscale("log")
    ax.set_ylabel(r"Relative RMSE Difference")
    ax.legend()

    plt.savefig(os.path.join(rmse_dir, "rmse_relative_difference_ica_vs_homl.svg"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_rmse_by_parameter(
    all_results: List[Dict],
    output_dir: str,
    x_param: str = "n_samples",
) -> None:
    """Plot RMSE comparison grouped by a parameter.

    Args:
        all_results: List of all experiment results
        output_dir: Output directory for plots
        x_param: Parameter to use as x-axis (n_samples, support_size, beta)
    """
    rmse_dir = os.path.join(output_dir, "rmse_comparison")
    os.makedirs(rmse_dir, exist_ok=True)

    # Extract unique x values
    x_values = sorted(set(res[x_param] for res in all_results))

    # Group results by x_param
    grouped_data = {x: {"ica_rmse": [], "homl_rmse": []} for x in x_values}

    for res in all_results:
        x_val = res[x_param]
        ica_rmse = calculate_rmse(
            res["biases"][OMLPlottingConfig.METHOD_ICA_START],
            res["sigmas"][OMLPlottingConfig.METHOD_ICA_START],
        )
        homl_rmse = calculate_rmse(
            res["biases"][OMLPlottingConfig.METHOD_ROBUST_ORTHO_EST_SPLIT_ML],
            res["sigmas"][OMLPlottingConfig.METHOD_ROBUST_ORTHO_EST_SPLIT_ML],
        )
        grouped_data[x_val]["ica_rmse"].append(ica_rmse)
        grouped_data[x_val]["homl_rmse"].append(homl_rmse)

    # Calculate means and stds
    ica_means = [np.mean(grouped_data[x]["ica_rmse"]) for x in x_values]
    ica_stds = [np.std(grouped_data[x]["ica_rmse"]) for x in x_values]
    homl_means = [np.mean(grouped_data[x]["homl_rmse"]) for x in x_values]
    homl_stds = [np.std(grouped_data[x]["homl_rmse"]) for x in x_values]

    # Plot RMSE with error bars
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.errorbar(x_values, ica_means, yerr=ica_stds, fmt="o-", color="blue", alpha=0.75, label="ICA", capsize=4)
    ax.errorbar(x_values, homl_means, yerr=homl_stds, fmt="o-", color="red", alpha=0.75, label="HOML", capsize=4)

    x_label_map = {
        "n_samples": r"Sample size $n$",
        "support_size": r"Covariate dimension $d$",
        "beta": r"Gen. normal param. $\beta$",
    }
    ax.set_xlabel(x_label_map.get(x_param, x_param))
    ax.set_ylabel(r"RMSE $\sqrt{\text{bias}^2 + \sigma^2}$")
    ax.legend()

    plt.savefig(os.path.join(rmse_dir, f"rmse_by_{x_param}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot RMSE difference with error bars
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    diff_means = [
        np.mean(np.array(grouped_data[x]["ica_rmse"]) - np.array(grouped_data[x]["homl_rmse"])) for x in x_values
    ]
    diff_stds = [
        np.std(np.array(grouped_data[x]["ica_rmse"]) - np.array(grouped_data[x]["homl_rmse"])) for x in x_values
    ]

    ax.errorbar(
        x_values, diff_means, yerr=diff_stds, fmt="o-", color="purple", alpha=0.75, label="ICA - HOML", capsize=4
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(x_label_map.get(x_param, x_param))
    ax.set_ylabel(r"RMSE Difference (ICA - HOML)")
    ax.legend()

    plt.savefig(os.path.join(rmse_dir, f"rmse_difference_by_{x_param}.svg"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_oml_plots(
    all_results: List[Dict],
    config: OMLExperimentConfig,
    param_grid: OMLParameterGrid,
    treatment_effects: List[float],
) -> None:
    """Generate all standard OML plots.

    This orchestrates the full plotting pipeline for OML experiments.

    Args:
        all_results: List of all experiment results
        config: Experiment configuration
        param_grid: Parameter grid
        treatment_effects: List of treatment effects
    """
    from plot_utils import plot_asymptotic_var_comparison, plot_multi_treatment

    print("\nGenerating plots...")

    # Generate gennorm plots if applicable
    if config.covariate_pdf == "gennorm" and not config.asymptotic_var:
        plot_ica_vs_homl_comparison(all_results, config.output_dir)
        plot_gennorm_suite(all_results, config, param_grid, save_subfolder=False)

    # Generate asymptotic variance comparison
    plot_asymptotic_var_comparison(all_results, config, save_subfolder=False)

    # Generate multi-treatment plots
    plot_multi_treatment(all_results, config, treatment_effects)

    # Generate RMSE comparison plots
    print("Generating RMSE comparison plots...")
    plot_ica_vs_homl_rmse_comparison(all_results, config.output_dir)

    # Generate RMSE by parameter plots for available parameters
    available_params = set()
    if all_results:
        first_result = all_results[0]
        for param in ["n_samples", "support_size", "beta"]:
            if param in first_result:
                available_params.add(param)

    for param in available_params:
        # Check if there are multiple unique values for this parameter
        unique_values = set(res[param] for res in all_results)
        if len(unique_values) > 1:
            plot_rmse_by_parameter(all_results, config.output_dir, x_param=param)

    print("All plots generated successfully!")


def extract_method_results(
    results: List[tuple], treatment_effect: float, true_coef_treatment: np.ndarray, true_coef_outcome: np.ndarray
) -> tuple:
    """Extract and structure method results from raw experiment outputs.

    Args:
        results: List of raw experiment results
        treatment_effect: True treatment effect
        true_coef_treatment: True treatment coefficients
        true_coef_outcome: True outcome coefficients

    Returns:
        Tuple of (ortho_rec_tau, first_stage_mse)
    """
    ortho_rec_tau = [
        [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml]
        + ica_treatment_effect_estimate.tolist()
        for (
            ortho_ml,
            robust_ortho_ml,
            robust_ortho_est_ml,
            robust_ortho_est_split_ml,
            _,
            _,
            ica_treatment_effect_estimate,
            _,
        ) in results
    ]

    first_stage_mse = [
        [
            np.linalg.norm(true_coef_treatment - coef_treatment),
            np.linalg.norm(true_coef_outcome - coef_outcome),
            np.linalg.norm(ica_treatment_effect_estimate - treatment_effect),
            ica_mcc,
        ]
        for (
            _,
            _,
            _,
            _,
            coef_treatment,
            coef_outcome,
            ica_treatment_effect_estimate,
            ica_mcc,
        ) in results
    ]

    return ortho_rec_tau, first_stage_mse


def save_results_with_metadata(
    all_results: List[Dict], output_dir: str, results_filename: str, config: OMLExperimentConfig
) -> None:
    """Save results with metadata and summary statistics.

    Args:
        all_results: List of all experiment results
        output_dir: Output directory
        results_filename: Filename for results
        config: Experiment configuration
    """
    results_file_path = os.path.join(output_dir, results_filename)
    np.save(results_file_path, all_results)

    print(f"\nAll results with noise parameters have been saved to {results_file_path}")
    print(f"Total configurations run: {len(all_results)}")
    print(f"Experiments per configuration: {config.n_experiments}")
    print(f"Total experiments: {len(all_results) * config.n_experiments}")
