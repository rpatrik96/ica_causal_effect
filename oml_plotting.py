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
    for treatment_effect in treatment_effects:
        plot_multi_treatment(all_results, config, treatment_effect)

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
