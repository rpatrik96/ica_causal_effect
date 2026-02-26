"""
Plotting utilities for OML experiments.

This module provides consolidated plotting functions and configurations
for Orthogonal Machine Learning experiments.
"""

import os
from typing import Dict, List

import numpy as np

from oml_utils import OMLExperimentConfig, OMLParameterGrid


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

    # Get available values in results
    available_betas = {res["beta"] for res in all_results}
    available_supports = {res["support_size"] for res in all_results}

    # Define plotting configurations
    plot_configs = [
        # Bias plots
        {
            "filter_type": "beta",
            "filter_value": param_grid.beta_filter,
            "compare_method": "homl",
            "plot_type": "bias",
            "available_values": available_betas,
        },
        {
            "filter_type": "support",
            "filter_value": param_grid.support_filter,
            "compare_method": "homl",
            "plot_type": "bias",
            "available_values": available_supports,
        },
        # RMSE difference plots (ICA - HOML)
        {
            "filter_type": "beta",
            "filter_value": param_grid.beta_filter,
            "compare_method": "homl",
            "plot_type": "rmse",
            "available_values": available_betas,
        },
        {
            "filter_type": "support",
            "filter_value": param_grid.support_filter,
            "compare_method": "homl",
            "plot_type": "rmse",
            "available_values": available_supports,
        },
    ]

    # Execute each plotting configuration
    for plot_config in plot_configs:
        filter_value = plot_config["filter_value"]
        available_values = plot_config["available_values"]

        # Skip if filter value not in results
        if filter_value not in available_values:
            print(
                f"Skipping {plot_config['filter_type']} plot: filter value {filter_value} "
                f"not in available values {sorted(available_values)}"
            )
            continue

        plot_gennorm(
            all_results,
            config,
            filter_type=plot_config["filter_type"],
            filter_value=filter_value,
            compare_method=plot_config["compare_method"],
            plot_type=plot_config["plot_type"],
            save_subfolder=save_subfolder,
        )


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

    # Generate gennorm plots (including RMSE heatmaps) if applicable
    if config.covariate_pdf == "gennorm" and not config.asymptotic_var:
        plot_gennorm_suite(all_results, config, param_grid, save_subfolder=False)

    # Generate asymptotic variance comparison
    plot_asymptotic_var_comparison(all_results, config, save_subfolder=False)

    # Generate multi-treatment plots
    plot_multi_treatment(all_results, config, treatment_effects)

    print("All plots generated successfully!")


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
