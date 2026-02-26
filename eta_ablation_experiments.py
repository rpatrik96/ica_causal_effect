"""
Experiment and computation functions for noise distribution and coefficient ablation studies.

This module contains functions for running treatment effect estimation experiments
comparing different noise distributions, coefficient configurations, and variance settings.
"""

import os
from itertools import product
from typing import List, Tuple

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

from ablation_utils import (
    HOML_IDX,
    ICA_IDX,
    calculate_homl_moments,
    calculate_ica_moments,
    compute_estimation_statistics,
    compute_estimation_statistics_varying_te,
    create_covariate_sampler,
    create_outcome_noise_sampler,
    extract_treatment_estimates,
    run_parallel_experiments,
)
from oml_runner import setup_treatment_noise

# =============================================================================
# Coefficient Utilities
# =============================================================================


def compute_ica_var_coeff(
    treatment_coef_scalar: float,
    outcome_coef_scalar: float,
    treatment_effect: float,
) -> float:
    """Compute the ICA variance coefficient.

    The ICA variance coefficient is:
      ica_var_coeff = 1 + ||outcome_coef + treatment_coef * treatment_effect||^2

    For scalar coefficients (only first element non-zero):
      ica_var_coeff = 1 + (outcome_coef_scalar + treatment_coef_scalar * treatment_effect)^2

    Args:
        treatment_coef_scalar: Scalar value for treatment coefficient
        outcome_coef_scalar: Scalar value for outcome coefficient
        treatment_effect: True treatment effect

    Returns:
        ICA variance coefficient
    """
    return 1 + (outcome_coef_scalar + treatment_coef_scalar * treatment_effect) ** 2


def compute_constrained_treatment_coef(
    target_ica_var_coeff: float,
    treatment_effect: float = 1.0,
    outcome_coef_scalar: float = 0.0,
) -> float:
    """Compute treatment coefficient scalar to achieve target ICA variance coefficient.

    Solves for treatment_coef_scalar in:
      target_ica_var_coeff = 1 + (outcome_coef_scalar + treatment_coef_scalar * treatment_effect)^2

    Args:
        target_ica_var_coeff: Target ICA variance coefficient (must be >= 1)
        treatment_effect: True treatment effect (must be non-zero)
        outcome_coef_scalar: Scalar value for outcome coefficient

    Returns:
        Treatment coefficient scalar that achieves the target ICA variance coefficient

    Raises:
        ValueError: If target_ica_var_coeff < 1 or treatment_effect == 0
    """
    if target_ica_var_coeff < 1:
        raise ValueError(f"target_ica_var_coeff must be >= 1, got {target_ica_var_coeff}")
    if treatment_effect == 0:
        raise ValueError("treatment_effect must be non-zero")

    # Solve: target = 1 + (outcome_coef_scalar + treatment_coef_scalar * treatment_effect)^2
    # => (outcome_coef_scalar + treatment_coef_scalar * treatment_effect)^2 = target - 1
    # => outcome_coef_scalar + treatment_coef_scalar * treatment_effect = +/- sqrt(target - 1)
    # => treatment_coef_scalar = (sqrt(target - 1) - outcome_coef_scalar) / treatment_effect
    # We choose the positive root for simplicity
    coef_sum = np.sqrt(target_ica_var_coeff - 1)
    treatment_coef_scalar = (coef_sum - outcome_coef_scalar) / treatment_effect

    return treatment_coef_scalar


# =============================================================================
# Distribution Parsing
# =============================================================================


def parse_distribution_spec(dist_spec: str) -> tuple:
    """Parse distribution specification string.

    Supports:
    - Simple distribution names: "discrete", "uniform", "laplace", etc.
    - Parameterized gennorm: "gennorm:1.5" (with beta=1.5)

    Args:
        dist_spec: Distribution specification string

    Returns:
        Tuple of (distribution_name, beta_value_or_none)
    """
    if ":" in dist_spec:
        parts = dist_spec.split(":")
        dist_name = parts[0]
        if dist_name == "gennorm":
            beta_val = float(parts[1])
            return "gennorm", beta_val
        raise ValueError(f"Parameterized format only supported for 'gennorm', got: {dist_name}")
    return dist_spec, None


# =============================================================================
# Noise Ablation Experiments
# =============================================================================


def run_noise_ablation_experiments(
    noise_distributions: list,
    n_samples: int = 50000,
    n_experiments: int = 10,
    support_size: int = 10,
    treatment_effect: float = 1.0,
    beta: float = 1.0,
    sigma_outcome: float = np.sqrt(3.0),
    covariate_pdf: str = "gennorm",
    check_convergence: bool = False,
    verbose: bool = False,
    seed: int = 12143,
    randomize_coeffs: bool = False,
    n_random_configs: int = 10,
    treatment_effect_range: Tuple[float, float] = (0.1, 5.0),
    treatment_coef_range: Tuple[float, float] = (-2.0, 2.0),
    outcome_coef_range: Tuple[float, float] = (-2.0, 2.0),
    oracle_support: bool = True,
):
    """Run experiments across different noise distributions.

    Args:
        noise_distributions: List of noise distribution names to test
        n_samples: Number of samples per experiment
        n_experiments: Number of Monte Carlo replications
        support_size: Size of support for coefficients
        treatment_effect: True treatment effect value (ignored if randomize_coeffs=True)
        beta: Beta parameter for gennorm covariate distribution
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
        seed: Random seed
        randomize_coeffs: If True, randomize coefficients for each experiment
        n_random_configs: Number of random configurations per distribution
        treatment_effect_range: Range for random treatment effect [min, max]
        treatment_coef_range: Range for random treatment coefficients [min, max]
        outcome_coef_range: Range for random outcome coefficients [min, max]
        oracle_support: If True, ICA receives x[:, support]. If False, both
            OML and ICA receive full x matrix.

    Returns:
        Dictionary with results for each noise distribution.
        If randomize_coeffs=True, includes per-experiment coefficient info.
    """
    np.random.seed(seed)

    # Setup samplers using shared utilities
    x_sample = create_covariate_sampler(covariate_pdf, beta)
    epsilon_sample = create_outcome_noise_sampler(sigma_outcome)

    # Setup base coefficients
    cov_dim_max = support_size
    treatment_support = outcome_support = np.array(range(support_size))

    # Compute regularization parameter
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    all_results = {}

    # Generate random coefficient configurations if needed
    if randomize_coeffs:
        random_configs = []
        for _ in range(n_random_configs):
            tc_val = np.random.uniform(treatment_coef_range[0], treatment_coef_range[1])
            oc_val = np.random.uniform(outcome_coef_range[0], outcome_coef_range[1])
            te_val = np.random.uniform(treatment_effect_range[0], treatment_effect_range[1])
            random_configs.append((tc_val, oc_val, te_val))
        print(f"Generated {n_random_configs} random coefficient configurations")
    else:
        # Single fixed configuration
        random_configs = [(1.0, 1.0, treatment_effect)]

    for noise_dist_spec in noise_distributions:
        print(f"\nRunning experiments for noise distribution: {noise_dist_spec}")

        # Parse distribution specification
        noise_dist, gennorm_beta = parse_distribution_spec(noise_dist_spec)

        # Setup treatment noise
        params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(
            distribution=noise_dist, gennorm_beta=gennorm_beta
        )

        # Calculate base moments using shared utility
        moments = calculate_homl_moments(noise_dist, params_or_discounts, mean_discount, probs)
        eta_second_moment = moments["eta_second_moment"]
        eta_third_moment = moments["eta_third_moment"]
        eta_cubed_variance = moments["eta_cubed_variance"]
        eta_fourth_moment = moments["eta_fourth_moment"]
        homl_asymptotic_var = moments["homl_asymptotic_var"]

        # Calculate empirical kurtosis
        empirical_eta_samples = eta_sample(100000)
        empirical_excess_kurtosis = scipy_kurtosis(empirical_eta_samples, fisher=True)
        print(f"  Theoretical excess kurtosis: {eta_cubed_variance / eta_second_moment**2 - 3:.4f}")
        print(f"  Empirical excess kurtosis: {empirical_excess_kurtosis:.4f}")

        # Store per-config results
        config_results = []

        for config_idx, (tc_val, oc_val, te_val) in enumerate(random_configs):
            # Create coefficient arrays
            treatment_coef = np.zeros(support_size)
            treatment_coef[0] = tc_val
            outcome_coef = np.zeros(support_size)
            outcome_coef[0] = oc_val
            current_treatment_effect = te_val

            # Calculate ICA variance coefficient
            ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * current_treatment_effect) ** 2

            if randomize_coeffs:
                print(
                    f"  Config {config_idx + 1}/{len(random_configs)}: "
                    f"tc={tc_val:.3f}, oc={oc_val:.3f}, te={te_val:.3f}, ica_var_coeff={ica_var_coeff:.3f}"
                )

            # Calculate ICA moments using shared utility
            ica_moments = calculate_ica_moments(
                noise_dist,
                treatment_coef,
                outcome_coef,
                current_treatment_effect,
                params_or_discounts,
                mean_discount,
                probs,
                eta_cubed_variance,
            )
            eta_excess_kurtosis = ica_moments["eta_excess_kurtosis"]
            eta_skewness_squared = ica_moments["eta_skewness_squared"]
            ica_asymptotic_var = ica_moments["ica_asymptotic_var"]

            # Run parallel experiments using shared utility
            results = run_parallel_experiments(
                n_experiments=n_experiments,
                x_sample=x_sample,
                eta_sample=eta_sample,
                epsilon_sample=epsilon_sample,
                n_samples=n_samples,
                cov_dim_max=cov_dim_max,
                treatment_effect=current_treatment_effect,
                treatment_support=treatment_support,
                treatment_coef=treatment_coef,
                outcome_support=outcome_support,
                outcome_coef=outcome_coef,
                eta_second_moment=eta_second_moment,
                eta_third_moment=eta_third_moment,
                lambda_reg=lambda_reg,
                check_convergence=check_convergence,
                verbose=verbose,
                oracle_support=oracle_support,
            )

            if len(results) == 0:
                print("    No experiments converged")
                continue

            # Extract and compute statistics using shared utilities
            ortho_rec_tau = extract_treatment_estimates(results)
            biases, sigmas, rmse = compute_estimation_statistics(ortho_rec_tau, current_treatment_effect)

            config_result = {
                "ortho_rec_tau": ortho_rec_tau,
                "biases": biases,
                "sigmas": sigmas,
                "rmse": rmse,
                "n_experiments": len(results),
                "treatment_coef_scalar": tc_val,
                "outcome_coef_scalar": oc_val,
                "treatment_effect": current_treatment_effect,
                "ica_var_coeff": ica_var_coeff,
                "ica_asymptotic_var": ica_asymptotic_var,
            }
            config_results.append(config_result)

        if len(config_results) == 0:
            print(f"No experiments converged for {noise_dist_spec}")
            continue

        print(
            f"Experiments kept: {sum(r['n_experiments'] for r in config_results)} total "
            f"across {len(config_results)} configs"
        )

        # Aggregate results across configs
        if randomize_coeffs:
            all_ortho_rec_tau = []
            all_treatment_effects = []
            for cr in config_results:
                all_ortho_rec_tau.extend(cr["ortho_rec_tau"])
                all_treatment_effects.extend([cr["treatment_effect"]] * len(cr["ortho_rec_tau"]))

            all_treatment_effects = np.array(all_treatment_effects)
            biases, sigmas, rmse = compute_estimation_statistics_varying_te(all_ortho_rec_tau, all_treatment_effects)

            avg_ica_var_coeff = np.mean([cr["ica_var_coeff"] for cr in config_results])
            avg_ica_asymptotic_var = np.mean([cr["ica_asymptotic_var"] for cr in config_results])
        else:
            cr = config_results[0]
            biases = cr["biases"]
            sigmas = cr["sigmas"]
            rmse = cr["rmse"]
            avg_ica_var_coeff = cr["ica_var_coeff"]
            avg_ica_asymptotic_var = cr["ica_asymptotic_var"]
            all_ortho_rec_tau = cr["ortho_rec_tau"]

        # Store results
        all_results[noise_dist_spec] = {
            "ortho_rec_tau": all_ortho_rec_tau,
            "biases": biases,
            "sigmas": sigmas,
            "rmse": rmse,
            "n_samples": n_samples,
            "n_experiments": sum(r["n_experiments"] for r in config_results),
            "eta_second_moment": eta_second_moment,
            "eta_third_moment": eta_third_moment,
            "eta_fourth_moment": eta_fourth_moment,
            "eta_excess_kurtosis": eta_excess_kurtosis,
            "eta_empirical_excess_kurtosis": empirical_excess_kurtosis,
            "eta_skewness_squared": eta_skewness_squared,
            "homl_asymptotic_var": homl_asymptotic_var,
            "ica_asymptotic_var": avg_ica_asymptotic_var,
            "ica_var_coeff": avg_ica_var_coeff,
            "gennorm_beta": gennorm_beta,
            "randomize_coeffs": randomize_coeffs,
            "config_results": config_results if randomize_coeffs else None,
        }

        # Print summary
        method_names = ["Ortho ML", "Robust Ortho ML", "Robust Ortho Est", "Robust Ortho Split", "ICA"]
        print(f"\nResults for {noise_dist_spec}:")
        print(f"  Excess kurtosis: {eta_excess_kurtosis:.4f}")
        print(f"  Third moment: {eta_third_moment:.4f}")
        if randomize_coeffs:
            print(f"  Avg ICA var coeff: {avg_ica_var_coeff:.4f}")
        for i, name in enumerate(method_names):
            if i < len(biases):
                print(f"  {name}: bias={biases[i]:.4f}, std={sigmas[i]:.4f}, rmse={rmse[i]:.4f}")

    return all_results


# =============================================================================
# Variance Ablation Experiments
# =============================================================================


def run_variance_ablation_experiments(
    beta_values: List[float] = None,
    variance_values: List[float] = None,
    n_samples: int = 5000,
    n_experiments: int = 20,
    support_size: int = 10,
    treatment_effect: float = 1.0,
    covariate_beta: float = 1.0,
    sigma_outcome: float = np.sqrt(3.0),
    covariate_pdf: str = "gennorm",
    check_convergence: bool = False,
    verbose: bool = False,
    seed: int = 12143,
    oracle_support: bool = True,
) -> dict:
    """Run experiments varying gennorm beta and variance.

    Creates a grid of experiments over beta (shape) and variance values for
    the treatment noise distribution.

    Args:
        beta_values: List of gennorm beta values (shape parameter)
        variance_values: List of variance values (scale^2)
        n_samples: Number of samples per experiment
        n_experiments: Number of Monte Carlo replications
        support_size: Size of support for coefficients
        treatment_effect: True treatment effect value
        covariate_beta: Beta parameter for gennorm covariate distribution
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
        seed: Random seed
        oracle_support: If True, ICA receives x[:, support]. If False, both
            OML and ICA receive full x matrix.

    Returns:
        Dictionary with results organized by (beta, variance) pairs
    """
    # Default grids
    if beta_values is None:
        beta_values = [0.5, 1.0, 1.5, 2.5, 3.0, 4.0]
    if variance_values is None:
        variance_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    np.random.seed(seed)

    # Setup samplers using shared utilities
    x_sample = create_covariate_sampler(covariate_pdf, covariate_beta)
    epsilon_sample = create_outcome_noise_sampler(sigma_outcome)

    # Setup base coefficients
    cov_dim_max = support_size
    treatment_support = outcome_support = np.array(range(support_size))

    # Fixed scalar coefficients
    treatment_coef = np.zeros(support_size)
    treatment_coef[0] = 1.0
    outcome_coef = np.zeros(support_size)
    outcome_coef[0] = 1.0

    # Compute regularization parameter
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    # Calculate ICA variance coefficient (fixed for this experiment)
    ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect) ** 2

    all_results = {
        "beta_values": np.array(beta_values),
        "variance_values": np.array(variance_values),
        "treatment_effect": treatment_effect,
        "ica_var_coeff": ica_var_coeff,
        "grid_results": {},
    }

    total_configs = len(beta_values) * len(variance_values)
    config_idx = 0

    for beta_val in beta_values:
        for var_val in variance_values:
            config_idx += 1
            scale_val = np.sqrt(var_val)  # variance = scale^2

            print(f"\n[{config_idx}/{total_configs}] beta={beta_val}, variance={var_val} (scale={scale_val:.4f})")

            # Skip Gaussian (beta=2) as ICA doesn't work
            if abs(beta_val - 2.0) < 0.01:
                print("  Skipping beta=2.0 (Gaussian) - ICA not applicable")
                all_results["grid_results"][(beta_val, var_val)] = None
                continue

            # Setup treatment noise with custom beta and scale
            params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(
                distribution="gennorm", gennorm_beta=beta_val, scale=scale_val
            )

            # Calculate moments using shared utility
            moments = calculate_homl_moments("gennorm", params_or_discounts, mean_discount, probs)
            eta_second_moment = moments["eta_second_moment"]
            eta_third_moment = moments["eta_third_moment"]
            eta_cubed_variance = moments["eta_cubed_variance"]
            homl_asymptotic_var = moments["homl_asymptotic_var"]

            # Calculate ICA moments
            ica_moments = calculate_ica_moments(
                "gennorm",
                treatment_coef,
                outcome_coef,
                treatment_effect,
                params_or_discounts,
                mean_discount,
                probs,
                eta_cubed_variance,
            )
            eta_excess_kurtosis = ica_moments["eta_excess_kurtosis"]
            ica_asymptotic_var = ica_moments["ica_asymptotic_var"]

            # Run parallel experiments
            results = run_parallel_experiments(
                n_experiments=n_experiments,
                x_sample=x_sample,
                eta_sample=eta_sample,
                epsilon_sample=epsilon_sample,
                n_samples=n_samples,
                cov_dim_max=cov_dim_max,
                treatment_effect=treatment_effect,
                treatment_support=treatment_support,
                treatment_coef=treatment_coef,
                outcome_support=outcome_support,
                outcome_coef=outcome_coef,
                eta_second_moment=eta_second_moment,
                eta_third_moment=eta_third_moment,
                lambda_reg=lambda_reg,
                check_convergence=check_convergence,
                verbose=verbose,
                oracle_support=oracle_support,
            )

            print(f"  Experiments kept: {len(results)} out of {n_experiments}")

            if len(results) == 0:
                print("  No experiments converged")
                all_results["grid_results"][(beta_val, var_val)] = None
                continue

            # Extract and compute statistics
            ortho_rec_tau = extract_treatment_estimates(results)
            biases, sigmas, rmse = compute_estimation_statistics(ortho_rec_tau, treatment_effect)

            # Store results
            all_results["grid_results"][(beta_val, var_val)] = {
                "biases": biases,
                "sigmas": sigmas,
                "rmse": rmse,
                "n_experiments": len(results),
                "eta_excess_kurtosis": eta_excess_kurtosis,
                "homl_asymptotic_var": homl_asymptotic_var,
                "ica_asymptotic_var": ica_asymptotic_var,
            }

            # Print summary
            print(f"  HOML: bias={biases[HOML_IDX]:.4f}, std={sigmas[HOML_IDX]:.4f}, rmse={rmse[HOML_IDX]:.4f}")
            if ICA_IDX < len(biases):
                print(f"  ICA:  bias={biases[ICA_IDX]:.4f}, std={sigmas[ICA_IDX]:.4f}, rmse={rmse[ICA_IDX]:.4f}")

    return all_results


# =============================================================================
# Coefficient Ablation Experiments
# =============================================================================


def run_coefficient_ablation_experiments(
    noise_distribution: str = "discrete",
    treatment_coef_values: List[float] = None,
    outcome_coef_values: List[float] = None,
    treatment_effect_values: List[float] = None,
    n_samples: int = 5000,
    n_experiments: int = 20,
    support_size: int = 10,
    beta: float = 1.0,
    sigma_outcome: float = np.sqrt(3.0),
    covariate_pdf: str = "gennorm",
    check_convergence: bool = False,
    verbose: bool = False,
    seed: int = 12143,
    oracle_support: bool = True,
) -> List[dict]:
    """Run experiments with varying coefficient configurations.

    Args:
        noise_distribution: Noise distribution for treatment
        treatment_coef_values: List of scalar values for treatment coefficient
        outcome_coef_values: List of scalar values for outcome coefficient
        treatment_effect_values: List of treatment effect values
        n_samples: Number of samples per experiment
        n_experiments: Number of Monte Carlo replications
        support_size: Size of support for coefficients
        beta: Beta parameter for gennorm covariate distribution
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
        seed: Random seed
        oracle_support: If True, ICA receives x[:, support]. If False, both
            OML and ICA receive full x matrix.

    Returns:
        List of result dictionaries, one per coefficient configuration
    """
    # Default coefficient grids
    if treatment_coef_values is None:
        treatment_coef_values = [-0.002, 0.05, -0.43, 1.56]
    if outcome_coef_values is None:
        outcome_coef_values = [0.003, -0.02, 0.63, -1.45]
    if treatment_effect_values is None:
        treatment_effect_values = [0.01, 0.1, 0.5, 1.0, 3.0, 10]

    np.random.seed(seed)

    # Parse distribution
    noise_dist, gennorm_beta = parse_distribution_spec(noise_distribution)

    # Setup samplers using shared utilities
    x_sample = create_covariate_sampler(covariate_pdf, beta)
    epsilon_sample = create_outcome_noise_sampler(sigma_outcome)

    # Setup treatment noise
    params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(
        distribution=noise_dist, gennorm_beta=gennorm_beta
    )

    # Compute regularization parameter
    cov_dim_max = support_size
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    # Setup support indices
    treatment_support = outcome_support = np.array(range(support_size))

    # Calculate base moments using shared utility
    moments = calculate_homl_moments(noise_dist, params_or_discounts, mean_discount, probs)
    eta_second_moment = moments["eta_second_moment"]
    eta_third_moment = moments["eta_third_moment"]
    homl_asymptotic_var = moments["homl_asymptotic_var"]

    all_results = []

    # Create coefficient grid
    coef_grid = list(product(treatment_coef_values, outcome_coef_values, treatment_effect_values))
    total_configs = len(coef_grid)

    print(f"\nRunning coefficient ablation with {total_configs} configurations")
    print(f"Noise distribution: {noise_distribution}")

    for config_idx, (tc_val, oc_val, te_val) in enumerate(coef_grid):
        # Create coefficient arrays
        treatment_coef = np.zeros(support_size)
        treatment_coef[0] = tc_val
        outcome_coef = np.zeros(support_size)
        outcome_coef[0] = oc_val
        current_treatment_effect = te_val

        # Calculate ICA variance coefficient
        ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * current_treatment_effect) ** 2

        print(
            f"\n[{config_idx + 1}/{total_configs}] tc={tc_val}, oc={oc_val}, "
            f"te={te_val}, ica_var_coeff={ica_var_coeff:.4f}"
        )

        # Run parallel experiments using shared utility
        results = run_parallel_experiments(
            n_experiments=n_experiments,
            x_sample=x_sample,
            eta_sample=eta_sample,
            epsilon_sample=epsilon_sample,
            n_samples=n_samples,
            cov_dim_max=cov_dim_max,
            treatment_effect=current_treatment_effect,
            treatment_support=treatment_support,
            treatment_coef=treatment_coef,
            outcome_support=outcome_support,
            outcome_coef=outcome_coef,
            eta_second_moment=eta_second_moment,
            eta_third_moment=eta_third_moment,
            lambda_reg=lambda_reg,
            check_convergence=check_convergence,
            verbose=verbose,
            oracle_support=oracle_support,
        )

        print(f"  Experiments kept: {len(results)} out of {n_experiments}")

        if len(results) == 0:
            print("  No experiments converged")
            continue

        # Extract and compute statistics using shared utilities
        ortho_rec_tau = extract_treatment_estimates(results)
        biases, sigmas, rmse = compute_estimation_statistics(ortho_rec_tau, current_treatment_effect)

        # Store results
        result_dict = {
            "treatment_coef_scalar": tc_val,
            "outcome_coef_scalar": oc_val,
            "treatment_effect": current_treatment_effect,
            "ica_var_coeff": ica_var_coeff,
            "ortho_rec_tau": ortho_rec_tau,
            "biases": biases,
            "sigmas": sigmas,
            "rmse": rmse,
            "n_samples": n_samples,
            "n_experiments": len(results),
            "noise_distribution": noise_distribution,
            "eta_second_moment": eta_second_moment,
            "eta_third_moment": eta_third_moment,
            "homl_asymptotic_var": homl_asymptotic_var,
        }
        all_results.append(result_dict)

        # Print summary for HOML and ICA only
        print(f"  HOML: bias={biases[HOML_IDX]:.4f}, std={sigmas[HOML_IDX]:.4f}, rmse={rmse[HOML_IDX]:.4f}")
        if ICA_IDX < len(biases):
            print(f"  ICA:  bias={biases[ICA_IDX]:.4f}, std={sigmas[ICA_IDX]:.4f}, rmse={rmse[ICA_IDX]:.4f}")

    return all_results


def run_sample_dimension_grid_experiments(
    sample_sizes: List[int] = None,
    dimension_values: List[int] = None,
    beta_values: List[float] = None,
    axis_mode: str = "d_vs_n",
    fixed_beta: float = 1.0,
    fixed_dimension: int = 10,
    noise_distribution: str = "discrete",
    n_experiments: int = 20,
    treatment_effect: float = 1.0,
    treatment_coef_scalar: float = 1.0,
    outcome_coef_scalar: float = 1.0,
    sigma_outcome: float = np.sqrt(3.0),
    covariate_pdf: str = "gennorm",
    check_convergence: bool = False,
    verbose: bool = False,
    seed: int = 12143,
    oracle_support: bool = True,
) -> List[dict]:
    """Run experiments over sample size and dimension/beta grid.

    Args:
        sample_sizes: List of sample sizes to test
        dimension_values: List of covariate dimensions (for d_vs_n mode)
        beta_values: List of gennorm beta values (for beta_vs_n mode)
        axis_mode: "d_vs_n" (dimension vs sample size) or "beta_vs_n" (beta vs sample size)
        fixed_beta: Fixed beta when axis_mode="d_vs_n"
        fixed_dimension: Fixed dimension when axis_mode="beta_vs_n"
        noise_distribution: Noise distribution for treatment
        n_experiments: Number of Monte Carlo replications
        treatment_effect: True treatment effect
        treatment_coef_scalar: Scalar value for treatment coefficient
        outcome_coef_scalar: Scalar value for outcome coefficient
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
        seed: Random seed
        oracle_support: If True, ICA receives x[:, support]. If False, both
            OML and ICA receive full x matrix.

    Returns:
        List of result dictionaries, one per (sample_size, dimension/beta) configuration
    """
    # Default grids
    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 5000, 10000]

    if axis_mode == "d_vs_n":
        if dimension_values is None:
            dimension_values = [5, 10, 20, 50]
        grid_values = dimension_values
        grid_key = "support_size"
        covariate_beta = fixed_beta
    elif axis_mode == "beta_vs_n":
        if beta_values is None:
            beta_values = [0.5, 1.0, 2.0, 3.0, 4.0]
        grid_values = beta_values
        grid_key = "beta"
    else:
        raise ValueError(f"Invalid axis_mode: {axis_mode}. Use 'd_vs_n' or 'beta_vs_n'.")

    np.random.seed(seed)

    # Parse distribution
    noise_dist, gennorm_beta = parse_distribution_spec(noise_distribution)

    # Setup treatment noise
    params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(
        distribution=noise_dist, gennorm_beta=gennorm_beta
    )

    # Calculate base moments
    moments = calculate_homl_moments(noise_dist, params_or_discounts, mean_discount, probs)
    eta_second_moment = moments["eta_second_moment"]
    eta_third_moment = moments["eta_third_moment"]
    homl_asymptotic_var = moments["homl_asymptotic_var"]

    all_results = []

    # Create grid
    grid = list(product(sample_sizes, grid_values))
    total_configs = len(grid)

    print(f"\nRunning sample-dimension grid experiments with {total_configs} configurations")
    print(f"Axis mode: {axis_mode}")
    print(f"Noise distribution: {noise_distribution}")
    print(f"Covariate distribution: {covariate_pdf}")

    for config_idx, (n_samples, grid_val) in enumerate(grid):
        # Set dimension and beta based on axis mode
        if axis_mode == "d_vs_n":
            support_size = grid_val
            covariate_beta = fixed_beta
        else:  # beta_vs_n
            support_size = fixed_dimension
            covariate_beta = grid_val

        # Create coefficient arrays
        treatment_coef = np.zeros(support_size)
        treatment_coef[0] = treatment_coef_scalar
        outcome_coef = np.zeros(support_size)
        outcome_coef[0] = outcome_coef_scalar

        # Calculate ICA variance coefficient
        ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect) ** 2

        # Setup samplers
        x_sample = create_covariate_sampler(covariate_pdf, covariate_beta)
        epsilon_sample = create_outcome_noise_sampler(sigma_outcome)

        # Compute regularization parameter
        lambda_reg = np.sqrt(np.log(support_size) / n_samples)

        # Setup support indices
        treatment_support = outcome_support = np.array(range(support_size))

        print(
            f"\n[{config_idx + 1}/{total_configs}] n={n_samples}, {grid_key}={grid_val}, "
            f"ica_var_coeff={ica_var_coeff:.4f}"
        )

        # Run parallel experiments
        results = run_parallel_experiments(
            n_experiments=n_experiments,
            x_sample=x_sample,
            eta_sample=eta_sample,
            epsilon_sample=epsilon_sample,
            n_samples=n_samples,
            cov_dim_max=support_size,
            treatment_effect=treatment_effect,
            treatment_support=treatment_support,
            treatment_coef=treatment_coef,
            outcome_support=outcome_support,
            outcome_coef=outcome_coef,
            eta_second_moment=eta_second_moment,
            eta_third_moment=eta_third_moment,
            lambda_reg=lambda_reg,
            check_convergence=check_convergence,
            verbose=verbose,
            oracle_support=oracle_support,
        )

        print(f"  Experiments kept: {len(results)} out of {n_experiments}")

        if len(results) == 0:
            print("  No experiments converged")
            continue

        # Extract and compute statistics
        ortho_rec_tau = extract_treatment_estimates(results)
        biases, sigmas, rmse = compute_estimation_statistics(ortho_rec_tau, treatment_effect)

        # Store results
        result_dict = {
            "n_samples": n_samples,
            "support_size": support_size,
            "beta": covariate_beta,
            "treatment_effect": treatment_effect,
            "treatment_coef_scalar": treatment_coef_scalar,
            "outcome_coef_scalar": outcome_coef_scalar,
            "ica_var_coeff": ica_var_coeff,
            "ortho_rec_tau": ortho_rec_tau,
            "biases": biases,
            "sigmas": sigmas,
            "rmse": rmse,
            "n_experiments_kept": len(results),
            "noise_distribution": noise_distribution,
            "eta_second_moment": eta_second_moment,
            "eta_third_moment": eta_third_moment,
            "homl_asymptotic_var": homl_asymptotic_var,
        }
        all_results.append(result_dict)

        # Print summary
        print(f"  HOML: bias={biases[HOML_IDX]:.4f}, std={sigmas[HOML_IDX]:.4f}, rmse={rmse[HOML_IDX]:.4f}")
        if ICA_IDX < len(biases):
            print(f"  ICA:  bias={biases[ICA_IDX]:.4f}, std={sigmas[ICA_IDX]:.4f}, rmse={rmse[ICA_IDX]:.4f}")

    return all_results


# =============================================================================
# Summary Printing Functions
# =============================================================================


def print_variance_ablation_summary(results: dict, opts) -> None:
    """Print summary table for variance ablation study.

    Args:
        results: Dictionary with variance ablation results
        opts: Parsed command-line options
    """
    print("\n" + "=" * 140)
    print("SUMMARY: Variance Ablation Study")
    print("=" * 140)
    print("\nExperiment settings:")
    print(f"  n_samples: {opts.n_samples}")
    print(f"  n_experiments: {opts.n_experiments}")
    print(f"  treatment_effect: {opts.treatment_effect}")
    print(f"  beta_values: {list(results['beta_values'])}")
    print(f"  variance_values: {list(results['variance_values'])}")

    print("\n" + "-" * 140)
    print(
        f"{'Beta':>6} {'Var':>8} {'Kurt':>10} "
        f"{'HOML Bias':>10} {'HOML Std':>10} {'HOML RMSE':>10} "
        f"{'ICA Bias':>10} {'ICA Std':>10} {'ICA RMSE':>10} {'Winner':>8}"
    )
    print("-" * 140)

    for beta_val in results["beta_values"]:
        for var_val in results["variance_values"]:
            res = results["grid_results"].get((beta_val, var_val))
            if res is None:
                print(f"{beta_val:>6.1f} {var_val:>8.2f} {'N/A':>10} " + "N/A" * 7)
                continue

            homl_bias = res["biases"][HOML_IDX]
            homl_std = res["sigmas"][HOML_IDX]
            homl_rmse = res["rmse"][HOML_IDX]
            ica_bias = res["biases"][ICA_IDX] if ICA_IDX < len(res["biases"]) else np.nan
            ica_std = res["sigmas"][ICA_IDX] if ICA_IDX < len(res["sigmas"]) else np.nan
            ica_rmse = res["rmse"][ICA_IDX] if ICA_IDX < len(res["rmse"]) else np.nan
            kurtosis = res.get("eta_excess_kurtosis", np.nan)
            winner = "ICA" if ica_rmse < homl_rmse else "HOML"

            print(
                f"{beta_val:>6.1f} {var_val:>8.2f} {kurtosis:>10.4f} "
                f"{homl_bias:>10.4f} {homl_std:>10.4f} {homl_rmse:>10.4f} "
                f"{ica_bias:>10.4f} {ica_std:>10.4f} {ica_rmse:>10.4f} {winner:>8}"
            )

    print("=" * 140)

    # Save markdown summary
    md_lines = []
    md_lines.append("# Variance Ablation Study Results\n")
    md_lines.append("## Experiment Settings\n")
    md_lines.append(f"- **n_samples**: {opts.n_samples}")
    md_lines.append(f"- **n_experiments**: {opts.n_experiments}")
    md_lines.append(f"- **treatment_effect**: {opts.treatment_effect}")
    md_lines.append(f"- **beta_values**: {list(results['beta_values'])}")
    md_lines.append(f"- **variance_values**: {list(results['variance_values'])}")
    md_lines.append("")
    md_lines.append("## Results Summary\n")
    md_lines.append(
        "| Beta | Variance | Kurtosis | HOML Bias | HOML Std | HOML RMSE | ICA Bias | ICA Std | ICA RMSE | Winner |"
    )
    md_lines.append(
        "|-----:|---------:|---------:|----------:|---------:|----------:|---------:|--------:|---------:|:------:|"
    )

    for beta_val in results["beta_values"]:
        for var_val in results["variance_values"]:
            res = results["grid_results"].get((beta_val, var_val))
            if res is None:
                md_lines.append(f"| {beta_val:.1f} | {var_val:.2f} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue

            homl_bias = res["biases"][HOML_IDX]
            homl_std = res["sigmas"][HOML_IDX]
            homl_rmse = res["rmse"][HOML_IDX]
            ica_bias = res["biases"][ICA_IDX] if ICA_IDX < len(res["biases"]) else np.nan
            ica_std = res["sigmas"][ICA_IDX] if ICA_IDX < len(res["sigmas"]) else np.nan
            ica_rmse = res["rmse"][ICA_IDX] if ICA_IDX < len(res["rmse"]) else np.nan
            kurtosis = res.get("eta_excess_kurtosis", np.nan)
            winner = "ICA" if ica_rmse < homl_rmse else "HOML"

            md_lines.append(
                f"| {beta_val:.1f} | {var_val:.2f} | {kurtosis:.4f} | "
                f"{homl_bias:.4f} | {homl_std:.4f} | {homl_rmse:.4f} | "
                f"{ica_bias:.4f} | {ica_std:.4f} | {ica_rmse:.4f} | **{winner}** |"
            )

    md_file_path = os.path.join(opts.output_dir, "variance_ablation_summary.md")
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown summary saved to: {md_file_path}")


def print_coefficient_ablation_summary(coef_results: List[dict], opts) -> None:
    """Print summary table for coefficient ablation study.

    Args:
        coef_results: List of result dictionaries from coefficient ablation
        opts: Parsed command-line options
    """
    print("\n" + "=" * 100)
    print("SUMMARY: Coefficient Ablation Study")
    print("=" * 100)
    print("\nExperiment settings:")
    print(f"  noise_distribution: {opts.noise_distribution}")
    print(f"  n_samples: {opts.n_samples}")
    print(f"  n_experiments: {opts.n_experiments}")

    print("\n" + "-" * 100)
    print(f"{'tc':>6} {'oc':>6} {'te':>6} {'ICA Var Coeff':>15} {'HOML RMSE':>12} {'ICA RMSE':>12} {'Winner':>10}")
    print("-" * 100)

    for res in coef_results:
        homl_rmse = res["rmse"][HOML_IDX]
        ica_rmse = res["rmse"][ICA_IDX] if ICA_IDX < len(res["rmse"]) else np.nan
        winner = "ICA" if ica_rmse < homl_rmse else "HOML"
        print(
            f"{res['treatment_coef_scalar']:>6.2f} {res['outcome_coef_scalar']:>6.2f} "
            f"{res['treatment_effect']:>6.2f} {res['ica_var_coeff']:>15.4f} "
            f"{homl_rmse:>12.4f} {ica_rmse:>12.4f} {winner:>10}"
        )

    print("=" * 100)


def print_noise_ablation_summary(results: dict, opts) -> None:
    """Print summary table for noise ablation study and save markdown.

    Args:
        results: Dictionary with results for each noise distribution
        opts: Parsed command-line options
    """
    print("\n" + "=" * 180)
    print("SUMMARY: Noise Distribution Ablation Study (HOML and ICA)")
    print("=" * 180)
    print("\nExperiment settings:")
    print(f"  n_samples: {opts.n_samples}")
    print(f"  n_experiments: {opts.n_experiments}")
    if opts.randomize_coeffs:
        print(f"  randomize_coeffs: True (n_random_configs={opts.n_random_configs})")
        print(f"  treatment_effect_range: {opts.treatment_effect_range}")
        print(f"  treatment_coef_range: {opts.treatment_coef_range}")
        print(f"  outcome_coef_range: {opts.outcome_coef_range}")
    else:
        print(f"  treatment_effect: {opts.treatment_effect}")
    print(f"  covariate_pdf: {opts.covariate_pdf}")

    print("\n" + "-" * 180)
    print(
        f"{'Distribution':<16} {'Emp.Kurt':>10} {'HOML AVar':>10} {'ICA AVar':>10} {'Ratio':>8} "
        f"{'HOML Bias':>10} {'HOML Std':>10} {'HOML RMSE':>10} "
        f"{'ICA Bias':>10} {'ICA Std':>10} {'ICA RMSE':>10} {'Winner':>8}"
    )
    print("-" * 180)

    # Build markdown content for saving
    md_lines = []
    md_lines.append("# Noise Distribution Ablation Study Results\n")
    md_lines.append("## Experiment Settings\n")
    md_lines.append(f"- **n_samples**: {opts.n_samples}")
    md_lines.append(f"- **n_experiments**: {opts.n_experiments}")
    if opts.randomize_coeffs:
        md_lines.append(f"- **randomize_coeffs**: True (n_random_configs={opts.n_random_configs})")
        md_lines.append(f"- **treatment_effect_range**: {opts.treatment_effect_range}")
        md_lines.append(f"- **treatment_coef_range**: {opts.treatment_coef_range}")
        md_lines.append(f"- **outcome_coef_range**: {opts.outcome_coef_range}")
    else:
        md_lines.append(f"- **treatment_effect**: {opts.treatment_effect}")
    md_lines.append(f"- **covariate_pdf**: {opts.covariate_pdf}")
    md_lines.append("")
    md_lines.append("## Results Summary\n")
    md_lines.append(
        "| Distribution | Emp. Kurt. | HOML AVar | ICA AVar | Ratio | "
        "HOML Bias | HOML Std | HOML RMSE | ICA Bias | ICA Std | ICA RMSE | Winner |"
    )
    md_lines.append(
        "|:-------------|----------:|----------:|----------:|------:|"
        "----------:|---------:|----------:|---------:|--------:|---------:|:------:|"
    )

    for dist, res in results.items():
        emp_kurtosis = res.get("eta_empirical_excess_kurtosis", np.nan)
        homl_avar = res.get("homl_asymptotic_var", np.nan)
        ica_avar = res.get("ica_asymptotic_var", np.nan)
        avar_ratio = ica_avar / homl_avar if homl_avar != 0 and not np.isnan(homl_avar) else np.nan
        homl_bias = res["biases"][HOML_IDX] if len(res["biases"]) > HOML_IDX else np.nan
        homl_std = res["sigmas"][HOML_IDX] if len(res["sigmas"]) > HOML_IDX else np.nan
        homl_rmse = res["rmse"][HOML_IDX] if len(res["rmse"]) > HOML_IDX else np.nan
        ica_bias = res["biases"][ICA_IDX] if len(res["biases"]) > ICA_IDX else np.nan
        ica_std = res["sigmas"][ICA_IDX] if len(res["sigmas"]) > ICA_IDX else np.nan
        ica_rmse = res["rmse"][ICA_IDX] if len(res["rmse"]) > ICA_IDX else np.nan
        winner = "ICA" if ica_rmse < homl_rmse else "HOML"

        emp_kurt_str = f"{emp_kurtosis:.4f}" if not np.isnan(emp_kurtosis) else "N/A"
        homl_avar_str = f"{homl_avar:.4f}" if not np.isnan(homl_avar) else "N/A"
        ica_avar_str = f"{ica_avar:.4f}" if not np.isnan(ica_avar) else "N/A"
        ratio_str = f"{avar_ratio:.4f}" if not np.isnan(avar_ratio) else "N/A"

        # Console output
        print(
            f"{dist:<16} {emp_kurt_str:>10} {homl_avar_str:>10} {ica_avar_str:>10} {ratio_str:>8} "
            f"{homl_bias:>10.4f} {homl_std:>10.4f} {homl_rmse:>10.4f} "
            f"{ica_bias:>10.4f} {ica_std:>10.4f} {ica_rmse:>10.4f} {winner:>8}"
        )

        # Markdown row
        md_lines.append(
            f"| {dist} | {emp_kurt_str} | {homl_avar_str} | {ica_avar_str} | {ratio_str} | "
            f"{homl_bias:.4f} | {homl_std:.4f} | {homl_rmse:.4f} | "
            f"{ica_bias:.4f} | {ica_std:.4f} | {ica_rmse:.4f} | **{winner}** |"
        )

    print("=" * 180)

    # Save markdown summary to file
    md_file_path = os.path.join(opts.output_dir, "noise_ablation_summary.md")
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown summary saved to: {md_file_path}")
