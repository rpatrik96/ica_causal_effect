"""
Noise distribution and coefficient ablation study for treatment effect estimation.

This script runs experiments comparing:
1. Different noise distributions for eta (treatment noise)
2. Different coefficient configurations (treatment_coef, outcome_coef, treatment_effect)

Noise distributions:
- Discrete (default asymmetric)
- Heavy-tailed: Laplace, gennorm (beta<2)
- Bounded: Uniform, Rademacher
- Light-tailed: gennorm (beta>2)

Supports flexible gennorm distributions with configurable beta parameter:
- gennorm:0.5  - Very heavy tails
- gennorm:1.0  - Equivalent to Laplace (gennorm_heavy)
- gennorm:2.0  - Equivalent to Gaussian
- gennorm:4.0  - Lighter tails (gennorm_light)
- gennorm:8.0  - Approaching uniform

Coefficient ablation varies the ICA variance coefficient:
  ica_var_coeff = 1 + ||outcome_coef + treatment_coef * treatment_effect||^2

This allows studying how the relationship between coefficients affects estimation error.
"""

import os
import sys
from itertools import product
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kurtosis as scipy_kurtosis
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
    calculate_homl_moments,
    calculate_ica_moments,
    compute_estimation_statistics,
    compute_estimation_statistics_varying_te,
    create_covariate_sampler,
    create_outcome_noise_sampler,
    extract_treatment_estimates,
    get_distribution_label,
    run_parallel_experiments,
)
from oml_runner import setup_treatment_noise
from plot_utils import add_legend_outside, plot_typography

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
        fixed_value = fixed_beta
        covariate_beta = fixed_beta
    elif axis_mode == "beta_vs_n":
        if beta_values is None:
            beta_values = [0.5, 1.0, 2.0, 3.0, 4.0]
        grid_values = beta_values
        grid_key = "beta"
        fixed_value = fixed_dimension
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


# =============================================================================
# Bar Plot Functions
# =============================================================================


def plot_metric_comparison_bars(
    results: dict,
    metric: str,
    output_dir: str,
    filename: str,
    ylabel: str,
    use_abs: bool = False,
):
    """Plot bar chart comparing HOML and ICA for a metric across distributions.

    Args:
        results: Results dictionary
        metric: Key in results ('rmse', 'biases', 'sigmas')
        output_dir: Output directory
        filename: Output filename
        ylabel: Y-axis label
        use_abs: Whether to use absolute values
    """
    _setup_plot()

    noise_dists = list(results.keys())
    x = np.arange(len(noise_dists))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 6))

    for i, (method_name, method_idx, color) in enumerate([("HOML", HOML_IDX, OML_COLOR), ("ICA", ICA_IDX, ICA_COLOR)]):
        values = []
        for dist in noise_dists:
            val = results[dist][metric][method_idx] if method_idx < len(results[dist][metric]) else np.nan
            if use_abs:
                val = np.abs(val)
            values.append(val)
        ax.bar(x + i * width, values, width, label=method_name, color=color)

    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([get_distribution_label(d) for d in noise_dists], rotation=45, ha="right")
    add_legend_outside(ax)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_difference_bars(
    results: dict,
    metric: str,
    output_dir: str,
    filename: str,
    ylabel: str,
    title: str,
    use_abs: bool = False,
):
    """Plot bar chart of ICA - HOML difference for a metric.

    Args:
        results: Results dictionary
        metric: Key in results ('rmse', 'biases', 'sigmas')
        output_dir: Output directory
        filename: Output filename
        ylabel: Y-axis label
        title: Plot title
        use_abs: Whether to use absolute values before differencing
    """
    _setup_plot()

    noise_dists = list(results.keys())
    x = np.arange(len(noise_dists))

    diffs = []
    for dist in noise_dists:
        homl_val = results[dist][metric][HOML_IDX] if HOML_IDX < len(results[dist][metric]) else np.nan
        ica_val = results[dist][metric][ICA_IDX] if ICA_IDX < len(results[dist][metric]) else np.nan
        if use_abs:
            homl_val, ica_val = np.abs(homl_val), np.abs(ica_val)
        diffs.append(ica_val - homl_val)

    colors = [_get_comparison_color(d) for d in diffs]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, diffs, width=0.6, color=colors)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([get_distribution_label(d) for d in noise_dists], rotation=45, ha="right")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Scatter Plot Functions
# =============================================================================


def plot_metric_vs_kurtosis(
    results: dict,
    metric: str,
    output_dir: str,
    filename: str,
    ylabel: str,
    title: str,
    use_abs: bool = False,
):
    """Plot scatter of metric vs excess kurtosis.

    Args:
        results: Results dictionary
        metric: Key in results ('rmse', 'biases', 'sigmas')
        output_dir: Output directory
        filename: Output filename
        ylabel: Y-axis label
        title: Plot title
        use_abs: Whether to use absolute values
    """
    _setup_plot()

    noise_dists = list(results.keys())

    # Get kurtosis values
    kurtosis_values = []
    for dist in noise_dists:
        emp_kurt = results[dist].get("eta_empirical_excess_kurtosis", np.nan)
        kurtosis_values.append(emp_kurt if not np.isnan(emp_kurt) else results[dist]["eta_excess_kurtosis"])

    homl_values = []
    ica_values = []
    for dist in noise_dists:
        homl_val = results[dist][metric][HOML_IDX] if HOML_IDX < len(results[dist][metric]) else np.nan
        ica_val = results[dist][metric][ICA_IDX] if ICA_IDX < len(results[dist][metric]) else np.nan
        if use_abs:
            homl_val, ica_val = np.abs(homl_val), np.abs(ica_val)
        homl_values.append(homl_val)
        ica_values.append(ica_val)

    _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(kurtosis_values, homl_values, c=OML_COLOR, s=80, alpha=0.8, label="HOML", marker="o")
    ax.scatter(kurtosis_values, ica_values, c=ICA_COLOR, s=80, alpha=0.8, label="ICA", marker="s")

    # Add labels
    for i, dist in enumerate(noise_dists):
        ax.annotate(
            get_distribution_label(dist)[:8],
            (kurtosis_values[i], homl_values[i]),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(r"Excess Kurtosis $\kappa$")
    ax.set_ylabel(ylabel)
    add_legend_outside(ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_diff_vs_kurtosis(
    results: dict,
    metric: str,
    output_dir: str,
    filename: str,
    ylabel: str,
    title: str,
    use_abs: bool = False,
):
    """Plot scatter of ICA - HOML difference vs excess kurtosis.

    Args:
        results: Results dictionary
        metric: Key in results ('rmse', 'biases', 'sigmas')
        output_dir: Output directory
        filename: Output filename
        ylabel: Y-axis label
        title: Plot title
        use_abs: Whether to use absolute values before differencing
    """
    _setup_plot()

    noise_dists = list(results.keys())

    # Get kurtosis and diff values
    kurtosis_values = []
    diffs = []
    for dist in noise_dists:
        emp_kurt = results[dist].get("eta_empirical_excess_kurtosis", np.nan)
        kurtosis_values.append(emp_kurt if not np.isnan(emp_kurt) else results[dist]["eta_excess_kurtosis"])

        homl_val = results[dist][metric][HOML_IDX] if HOML_IDX < len(results[dist][metric]) else np.nan
        ica_val = results[dist][metric][ICA_IDX] if ICA_IDX < len(results[dist][metric]) else np.nan
        if use_abs:
            homl_val, ica_val = np.abs(homl_val), np.abs(ica_val)
        diffs.append(ica_val - homl_val)

    _, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_markers(ax, kurtosis_values, diffs, diffs, alpha=0.8, s=80)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    for i, dist in enumerate(noise_dists):
        ax.annotate(
            get_distribution_label(dist)[:8],
            (kurtosis_values[i], diffs[i]),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(r"Excess Kurtosis $\kappa$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add legend
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Asymptotic Variance Plots
# =============================================================================


def plot_asymptotic_variance_comparison(results: dict, output_dir: str):
    """Plot asymptotic variance comparison plots.

    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    _setup_plot()

    noise_dists = list(results.keys())
    x = np.arange(len(noise_dists))

    homl_avar = [results[dist].get("homl_asymptotic_var", np.nan) for dist in noise_dists]
    ica_avar = [results[dist].get("ica_asymptotic_var", np.nan) for dist in noise_dists]
    avar_ratio = [
        ica / homl if homl != 0 and not np.isnan(homl) and not np.isnan(ica) else np.nan
        for ica, homl in zip(ica_avar, homl_avar)
    ]

    # Get kurtosis values
    kurtosis_values = []
    for dist in noise_dists:
        emp_kurt = results[dist].get("eta_empirical_excess_kurtosis", np.nan)
        kurtosis_values.append(emp_kurt if not np.isnan(emp_kurt) else results[dist]["eta_excess_kurtosis"])

    # Plot 1: Ratio bar chart
    _, ax = plt.subplots(figsize=(10, 6))
    colors_ratio = [ICA_BETTER_COLOR if r < 1 else OML_BETTER_COLOR for r in avar_ratio]
    ax.bar(x, avar_ratio, width=0.6, color=colors_ratio)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Equal variance")
    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel(r"Asymptotic Variance Ratio (ICA / OML)")
    ax.set_xticks(x)
    ax.set_xticklabels([get_distribution_label(d) for d in noise_dists], rotation=45, ha="right")
    ax.set_title("Asymptotic Variance Ratio: ICA / OML\n(Blue = ICA lower, Red = OML lower)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "asymptotic_var_ratio.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Side by side comparison
    _, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    ax.bar(x - width / 2, homl_avar, width, label="HOML", color=OML_COLOR)
    ax.bar(x + width / 2, ica_avar, width, label="ICA", color=ICA_COLOR)
    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel("Asymptotic Variance")
    ax.set_xticks(x)
    ax.set_xticklabels([get_distribution_label(d) for d in noise_dists], rotation=45, ha="right")
    add_legend_outside(ax)
    ax.set_title("Asymptotic Variance: HOML vs ICA")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "asymptotic_var_comparison.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Ratio vs kurtosis scatter
    # For ratio, ICA is better when ratio < 1 (use ratio - 1 as diff: negative = ICA better)
    _, ax = plt.subplots(figsize=(8, 6))
    ratio_diffs = np.array([r - 1 if not np.isnan(r) else np.nan for r in avar_ratio])
    _scatter_with_markers(ax, kurtosis_values, avar_ratio, ratio_diffs, alpha=0.8, s=80)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1)
    for i, dist in enumerate(noise_dists):
        if not np.isnan(avar_ratio[i]):
            ax.annotate(
                get_distribution_label(dist)[:8],
                (kurtosis_values[i], avar_ratio[i]),
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )
    ax.set_xlabel(r"Excess Kurtosis $\kappa$")
    ax.set_ylabel(r"Asymptotic Variance Ratio (ICA / OML)")
    ax.set_title("Asymptotic Variance Ratio vs Kurtosis\n(Blue = ICA lower, Red = OML lower)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add legend
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "asymptotic_var_ratio_vs_kurtosis.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Variances vs kurtosis scatter
    _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(kurtosis_values, homl_avar, c=OML_COLOR, s=80, alpha=0.8, label="HOML", marker="o")
    ax.scatter(kurtosis_values, ica_avar, c=ICA_COLOR, s=80, alpha=0.8, label="ICA", marker="s")
    for i, dist in enumerate(noise_dists):
        if not np.isnan(homl_avar[i]):
            ax.annotate(
                get_distribution_label(dist)[:8],
                (kurtosis_values[i], homl_avar[i]),
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )
    ax.set_xlabel(r"Excess Kurtosis $\kappa$")
    ax.set_ylabel("Asymptotic Variance")
    add_legend_outside(ax)
    ax.set_title("Asymptotic Variance vs Excess Kurtosis")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "asymptotic_var_vs_kurtosis.svg"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Summary Table Plot
# =============================================================================


def plot_summary_table(results: dict, output_dir: str):
    """Plot summary table of distribution properties.

    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    _setup_plot()

    noise_dists = list(results.keys())

    _, ax = plt.subplots(figsize=(20, 4))
    ax.axis("off")

    table_data = []
    headers = [
        "Distribution",
        "Emp. Kurt.",
        "HOML AVar",
        "ICA AVar",
        "Ratio",
        "HOML Bias",
        "HOML Std",
        "HOML RMSE",
        "ICA Bias",
        "ICA Std",
        "ICA RMSE",
    ]

    for dist in noise_dists:
        homl_avar_val = results[dist].get("homl_asymptotic_var", np.nan)
        ica_avar_val = results[dist].get("ica_asymptotic_var", np.nan)
        avar_ratio_val = ica_avar_val / homl_avar_val if homl_avar_val != 0 and not np.isnan(homl_avar_val) else np.nan
        homl_bias = results[dist]["biases"][HOML_IDX] if HOML_IDX < len(results[dist]["biases"]) else np.nan
        homl_std = results[dist]["sigmas"][HOML_IDX] if HOML_IDX < len(results[dist]["sigmas"]) else np.nan
        homl_rmse = results[dist]["rmse"][HOML_IDX] if HOML_IDX < len(results[dist]["rmse"]) else np.nan
        ica_bias = results[dist]["biases"][ICA_IDX] if ICA_IDX < len(results[dist]["biases"]) else np.nan
        ica_std = results[dist]["sigmas"][ICA_IDX] if ICA_IDX < len(results[dist]["sigmas"]) else np.nan
        ica_rmse = results[dist]["rmse"][ICA_IDX] if ICA_IDX < len(results[dist]["rmse"]) else np.nan
        empirical_kurt = results[dist].get("eta_empirical_excess_kurtosis", np.nan)

        row = [
            get_distribution_label(dist),
            f"{empirical_kurt:.3f}" if not np.isnan(empirical_kurt) else "N/A",
            f"{homl_avar_val:.3f}" if not np.isnan(homl_avar_val) else "N/A",
            f"{ica_avar_val:.3f}" if not np.isnan(ica_avar_val) else "N/A",
            f"{avar_ratio_val:.3f}" if not np.isnan(avar_ratio_val) else "N/A",
            f"{homl_bias:.4f}" if not np.isnan(homl_bias) else "N/A",
            f"{homl_std:.4f}" if not np.isnan(homl_std) else "N/A",
            f"{homl_rmse:.4f}" if not np.isnan(homl_rmse) else "N/A",
            f"{ica_bias:.4f}" if not np.isnan(ica_bias) else "N/A",
            f"{ica_std:.4f}" if not np.isnan(ica_std) else "N/A",
            f"{ica_rmse:.4f}" if not np.isnan(ica_rmse) else "N/A",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_properties_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main Plotting Function
# =============================================================================


def plot_noise_ablation_results(results: dict, output_dir: str = "figures/noise_ablation"):
    """Plot results from noise ablation study.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # RMSE comparison bar
    plot_metric_comparison_bars(results, "rmse", output_dir, "rmse_comparison_homl_ica.svg", "RMSE")

    # Bias comparison bar
    plot_metric_comparison_bars(
        results, "biases", output_dir, "bias_comparison_homl_ica.svg", r"$|\mathrm{Bias}|$", use_abs=True
    )

    # Std comparison bar
    plot_metric_comparison_bars(results, "sigmas", output_dir, "std_comparison_homl_ica.svg", "Standard Deviation")

    # RMSE difference bar
    plot_metric_difference_bars(
        results,
        "rmse",
        output_dir,
        "rmse_diff_homl_ica.svg",
        "RMSE Difference (ICA - HOML)",
        "RMSE Difference: ICA - HOML\n(Blue = ICA better, Red = HOML better)",
    )

    # Bias difference bar
    plot_metric_difference_bars(
        results,
        "biases",
        output_dir,
        "bias_diff_homl_ica.svg",
        r"$|\mathrm{Bias}|$ Difference (ICA - HOML)",
        "Absolute Bias Difference: ICA - HOML\n(Blue = ICA better, Red = HOML better)",
        use_abs=True,
    )

    # RMSE vs kurtosis scatter
    plot_metric_vs_kurtosis(results, "rmse", output_dir, "rmse_vs_kurtosis.svg", "RMSE", "RMSE vs Excess Kurtosis")

    # Bias vs kurtosis scatter
    plot_metric_vs_kurtosis(
        results,
        "biases",
        output_dir,
        "bias_vs_kurtosis.svg",
        r"$|\mathrm{Bias}|$",
        "Absolute Bias vs Excess Kurtosis",
        use_abs=True,
    )

    # Std vs kurtosis scatter
    plot_metric_vs_kurtosis(
        results,
        "sigmas",
        output_dir,
        "std_vs_kurtosis.svg",
        "Standard Deviation",
        "Standard Deviation vs Excess Kurtosis",
    )

    # RMSE diff vs kurtosis
    plot_diff_vs_kurtosis(
        results,
        "rmse",
        output_dir,
        "diff_vs_kurtosis_rmse.svg",
        "RMSE Diff (ICA - HOML)",
        "RMSE Difference vs Kurtosis",
    )

    # Bias diff vs kurtosis
    plot_diff_vs_kurtosis(
        results,
        "biases",
        output_dir,
        "diff_vs_kurtosis_bias.svg",
        r"$|\mathrm{Bias}|$ Diff (ICA - HOML)",
        "Bias Difference vs Kurtosis",
        use_abs=True,
    )

    # Asymptotic variance plots
    plot_asymptotic_variance_comparison(results, output_dir)

    # Summary table
    plot_summary_table(results, output_dir)

    # Distribution diff heatmap (kurtosis-based)
    plot_distribution_diff_heatmap(results, output_dir)

    print(f"\nPlots saved to {output_dir}")


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
    ax.set_xticklabels([f"{lbl}\n(κ={k:.2f})" for lbl, k in zip(labels, kurtosis_vals)], rotation=45, ha="right")
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

    plt.tight_layout()
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
        print(
            f"| {d['label']:20s} | {d['kurtosis']:8.4f} | {rmse_diff:9.4f} | {bias_diff:10.4f} | {std_diff:8.4f} | {winner:6s} |"
        )

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
                f"| {d['label']:20s} | {d['kurtosis']:.4f} | {rmse_diff:.4f} | {bias_diff:.4f} | {std_diff:.4f} | {winner} |\n"
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
    ax.set_ylabel("RMSE Diff (ICA - HOML)")
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
        x_labels = [f"{dist_data[d]['label']}\n(κ={dist_data[d]['kurtosis']:.2f})" for d in sorted_dists]
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

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Create and plot heatmaps
    rmse_heatmap = create_heatmap_data("rmse_diffs")
    bias_heatmap = create_heatmap_data("bias_diffs")
    std_heatmap = create_heatmap_data("std_diffs")

    plot_single_heatmap(
        rmse_heatmap,
        "RMSE Difference (ICA - HOML)\n(Blue = ICA better, Red = HOML better)",
        f"heatmap_rmse_diff{suffix}.svg",
        "RMSE Diff",
    )

    plot_single_heatmap(
        bias_heatmap,
        r"$|\mathrm{Bias}|$ Difference (ICA - HOML)" + "\n(Blue = ICA better, Red = HOML better)",
        f"heatmap_bias_diff{suffix}.svg",
        r"$|\mathrm{Bias}|$ Diff",
    )

    plot_single_heatmap(
        std_heatmap,
        "Std Difference (ICA - HOML)\n(Blue = ICA better, Red = HOML better)",
        f"heatmap_std_diff{suffix}.svg",
        "Std Diff",
    )

    # Also create a combined 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

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
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(np.arange(n_dists))
        x_labels = [f"{dist_data[d]['label']}\n(κ={dist_data[d]['kurtosis']:.1f})" for d in sorted_dists]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(0, n_outcome_bins, 2))
        ax.set_yticklabels([f"{outcome_centers[i]:.2f}" for i in range(0, n_outcome_bins, 2)])
        ax.set_xlabel("Distribution")
        ax.set_ylabel(r"Outcome Coef $b$")
        ax.set_title(title)

        # Add colorbar next to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    fig.suptitle("ICA - HOML Differences vs Distribution and Outcome Coef\n(Blue = ICA better, Red = HOML better)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"heatmap_combined{suffix}.svg"), dpi=300, bbox_inches="tight")
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
    ax.scatter(ica_var_coeffs, homl_rmse, c=OML_COLOR, alpha=0.7, label="HOML", s=60, marker="o")
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

        plt.tight_layout()
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
        r"$|\mathrm{Bias}|$ Difference (ICA - HOML)" + "\n(Blue = ICA better, Red = HOML better)",
        "bias_diff_heatmap.svg",
        r"$|\mathrm{Bias}|$ Diff",
        diverging=True,
    )
    plot_single_heatmap(
        std_diff_grid,
        "Std Difference (ICA - HOML)\n(Blue = ICA better, Red = HOML better)",
        "std_diff_heatmap.svg",
        "Std Diff",
        diverging=True,
    )
    plot_single_heatmap(
        rmse_diff_grid,
        "RMSE Difference (ICA - HOML)\n(Blue = ICA better, Red = HOML better)",
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
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "combined_metrics_heatmap.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create combined 1x3 figure for differences
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    diff_metrics = [
        (bias_diff_grid, r"$|\mathrm{Bias}|$ Diff"),
        (std_diff_grid, "Std Diff"),
        (rmse_diff_grid, "RMSE Diff"),
    ]

    for ax, (data, title) in zip(axes, diff_metrics):
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(n_betas))
        ax.set_xticklabels([f"{b:.1f}" for b in beta_values])
        ax.set_yticks(np.arange(n_vars))
        ax.set_yticklabels([f"{v:.2f}" for v in variance_values])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"Var")
        ax.set_title(title)

        # Add value annotations to each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    # Use white text on dark backgrounds, black on light
                    val_norm = abs(val) / (vmax + 1e-10)
                    color = "white" if val_norm > 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
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
    1. HOML RMSE heatmap
    2. ICA RMSE heatmap
    3. RMSE difference heatmap (ICA - HOML)

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
        f"\nFiltered {len(filtered_results)}/{len(results)} results with ica_var_coeff {'<=' if filter_below else '>'} {ica_var_threshold}"
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

    # Create heatmap data matrices for HOML, ICA, and difference
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
            cbar_label = r"RMSE diff (ICA $-$ HOML)"
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
    with open(summary_file, "w") as f:
        f.write(f"ICA Variance Coefficient Filtering Summary\n")
        f.write(f"==========================================\n")
        f.write(f"Threshold: {ica_var_threshold}\n")
        f.write(f"Filter mode: {'<=' if filter_below else '>'} threshold\n")
        f.write(f"Results kept: {len(filtered_results)} / {len(results)}\n\n")
        f.write(f"ICA var coeff range in filtered data:\n")
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

    # Create difference heatmap (ICA - HOML) for reference
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
    cbar.set_label(r"$|\mathrm{Bias}|$ diff (ICA $-$ HOML)")

    diff_filename = f"bias_diff_sample_size_vs_{filename_suffix}_filtered_{filter_desc}.svg"
    plt.savefig(os.path.join(output_dir, diff_filename), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved bias difference heatmap to {os.path.join(output_dir, diff_filename)}")

    # Save filtering summary
    summary_file = os.path.join(output_dir, f"bias_filtering_summary_{filter_desc}.txt")
    with open(summary_file, "w") as f:
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


# =============================================================================
# Summary Printing Functions
# =============================================================================


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


# =============================================================================
# Main Entry Point
# =============================================================================


def main(args=None):
    """Main function for noise distribution and coefficient ablation study."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Noise distribution and coefficient ablation study for treatment effect estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default noise distribution ablation
  python eta_noise_ablation_refactored.py

  # Run noise ablation with specific gennorm beta values
  python eta_noise_ablation_refactored.py --distributions gennorm:0.5 gennorm:1.0 gennorm:2.0

  # Run coefficient ablation
  python eta_noise_ablation_refactored.py --coefficient_ablation

  # Run variance ablation (beta vs variance heatmaps)
  python eta_noise_ablation_refactored.py --variance_ablation

  # Run variance ablation with custom grid
  python eta_noise_ablation_refactored.py --variance_ablation --variance_beta_values 0.5 1.0 1.5 2.5 3.0 --variance_values 0.5 1.0 2.0 4.0
        """,
    )

    # Common arguments
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples per experiment")
    parser.add_argument("--n_experiments", type=int, default=20, help="Number of Monte Carlo replications")
    parser.add_argument("--support_size", type=int, default=10, help="Support size for coefficients")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for gennorm covariates")
    parser.add_argument("--sigma_outcome", type=float, default=np.sqrt(3.0), help="Outcome noise std")
    parser.add_argument("--covariate_pdf", type=str, default="gennorm", help="Covariate distribution")
    parser.add_argument("--check_convergence", action="store_true", help="Check ICA convergence")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=12143, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="figures/noise_ablation", help="Output directory")

    # Noise ablation arguments
    parser.add_argument("--treatment_effect", type=float, default=1.0, help="True treatment effect")
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["discrete", "laplace", "uniform", "rademacher", "gennorm_heavy", "gennorm_light"],
        help="Noise distributions to test",
    )
    parser.add_argument("--gennorm_betas", nargs="+", type=float, default=None, help="Gennorm beta values to add")
    parser.add_argument("--randomize_coeffs", action="store_true", help="Randomize coefficients")
    parser.add_argument("--n_random_configs", type=int, default=20, help="Number of random configs")
    parser.add_argument(
        "--treatment_effect_range", nargs=2, type=float, default=[0.001, 0.2], help="Treatment effect range"
    )
    parser.add_argument(
        "--treatment_coef_range", nargs=2, type=float, default=[-10.0, 10.0], help="Treatment coef range"
    )
    parser.add_argument("--outcome_coef_range", nargs=2, type=float, default=[-0.5, 0.5], help="Outcome coef range")

    # Coefficient ablation arguments
    parser.add_argument("--coefficient_ablation", action="store_true", help="Run coefficient ablation")
    parser.add_argument(
        "--noise_distribution", type=str, default="discrete", help="Noise distribution for coef ablation"
    )

    # Variance ablation arguments
    parser.add_argument("--variance_ablation", action="store_true", help="Run variance ablation")
    parser.add_argument(
        "--variance_beta_values",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.5, 3.0, 4.0],
        help="Gennorm beta values for variance ablation",
    )
    parser.add_argument(
        "--variance_values",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Variance values (scale^2) for variance ablation",
    )

    # Filtered heatmap arguments
    parser.add_argument("--filtered_heatmap", action="store_true", help="Run filtered RMSE heatmap experiments")
    parser.add_argument(
        "--heatmap_axis_mode",
        type=str,
        default="d_vs_n",
        choices=["d_vs_n", "beta_vs_n"],
        help="Axis mode: 'd_vs_n' (dimension vs sample size) or 'beta_vs_n' (beta vs sample size)",
    )
    parser.add_argument(
        "--heatmap_sample_sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 5000, 10000],
        help="Sample sizes for heatmap",
    )
    parser.add_argument(
        "--heatmap_dimensions",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50],
        help="Covariate dimensions for d_vs_n mode",
    )
    parser.add_argument(
        "--heatmap_betas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0, 4.0],
        help="Beta values for beta_vs_n mode",
    )
    parser.add_argument(
        "--ica_var_threshold",
        type=float,
        default=1.5,
        help="ICA variance coefficient threshold for filtering",
    )
    parser.add_argument(
        "--fixed_beta",
        type=float,
        default=1.0,
        help="Fixed beta for d_vs_n mode",
    )
    parser.add_argument(
        "--fixed_dimension",
        type=int,
        default=10,
        help="Fixed covariate dimension for beta_vs_n mode",
    )
    parser.add_argument(
        "--heatmap_treatment_coef",
        type=float,
        default=0.5,
        help="Treatment coefficient scalar for filtered heatmap experiments (default 0.5 gives ica_var_coeff=1.25)",
    )
    parser.add_argument(
        "--heatmap_outcome_coef",
        type=float,
        default=0.0,
        help="Outcome coefficient scalar for filtered heatmap experiments",
    )
    parser.add_argument(
        "--constrain_ica_var",
        action="store_true",
        help="Automatically set treatment coefficient to achieve ica_var_coeff = ica_var_threshold. "
        "When enabled, --heatmap_treatment_coef is ignored and computed from --ica_var_threshold.",
    )

    if args is None:
        args = sys.argv[1:]
    opts = parser.parse_args(args)

    if opts.variance_ablation:
        # Run variance ablation
        var_output_dir = os.path.join(opts.output_dir, "variance_ablation")
        results_file = os.path.join(var_output_dir, "variance_ablation_results.npy")

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            var_results = np.load(results_file, allow_pickle=True).item()
        else:
            var_results = run_variance_ablation_experiments(
                beta_values=opts.variance_beta_values,
                variance_values=opts.variance_values,
                n_samples=opts.n_samples,
                n_experiments=opts.n_experiments,
                support_size=opts.support_size,
                treatment_effect=opts.treatment_effect,
                covariate_beta=opts.beta,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
            )

            os.makedirs(var_output_dir, exist_ok=True)
            np.save(results_file, var_results)
            print(f"Results saved to {results_file}")

        plot_variance_ablation_heatmaps(var_results, var_output_dir)

        # Print summary
        print_variance_ablation_summary(var_results, opts)

    elif opts.coefficient_ablation:
        # Run coefficient ablation
        coef_output_dir = os.path.join(opts.output_dir, "coefficient_ablation")
        results_file = os.path.join(coef_output_dir, "coefficient_ablation_results.npy")

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            coef_results = np.load(results_file, allow_pickle=True).tolist()
        else:
            coef_results = run_coefficient_ablation_experiments(
                noise_distribution=opts.noise_distribution,
                n_samples=opts.n_samples,
                n_experiments=opts.n_experiments,
                support_size=opts.support_size,
                beta=opts.beta,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
            )

            os.makedirs(coef_output_dir, exist_ok=True)
            np.save(results_file, coef_results)
            print(f"Results saved to {results_file}")

        plot_coefficient_ablation_results(coef_results, coef_output_dir)

        # Print summary
        print_coefficient_ablation_summary(coef_results, opts)

    elif opts.filtered_heatmap:
        # Run filtered heatmap experiments
        heatmap_output_dir = os.path.join(opts.output_dir, "filtered_heatmap")

        # Compute treatment coefficient if constrain_ica_var is enabled
        if opts.constrain_ica_var:
            # Ensure outcome_coef is non-zero (default to 0.2 if zero)
            outcome_coef_scalar = opts.heatmap_outcome_coef
            if outcome_coef_scalar == 0.0:
                # Set a default non-zero outcome coefficient
                # Use 30% of the target coefficient sum to ensure treatment_coef is also non-zero
                target_coef_sum = np.sqrt(opts.ica_var_threshold - 1)
                outcome_coef_scalar = 0.3 * target_coef_sum
                print(f"Setting outcome_coef_scalar to {outcome_coef_scalar:.6f} (30% of target sum)")

            treatment_coef_scalar = compute_constrained_treatment_coef(
                target_ica_var_coeff=opts.ica_var_threshold,
                treatment_effect=opts.treatment_effect,
                outcome_coef_scalar=outcome_coef_scalar,
            )

            # Validate both coefficients are non-zero
            if treatment_coef_scalar == 0.0:
                raise ValueError(
                    f"Computed treatment_coef_scalar is zero. "
                    f"Adjust outcome_coef_scalar ({outcome_coef_scalar}) or ica_var_threshold ({opts.ica_var_threshold})."
                )
            if outcome_coef_scalar == 0.0:
                raise ValueError("outcome_coef_scalar must be non-zero when --constrain_ica_var is enabled.")

            computed_ica_var = compute_ica_var_coeff(treatment_coef_scalar, outcome_coef_scalar, opts.treatment_effect)
            print(f"Constraining ICA variance coefficient to {opts.ica_var_threshold}")
            print(f"  Computed treatment_coef_scalar: {treatment_coef_scalar:.6f}")
            print(f"  Outcome_coef_scalar: {outcome_coef_scalar:.6f}")
            print(f"  Resulting ica_var_coeff: {computed_ica_var:.6f}")
        else:
            treatment_coef_scalar = opts.heatmap_treatment_coef
            outcome_coef_scalar = opts.heatmap_outcome_coef

        results_file = os.path.join(
            heatmap_output_dir,
            f"filtered_heatmap_results_{opts.heatmap_axis_mode}.npy",
        )

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            heatmap_results = np.load(results_file, allow_pickle=True).tolist()
        else:
            heatmap_results = run_sample_dimension_grid_experiments(
                sample_sizes=opts.heatmap_sample_sizes,
                dimension_values=opts.heatmap_dimensions if opts.heatmap_axis_mode == "d_vs_n" else None,
                beta_values=opts.heatmap_betas if opts.heatmap_axis_mode == "beta_vs_n" else None,
                axis_mode=opts.heatmap_axis_mode,
                fixed_beta=opts.fixed_beta,
                fixed_dimension=opts.fixed_dimension,
                noise_distribution=opts.noise_distribution,
                n_experiments=opts.n_experiments,
                treatment_effect=opts.treatment_effect,
                treatment_coef_scalar=treatment_coef_scalar,
                outcome_coef_scalar=outcome_coef_scalar,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
            )

            os.makedirs(heatmap_output_dir, exist_ok=True)
            np.save(results_file, heatmap_results)
            print(f"Results saved to {results_file}")

        # Validate that all results are below the threshold when constrain_ica_var is enabled
        if opts.constrain_ica_var:
            ica_var_coeffs = [r["ica_var_coeff"] for r in heatmap_results]
            max_ica_var = max(ica_var_coeffs)
            min_ica_var = min(ica_var_coeffs)
            results_above_threshold = [c for c in ica_var_coeffs if c > opts.ica_var_threshold]

            if results_above_threshold:
                print(
                    f"\nWARNING: {len(results_above_threshold)}/{len(ica_var_coeffs)} results have "
                    f"ica_var_coeff > {opts.ica_var_threshold}"
                )
                print(f"  Max ica_var_coeff: {max_ica_var:.6f}")
                print(f"  Min ica_var_coeff: {min_ica_var:.6f}")
                raise ValueError(
                    f"ICA variance constraint violated: {len(results_above_threshold)} results "
                    f"exceed threshold {opts.ica_var_threshold}. Max: {max_ica_var:.6f}"
                )
            else:
                print(
                    f"\nValidation passed: All {len(ica_var_coeffs)} results have "
                    f"ica_var_coeff <= {opts.ica_var_threshold}"
                )
                print(f"  Range: [{min_ica_var:.6f}, {max_ica_var:.6f}]")

        # Plot filtered RMSE heatmap
        plot_ica_var_filtered_rmse_heatmap(
            heatmap_results,
            heatmap_output_dir,
            axis_mode=opts.heatmap_axis_mode,
            ica_var_threshold=opts.ica_var_threshold,
            filter_below=True,
        )

        # Plot filtered bias heatmaps
        plot_ica_var_filtered_bias_heatmaps(
            heatmap_results,
            heatmap_output_dir,
            axis_mode=opts.heatmap_axis_mode,
            ica_var_threshold=opts.ica_var_threshold,
            filter_below=True,
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print("SUMMARY: Filtered Heatmap Experiments")
        print(f"{'=' * 80}")
        print(f"Axis mode: {opts.heatmap_axis_mode}")
        print(f"Sample sizes: {opts.heatmap_sample_sizes}")
        if opts.heatmap_axis_mode == "d_vs_n":
            print(f"Dimensions: {opts.heatmap_dimensions}")
            print(f"Fixed beta: {opts.fixed_beta}")
        else:
            print(f"Beta values: {opts.heatmap_betas}")
            print(f"Fixed dimension: {opts.fixed_dimension}")
        print(f"Covariate distribution: {opts.covariate_pdf}")
        print(f"Noise distribution: {opts.noise_distribution}")
        print(f"ICA var threshold: {opts.ica_var_threshold}")
        print(f"Constrain ICA var: {opts.constrain_ica_var}")
        print(f"Treatment coef: {treatment_coef_scalar:.6f}" + (" (computed)" if opts.constrain_ica_var else ""))
        outcome_coef_computed = opts.constrain_ica_var and opts.heatmap_outcome_coef == 0.0
        print(f"Outcome coef: {outcome_coef_scalar:.6f}" + (" (computed)" if outcome_coef_computed else ""))
        print(f"Treatment effect: {opts.treatment_effect}")
        actual_ica_var = compute_ica_var_coeff(treatment_coef_scalar, outcome_coef_scalar, opts.treatment_effect)
        print(f"Resulting ICA var coeff: {actual_ica_var:.6f}")
        print(f"Total configurations: {len(heatmap_results)}")

        # Count filtered results
        filtered_count = sum(1 for r in heatmap_results if r["ica_var_coeff"] <= opts.ica_var_threshold)
        print(f"Results with ica_var_coeff <= {opts.ica_var_threshold}: {filtered_count}/{len(heatmap_results)}")
        print(f"Output directory: {heatmap_output_dir}")

    else:
        # Run noise distribution ablation
        distributions = list(opts.distributions)
        if opts.gennorm_betas is not None:
            for beta_val in opts.gennorm_betas:
                gennorm_spec = f"gennorm:{beta_val}"
                if gennorm_spec not in distributions:
                    distributions.append(gennorm_spec)
            print(f"Final distribution list: {distributions}")

        # Setup results file path
        if opts.randomize_coeffs:
            tc_range = tuple(opts.treatment_coef_range)
            oc_range = tuple(opts.outcome_coef_range)
            te_range = tuple(opts.treatment_effect_range)
            results_file = os.path.join(
                opts.output_dir,
                f"noise_ablation_results_n{opts.n_random_configs}"
                f"_tc{tc_range[0]:.1f}to{tc_range[1]:.1f}"
                f"_oc{oc_range[0]:.1f}to{oc_range[1]:.1f}"
                f"_te{te_range[0]:.1f}to{te_range[1]:.1f}.npy",
            )
        else:
            results_file = os.path.join(opts.output_dir, "noise_ablation_results.npy")

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            results = np.load(results_file, allow_pickle=True).item()
        else:
            results = run_noise_ablation_experiments(
                noise_distributions=distributions,
                n_samples=opts.n_samples,
                n_experiments=opts.n_experiments,
                support_size=opts.support_size,
                treatment_effect=opts.treatment_effect,
                beta=opts.beta,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
                randomize_coeffs=opts.randomize_coeffs,
                n_random_configs=opts.n_random_configs,
                treatment_effect_range=tuple(opts.treatment_effect_range),
                treatment_coef_range=tuple(opts.treatment_coef_range),
                outcome_coef_range=tuple(opts.outcome_coef_range),
            )

            os.makedirs(opts.output_dir, exist_ok=True)
            np.save(results_file, results)
            print(f"Results saved to {results_file}")

        plot_noise_ablation_results(results, opts.output_dir)

        if opts.randomize_coeffs:
            plot_noise_ablation_coeff_scatter(
                results,
                opts.output_dir,
                n_configs=opts.n_random_configs,
                treatment_coef_range=tc_range,
                outcome_coef_range=oc_range,
                treatment_effect_range=te_range,
            )
            plot_noise_ablation_std_scatter(
                results,
                opts.output_dir,
                n_configs=opts.n_random_configs,
                treatment_coef_range=tc_range,
                outcome_coef_range=oc_range,
                treatment_effect_range=te_range,
            )
            plot_diff_heatmaps(
                results,
                opts.output_dir,
                n_configs=opts.n_random_configs,
                treatment_coef_range=tc_range,
                outcome_coef_range=oc_range,
                treatment_effect_range=te_range,
            )

        # Print summary
        print_noise_ablation_summary(results, opts)


if __name__ == "__main__":
    main()
