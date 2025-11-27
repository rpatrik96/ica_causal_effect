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
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from tueplots import bundles

from ica import ica_treatment_effect_estimation
from main_estimation import all_together_cross_fitting
from oml_runner import setup_covariate_pdf, setup_treatment_noise, setup_treatment_outcome_coefs
from oml_utils import (
    AsymptoticVarianceCalculator,
    OMLExperimentConfig,
    OMLParameterGrid,
    OMLResultsManager,
    compute_distribution_moments,
)
from plot_utils import plot_typography


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


def experiment_with_noise_dist(
    x,
    eta,
    epsilon,
    treatment_effect,
    treatment_support,
    treatment_coef,
    outcome_support,
    outcome_coef,
    eta_second_moment,
    eta_third_moment,
    lambda_reg,
    check_convergence=False,
    verbose=False,
):
    """Run a single OML experiment with specified noise samples.

    Args:
        x: Covariate matrix
        eta: Treatment noise samples
        epsilon: Outcome noise
        treatment_effect: True treatment effect
        treatment_support: Support indices for treatment
        treatment_coef: Treatment coefficients
        outcome_support: Support indices for outcome
        outcome_coef: Outcome coefficients
        eta_second_moment: Second moment of treatment noise
        eta_third_moment: Third moment of treatment noise
        lambda_reg: Regularization parameter
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output

    Returns:
        Tuple of estimation results from all methods
    """
    # Generate treatment as a function of covariates
    treatment = np.dot(x[:, treatment_support], treatment_coef) + eta

    # Generate outcome as a function of treatment and covariates
    outcome = treatment_effect * treatment + np.dot(x[:, outcome_support], outcome_coef) + epsilon

    model_treatment = Lasso(alpha=lambda_reg)
    model_outcome = Lasso(alpha=lambda_reg)

    assert (treatment_support == outcome_support).all()

    try:
        ica_treatment_effect_estimate, ica_mcc = ica_treatment_effect_estimation(
            np.hstack((x[:, treatment_support], treatment.reshape(-1, 1), outcome.reshape(-1, 1))),
            np.hstack((x[:, treatment_support], eta.reshape(-1, 1), epsilon.reshape(-1, 1))),
            check_convergence=check_convergence,
            verbose=verbose,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An error occurred during ICA treatment effect estimation: {e}")
        ica_mcc = None
        ica_treatment_effect_estimate = None

    if verbose:
        print(f"Estimated vs true treatment effect: {ica_treatment_effect_estimate}, {treatment_effect}")

    return (
        *all_together_cross_fitting(
            x,
            treatment,
            outcome,
            eta_second_moment,
            eta_third_moment,
            model_treatment=model_treatment,
            model_outcome=model_outcome,
        ),
        ica_treatment_effect_estimate,
        ica_mcc,
    )


def run_noise_ablation_experiments(
    noise_distributions: list,
    n_samples: int = 5000,
    n_experiments: int = 20,
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
    coef_range: Tuple[float, float] = (-2.0, 2.0),
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
        coef_range: Range for random treatment/outcome coefficients [min, max]

    Returns:
        Dictionary with results for each noise distribution.
        If randomize_coeffs=True, includes per-experiment coefficient info.
    """
    np.random.seed(seed)

    # Setup covariate sampling
    config = OMLExperimentConfig(covariate_pdf=covariate_pdf)

    if covariate_pdf == "gennorm":
        from scipy.stats import gennorm

        x_sample = lambda n, d: gennorm.rvs(beta, size=(n, d))
    elif covariate_pdf == "gauss":
        x_sample = lambda n, d: np.random.normal(size=(n, d))
    elif covariate_pdf == "uniform":
        x_sample = lambda n, d: np.random.uniform(-1, 1, size=(n, d))
    else:
        raise ValueError(f"Unknown covariate PDF: {covariate_pdf}")

    # Setup base coefficients (used if not randomizing)
    cov_dim_max = support_size
    treatment_support = outcome_support = np.array(range(support_size))

    # Outcome noise distribution
    epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

    # Compute regularization parameter
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    all_results = {}

    # Generate random coefficient configurations if needed
    if randomize_coeffs:
        random_configs = []
        for _ in range(n_random_configs):
            tc_val = np.random.uniform(coef_range[0], coef_range[1])
            oc_val = np.random.uniform(coef_range[0], coef_range[1])
            te_val = np.random.uniform(treatment_effect_range[0], treatment_effect_range[1])
            random_configs.append((tc_val, oc_val, te_val))
        print(f"Generated {n_random_configs} random coefficient configurations")
    else:
        # Single fixed configuration
        random_configs = [(1.0, 1.0, treatment_effect)]

    for noise_dist_spec in noise_distributions:
        print(f"\nRunning experiments for noise distribution: {noise_dist_spec}")

        # Parse distribution specification (handles "gennorm:1.5" format)
        noise_dist, gennorm_beta = parse_distribution_spec(noise_dist_spec)

        # Setup treatment noise for this distribution
        params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(
            distribution=noise_dist, beta=gennorm_beta
        )

        # Calculate base moments for asymptotic variance (distribution-specific, not coef-dependent)
        var_calculator = AsymptoticVarianceCalculator()

        if noise_dist == "discrete":
            (
                eta_cubed_variance,
                eta_fourth_moment,
                eta_non_gauss_cond,
                eta_second_moment,
                eta_third_moment,
                homl_asymptotic_var,
                homl_asymptotic_var_num,
            ) = var_calculator.calc_homl_asymptotic_var(params_or_discounts, mean_discount, probs)
        else:
            (
                eta_cubed_variance,
                eta_fourth_moment,
                eta_non_gauss_cond,
                eta_second_moment,
                eta_third_moment,
                homl_asymptotic_var,
                homl_asymptotic_var_num,
            ) = var_calculator.calc_homl_asymptotic_var_from_distribution(noise_dist, params_or_discounts, probs)

        # Store per-config results when randomizing
        config_results = []

        for config_idx, (tc_val, oc_val, te_val) in enumerate(random_configs):
            # Create coefficient arrays
            treatment_coef = np.zeros(support_size)
            treatment_coef[0] = tc_val
            outcome_coef = np.zeros(support_size)
            outcome_coef[0] = oc_val
            current_treatment_effect = te_val

            # Calculate ICA variance coefficient for this config
            ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * current_treatment_effect) ** 2

            if randomize_coeffs:
                print(
                    f"  Config {config_idx + 1}/{len(random_configs)}: tc={tc_val:.3f}, oc={oc_val:.3f}, te={te_val:.3f}, ica_var_coeff={ica_var_coeff:.3f}"
                )

            # Calculate ICA asymptotic variance for this config
            if noise_dist == "discrete":
                (
                    eta_excess_kurtosis,
                    eta_skewness_squared,
                    ica_asymptotic_var,
                    ica_asymptotic_var_hyvarinen,
                    ica_asymptotic_var_num,
                    _,
                ) = var_calculator.calc_ica_asymptotic_var(
                    treatment_coef,
                    outcome_coef,
                    current_treatment_effect,
                    params_or_discounts,
                    mean_discount,
                    probs,
                    eta_cubed_variance,
                )
            else:
                (
                    eta_excess_kurtosis,
                    eta_skewness_squared,
                    ica_asymptotic_var,
                    ica_asymptotic_var_hyvarinen,
                    ica_asymptotic_var_num,
                    _,
                ) = var_calculator.calc_ica_asymptotic_var_from_distribution(
                    treatment_coef, outcome_coef, current_treatment_effect, noise_dist, params_or_discounts, probs
                )

            # Run parallel experiments for this config
            results = [
                r
                for r in Parallel(n_jobs=-1, verbose=0)(
                    delayed(experiment_with_noise_dist)(
                        x_sample(n_samples, cov_dim_max),
                        eta_sample(n_samples),
                        epsilon_sample(n_samples),
                        current_treatment_effect,
                        treatment_support,
                        treatment_coef,
                        outcome_support,
                        outcome_coef,
                        eta_second_moment,
                        eta_third_moment,
                        lambda_reg,
                        check_convergence,
                        verbose,
                    )
                    for _ in range(n_experiments)
                )
                if (check_convergence is False or r[-1] is not None)
            ]

            if len(results) == 0:
                print(f"    No experiments converged")
                continue

            # Extract results
            ortho_rec_tau = [
                [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] + ica_estimate.tolist()
                for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_estimate, _ in results
            ]

            # Compute statistics
            ortho_rec_tau_array = np.array(ortho_rec_tau)
            biases = np.mean(ortho_rec_tau_array - current_treatment_effect, axis=0)
            sigmas = np.std(ortho_rec_tau_array, axis=0)
            rmse = np.sqrt(biases**2 + sigmas**2)

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
            f"Experiments kept: {sum(r['n_experiments'] for r in config_results)} total across {len(config_results)} configs"
        )

        # Aggregate results across configs
        if randomize_coeffs:
            # Store all config results and compute aggregate statistics
            all_ortho_rec_tau = []
            all_treatment_effects = []
            for cr in config_results:
                all_ortho_rec_tau.extend(cr["ortho_rec_tau"])
                all_treatment_effects.extend([cr["treatment_effect"]] * len(cr["ortho_rec_tau"]))

            ortho_rec_tau_array = np.array(all_ortho_rec_tau)
            all_treatment_effects = np.array(all_treatment_effects)
            # Compute bias relative to each experiment's true treatment effect
            biases = np.mean(ortho_rec_tau_array - all_treatment_effects[:, None], axis=0)
            sigmas = np.std(ortho_rec_tau_array, axis=0)
            rmse = np.sqrt(np.mean((ortho_rec_tau_array - all_treatment_effects[:, None]) ** 2, axis=0))

            # Average ICA var coeff across configs
            avg_ica_var_coeff = np.mean([cr["ica_var_coeff"] for cr in config_results])
        else:
            # Single config - use its results directly
            cr = config_results[0]
            biases = cr["biases"]
            sigmas = cr["sigmas"]
            rmse = cr["rmse"]
            avg_ica_var_coeff = cr["ica_var_coeff"]
            all_ortho_rec_tau = cr["ortho_rec_tau"]

        # Store results with full spec as key (preserves gennorm:beta format)
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
            "eta_skewness_squared": eta_skewness_squared,
            "homl_asymptotic_var": homl_asymptotic_var,
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

    Ablates over treatment_coef, outcome_coef, and treatment_effect to study
    how the ICA variance coefficient affects estimation error.

    Args:
        noise_distribution: Noise distribution for treatment (e.g., "discrete", "gennorm:1.5")
        treatment_coef_values: List of scalar values for treatment coefficient (default: [0.5, 1.0, 2.0])
        outcome_coef_values: List of scalar values for outcome coefficient (default: [0.5, 1.0, 2.0])
        treatment_effect_values: List of treatment effect values (default: [0.5, 1.0, 2.0])
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

    # Parse distribution specification
    noise_dist, gennorm_beta = parse_distribution_spec(noise_distribution)

    # Setup covariate sampling
    if covariate_pdf == "gennorm":
        from scipy.stats import gennorm

        x_sample = lambda n, d: gennorm.rvs(beta, size=(n, d))
    elif covariate_pdf == "gauss":
        x_sample = lambda n, d: np.random.normal(size=(n, d))
    elif covariate_pdf == "uniform":
        x_sample = lambda n, d: np.random.uniform(-1, 1, size=(n, d))
    else:
        raise ValueError(f"Unknown covariate PDF: {covariate_pdf}")

    # Setup treatment noise
    params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(
        distribution=noise_dist, beta=gennorm_beta
    )

    # Outcome noise distribution
    epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

    # Compute regularization parameter
    cov_dim_max = support_size
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    # Setup support indices
    treatment_support = outcome_support = np.array(range(support_size))

    # Calculate base moments for the noise distribution
    var_calculator = AsymptoticVarianceCalculator()

    if noise_dist == "discrete":
        (
            eta_cubed_variance,
            eta_fourth_moment,
            eta_non_gauss_cond,
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            homl_asymptotic_var_num,
        ) = var_calculator.calc_homl_asymptotic_var(params_or_discounts, mean_discount, probs)
    else:
        (
            eta_cubed_variance,
            eta_fourth_moment,
            eta_non_gauss_cond,
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            homl_asymptotic_var_num,
        ) = var_calculator.calc_homl_asymptotic_var_from_distribution(noise_dist, params_or_discounts, probs)

    all_results = []

    # Create coefficient grid
    coef_grid = list(product(treatment_coef_values, outcome_coef_values, treatment_effect_values))
    total_configs = len(coef_grid)

    print(f"\nRunning coefficient ablation with {total_configs} configurations")
    print(f"Noise distribution: {noise_distribution}")

    for config_idx, (tc_val, oc_val, te_val) in enumerate(coef_grid):
        # Create coefficient arrays (scalar coefficient at first position)
        treatment_coef = np.zeros(support_size)
        treatment_coef[0] = tc_val
        outcome_coef = np.zeros(support_size)
        outcome_coef[0] = oc_val
        treatment_effect = te_val

        # Calculate ICA variance coefficient
        ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect) ** 2

        print(
            f"\n[{config_idx + 1}/{total_configs}] tc={tc_val}, oc={oc_val}, te={te_val}, ica_var_coeff={ica_var_coeff:.4f}"
        )

        # Run parallel experiments
        results = [
            r
            for r in Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_with_noise_dist)(
                    x_sample(n_samples, cov_dim_max),
                    eta_sample(n_samples),
                    epsilon_sample(n_samples),
                    treatment_effect,
                    treatment_support,
                    treatment_coef,
                    outcome_support,
                    outcome_coef,
                    eta_second_moment,
                    eta_third_moment,
                    lambda_reg,
                    check_convergence,
                    verbose,
                )
                for _ in range(n_experiments)
            )
            if (check_convergence is False or r[-1] is not None)
        ]

        print(f"  Experiments kept: {len(results)} out of {n_experiments}")

        if len(results) == 0:
            print(f"  No experiments converged")
            continue

        # Extract results - indices: 0=OrthoML, 1=RobustOrthoML(HOML), 2=RobustOrthoEst, 3=RobustOrthoSplit, 4+=ICA
        ortho_rec_tau = [
            [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] + ica_estimate.tolist()
            for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_estimate, _ in results
        ]

        # Compute statistics
        ortho_rec_tau_array = np.array(ortho_rec_tau)
        biases = np.mean(ortho_rec_tau_array - treatment_effect, axis=0)
        sigmas = np.std(ortho_rec_tau_array, axis=0)
        rmse = np.sqrt(biases**2 + sigmas**2)

        # Store results
        result_dict = {
            "treatment_coef_scalar": tc_val,
            "outcome_coef_scalar": oc_val,
            "treatment_effect": treatment_effect,
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
        homl_idx = 1  # Robust Ortho ML
        ica_idx = 4  # ICA (first ICA estimate)
        print(f"  HOML: bias={biases[homl_idx]:.4f}, std={sigmas[homl_idx]:.4f}, rmse={rmse[homl_idx]:.4f}")
        if ica_idx < len(biases):
            print(f"  ICA:  bias={biases[ica_idx]:.4f}, std={sigmas[ica_idx]:.4f}, rmse={rmse[ica_idx]:.4f}")

    return all_results


def plot_noise_ablation_results(results: dict, output_dir: str = "figures/noise_ablation"):
    """Plot results from noise ablation study.

    Shows only HOML (Robust Ortho ML) and ICA methods.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    # Method indices: 0=OrthoML, 1=RobustOrthoML(HOML), 2=RobustOrthoEst, 3=RobustOrthoSplit, 4=ICA
    HOML_IDX = 1
    ICA_IDX = 4
    method_names = ["HOML", "ICA"]
    method_indices = [HOML_IDX, ICA_IDX]
    method_colors = ["#1f77b4", "#ff7f0e"]  # Blue for HOML, Orange for ICA

    noise_dists = list(results.keys())

    def get_dist_label(dist_key):
        """Generate a nice label for distribution key."""
        label_map = {
            "discrete": "Discrete",
            "laplace": "Laplace",
            "uniform": "Uniform",
            "rademacher": "Rademacher",
            "gennorm_heavy": r"GenNorm ($\beta$=1)",
            "gennorm_light": r"GenNorm ($\beta$=4)",
        }
        if dist_key in label_map:
            return label_map[dist_key]
        if dist_key.startswith("gennorm:"):
            beta_val = dist_key.split(":")[1]
            return rf"GenNorm ($\beta$={beta_val})"
        return dist_key

    label_map = {d: get_dist_label(d) for d in noise_dists}

    # Plot 1: RMSE comparison (HOML vs ICA only)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(noise_dists))
    width = 0.35

    for i, (method_name, method_idx) in enumerate(zip(method_names, method_indices)):
        rmse_values = [
            results[dist]["rmse"][method_idx] if method_idx < len(results[dist]["rmse"]) else np.nan
            for dist in noise_dists
        ]
        ax.bar(x + i * width, rmse_values, width, label=method_name, color=method_colors[i])

    ax.set_xlabel("Noise Distribution", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=9)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_comparison_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Bias comparison (HOML vs ICA only)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method_name, method_idx) in enumerate(zip(method_names, method_indices)):
        bias_values = [
            np.abs(results[dist]["biases"][method_idx]) if method_idx < len(results[dist]["biases"]) else np.nan
            for dist in noise_dists
        ]
        ax.bar(x + i * width, bias_values, width, label=method_name, color=method_colors[i])

    ax.set_xlabel("Noise Distribution", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$", fontsize=9)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_comparison_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Standard deviation comparison (HOML vs ICA only)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method_name, method_idx) in enumerate(zip(method_names, method_indices)):
        std_values = [
            results[dist]["sigmas"][method_idx] if method_idx < len(results[dist]["sigmas"]) else np.nan
            for dist in noise_dists
        ]
        ax.bar(x + i * width, std_values, width, label=method_name, color=method_colors[i])

    ax.set_xlabel("Noise Distribution", fontsize=9)
    ax.set_ylabel("Standard Deviation", fontsize=9)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "std_comparison_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: RMSE difference (ICA - HOML) bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    rmse_diff = [
        (results[dist]["rmse"][ICA_IDX] if ICA_IDX < len(results[dist]["rmse"]) else np.nan)
        - (results[dist]["rmse"][HOML_IDX] if HOML_IDX < len(results[dist]["rmse"]) else np.nan)
        for dist in noise_dists
    ]
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in rmse_diff]  # Green if ICA better, red if HOML better

    ax.bar(x, rmse_diff, width=0.6, color=colors)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Noise Distribution", fontsize=9)
    ax.set_ylabel("RMSE Difference (ICA - HOML)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("RMSE Difference: ICA - HOML\n(Green = ICA better, Red = HOML better)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_diff_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 5: Bias difference (|ICA bias| - |HOML bias|) bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bias_diff = [
        (np.abs(results[dist]["biases"][ICA_IDX]) if ICA_IDX < len(results[dist]["biases"]) else np.nan)
        - (np.abs(results[dist]["biases"][HOML_IDX]) if HOML_IDX < len(results[dist]["biases"]) else np.nan)
        for dist in noise_dists
    ]
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in bias_diff]

    ax.bar(x, bias_diff, width=0.6, color=colors)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Noise Distribution", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Difference (ICA - HOML)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Absolute Bias Difference: ICA - HOML\n(Green = ICA better, Red = HOML better)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_diff_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 6: RMSE/Bias diff vs ICA variance coefficient (scatter)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ica_var_coeffs = [results[dist].get("ica_var_coeff", np.nan) for dist in noise_dists]

    # RMSE diff vs ICA var coeff
    ax = axes[0]
    colors_scatter = ["#2ca02c" if d < 0 else "#d62728" for d in rmse_diff]
    ax.scatter(ica_var_coeffs, rmse_diff, c=colors_scatter, s=80, alpha=0.8)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff: $1 + \|b + a\theta\|_2^2$", fontsize=9)
    ax.set_ylabel("RMSE Diff (ICA - HOML)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("RMSE Diff vs ICA Var Coeff", fontsize=10)
    ax.grid(True, alpha=0.3)
    # Add distribution labels
    for i, dist in enumerate(noise_dists):
        if not np.isnan(ica_var_coeffs[i]):
            ax.annotate(
                label_map.get(dist, dist)[:8], (ica_var_coeffs[i], rmse_diff[i]), fontsize=6, ha="center", va="bottom"
            )

    # Bias diff vs ICA var coeff
    ax = axes[1]
    colors_scatter = ["#2ca02c" if d < 0 else "#d62728" for d in bias_diff]
    ax.scatter(ica_var_coeffs, bias_diff, c=colors_scatter, s=80, alpha=0.8)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff: $1 + \|b + a\theta\|_2^2$", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff (ICA - HOML)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Bias Diff vs ICA Var Coeff", fontsize=10)
    ax.grid(True, alpha=0.3)
    for i, dist in enumerate(noise_dists):
        if not np.isnan(ica_var_coeffs[i]):
            ax.annotate(
                label_map.get(dist, dist)[:8], (ica_var_coeffs[i], bias_diff[i]), fontsize=6, ha="center", va="bottom"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diff_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 7: Distribution properties table (simplified for HOML and ICA)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table_data = []
    headers = ["Distribution", "Kurtosis", "3rd Moment", "ICA Var Coeff", "HOML RMSE", "ICA RMSE"]

    for dist in noise_dists:
        ica_var_coeff = results[dist].get("ica_var_coeff", "N/A")
        homl_rmse = results[dist]["rmse"][HOML_IDX] if HOML_IDX < len(results[dist]["rmse"]) else np.nan
        ica_rmse = results[dist]["rmse"][ICA_IDX] if ICA_IDX < len(results[dist]["rmse"]) else np.nan
        row = [
            label_map.get(dist, dist),
            f"{results[dist]['eta_excess_kurtosis']:.3f}",
            f"{results[dist]['eta_third_moment']:.3f}",
            f"{ica_var_coeff:.3f}" if isinstance(ica_var_coeff, (int, float)) else ica_var_coeff,
            f"{homl_rmse:.4f}" if not np.isnan(homl_rmse) else "N/A",
            f"{ica_rmse:.4f}" if not np.isnan(ica_rmse) else "N/A",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_properties_homl_ica.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {output_dir}")


def plot_noise_ablation_coeff_scatter(
    results: dict,
    output_dir: str = "figures/noise_ablation",
    n_configs: int = 0,
    coef_range: Tuple[float, float] = (-2.0, 2.0),
    treatment_effect_range: Tuple[float, float] = (0.1, 5.0),
):
    """Plot RMSE/bias differences vs coefficient values when using randomized coefficients.

    Creates scatter plots showing how RMSE and bias differences vary with
    treatment effect, treatment coefficient, outcome coefficient, and ICA variance coefficient.

    Args:
        results: Dictionary with results for each noise distribution (must have config_results)
        output_dir: Directory to save figures
        n_configs: Number of random configs (for filename)
        coef_range: Range for coefficients (for filename)
        treatment_effect_range: Range for treatment effect (for filename)
    """
    # Create filename suffix with config info
    suffix = f"_n{n_configs}_coef{coef_range[0]:.1f}to{coef_range[1]:.1f}_te{treatment_effect_range[0]:.1f}to{treatment_effect_range[1]:.1f}"

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    # Method indices
    HOML_IDX = 1
    ICA_IDX = 4

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

    # Convert to arrays for plotting
    ica_var_coeffs = np.array([d["ica_var_coeff"] for d in all_data])
    rmse_diffs = np.array([d["rmse_diff"] for d in all_data])
    bias_diffs = np.array([d["bias_diff"] for d in all_data])
    treatment_effects = np.array([d["treatment_effect"] for d in all_data])
    treatment_coefs = np.array([d["treatment_coef"] for d in all_data])
    outcome_coefs = np.array([d["outcome_coef"] for d in all_data])
    distributions = [d["distribution"] for d in all_data]

    # Get unique distributions for coloring
    unique_dists = list(results.keys())
    cmap = plt.cm.tab10
    dist_colors = {d: cmap(i % 10) for i, d in enumerate(unique_dists)}

    # Plot 1: RMSE diff vs ICA variance coefficient (colored by distribution)
    fig, ax = plt.subplots(figsize=(8, 5))
    for dist in unique_dists:
        mask = [d == dist for d in distributions]
        if not any(mask):
            continue
        x_vals = ica_var_coeffs[mask]
        y_vals = rmse_diffs[mask]
        ax.scatter(x_vals, y_vals, c=[dist_colors[dist]], alpha=0.6, s=40, label=dist[:12])

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff: $1 + \|b + a\theta\|_2^2$", fontsize=9)
    ax.set_ylabel("RMSE Diff (ICA - HOML)", fontsize=9)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title("RMSE Diff vs ICA Var Coeff (by distribution)", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_rmse_vs_ica_var{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Bias diff vs ICA variance coefficient
    fig, ax = plt.subplots(figsize=(8, 5))
    for dist in unique_dists:
        mask = [d == dist for d in distributions]
        if not any(mask):
            continue
        x_vals = ica_var_coeffs[mask]
        y_vals = bias_diffs[mask]
        ax.scatter(x_vals, y_vals, c=[dist_colors[dist]], alpha=0.6, s=40, label=dist[:12])

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff: $1 + \|b + a\theta\|_2^2$", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff (ICA - HOML)", fontsize=9)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title("Bias Diff vs ICA Var Coeff (by distribution)", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_bias_vs_ica_var{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: 2x2 grid showing RMSE diff vs each coefficient
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in rmse_diffs]

    # RMSE diff vs treatment effect
    ax = axes[0, 0]
    ax.scatter(treatment_effects, rmse_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Effect $\theta$", fontsize=9)
    ax.set_ylabel("RMSE Diff", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"RMSE Diff vs $\theta$", fontsize=10)
    ax.grid(True, alpha=0.3)

    # RMSE diff vs treatment coef
    ax = axes[0, 1]
    ax.scatter(treatment_coefs, rmse_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Coef $a$", fontsize=9)
    ax.set_ylabel("RMSE Diff", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"RMSE Diff vs $a$", fontsize=10)
    ax.grid(True, alpha=0.3)

    # RMSE diff vs outcome coef
    ax = axes[1, 0]
    ax.scatter(outcome_coefs, rmse_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Outcome Coef $b$", fontsize=9)
    ax.set_ylabel("RMSE Diff", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"RMSE Diff vs $b$", fontsize=10)
    ax.grid(True, alpha=0.3)

    # RMSE diff vs ICA var coeff
    ax = axes[1, 1]
    ax.scatter(ica_var_coeffs, rmse_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff", fontsize=9)
    ax.set_ylabel("RMSE Diff", fontsize=9)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"RMSE Diff vs ICA Var Coeff", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "RMSE Difference (ICA - HOML) vs Coefficient Values\n(Green = ICA better, Red = HOML better)", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_rmse_grid{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: 2x2 grid showing Bias diff vs each coefficient
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in bias_diffs]

    # Bias diff vs treatment effect
    ax = axes[0, 0]
    ax.scatter(treatment_effects, bias_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Effect $\theta$", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"Bias Diff vs $\theta$", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bias diff vs treatment coef
    ax = axes[0, 1]
    ax.scatter(treatment_coefs, bias_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Treatment Coef $a$", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"Bias Diff vs $a$", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bias diff vs outcome coef
    ax = axes[1, 0]
    ax.scatter(outcome_coefs, bias_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Outcome Coef $b$", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"Bias Diff vs $b$", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bias diff vs ICA var coeff
    ax = axes[1, 1]
    ax.scatter(ica_var_coeffs, bias_diffs, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"ICA Var Coeff", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{Bias}|$ Diff", fontsize=9)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(r"Bias Diff vs ICA Var Coeff", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"$|\mathrm{Bias}|$ Difference (ICA - HOML) vs Coefficient Values"
        + "\n(Green = ICA better, Red = HOML better)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"coeff_scatter_bias_grid{suffix}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nCoefficient scatter plots saved to {output_dir}")


def plot_coefficient_ablation_results(results: List[dict], output_dir: str = "figures/coefficient_ablation"):
    """Plot results from coefficient ablation study.

    Creates scatter plots of error vs ICA variance coefficient for HOML and ICA.

    Args:
        results: List of result dictionaries from coefficient ablation
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    # Method indices: 1=RobustOrthoML(HOML), 4=ICA
    HOML_IDX = 1
    ICA_IDX = 4

    # Extract data
    ica_var_coeffs = [r["ica_var_coeff"] for r in results]
    homl_rmse = [r["rmse"][HOML_IDX] for r in results]
    ica_rmse = [r["rmse"][ICA_IDX] if ICA_IDX < len(r["rmse"]) else np.nan for r in results]
    homl_bias = [np.abs(r["biases"][HOML_IDX]) for r in results]
    ica_bias = [np.abs(r["biases"][ICA_IDX]) if ICA_IDX < len(r["biases"]) else np.nan for r in results]

    # Plot 1: RMSE vs ICA variance coefficient
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(ica_var_coeffs, homl_rmse, c="#1f77b4", alpha=0.7, label="HOML", s=60, marker="o")
    ax.scatter(ica_var_coeffs, ica_rmse, c="#ff7f0e", alpha=0.7, label="ICA", s=60, marker="s")

    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel("RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.set_title("RMSE vs ICA Variance Coefficient")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Bias vs ICA variance coefficient
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(ica_var_coeffs, homl_bias, c="#1f77b4", alpha=0.7, label="HOML", s=60, marker="o")
    ax.scatter(ica_var_coeffs, ica_bias, c="#ff7f0e", alpha=0.7, label="ICA", s=60, marker="s")

    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel(r"$|\mathrm{Bias}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.set_title("Absolute Bias vs ICA Variance Coefficient")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Error difference (ICA - HOML) vs ICA variance coefficient
    fig, ax = plt.subplots(figsize=(8, 6))

    rmse_diff = np.array(ica_rmse) - np.array(homl_rmse)
    colors = ["green" if d < 0 else "red" for d in rmse_diff]

    ax.scatter(ica_var_coeffs, rmse_diff, c=colors, alpha=0.7, s=60)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel("RMSE Difference (ICA - HOML)")
    ax.set_xscale("log")
    ax.set_title("RMSE Difference: ICA vs HOML\n(Green = ICA better, Red = HOML better)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_diff_vs_ica_var_coeff.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Combined plot with treatment effect as color
    fig, ax = plt.subplots(figsize=(10, 6))

    treatment_effects = [r["treatment_effect"] for r in results]
    unique_tes = sorted(set(treatment_effects))
    cmap = plt.cm.viridis
    te_colors = {te: cmap(i / (len(unique_tes) - 1)) for i, te in enumerate(unique_tes)}

    for te in unique_tes:
        mask = [r["treatment_effect"] == te for r in results]
        te_ica_var = [ica_var_coeffs[i] for i, m in enumerate(mask) if m]
        te_homl_rmse = [homl_rmse[i] for i, m in enumerate(mask) if m]
        te_ica_rmse = [ica_rmse[i] for i, m in enumerate(mask) if m]

        ax.scatter(
            te_ica_var, te_homl_rmse, c=[te_colors[te]], alpha=0.7, s=60, marker="o", label=rf"HOML ($\theta$={te})"
        )
        ax.scatter(te_ica_var, te_ica_rmse, c=[te_colors[te]], alpha=0.7, s=60, marker="s")

    ax.set_xlabel(r"ICA Variance Coefficient: $1 + \|b + a\theta\|_2^2$")
    ax.set_ylabel("RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_title(r"RMSE vs ICA Var Coeff (circle=HOML, square=ICA, color=$\theta$)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_vs_ica_var_coeff_by_treatment_effect.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nCoefficient ablation plots saved to {output_dir}")


def main(args=None):
    """Main function for noise distribution and coefficient ablation study."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Noise distribution and coefficient ablation study for treatment effect estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default noise distribution ablation
  python eta_noise_ablation.py

  # Run noise ablation with specific gennorm beta values
  python eta_noise_ablation.py --distributions gennorm:0.5 gennorm:1.0 gennorm:2.0 gennorm:4.0

  # Sweep over gennorm beta values
  python eta_noise_ablation.py --gennorm_betas 0.5 1.0 1.5 2.0 3.0 4.0

  # Run coefficient ablation (varies treatment_coef, outcome_coef, treatment_effect)
  python eta_noise_ablation.py --coefficient_ablation

  # Coefficient ablation with custom coefficient values
  python eta_noise_ablation.py --coefficient_ablation \\
      --treatment_coef_values 0.5 1.0 2.0 5.0 \\
      --outcome_coef_values 0.5 1.0 2.0 5.0 \\
      --treatment_effect_values 0.5 1.0 2.0 5.0

  # Coefficient ablation with specific noise distribution
  python eta_noise_ablation.py --coefficient_ablation --noise_distribution gennorm:1.5
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

    # Noise distribution ablation arguments
    parser.add_argument(
        "--treatment_effect", type=float, default=1.0, help="True treatment effect (for noise ablation)"
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["discrete", "laplace", "uniform", "rademacher", "gennorm_heavy", "gennorm_light"],
        help="Noise distributions to test. Use 'gennorm:beta' for custom beta values",
    )
    parser.add_argument(
        "--gennorm_betas",
        nargs="+",
        type=float,
        default=None,
        help="Convenience flag to add gennorm distributions with specified beta values",
    )
    parser.add_argument(
        "--randomize_coeffs",
        action="store_true",
        help="Randomize treatment_coef, outcome_coef, and treatment_effect for each experiment",
    )
    parser.add_argument(
        "--n_random_configs",
        type=int,
        default=20,
        help="Number of random coefficient configurations per distribution (when --randomize_coeffs)",
    )
    parser.add_argument(
        "--treatment_effect_range",
        nargs=2,
        type=float,
        default=[0.01, 5.0],
        help="Range for random treatment effect [min, max]",
    )
    parser.add_argument(
        "--coef_range",
        nargs=2,
        type=float,
        default=[-1.5, 1.5],
        help="Range for random treatment/outcome coefficients [min, max]",
    )

    # Coefficient ablation arguments
    parser.add_argument(
        "--coefficient_ablation",
        action="store_true",
        help="Run coefficient ablation instead of noise distribution ablation",
    )
    parser.add_argument(
        "--noise_distribution",
        type=str,
        default="discrete",
        help="Noise distribution for coefficient ablation (e.g., 'discrete', 'gennorm:1.5')",
    )

    if args is None:
        args = sys.argv[1:]
    opts = parser.parse_args(args)

    if opts.coefficient_ablation:
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

        # Generate plots
        plot_coefficient_ablation_results(coef_results, coef_output_dir)

        # Print summary
        print("\n" + "=" * 100)
        print("SUMMARY: Coefficient Ablation Study")
        print("=" * 100)
        print(f"\nExperiment settings:")
        print(f"  noise_distribution: {opts.noise_distribution}")
        print(f"  n_samples: {opts.n_samples}")
        print(f"  n_experiments: {opts.n_experiments}")
        # print(f"  treatment_coef_values: {opts.treatment_coef_values}")
        # print(f"  outcome_coef_values: {opts.outcome_coef_values}")
        # print(f"  treatment_effect_values: {opts.treatment_effect_values}")

        print("\n" + "-" * 100)
        print(f"{'tc':>6} {'oc':>6} {'te':>6} {'ICA Var Coeff':>15} {'HOML RMSE':>12} {'ICA RMSE':>12} {'Winner':>10}")
        print("-" * 100)

        for res in coef_results:
            homl_rmse = res["rmse"][1]
            ica_rmse = res["rmse"][4] if 4 < len(res["rmse"]) else np.nan
            winner = "ICA" if ica_rmse < homl_rmse else "HOML"
            print(
                f"{res['treatment_coef_scalar']:>6.2f} {res['outcome_coef_scalar']:>6.2f} "
                f"{res['treatment_effect']:>6.2f} {res['ica_var_coeff']:>15.4f} "
                f"{homl_rmse:>12.4f} {ica_rmse:>12.4f} {winner:>10}"
            )

        print("=" * 100)

    else:
        # Run noise distribution ablation
        distributions = list(opts.distributions)
        if opts.gennorm_betas is not None:
            for beta_val in opts.gennorm_betas:
                gennorm_spec = f"gennorm:{beta_val}"
                if gennorm_spec not in distributions:
                    distributions.append(gennorm_spec)
            print(f"Final distribution list: {distributions}")

        # Use different results file for randomized vs fixed coefficients
        if opts.randomize_coeffs:
            coef_range = tuple(opts.coef_range)
            te_range = tuple(opts.treatment_effect_range)
            results_file = os.path.join(
                opts.output_dir,
                f"noise_ablation_results_n{opts.n_random_configs}_coef{coef_range[0]:.1f}to{coef_range[1]:.1f}_te{te_range[0]:.1f}to{te_range[1]:.1f}.npy",
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
                coef_range=tuple(opts.coef_range),
            )

            os.makedirs(opts.output_dir, exist_ok=True)
            np.save(results_file, results)
            print(f"Results saved to {results_file}")

        # Generate plots
        plot_noise_ablation_results(results, opts.output_dir)

        # Generate additional coefficient-specific plots if randomized
        if opts.randomize_coeffs:
            plot_noise_ablation_coeff_scatter(
                results,
                opts.output_dir,
                n_configs=opts.n_random_configs,
                coef_range=coef_range,
                treatment_effect_range=te_range,
            )

        # Print summary table (HOML and ICA only)
        print("\n" + "=" * 90)
        print("SUMMARY: Noise Distribution Ablation Study (HOML and ICA)")
        print("=" * 90)
        print(f"\nExperiment settings:")
        print(f"  n_samples: {opts.n_samples}")
        print(f"  n_experiments: {opts.n_experiments}")
        if opts.randomize_coeffs:
            print(f"  randomize_coeffs: True (n_random_configs={opts.n_random_configs})")
            print(f"  treatment_effect_range: {opts.treatment_effect_range}")
            print(f"  coef_range: {opts.coef_range}")
        else:
            print(f"  treatment_effect: {opts.treatment_effect}")
        print(f"  covariate_pdf: {opts.covariate_pdf}")

        print("\n" + "-" * 90)
        print(
            f"{'Distribution':<20} {'Kurtosis':>10} {'ICA Var Coeff':>14} {'HOML RMSE':>12} {'ICA RMSE':>12} {'Winner':>10}"
        )
        print("-" * 90)

        for dist, res in results.items():
            kurtosis = res["eta_excess_kurtosis"]
            ica_var_coeff = res.get("ica_var_coeff", np.nan)
            homl_rmse = res["rmse"][1] if len(res["rmse"]) > 1 else np.nan
            ica_rmse = res["rmse"][4] if len(res["rmse"]) > 4 else np.nan
            winner = "ICA" if ica_rmse < homl_rmse else "HOML"
            ica_var_str = f"{ica_var_coeff:.4f}" if not np.isnan(ica_var_coeff) else "N/A"
            print(f"{dist:<20} {kurtosis:>10.4f} {ica_var_str:>14} {homl_rmse:>12.4f} {ica_rmse:>12.4f} {winner:>10}")

        print("=" * 90)

    return 0


if __name__ == "__main__":
    sys.exit(main())
