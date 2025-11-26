"""
Noise distribution ablation study for treatment effect estimation.

This script runs experiments comparing different noise distributions for eta:
- Discrete (default asymmetric)
- Heavy-tailed: Laplace, gennorm_heavy (beta=1)
- Bounded: Uniform, Rademacher
- Light-tailed: gennorm_light (beta=4)
"""

import os
import sys

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
):
    """Run experiments across different noise distributions.

    Args:
        noise_distributions: List of noise distribution names to test
        n_samples: Number of samples per experiment
        n_experiments: Number of Monte Carlo replications
        support_size: Size of support for coefficients
        treatment_effect: True treatment effect value
        beta: Beta parameter for gennorm covariate distribution
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
        seed: Random seed

    Returns:
        Dictionary with results for each noise distribution
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

    # Setup coefficients
    cov_dim_max = support_size
    treatment_support = outcome_support = np.array(range(support_size))
    treatment_coef = np.zeros(support_size)
    treatment_coef[0] = 1.0  # Simple scalar coefficient
    outcome_coef = np.zeros(support_size)
    outcome_coef[0] = 1.0

    # Outcome noise distribution
    epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

    # Compute regularization parameter
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    all_results = {}

    for noise_dist in noise_distributions:
        print(f"\nRunning experiments for noise distribution: {noise_dist}")

        # Setup treatment noise for this distribution
        params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise(distribution=noise_dist)

        # Calculate moments for asymptotic variance
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

            (
                eta_excess_kurtosis,
                eta_skewness_squared,
                ica_asymptotic_var,
                ica_asymptotic_var_hyvarinen,
                ica_asymptotic_var_num,
                ica_var_coeff,
            ) = var_calculator.calc_ica_asymptotic_var(
                treatment_coef,
                outcome_coef,
                treatment_effect,
                params_or_discounts,
                mean_discount,
                probs,
                eta_cubed_variance,
            )
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

            (
                eta_excess_kurtosis,
                eta_skewness_squared,
                ica_asymptotic_var,
                ica_asymptotic_var_hyvarinen,
                ica_asymptotic_var_num,
                ica_var_coeff,
            ) = var_calculator.calc_ica_asymptotic_var_from_distribution(
                treatment_coef, outcome_coef, treatment_effect, noise_dist, params_or_discounts, probs
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

        print(f"Experiments kept: {len(results)} out of {n_experiments} seeds")

        if len(results) == 0:
            print(f"No experiments converged for {noise_dist}")
            continue

        # Extract results
        ortho_rec_tau = [
            [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] + ica_estimate.tolist()
            for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_estimate, _ in results
        ]

        # Compute statistics
        ortho_rec_tau_array = np.array(ortho_rec_tau)
        biases = np.mean(ortho_rec_tau_array - treatment_effect, axis=0)
        sigmas = np.std(ortho_rec_tau_array, axis=0)
        rmse = np.sqrt(biases**2 + sigmas**2)

        all_results[noise_dist] = {
            "ortho_rec_tau": ortho_rec_tau,
            "biases": biases,
            "sigmas": sigmas,
            "rmse": rmse,
            "n_samples": n_samples,
            "n_experiments": len(results),
            "eta_second_moment": eta_second_moment,
            "eta_third_moment": eta_third_moment,
            "eta_fourth_moment": eta_fourth_moment,
            "eta_excess_kurtosis": eta_excess_kurtosis,
            "eta_skewness_squared": eta_skewness_squared,
            "homl_asymptotic_var": homl_asymptotic_var,
            "ica_asymptotic_var": ica_asymptotic_var,
        }

        # Print summary
        method_names = ["Ortho ML", "Robust Ortho ML", "Robust Ortho Est", "Robust Ortho Split", "ICA"]
        print(f"\nResults for {noise_dist}:")
        print(f"  Excess kurtosis: {eta_excess_kurtosis:.4f}")
        print(f"  Third moment: {eta_third_moment:.4f}")
        for i, name in enumerate(method_names):
            if i < len(biases):
                print(f"  {name}: bias={biases[i]:.4f}, std={sigmas[i]:.4f}, rmse={rmse[i]:.4f}")

    return all_results


def plot_noise_ablation_results(results: dict, output_dir: str = "figures/noise_ablation"):
    """Plot results from noise ablation study.

    Args:
        results: Dictionary with results for each noise distribution
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    method_names = ["Ortho ML", "Robust Ortho ML", "Robust Ortho Est", "Robust Ortho Split", "ICA"]
    noise_dists = list(results.keys())

    # Create label mapping for nicer display
    label_map = {
        "discrete": "Discrete",
        "laplace": "Laplace",
        "uniform": "Uniform",
        "rademacher": "Rademacher",
        "gennorm_heavy": r"GenNorm ($\beta$=1)",
        "gennorm_light": r"GenNorm ($\beta$=4)",
    }

    # Plot 1: RMSE comparison across distributions
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(noise_dists))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for i, method in enumerate(method_names):
        rmse_values = [results[dist]["rmse"][i] if i < len(results[dist]["rmse"]) else np.nan for dist in noise_dists]
        ax.bar(x + i * width, rmse_values, width, label=method, color=colors[i])

    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_comparison.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Bias comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(method_names):
        bias_values = [
            np.abs(results[dist]["biases"][i]) if i < len(results[dist]["biases"]) else np.nan for dist in noise_dists
        ]
        ax.bar(x + i * width, bias_values, width, label=method, color=colors[i])

    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel(r"$|\mathrm{Bias}|$")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_comparison.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Standard deviation comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(method_names):
        std_values = [results[dist]["sigmas"][i] if i < len(results[dist]["sigmas"]) else np.nan for dist in noise_dists]
        ax.bar(x + i * width, std_values, width, label=method, color=colors[i])

    ax.set_xlabel("Noise Distribution")
    ax.set_ylabel("Standard Deviation")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([label_map.get(d, d) for d in noise_dists])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "std_comparison.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Distribution properties table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table_data = []
    headers = ["Distribution", "Kurtosis", "3rd Moment", "HOML AsymVar", "ICA AsymVar"]

    for dist in noise_dists:
        row = [
            label_map.get(dist, dist),
            f"{results[dist]['eta_excess_kurtosis']:.3f}",
            f"{results[dist]['eta_third_moment']:.3f}",
            f"{results[dist]['homl_asymptotic_var']:.3f}"
            if results[dist]["homl_asymptotic_var"] < 1e10
            else r"$\infty$",
            f"{results[dist]['ica_asymptotic_var']:.3f}" if results[dist]["ica_asymptotic_var"] < 1e10 else r"$\infty$",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_properties.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {output_dir}")


def main(args=None):
    """Main function for noise distribution ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Noise distribution ablation study for treatment effect estimation")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples per experiment")
    parser.add_argument("--n_experiments", type=int, default=20, help="Number of Monte Carlo replications")
    parser.add_argument("--support_size", type=int, default=10, help="Support size for coefficients")
    parser.add_argument("--treatment_effect", type=float, default=1.0, help="True treatment effect")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for gennorm covariates")
    parser.add_argument("--sigma_outcome", type=float, default=np.sqrt(3.0), help="Outcome noise std")
    parser.add_argument("--covariate_pdf", type=str, default="gennorm", help="Covariate distribution")
    parser.add_argument("--check_convergence", action="store_true", help="Check ICA convergence")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=12143, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="figures/noise_ablation", help="Output directory")
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["discrete", "laplace", "uniform", "rademacher", "gennorm_heavy", "gennorm_light"],
        help="Noise distributions to test",
    )

    if args is None:
        args = sys.argv[1:]
    opts = parser.parse_args(args)

    # Check for existing results
    results_file = os.path.join(opts.output_dir, "noise_ablation_results.npy")

    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        results = np.load(results_file, allow_pickle=True).item()
    else:
        # Run experiments
        results = run_noise_ablation_experiments(
            noise_distributions=opts.distributions,
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
        )

        # Save results
        os.makedirs(opts.output_dir, exist_ok=True)
        np.save(results_file, results)
        print(f"Results saved to {results_file}")

    # Generate plots
    plot_noise_ablation_results(results, opts.output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Noise Distribution Ablation Study")
    print("=" * 80)
    print(f"\nExperiment settings:")
    print(f"  n_samples: {opts.n_samples}")
    print(f"  n_experiments: {opts.n_experiments}")
    print(f"  treatment_effect: {opts.treatment_effect}")
    print(f"  covariate_pdf: {opts.covariate_pdf}")

    print("\n" + "-" * 80)
    print(f"{'Distribution':<15} {'Kurtosis':>10} {'3rd Moment':>12} {'Ortho RMSE':>12} {'ICA RMSE':>12}")
    print("-" * 80)

    for dist, res in results.items():
        kurtosis = res["eta_excess_kurtosis"]
        third_moment = res["eta_third_moment"]
        ortho_rmse = res["rmse"][0] if len(res["rmse"]) > 0 else np.nan
        ica_rmse = res["rmse"][4] if len(res["rmse"]) > 4 else np.nan
        print(f"{dist:<15} {kurtosis:>10.4f} {third_moment:>12.4f} {ortho_rmse:>12.4f} {ica_rmse:>12.4f}")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
