"""
Monte Carlo experiments for Orthogonal Machine Learning.

This script runs Monte Carlo experiments comparing different OML methods
including standard orthogonal ML, robust variants, and ICA-based estimation.
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
from oml_plotting import generate_all_oml_plots, plot_method_comparison_both_errors, save_results_with_metadata
from oml_runner import setup_covariate_pdf, setup_treatment_noise, setup_treatment_outcome_coefs
from oml_utils import (
    AsymptoticVarianceCalculator,
    OMLExperimentConfig,
    OMLParameterGrid,
    OMLResultsManager,
    setup_output_dir,
    setup_results_filename,
)
from plot_utils import plot_and_save_model_errors, plot_typography


def experiment(
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
    """Run a single OML experiment.

    Args:
        x: Covariate matrix
        eta: Treatment noise
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


def run_experiments_for_configuration(
    config: OMLExperimentConfig,
    param_grid: OMLParameterGrid,
    n_samples: int,
    support_size: int,
    beta: float,
    treatment_effect: float,
    treatment_coefficient: float,
    outcome_coefficient: float,
    treatment_coef_array: np.ndarray,
    outcome_coef_array: np.ndarray,
) -> dict:
    """Run experiments for a single parameter configuration.

    Args:
        config: Experiment configuration
        param_grid: Parameter grid
        n_samples: Number of samples
        support_size: Support size
        beta: Beta parameter
        treatment_effect: Treatment effect value
        treatment_coefficient: Treatment coefficient
        outcome_coefficient: Outcome coefficient
        treatment_coef_array: Treatment coefficient array
        outcome_coef_array: Outcome coefficient array

    Returns:
        Dictionary with experiment results
    """
    cov_dim_max = param_grid.support_sizes[-1]

    # Setup coefficients
    treatment_coef_list = np.zeros(cov_dim_max)
    outcome_coef_list = np.zeros(cov_dim_max)

    outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support = (
        setup_treatment_outcome_coefs(
            cov_dim_max,
            config,
            outcome_coef_array,
            outcome_coef_list,
            outcome_coefficient,
            support_size,
            treatment_coef_array,
            treatment_coef_list,
            treatment_coefficient,
        )
    )

    if config.verbose:
        print(f"Treatment support: {treatment_support}")
        print(f"Treatment coefficients: {treatment_coef}")

    # Setup noise distributions based on config
    eta_noise_dist = getattr(config, "eta_noise_dist", "discrete")
    discounts_or_params, eta_sample, mean_discount, probs = setup_treatment_noise(distribution=eta_noise_dist)

    # Calculate asymptotic variances
    var_calculator = AsymptoticVarianceCalculator()

    # Use distribution-aware methods for asymptotic variance calculation
    if eta_noise_dist == "discrete":
        # Use original discrete-based calculation
        (
            eta_cubed_variance,
            eta_fourth_moment,
            eta_non_gauss_cond,
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            homl_asymptotic_var_num,
        ) = var_calculator.calc_homl_asymptotic_var(discounts_or_params, mean_discount, probs)

        (
            eta_excess_kurtosis,
            eta_skewness_squared,
            ica_asymptotic_var,
            ica_asymptotic_var_hyvarinen,
            ica_asymptotic_var_num,
            ica_var_coeff,
        ) = var_calculator.calc_ica_asymptotic_var(
            treatment_coef, outcome_coef, treatment_effect, discounts_or_params, mean_discount, probs, eta_cubed_variance
        )
    else:
        # Use distribution-based calculation for continuous distributions
        (
            eta_cubed_variance,
            eta_fourth_moment,
            eta_non_gauss_cond,
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            homl_asymptotic_var_num,
        ) = var_calculator.calc_homl_asymptotic_var_from_distribution(eta_noise_dist, discounts_or_params, probs)

        (
            eta_excess_kurtosis,
            eta_skewness_squared,
            ica_asymptotic_var,
            ica_asymptotic_var_hyvarinen,
            ica_asymptotic_var_num,
            ica_var_coeff,
        ) = var_calculator.calc_ica_asymptotic_var_from_distribution(
            treatment_coef, outcome_coef, treatment_effect, eta_noise_dist, discounts_or_params, probs
        )

    # Outcome noise distribution
    epsilon_sample = lambda x: np.random.uniform(  # pylint: disable=unnecessary-lambda-assignment
        -config.sigma_outcome, config.sigma_outcome, size=x
    )

    # True coefficients
    true_coef_treatment = np.zeros(cov_dim_max)
    true_coef_treatment[treatment_support] = treatment_coef
    true_coef_outcome = np.zeros(cov_dim_max)
    true_coef_outcome[outcome_support] = outcome_coef
    true_coef_outcome[treatment_support] += treatment_effect * treatment_coef

    if config.verbose:
        print(f"True outcome coefficients on support: {true_coef_outcome[outcome_support]}")

    # Setup covariate sampling
    x_sample = setup_covariate_pdf(config, beta)

    # Compute regularization parameter
    lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

    # Run parallel experiments
    results = [
        r
        for r in Parallel(n_jobs=-1, verbose=0)(
            delayed(experiment)(
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
                config.check_convergence,
                verbose=config.verbose,
            )
            for _ in range(config.n_experiments)
        )
        if (config.check_convergence is False or r[-1] is not None)
    ]

    print(f"Experiments kept: {len(results)} out of {config.n_experiments} seeds")

    if len(results) == 0:
        print("Configuration skipped as no runs converged")
        return None

    # Extract results
    ortho_rec_tau = [
        [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] + ica_estimate.tolist()
        for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_estimate, _ in results
    ]

    first_stage_mse = [
        [
            np.linalg.norm(true_coef_treatment - coef_treatment),
            np.linalg.norm(true_coef_outcome - coef_outcome),
            np.linalg.norm(ica_estimate - treatment_effect),
            ica_mcc,
        ]
        for _, _, _, _, coef_treatment, coef_outcome, ica_estimate, ica_mcc in results
    ]

    # Compute method comparison statistics
    error_stats = plot_method_comparison_both_errors(
        ortho_rec_tau,
        treatment_effect,
        config.output_dir,
        n_samples,
        cov_dim_max,
        config.n_experiments,
        support_size,
        config.sigma_outcome,
        config.covariate_pdf,
        beta,
        plot=False,
        verbose=config.verbose,
    )

    # Plot model errors
    plot_and_save_model_errors(
        first_stage_mse,
        ortho_rec_tau,
        config.output_dir,
        n_samples,
        cov_dim_max,
        config.n_experiments,
        support_size,
        config.sigma_outcome,
        config.covariate_pdf,
        beta,
        plot=False,
    )

    # Assemble result dictionary
    result_dict = {
        "n_samples": n_samples,
        "support_size": support_size,
        "beta": beta,
        "treatment_effect": treatment_effect,
        "cov_dim_max": cov_dim_max,
        "sigma_outcome": config.sigma_outcome,
        "eta_noise_dist": eta_noise_dist,
        "ortho_rec_tau": ortho_rec_tau,
        "first_stage_mse": first_stage_mse,
        "biases": error_stats["absolute"][0],
        "sigmas": error_stats["absolute"][1],
        "biases_rel": error_stats["relative"][0],
        "sigmas_rel": error_stats["relative"][1],
        "eta_second_moment": eta_second_moment,
        "eta_third_moment": eta_third_moment,
        "eta_non_gauss_cond": eta_non_gauss_cond,
        "eta_cubed_variance": eta_cubed_variance,
        "eta_fourth_moment": eta_fourth_moment,
        "eta_skewness_squared": eta_skewness_squared,
        "eta_excess_kurtosis": eta_excess_kurtosis,
        "ica_var_coeff": ica_var_coeff,
        "ica_asymptotic_var": ica_asymptotic_var,
        "ica_asymptotic_var_hyvarinen": ica_asymptotic_var_hyvarinen,
        "ica_asymptotic_var_num": ica_asymptotic_var_num,
        "homl_asymptotic_var": homl_asymptotic_var,
        "homl_asymptotic_var_num": homl_asymptotic_var_num,
        "treatment_coefficient": treatment_coefficient if treatment_coefficient is not None else treatment_coef_array,
        "outcome_coefficient": outcome_coefficient if outcome_coefficient is not None else outcome_coef_array,
    }

    return result_dict


def main(args):
    """Main experiment execution function.

    Args:
        args: Command-line arguments
    """
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples", type=int, help="n_samples", default=500)
    parser.add_argument("--n_experiments", dest="n_experiments", type=int, help="n_experiments", default=20)
    parser.add_argument("--seed", dest="seed", type=int, help="seed", default=12143)
    parser.add_argument("--sigma_outcome", dest="sigma_outcome", type=float, help="sigma_outcome", default=np.sqrt(3.0))
    parser.add_argument("--covariate_pdf", dest="covariate_pdf", type=str, help="pdf of covariates", default="gennorm")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./figures")
    parser.add_argument(
        "--check_convergence", dest="check_convergence", type=bool, help="check convergence", default=False
    )
    parser.add_argument(
        "--asymptotic_var", dest="asymptotic_var", type=bool, help="Flag to ablate asymptotic variance", default=False
    )
    parser.add_argument("--tie_sample_dim", dest="tie_sample_dim", type=bool, help="Ties n=d**4", default=False)
    parser.add_argument("--verbose", dest="verbose", type=bool, help="Enable verbose output", default=False)
    parser.add_argument("--small_data", dest="small_data", type=bool, help="Flag to use a small dataset", default=False)
    parser.add_argument(
        "--matched_coefficients",
        dest="matched_coefficients",
        type=bool,
        help="Flag to indicate if treatment and outcome coefficients are matched",
        default=False,
    )
    parser.add_argument(
        "--scalar_coeffs",
        dest="scalar_coeffs",
        type=bool,
        help="Flag to indicate if only one coefficient is non-zero",
        default=False,
    )
    parser.add_argument(
        "--eta_noise_dist",
        dest="eta_noise_dist",
        type=str,
        help="Distribution for treatment noise eta: discrete, laplace, uniform, rademacher, gennorm_heavy, gennorm_light",
        default="discrete",
    )

    opts = parser.parse_args(args)

    # Create configuration from parsed arguments
    config = OMLExperimentConfig(
        n_samples=opts.n_samples,
        n_experiments=opts.n_experiments,
        seed=opts.seed,
        sigma_outcome=opts.sigma_outcome,
        covariate_pdf=opts.covariate_pdf,
        output_dir=opts.output_dir,
        check_convergence=opts.check_convergence,
        asymptotic_var=opts.asymptotic_var,
        tie_sample_dim=opts.tie_sample_dim,
        verbose=opts.verbose,
        small_data=opts.small_data,
        matched_coefficients=opts.matched_coefficients,
        scalar_coeffs=opts.scalar_coeffs,
        eta_noise_dist=opts.eta_noise_dist,
    )

    # Set random seed
    np.random.seed(config.seed)

    # Setup plotting
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    # Setup output directory
    config.output_dir = setup_output_dir(config)

    # Create parameter grid
    param_grid = OMLParameterGrid.create_from_config(config)

    # Setup results management
    results_filename = setup_results_filename(config)
    results_filepath = os.path.join(config.output_dir, results_filename)
    results_manager = OMLResultsManager(results_filepath)

    # Check if results already exist
    if results_manager.exists():
        print(f"\nResults file already exists. Loading data from {results_filename}")
        all_results = results_manager.load()
    else:
        print("\nRunning experiments...")
        all_results = []

        # Setup coefficient arrays for non-scalar mode
        cov_dim_max = param_grid.support_sizes[-1]
        if config.scalar_coeffs:
            treatment_coef_array = outcome_coef_array = None
        else:
            treatment_coef_array = np.random.uniform(-5, 5, size=cov_dim_max)
            outcome_coef_array = np.random.uniform(-5, 5, size=cov_dim_max)
            if config.matched_coefficients:
                treatment_coef_array = -outcome_coef_array

        # Run parameter sweep
        for n_samples in param_grid.data_samples:
            print(f"\n{n_samples=}")

            for treatment_coefficient in param_grid.treatment_coefs:
                for outcome_coefficient in param_grid.outcome_coefs:
                    for support_size in param_grid.support_sizes:
                        print(f"{support_size=}")

                        for beta in param_grid.beta_values:
                            print(f"{beta=}")

                            for treatment_effect in param_grid.treatment_effects:
                                print(f"{treatment_effect=}")

                                # Adjust sample size if tie_sample_dim is enabled
                                current_n_samples = n_samples
                                if config.asymptotic_var and config.tie_sample_dim:
                                    current_n_samples = support_size**4

                                # Run experiments for this configuration
                                result = run_experiments_for_configuration(
                                    config=config,
                                    param_grid=param_grid,
                                    n_samples=current_n_samples,
                                    support_size=support_size,
                                    beta=beta,
                                    treatment_effect=treatment_effect,
                                    treatment_coefficient=treatment_coefficient,
                                    outcome_coefficient=outcome_coefficient,
                                    treatment_coef_array=treatment_coef_array,
                                    outcome_coef_array=outcome_coef_array,
                                )

                                if result is not None:
                                    all_results.append(result)

        # Save results
        save_results_with_metadata(all_results, config.output_dir, results_filename, config)

    print("\nDone with all experiments!")

    # Generate all plots
    generate_all_oml_plots(all_results, config, param_grid, param_grid.treatment_effects)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
