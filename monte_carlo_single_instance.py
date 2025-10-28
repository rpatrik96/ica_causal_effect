import sys

import matplotlib

from plot_utils import plot_asymptotic_var_comparison, plot_gennorm, plot_multi_treatment, plot_typography

matplotlib.use("Agg")
import os

import numpy as np
from sklearn.linear_model import Lasso
from tueplots import bundles

from ica import ica_treatment_effect_estimation
from main_estimation import all_together_cross_fitting


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
    # Generate price as a function of co-variates
    treatment = np.dot(x[:, treatment_support], treatment_coef) + eta
    # Generate demand as a function of price and co-variates
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
    except Exception as e:
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


def main(args):
    import argparse
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from joblib import Parallel, delayed

    from plot_utils import plot_and_save_model_errors, plot_error_bar_stats, plot_method_comparison, plot_typography

    os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

    parser = argparse.ArgumentParser(description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples", type=int, help="n_samples", default=500)
    parser.add_argument("--n_experiments", dest="n_experiments", type=int, help="n_experiments", default=5)
    parser.add_argument("--seed", dest="seed", type=int, help="seed", default=12143)
    parser.add_argument("--sigma_outcome", dest="sigma_outcome", type=float, help="sigma_outcome", default=np.sqrt(3.0))
    parser.add_argument("--covariate_pdf", dest="covariate_pdf", type=str, help="pdf of covariates", default="gennorm")

    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./figures")
    parser.add_argument(
        "--check_convergence", dest="check_convergence", type=bool, help="check convergence", default=True
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
        default=True,
    )

    opts = parser.parse_args(args)

    np.random.seed(opts.seed)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    setup_output_dir(opts)

    """
    We will work with a sparse linear model with high dimensional co-variates
    """

    # How many experiments to run to see the distribution of the recovered coefficient between price and demand
    n_experiments = opts.n_experiments
    sigma_outcome = opts.sigma_outcome

    # Run experiments for different support sizes and beta values

    beta_filter = 4
    support_filter = 5 if opts.small_data else 10

    if opts.small_data:
        support_sizes = [2, 5, 10]
        data_samples = [20, 50, 100]
    else:
        support_sizes = [2, 5, 10, 20, 50] if opts.asymptotic_var is False and opts.scalar_coeffs is False else [10]
        data_samples = [100, 200, 500, 1000, 2000, 5000] if opts.asymptotic_var is False else [10**4]

    beta_values = (
        [1.0]
        if opts.covariate_pdf != "gennorm" or opts.asymptotic_var is True
        else [
            0.5,
            1.0,  # 1.5,
            2.0,  # 2.5,
            3.0,  # 3.5,
            4.0,  # 4.5,
            # 5.
        ]
    )
    # treatment_effects = [3.0] if opts.asymptotic_var is False else [-10, -2, -0.2, -.02, 0.01, .1, .5, 1., 3., 10]
    treatment_effects = [0.01, 0.1, 0.5, 1.0, 3.0, 10] if opts.matched_coefficients is False else [1.0]
    # Dimension of co-variates
    cov_dim_max = support_sizes[-1]

    if opts.asymptotic_var:
        treatment_coefs = [  # -.00005, 0.0006,
            -0.002,
            # 0.1, 0.23,
            -0.33,  # -0.47, 0.89,
            # -1.34, 1.78,
            1.26,
            # -2.56, 3.14,  # -8.67, 12.3
        ]
        outcome_coefs = [  # -.00001, 0.0002,
            # 0.003,
            -0.05,  # 0.3, -0.5,
            0.7,
            # -0.9,
            # 1.1,
            # -1.3,  # 1.5, -1.7,
            1.9,
            # -2.1,
            # 5., -4.,  # 13., -22,
            # 33., -56
        ]

        treatment_coef_array = outcome_coef_array = None
    elif opts.scalar_coeffs:
        treatment_coefs = [-0.002, 0.05, -0.43, 1.56]
        outcome_coefs = [0.003, -0.02, 0.63, -1.45]
        treatment_coef_array = outcome_coef_array = None
    else:
        treatment_coef_array = np.random.uniform(-5, 5, size=cov_dim_max)
        outcome_coef_array = np.random.uniform(-5, 5, size=cov_dim_max)

        if opts.matched_coefficients:
            treatment_coef_array = -outcome_coef_array

        # dummy variable to avoid loop
        treatment_coefs = [None]
        outcome_coefs = [None]

    results_file_path, results_filename = setup_filename(n_experiments, opts)

    all_results = []
    treatment_coef_list = np.zeros(cov_dim_max)
    outcome_coef_list = np.zeros(cov_dim_max)

    for n_samples in data_samples:

        # Check if the results file already exists
        if os.path.exists(results_file_path):
            print(f"Results file {results_file_path} already exists. Loading data instead of rerunning experiments.")
            structured_results = np.load(results_file_path, allow_pickle=True)  # .item()
            all_results = structured_results
            break

        print(f"{n_samples=}")

        for treatment_coefficient in treatment_coefs:
            for outcome_coefficient in outcome_coefs:
                for support_size in support_sizes:
                    print(f"{support_size=}")

                    outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support = (
                        setup_treatment_outcome_coefs(
                            cov_dim_max,
                            opts,
                            outcome_coef_array,
                            outcome_coef_list,
                            outcome_coefficient,
                            support_size,
                            treatment_coef_array,
                            treatment_coef_list,
                            treatment_coefficient,
                        )
                    )

                    for beta in beta_values:
                        print(f"{beta=}")

                        for treatment_effect in treatment_effects:
                            print(f"{treatment_effect=}")

                            if opts.asymptotic_var and opts.tie_sample_dim:
                                n_samples = support_size**4

                            if opts.verbose:
                                print("Support of treatment as function of co-variates: {}".format(treatment_support))
                                print("Coefficients of treatment as function of co-variates: {}".format(treatment_coef))

                            # Distribution of residuals of treatment
                            discounts, eta_sample, mean_discount, probs = setup_treatment_noise()

                            (
                                eta_cubed_variance,
                                eta_fourth_moment,
                                eta_non_gauss_cond,
                                eta_second_moment,
                                eta_third_moment,
                                homl_asymptotic_var,
                                homl_asymptotic_var_num,
                            ) = calc_homl_asymptotic_var(discounts, mean_discount, probs)

                            (
                                eta_excess_kurtosis,
                                eta_skewness_squared,
                                ica_asymptotic_var,
                                ica_asymptotic_var_hyvarinen,
                                ica_asymptotic_var_num,
                                ica_var_coeff,
                            ) = calc_ica_asymptotic_var(
                                eta_cubed_variance,
                                eta_fourth_moment,
                                eta_third_moment,
                                outcome_coef,
                                sigma_outcome,
                                treatment_coef,
                                treatment_effect,
                            )

                            # Distribution of outcome residuals
                            epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

                            true_coef_treatment = np.zeros(cov_dim_max)
                            true_coef_treatment[treatment_support] = treatment_coef
                            true_coef_outcome = np.zeros(cov_dim_max)
                            true_coef_outcome[outcome_support] = outcome_coef
                            true_coef_outcome[treatment_support] += treatment_effect * treatment_coef

                            if opts.verbose:
                                print(f"{true_coef_outcome[outcome_support]=}")

                            """
                            Run the experiments.
                            """

                            x_sample = setup_covariate_pdf(opts, beta)

                            # Coefficients recovered by orthogonal ML
                            lambda_reg = np.sqrt(np.log(cov_dim_max) / (n_samples))
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
                                        opts.check_convergence,
                                        verbose=opts.verbose,
                                    )
                                    for _ in range(n_experiments)
                                )
                                if (opts.check_convergence is False or r[-1] is not None)
                            ]

                            ortho_rec_tau = [
                                [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml]
                                + ica_treatment_effect_estimate.tolist()
                                for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_treatment_effect_estimate, _ in results  # if not np.isnan(ica_treatment_effect_estimate).any()
                            ]

                            print(f"Experiments kept: {len(ortho_rec_tau)} out of {n_experiments} seeds")
                            if len(ortho_rec_tau) == 0:
                                print("Configuration skipped as no runs converged")
                                continue

                            first_stage_mse = [
                                [
                                    np.linalg.norm(true_coef_treatment - coef_treatment),
                                    np.linalg.norm(true_coef_outcome - coef_outcome),
                                    np.linalg.norm(ica_treatment_effect_estimate - treatment_effect),
                                    ica_mcc,
                                ]
                                for _, _, _, _, coef_treatment, coef_outcome, ica_treatment_effect_estimate, ica_mcc in results  # if not np.isnan(ica_treatment_effect_estimate).any()
                            ]

                            all_results.append(
                                {
                                    "n_samples": n_samples,
                                    "support_size": support_size,
                                    "beta": beta,
                                    "ortho_rec_tau": ortho_rec_tau,
                                    "first_stage_mse": first_stage_mse,
                                    "eta_non_gauss_cond": eta_non_gauss_cond,
                                    "eta_skewness_squared": eta_skewness_squared,
                                    "eta_second_moment": eta_second_moment,
                                    "eta_third_moment": eta_third_moment,
                                    "eta_fourth_moment": eta_fourth_moment,
                                    "eta_excess_kurtosis": eta_excess_kurtosis,
                                    "eta_cubed_variance": eta_cubed_variance,
                                    "sigma_outcome": sigma_outcome,
                                    "homl_asymptotic_var_num": homl_asymptotic_var_num,
                                    "homl_asymptotic_var": homl_asymptotic_var,
                                    "ica_asymptotic_var": ica_asymptotic_var,
                                    "ica_asymptotic_var_num": ica_asymptotic_var_num,
                                    "ica_asymptotic_var_hyvarinen": ica_asymptotic_var_hyvarinen,
                                    "ica_var_coeff": ica_var_coeff,
                                    "treatment_effect": treatment_effect,
                                    "treatment_coefficient": (
                                        treatment_coefficient
                                        if treatment_coefficient is not None
                                        else treatment_coef_array
                                    ),
                                    "outcome_coefficient": (
                                        outcome_coefficient if outcome_coefficient is not None else outcome_coef_array
                                    ),
                                }
                            )

                            """
                            Plotting histograms
                            """

                            biases, sigmas = plot_method_comparison(
                                ortho_rec_tau,
                                treatment_effect,
                                opts.output_dir,
                                n_samples,
                                cov_dim_max,
                                n_experiments,
                                support_size,
                                sigma_outcome,
                                opts.covariate_pdf,
                                beta,
                                plot=False,
                                relative_error=False,
                                verbose=opts.verbose,
                            )

                            biases_rel, sigmas_rel = plot_method_comparison(
                                ortho_rec_tau,
                                treatment_effect,
                                opts.output_dir,
                                n_samples,
                                cov_dim_max,
                                n_experiments,
                                support_size,
                                sigma_outcome,
                                opts.covariate_pdf,
                                beta,
                                plot=False,
                                relative_error=True,
                                verbose=opts.verbose,
                            )
                            all_results[-1]["biases"] = biases
                            all_results[-1]["sigmas"] = sigmas

                            all_results[-1]["biases_rel"] = biases_rel
                            all_results[-1]["sigmas_rel"] = sigmas_rel

                            plot_and_save_model_errors(
                                first_stage_mse,
                                ortho_rec_tau,
                                opts.output_dir,
                                n_samples,
                                cov_dim_max,
                                n_experiments,
                                support_size,
                                sigma_outcome,
                                opts.covariate_pdf,
                                beta,
                                plot=False,
                            )

                        # plot_error_bar_stats(all_results, cov_dim_max, n_experiments, n_samples, opts, beta)

    # Define the output file path
    results_file_path = os.path.join(opts.output_dir, results_filename)
    np.save(results_file_path, all_results)

    print(f"All results with noise parameters have been saved to {results_file_path}")

    print("\nDone with all experiments!")

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:

        # filtered_results = [res for res in all_results if res['ica_var_coeff'] <= 50]

        # Prepare data for the scatter plot
        x_values_ica_var_coeff = [res["ica_var_coeff"] for res in all_results]
        # treatment_coef = [res['treatment_coefficient'] for res in all_results]
        # outcome_coef = [res['outcome_coefficient'] for res in all_results]
        y_values_ica_biases = [res["biases"][-1] for res in all_results]
        y_values_homl_biases = [res["biases"][3] for res in all_results]
        # y_errors_ica = [res['sigmas'][-1] / np.sqrt(res['n_samples']) for res in all_results]
        # y_errors_homl = [res['sigmas'][3] / np.sqrt(res['n_samples']) for res in all_results]

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))

        # Subplot 1: x-axis is x_values_ica_var_coeff
        y_diff_biases = np.array(y_values_ica_biases) - np.array(y_values_homl_biases)
        # y_diff_errors = np.sqrt(np.array(y_errors_ica) ** 2 + np.array(y_errors_homl) ** 2)
        axs.scatter(x_values_ica_var_coeff, y_diff_biases, color="purple", alpha=0.75, label="ICA - HOML")
        axs.set_xlabel(r"$1+\Vert b+a\theta\Vert_2^2$")
        axs.set_xscale("log")
        axs.set_ylabel(r"Differences $|\theta-\hat{\theta}|$")
        # axs.set_yscale('log')
        axs.legend()

        # plt.tight_layout()
        plt.savefig(os.path.join(opts.output_dir, "gennorm_asymp_var_overall.svg"), dpi=300, bbox_inches="tight")
        plt.close()

    plot_gennorm(
        all_results,
        opts,
        filter_type="beta",
        filter_value=beta_filter,
        compare_method="homl",
        plot_type="bias",
        save_subfolder=False,
    )
    plot_gennorm(
        all_results,
        opts,
        filter_type="support",
        filter_value=support_filter,
        compare_method="homl",
        plot_type="bias",
        save_subfolder=False,
    )

    # # Duplicate with ica_var_filter below 70
    # plot_gennorm(all_results, opts, filter_type='beta', filter_value=beta_filter, compare_method='homl',
    #              plot_type='bias', save_subfolder=False, filter_ica_var_coeff=True, ica_var_threshold=20, filter_below=True)
    # plot_gennorm(all_results, opts, filter_type='support', filter_value=support_filter, compare_method='homl',
    #              plot_type='bias', save_subfolder=False, filter_ica_var_coeff=True, ica_var_threshold=20, filter_below=True)
    #
    # # Duplicate with ica_var_filter above 70
    # plot_gennorm(all_results, opts, filter_type='beta', filter_value=beta_filter, compare_method='homl',
    #              plot_type='bias', save_subfolder=False, filter_ica_var_coeff=True, ica_var_threshold=20, filter_below=False)
    # plot_gennorm(all_results, opts, filter_type='support', filter_value=support_filter, compare_method='homl',
    #              plot_type='bias', save_subfolder=False, filter_ica_var_coeff=True, ica_var_threshold=20, filter_below=False)
    #

    # plot_gennorm(all_results, opts, filter_type='beta', filter_value=beta_filter, compare_method='oml',
    #              plot_type='bias', save_subfolder=False)
    # plot_gennorm(all_results, opts, filter_type='support', filter_value=support_filter, compare_method='oml',
    #              plot_type='bias', save_subfolder=False)
    #
    # plot_gennorm(all_results, opts, filter_type='beta', filter_value=beta_filter, compare_method=None,
    #              plot_type='bias',save_subfolder=False)
    # plot_gennorm(all_results, opts, filter_type='support', filter_value=support_filter, compare_method=None,
    #              plot_type='mcc',save_subfolder=False)
    # plot_gennorm(all_results, opts, filter_type='support', filter_value=support_filter, compare_method=None,
    #              plot_type='bias',save_subfolder=False)

    plot_asymptotic_var_comparison(all_results, opts, save_subfolder=False)

    # return 0

    for treatment_effect in treatment_effects:
        filtered_results = [result for result in all_results if result["treatment_effect"] == treatment_effect]

        # plot_gennorm(filtered_results, opts, filter_type='beta', filter_value=beta_filter, compare_method='homl',
        #              plot_type='bias')
        # plot_gennorm(filtered_results, opts, filter_type='support', filter_value=support_filter, compare_method='homl',
        #              plot_type='bias')

        # plot_gennorm(filtered_results, opts, filter_type='beta', filter_value=beta_filter, compare_method='oml', plot_type='bias')
        # plot_gennorm(filtered_results, opts, filter_type='support', filter_value=support_filter, compare_method='oml',
        #              plot_type='bias')

        # plot_gennorm(filtered_results, opts, filter_type='beta', filter_value=beta_filter, compare_method=None,
        #                        plot_type='bias')
        # plot_gennorm(filtered_results, opts, filter_type='support', filter_value=support_filter, compare_method=None,
        #              plot_type='mcc')
        # plot_gennorm(filtered_results, opts, filter_type='support', filter_value=support_filter, compare_method=None,
        #              plot_type='bias')

        plot_asymptotic_var_comparison(filtered_results, opts)

    # plot_multi_treatment(all_results, opts, treatment_effects)


def setup_output_dir(opts):
    opts.output_dir = os.path.join(
        opts.output_dir, f"n_exp_{opts.n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}"
    )
    if opts.check_convergence:
        opts.output_dir += "_convergence"
    if opts.small_data:
        opts.output_dir += "_small_data"
    if opts.matched_coefficients:
        opts.output_dir += "_matched_coefficients"
    if opts.scalar_coeffs:
        opts.output_dir += "_scalar_coeffs"
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)


def setup_filename(n_experiments, opts):
    # Define the output file path
    if opts.asymptotic_var:
        results_filename = f"all_results_asymptotic_var_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}"
    else:
        results_filename = (
            f"all_results_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}"
        )
    # Add check_convergence flag status to the filename if it is set to true
    if opts.check_convergence:
        results_filename += "_check_convergence"
    results_filename += ".npy"
    results_file_path = os.path.join(opts.output_dir, results_filename)
    return results_file_path, results_filename


def setup_treatment_outcome_coefs(
    n_dim,
    opts,
    outcome_coef_array,
    outcome_coef_list,
    outcome_coefficient,
    support_size,
    treatment_coef_array,
    treatment_coef_list,
    treatment_coefficient,
):
    if opts.asymptotic_var is True:
        outcome_coefficient = treatment_coefficient

    if opts.matched_coefficients:
        outcome_coefficient = -treatment_coefficient

    if opts.asymptotic_var is True or opts.scalar_coeffs is True:

        treatment_coef_list[0] = treatment_coefficient
        outcome_coef_list[0] = outcome_coefficient

        outcome_support = treatment_support = np.array(range(support_size))

        treatment_coef = treatment_coef_list[:support_size]
        outcome_coef = outcome_coef_list[:support_size]

        print(f"{treatment_coef=}")
        print(f"{outcome_coef=}")

    else:
        # this implicitly specifies sparsity via restricting the support
        outcome_support = treatment_support = np.random.choice(range(n_dim), size=support_size, replace=False)
        treatment_coef = treatment_coef_array[
            treatment_support
        ]  # Support and coefficients for treatment as function of co-variates
        outcome_coef = outcome_coef_array[
            outcome_support
        ]  # Support and coefficients for outcome as function of co-variates
    return outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support


def calc_ica_asymptotic_var(
    eta_cubed_variance,
    eta_fourth_moment,
    eta_third_moment,
    outcome_coef,
    sigma_outcome,
    treatment_coef,
    treatment_effect,
):
    eps_second_moment = (1 / 3) * sigma_outcome**2
    eps_fourth_moment = (1 / 5) * sigma_outcome**4
    eps_sixth_moment = (1 / 7) * sigma_outcome**6
    ica_asymptotic_var_num = eps_sixth_moment - eps_fourth_moment**2
    ica_asymptotic_var_hyvarinen = ica_asymptotic_var_num / (eps_fourth_moment - 3 * eps_second_moment) ** 2
    eta_excess_kurtosis = eta_fourth_moment - 3
    eta_skewness_squared = eta_third_moment**2
    ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect, ord=2) ** 2
    ica_asymptotic_var = ica_var_coeff * eta_cubed_variance / eta_excess_kurtosis**2
    return (
        eta_excess_kurtosis,
        eta_skewness_squared,
        ica_asymptotic_var,
        ica_asymptotic_var_hyvarinen,
        ica_asymptotic_var_num,
        ica_var_coeff,
    )


def calc_homl_asymptotic_var(discounts, mean_discount, probs):
    # Calculate moments of the residual distribution
    eta_second_moment = np.dot(probs, (discounts - mean_discount) ** 2)
    eta_third_moment = np.dot(probs, (discounts - mean_discount) ** 3)
    eta_fourth_moment = np.dot(probs, (discounts - mean_discount) ** 4)
    eta_non_gauss_cond = eta_fourth_moment - 3 * eta_second_moment
    # HOML asymptotic variance numerator
    eta_cubed_variance = np.dot(probs, ((discounts - mean_discount) ** 3 - eta_third_moment) ** 2)
    homl_asymptotic_var_num = eta_cubed_variance + 9 * eta_second_moment**2 + 3 * eta_fourth_moment * eta_second_moment
    homl_asymptotic_var = homl_asymptotic_var_num / eta_non_gauss_cond**2
    return (
        eta_cubed_variance,
        eta_fourth_moment,
        eta_non_gauss_cond,
        eta_second_moment,
        eta_third_moment,
        homl_asymptotic_var,
        homl_asymptotic_var_num,
    )


def setup_treatment_noise(rademacher=False):
    if rademacher is False:
        discounts = np.array([0, -0.5, -2.0, -4.0])
        probs = np.array([0.65, 0.2, 0.1, 0.05])
    else:
        discounts = np.array([1, -1])
        probs = np.array([0.5, 0.5])
    mean_discount = np.dot(discounts, probs)
    eta_sample = lambda x: np.array(
        [discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)]
    )
    return discounts, eta_sample, mean_discount, probs


def setup_covariate_pdf(opts, beta):
    if opts.covariate_pdf == "gauss":
        x_sample = lambda n_samples, n_dim: np.random.normal(size=(n_samples, n_dim))
    elif opts.covariate_pdf == "uniform":
        x_sample = lambda n_samples, n_dim: np.random.uniform(size=(n_samples, n_dim))
    if opts.covariate_pdf == "gennorm":
        from scipy.stats import gennorm

        loc = 0
        scale = 1
        if opts.asymptotic_var:
            scale = 1 / np.sqrt(2.0)

        x_sample = lambda n_samples, n_dim: gennorm.rvs(beta, loc=loc, scale=scale, size=(n_samples, n_dim))
    return x_sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()
    main(sys.argv[1:])
