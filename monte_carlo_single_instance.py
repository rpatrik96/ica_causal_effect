import sys

import matplotlib

from plot_utils import plot_typography, plot_ica_gennorm_beta_filter_bias, plot_ica_gennorm_support_filter_mcc, \
    plot_ica_gennorm_beta_filter, plot_oml_ica_comparison_gennorm_support_filter, \
    plot_oml_ica_comparison_gennorm_beta_filter, plot_homl_ica_comparison_gennorm_support_filter, \
    plot_homl_ica_comparison_gennorm_beta_filter, plot_multi_treatment, plot_asymptotic_var_comparison, plot_mse

matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import Lasso
from main_estimation import all_together_cross_fitting

from ica import ica_treatment_effect_estimation

from tueplots import bundles


def experiment(x, eta, epsilon, treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef,
               eta_second_moment, eta_third_moment, lambda_reg, check_convergence=False, verbose=False):
    # Generate price as a function of co-variates
    treatment = np.dot(x[:, treatment_support], treatment_coef) + eta
    # Generate demand as a function of price and co-variates
    outcome = treatment_effect * treatment + np.dot(x[:, outcome_support], outcome_coef) + epsilon
    model_treatment = Lasso(alpha=lambda_reg)
    model_outcome = Lasso(alpha=lambda_reg)

    assert (treatment_support == outcome_support).all()
    ica_treatment_effect_estimate, ica_mcc = ica_treatment_effect_estimation(
        np.hstack((x[:, treatment_support], treatment.reshape(-1, 1), outcome.reshape(-1, 1))),
        np.hstack((x[:, treatment_support], eta.reshape(-1, 1), epsilon.reshape(-1, 1))),
        check_convergence=check_convergence, verbose=verbose)

    if verbose:
        print(f"Estimated vs true treatment effect: {ica_treatment_effect_estimate}, {treatment_effect}")

    return *all_together_cross_fitting(x, treatment, outcome, eta_second_moment, eta_third_moment,
                                       model_treatment=model_treatment,
                                       model_outcome=model_outcome), ica_treatment_effect_estimate, ica_mcc


def main(args):
    import os

    import matplotlib.pyplot as plt

    from plot_utils import plot_method_comparison, plot_and_save_model_errors, plot_error_bar_stats, plot_typography

    import numpy as np
    import argparse
    from joblib import delayed, Parallel

    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='n_samples', default=500)
    parser.add_argument("--n_dim", dest="n_dim",
                        type=int, help='n_dim', default=50)
    parser.add_argument("--n_experiments", dest="n_experiments",
                        type=int, help='n_experiments', default=20)
    parser.add_argument("--seed", dest="seed",
                        type=int, help='seed', default=12143)
    parser.add_argument("--sigma_outcome", dest="sigma_outcome",
                        type=float, help='sigma_outcome', default=np.sqrt(3.))
    parser.add_argument("--covariate_pdf", dest="covariate_pdf",
                        type=str, help='pdf of covariates', default="gennorm")

    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./figures")
    parser.add_argument("--check_convergence", dest="check_convergence",
                        type=bool, help='check convergence', default=False)

    parser.add_argument("--asymptotic_var", dest="asymptotic_var", type=bool, help='Flag to ablate asymptotic variance',
                        default=False)
    parser.add_argument("--tie_sample_dim", dest="tie_sample_dim", type=bool, help='Ties n=d**4', default=False)
    parser.add_argument("--verbose", dest="verbose", type=bool, help='Enable verbose output', default=False)

    opts = parser.parse_args(args)

    np.random.seed(opts.seed)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    if opts.check_convergence:
        opts.output_dir += "_convergence"

    '''
    We will work with a sparse linear model with high dimensional co-variates
    '''
    # Dimension of co-variates
    n_dim = opts.n_dim
    # How many experiments to run to see the distribution of the recovered coefficient between price and demand
    n_experiments = opts.n_experiments

    sigma_outcome = opts.sigma_outcome

    # Run experiments for different support sizes and beta values

    support_sizes = [2, 5, 10, 20, 50] if opts.asymptotic_var is False else [10]  # [5, 10, 20]
    data_samples = [100, 200, 500, 1000, 2000, 5000] if opts.asymptotic_var is False else [10 ** 4]
    beta_values = [1.0] if opts.covariate_pdf != "gennorm" or opts.asymptotic_var is True else [0.5, 1.0, 1.5, 2.0, 2.5,
                                                                                                3., 3.5, 4., 4.5, 5]
    treatment_effects = [3.0] if opts.asymptotic_var is False else [
        3.]  # [-20, -10, -5, -2, -1, -.5, -0.2, -.1, .1,  0.2, .5, 1, 2, 5, 10, 20]

    if opts.asymptotic_var:
        treatment_coefs = [0.1, 0.23, -.33, -0.47, 0.89, -1.34, 1.78, -2.56, 3.14, -3.67, ]
        outcome_coefs = [0.3]  # , -0.5, 0.7, -0.9, 1.1, -1.3, 1.5, -1.7, 1.9, -2.1]
    else:
        treatment_coef_array = np.random.uniform(0, 5, size=support_sizes[-1])
        outcome_coef_array = np.random.uniform(0, 5, size=support_sizes[-1])

       
        # dummy variable to avoid loop
        treatment_coefs = [0.6]
        outcome_coefs = [0.3]

    import os

    # Define the output file path
    if opts.asymptotic_var:
        results_filename = f'all_results_asymptotic_var_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}'
    else:
        results_filename = f'all_results_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}'
    
    # Add check_convergence flag status to the filename if it is set to true
    if opts.check_convergence:
        results_filename += '_check_convergence'
    
    results_filename += '.npy'
    results_file_path = os.path.join(opts.output_dir, results_filename)

    all_results = []
    treatment_coef_list = np.zeros(support_sizes[-1])
    outcome_coef_list = np.zeros(support_sizes[-1])

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

                    outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support = setup_treatment_outcome_coefs(
                        n_dim, opts, outcome_coef_array, outcome_coef_list, outcome_coefficient, support_size,
                        treatment_coef_array, treatment_coef_list, treatment_coefficient)

                    for beta in beta_values:
                        print(f"{beta=}")

                        for treatment_effect in treatment_effects:
                            print(f"{treatment_effect=}")


                            if opts.asymptotic_var and opts.tie_sample_dim:
                                n_samples = support_size ** 4

                            if opts.verbose:
                                print("Support of treatment as function of co-variates: {}".format(treatment_support))
                                print("Coefficients of treatment as function of co-variates: {}".format(treatment_coef))

                            # Distribution of residuals of treatment
                            discounts, eta_sample, mean_discount, probs = setup_treatment_noise()

                            eta_cubed_variance, eta_fourth_moment, eta_non_gauss_cond, eta_second_moment, eta_third_moment, homl_asymptotic_var, homl_asymptotic_var_num = calc_homl_asymptotic_var(
                                discounts, mean_discount, probs)



                            # Distribution of outcome residuals
                            epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

                            eta_excess_kurtosis, eta_skewness_squared, ica_asymptotic_var, ica_asymptotic_var_hyvarinen, ica_asymptotic_var_num, ica_var_coeff = calc_ica_asymptotic_var(
                                eta_cubed_variance, eta_fourth_moment, eta_third_moment, outcome_coef, sigma_outcome,
                                treatment_coef, treatment_effect)

                            true_coef_treatment = np.zeros(n_dim)
                            true_coef_treatment[treatment_support] = treatment_coef
                            true_coef_outcome = np.zeros(n_dim)
                            true_coef_outcome[outcome_support] = outcome_coef
                            true_coef_outcome[treatment_support] += treatment_effect * treatment_coef

                            if opts.verbose:
                                print(f"{true_coef_outcome[outcome_support]=}")

                            '''
                            Run the experiments.
                            '''

                            x_sample = setup_covariate_pdf(opts, beta)

                            # Coefficients recovered by orthogonal ML
                            lambda_reg = np.sqrt(np.log(n_dim) / (n_samples))
                            results = [r for r in Parallel(n_jobs=-1, verbose=0)(
                                delayed(experiment)(x_sample(n_samples, n_dim), eta_sample(n_samples),
                                    epsilon_sample(n_samples), treatment_effect, treatment_support, treatment_coef,
                                    outcome_support, outcome_coef, eta_second_moment, eta_third_moment, lambda_reg) for
                                _ in range(n_experiments)) if (opts.check_convergence is False or r[-1] is not None)]

                            ortho_rec_tau = [[ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] + ica_treatment_effect_estimate.tolist() for
                                             ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_treatment_effect_estimate, _
                                             in results if not np.isnan(ica_treatment_effect_estimate).any()]

                            print(f"Experiments kept: {len(ortho_rec_tau)} out of {n_experiments} seeds")


                            first_stage_mse = [
                                [np.linalg.norm(true_coef_treatment - coef_treatment),
                                 np.linalg.norm(true_coef_outcome - coef_outcome),
                                 np.linalg.norm(ica_treatment_effect_estimate - treatment_effect), ica_mcc] for
                                _, _, _, _, coef_treatment, coef_outcome, ica_treatment_effect_estimate, ica_mcc in
                                results if not np.isnan(ica_treatment_effect_estimate).any()]

                            all_results.append({
                                'n_samples': n_samples,
                                'support_size': support_size,
                                'beta': beta,
                                'ortho_rec_tau': ortho_rec_tau,
                                'first_stage_mse': first_stage_mse,
                                'eta_non_gauss_cond': eta_non_gauss_cond,
                                'eta_skewness_squared': eta_skewness_squared,
                                'eta_second_moment': eta_second_moment,
                                'eta_third_moment': eta_third_moment,
                                'eta_fourth_moment': eta_fourth_moment,
                                'eta_excess_kurtosis': eta_excess_kurtosis,
                                'eta_cubed_variance': eta_cubed_variance,
                                'sigma_outcome': sigma_outcome,
                                'homl_asymptotic_var_num': homl_asymptotic_var_num,
                                'homl_asymptotic_var': homl_asymptotic_var,
                                'ica_asymptotic_var': ica_asymptotic_var,
                                'ica_asymptotic_var_num': ica_asymptotic_var_num,
                                'ica_asymptotic_var_hyvarinen': ica_asymptotic_var_hyvarinen,
                                'ica_var_coeff': ica_var_coeff,
                                'treatment_effect': treatment_effect,
                                'treatment_coefficient': treatment_coefficient,
                                'outcome_coefficient': outcome_coefficient,
                            })

                            '''
                            Plotting histograms
                            '''

                            biases, sigmas = plot_method_comparison(ortho_rec_tau, treatment_effect, opts.output_dir,
                                                                    n_samples, n_dim, n_experiments, support_size,
                                                                    sigma_outcome, opts.covariate_pdf, beta, plot=False,
                                                                    relative_error=False, verbose=opts.verbose)

                            biases_rel, sigmas_rel = plot_method_comparison(ortho_rec_tau, treatment_effect,
                                                                            opts.output_dir, n_samples, n_dim,
                                                                            n_experiments, support_size, sigma_outcome,
                                                                            opts.covariate_pdf, beta, plot=False,
                                                                            relative_error=True, verbose=opts.verbose)
                            all_results[-1]['biases'] = biases
                            all_results[-1]['sigmas'] = sigmas

                            all_results[-1]['biases_rel'] = biases_rel
                            all_results[-1]['sigmas_rel'] = sigmas_rel

                            plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, opts.output_dir, n_samples,
                                                       n_dim, n_experiments, support_size, sigma_outcome,
                                                       opts.covariate_pdf, beta, plot=False)

                    plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts, beta)

    # Define the output file path
    results_file_path = os.path.join(opts.output_dir, results_filename)
    np.save(results_file_path, all_results)

    print(f"All results with noise parameters have been saved to {results_file_path}")

    print("\nDone with all experiments!")

    

    for treatment_effect in treatment_effects:
        filtered_results = [result for result in all_results if result['treatment_effect'] == treatment_effect]

        plot_mse(filtered_results, data_samples, opts, support_sizes, beta_values)

        continue
        
        plot_homl_ica_comparison_gennorm_beta_filter(filtered_results, opts)
        plot_homl_ica_comparison_gennorm_support_filter(filtered_results, opts)
        plot_oml_ica_comparison_gennorm_beta_filter(filtered_results, opts)
        plot_oml_ica_comparison_gennorm_support_filter(filtered_results, opts)

        plot_ica_gennorm_beta_filter(filtered_results, opts)
        plot_ica_gennorm_support_filter_mcc(filtered_results, opts)
        plot_ica_gennorm_beta_filter_bias(filtered_results, opts)

    plot_asymptotic_var_comparison(all_results, opts)

    plot_multi_treatment(all_results, opts, treatment_effects)


def setup_treatment_outcome_coefs(n_dim, opts, outcome_coef_array, outcome_coef_list, outcome_coefficient, support_size,
                                  treatment_coef_array, treatment_coef_list, treatment_coefficient):
    if opts.asymptotic_var is True:
        outcome_coefficient = treatment_coefficient

        treatment_coef_list[0] = treatment_coefficient
        outcome_coef_list[0] = outcome_coefficient

        outcome_support = treatment_support = np.array(range(support_size))

        treatment_coef = treatment_coef_list[:support_size]
        outcome_coef = outcome_coef_list[:support_size]

    else:
        # this implicitly specifies sparsity via restricting the support
        outcome_support = treatment_support = np.random.choice(range(n_dim), size=support_size,
                                                               replace=False)
        treatment_coef = treatment_coef_array[
            treatment_support]  # Support and coefficients for treatment as function of co-variates
        outcome_coef = outcome_coef_array[
            outcome_support]  # Support and coefficients for outcome as function of co-variates
    return outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support


def calc_ica_asymptotic_var(eta_cubed_variance, eta_fourth_moment, eta_third_moment, outcome_coef, sigma_outcome,
                            treatment_coef, treatment_effect):
    eps_second_moment = (1 / 3) * sigma_outcome ** 2
    eps_fourth_moment = (1 / 5) * sigma_outcome ** 4
    eps_sixth_moment = (1 / 7) * sigma_outcome ** 6
    ica_asymptotic_var_num = eps_sixth_moment - eps_fourth_moment ** 2
    ica_asymptotic_var_hyvarinen = ica_asymptotic_var_num / (eps_fourth_moment - 3 * eps_second_moment) ** 2
    eta_excess_kurtosis = eta_fourth_moment - 3
    eta_skewness_squared = eta_third_moment ** 2
    ica_var_coeff = (1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect, ord=2) ** 2)
    ica_asymptotic_var = ica_var_coeff * eta_cubed_variance / eta_excess_kurtosis ** 2
    return eta_excess_kurtosis, eta_skewness_squared, ica_asymptotic_var, ica_asymptotic_var_hyvarinen, ica_asymptotic_var_num, ica_var_coeff


def calc_homl_asymptotic_var(discounts, mean_discount, probs):
    # Calculate moments of the residual distribution
    eta_second_moment = np.dot(probs, (discounts - mean_discount) ** 2)
    eta_third_moment = np.dot(probs, (discounts - mean_discount) ** 3)
    eta_fourth_moment = np.dot(probs, (discounts - mean_discount) ** 4)
    eta_non_gauss_cond = eta_fourth_moment - 3 * eta_second_moment
    # HOML asymptotic variance numerator
    eta_cubed_variance = np.dot(probs, ((discounts - mean_discount) ** 3 - eta_third_moment) ** 2)
    homl_asymptotic_var_num = eta_cubed_variance + 9 * eta_second_moment ** 2 + 3 * eta_fourth_moment * eta_second_moment
    homl_asymptotic_var = homl_asymptotic_var_num / eta_non_gauss_cond ** 2
    return eta_cubed_variance, eta_fourth_moment, eta_non_gauss_cond, eta_second_moment, eta_third_moment, homl_asymptotic_var, homl_asymptotic_var_num


def setup_treatment_noise(rademacher=True):
    if rademacher is False:
        discounts = np.array([0, -.5, -2., -4.])
        probs = np.array([.65, .2, .1, .05])
    else:
        discounts = np.array([1, -1])
        probs = np.array([.5, .5])
    mean_discount = np.dot(discounts, probs)
    eta_sample = lambda x: np.array(
        [discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)])
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
