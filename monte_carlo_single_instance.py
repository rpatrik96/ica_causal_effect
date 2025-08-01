import sys

import matplotlib

from plot_utils import plot_typography

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

    parser.add_argument("--asymptotic_var", dest="asymptotic_var",
                        type=bool, help='Flag to ablate asymptotic variance', default=False)
    parser.add_argument("--tie_sample_dim", dest="tie_sample_dim",
                        type=bool, help='Ties n=d**4', default=False)

    opts = parser.parse_args(args)

    np.random.seed(opts.seed)

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

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
        1.]  # [-20, -10, -5, -2, -1, -.5, -0.2, -.1, .1,  0.2, .5, 1, 2, 5, 10, 20]

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
        results_filename = f'all_results_asymptotic_var_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}.npy'
    else:
        results_filename = f'all_results_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}.npy'
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

        print(f"\nRunning experiments with sample size: {n_samples}")

        for treatment_coefficient in treatment_coefs:
            for outcome_coefficient in outcome_coefs:
                for support_size in support_sizes:
                    print(f"\nRunning experiments with support size: {support_size}")

                    if opts.asymptotic_var:
                        outcome_coefficient = treatment_coefficient

                        treatment_coef_list[0] = treatment_coefficient
                        outcome_coef_list[0] = outcome_coefficient

                        outcome_support = treatment_support = np.array(range(support_size))

                        treatment_coef = treatment_coef_list[:support_size]
                        outcome_coef = outcome_coef_list[:support_size]

                    else:

                        # this implicitly specifies sparsity via restricting the support
                        outcome_support = treatment_support = np.random.choice(range(n_dim), size=support_size, replace=False)


                        # Support and coefficients for treatment as function of co-variates
                        treatment_coef = treatment_coef_array[treatment_support]

                        # Support and coefficients for outcome as function of co-variates
                        outcome_coef = outcome_coef_array[outcome_support]

                    for beta in beta_values:
                        print(f"\nRunning experiments with beta: {beta}")



                        for treatment_effect in treatment_effects:
                            print(f"\nRunning experiments with treatment effect: {treatment_effect}")

                            '''
                            True parameters
                            '''

                            if opts.asymptotic_var and opts.tie_sample_dim:
                                n_samples = support_size ** 4



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
                            print(true_coef_outcome[outcome_support])

                            '''
                            Run the experiments.
                            '''

                            x_sample = setup_covariate_pdf(opts, beta)

                            # Coefficients recovered by orthogonal ML
                            lambda_reg = np.sqrt(np.log(n_dim) / (n_samples))
                            results = [r for r in Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(
                                x_sample(n_samples, n_dim),
                                eta_sample(n_samples),
                                epsilon_sample(n_samples),
                                treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef,
                                eta_second_moment,
                                eta_third_moment, lambda_reg
                            ) for _ in range(n_experiments)) if (opts.check_convergence is False or r[-1] is not None)]

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
                                "ica_mcc": ica_mcc,
                                'treatment_effect': treatment_effect,
                                'treatment_coefficient': treatment_coefficient,
                                'outcome_coefficient': outcome_coefficient,
                            })

                            '''
                            Plotting histograms
                            '''

                            biases, sigmas = plot_method_comparison(ortho_rec_tau, treatment_effect, opts.output_dir,
                                                                    n_samples, n_dim,
                                                                    n_experiments, support_size,
                                                                    sigma_outcome, opts.covariate_pdf, beta,  relative_error=False)

                            biases_rel, sigmas_rel = plot_method_comparison(ortho_rec_tau, treatment_effect, opts.output_dir,
                                                                    n_samples, n_dim,
                                                                    n_experiments, support_size,
                                                                    sigma_outcome, opts.covariate_pdf, beta, relative_error=True)
                            all_results[-1]['biases'] = biases
                            all_results[-1]['sigmas'] = sigmas

                            all_results[-1]['biases_rel'] = biases_rel
                            all_results[-1]['sigmas_rel'] = sigmas_rel

                            plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, opts.output_dir, n_samples,
                                                       n_dim, n_experiments,
                                                       support_size,
                                                       sigma_outcome, opts.covariate_pdf, beta)

                    plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts, beta)

    import seaborn as sns
    import matplotlib.pyplot as plt

    def plot_heatmap(data_matrix, x_labels, y_labels, xlabel, ylabel, filename, output_dir, cmap="coolwarm",
                     center=None):
        plot_typography()
        plt.figure(figsize=(10, 8))
        # Set the midpoint of the color scale to the specified center if provided
        sns.heatmap(data_matrix, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, annot=True, center=center)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def prepare_heatmap_data(all_results, x_key, y_key, value_key, diff_index=None, beta_filter=None,
                             support_size_filter=None, relative_error=False):
        x_values = sorted(set([res[x_key] for res in all_results if
                               (beta_filter is None or res['beta'] == beta_filter) and (
                                           support_size_filter is None or res['support_size'] == support_size_filter)]))
        y_values = sorted(set([res[y_key] for res in all_results if
                               (beta_filter is None or res['beta'] == beta_filter) and (
                                           support_size_filter is None or res['support_size'] == support_size_filter)]),
                          reverse=True)
        data_matrix = np.zeros((len(y_values), len(x_values)))
        data_matrix_mean = np.zeros((len(y_values), len(x_values)))
        data_matrix_std = np.zeros((len(y_values), len(x_values)))

        # Determine the keys based on whether relative error is considered
        value_key_suffix = '_rel' if relative_error else ''
        sigmas_key = 'sigmas' + value_key_suffix

        for i, x_val in enumerate(x_values):
            for j, y_val in enumerate(y_values):
                ica_mean = [res[value_key + value_key_suffix][-1] for res in all_results if (
                            res[x_key] == x_val and res[y_key] == y_val and (
                                beta_filter is None or res['beta'] == beta_filter) and (
                                        support_size_filter is None or res['support_size'] == support_size_filter))][0]
                ica_std = [res[sigmas_key][-1] for res in all_results if (
                            res[x_key] == x_val and res[y_key] == y_val and (
                                beta_filter is None or res['beta'] == beta_filter) and (
                                        support_size_filter is None or res['support_size'] == support_size_filter))][0]
                if diff_index is not None:
                    compare_mean = [res[value_key + value_key_suffix][diff_index] for res in all_results if (
                                res[x_key] == x_val and res[y_key] == y_val and (
                                    beta_filter is None or res['beta'] == beta_filter) and (
                                            support_size_filter is None or res[
                                        'support_size'] == support_size_filter))][0]
                    compare_std = [res[sigmas_key][diff_index] for res in all_results if (
                                res[x_key] == x_val and res[y_key] == y_val and (
                                    beta_filter is None or res['beta'] == beta_filter) and (
                                            support_size_filter is None or res[
                                        'support_size'] == support_size_filter))][0]
                    diffs = [res[value_key + value_key_suffix][-1] - res[value_key + value_key_suffix][diff_index] for res in all_results if (
                                res[x_key] == x_val and res[y_key] == y_val and (
                                    beta_filter is None or res['beta'] == beta_filter) and (
                                            support_size_filter is None or res['support_size'] == support_size_filter))]

                    if (ica_mean + ica_std) < (compare_mean - compare_std):
                        data_matrix[j, i] = -1
                    elif (ica_mean - ica_std) > (compare_mean + compare_std):
                        data_matrix[j, i] = 1
                    else:
                        data_matrix[j, i] = 0

                else:
                    diffs = ica_mean
                if diffs:
                    data_matrix_mean[j, i] = np.nanmean(diffs)
                    data_matrix_std[j, i] = ica_std

        return data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values,

    # Define the output file path
    results_file_path = os.path.join(opts.output_dir, results_filename)
    np.save(results_file_path, all_results)

    print(f"All results with noise parameters have been saved to {results_file_path}")

    print("\nDone with all experiments!")

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        import matplotlib.pyplot as plt

        # Prepare data for plotting
        # ica_mse_matrix, _, _, support_sizes, _ = prepare_heatmap_data(
        #     all_results, 'support_size', 'n_samples', 'biases', beta_filter=1)
        homl_bias_matrix, _, _, _, _ = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=None, beta_filter=1, relative_error=True)

        homl_sigma_matrix, _, _, _, _ = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'sigmas', diff_index=None, beta_filter=1, relative_error=True)

        # Calculate mean MSE across sample sizes for each support size

        homl_mse_matrix = homl_bias_matrix #**2 + homl_sigma_matrix**2
        

        # Plotting
        plt.figure(figsize=(10, 8))
        for idx, sample_size in enumerate(data_samples):
            plt.plot(support_sizes, homl_mse_matrix[idx, :], label=r'$n=' + str(sample_size) + '$', marker='x')
        plt.xlabel(r'$\dim X$')
        plt.ylabel(r'$\frac{|\theta - \hat{\theta}|}{\theta}$')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.22), ncol=len(data_samples)//2)
        plt.grid(True)
        plt.savefig(os.path.join(opts.output_dir, 'mse_vs_support_size_rel.svg'))

        homl_mse_matrix = homl_sigma_matrix

        plt.figure(figsize=(10, 8))
        for idx, sample_size in enumerate(data_samples):
            plt.plot(support_sizes, homl_mse_matrix[idx, :], label=r'$n=' + str(sample_size) + '$', marker='x')
        plt.xlabel(r'$\dim X$')
        plt.ylabel(r'$\sigma_{\frac{|\theta - \hat{\theta}|}{\theta}}$')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.22), ncol=len(data_samples)//2)
        plt.grid(True)
        plt.savefig(os.path.join(opts.output_dir, 'mse_vs_support_size_rel_std.svg'))

        homl_bias_matrix, _, _, _, _ = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=3, beta_filter=1, relative_error=True)
        homl_mse_matrix = homl_bias_matrix

        plt.figure(figsize=(10, 8))
        for idx, sample_size in enumerate(data_samples):
            plt.plot(support_sizes, homl_mse_matrix[idx, :], label=f'(n={sample_size})', marker='x')
        plt.xlabel('Support Size')
        plt.ylabel('ICA-HOML Relative MSE ')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.22), ncol=len(data_samples)//2)
        plt.grid(True)
        plt.savefig(os.path.join(opts.output_dir, 'mse_vs_support_size_rel_ica_vs_homl.svg'))

        homl_bias_matrix, _, _, _, _ = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=None, beta_filter=1, relative_error=False)

        homl_sigma_matrix, _, _, _, _ = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'sigmas', diff_index=None, beta_filter=1, relative_error=False)

        homl_mse_matrix = homl_bias_matrix

        # Plotting
        plt.figure(figsize=(10, 8))
        for idx, sample_size in enumerate(data_samples):
            plt.plot(support_sizes, homl_mse_matrix[idx, :], label=f'(n={sample_size})', marker='x')
        plt.xlabel('Support Size')
        plt.ylabel('ICA MSE ')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.22), ncol=len(data_samples)//2)
        plt.grid(True)
        plt.savefig(os.path.join(opts.output_dir, 'mse_vs_support_size.svg'))



    # Plot heatmaps for comparison with HOML Split, filtered for beta=1
    if opts.asymptotic_var is False:
        bias_diff_matrix_dim_homl_mean, bias_diff_matrix_dim_homl_std, bias_diff_matrix_dim_homl, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=3, beta_filter=1)
        plot_heatmap(bias_diff_matrix_dim_homl_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl_mean.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_homl_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl_std.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_homl, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl.svg', opts.output_dir, center=0)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        bias_diff_matrix_beta_homl_mean, bias_diff_matrix_beta_homl_std, bias_diff_matrix_beta_homl, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', diff_index=3, support_size_filter=10)
        plot_heatmap(bias_diff_matrix_beta_homl_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_mean.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_homl_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_std.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_homl, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl.svg', opts.output_dir, center=0)

    # Plot heatmaps for comparison with OML, filtered for beta=1
    if opts.asymptotic_var is False:
        bias_diff_matrix_dim_oml_mean, bias_diff_matrix_dim_oml_std, bias_diff_matrix_dim_oml, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=0, beta_filter=1)
        plot_heatmap(bias_diff_matrix_dim_oml_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_oml_mean.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_oml_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_oml_std.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_oml, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_oml.svg', opts.output_dir, center=0)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        bias_diff_matrix_beta_oml_mean, bias_diff_matrix_beta_oml_std, bias_diff_matrix_beta_oml, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', diff_index=0, support_size_filter=10)
        plot_heatmap(bias_diff_matrix_beta_oml_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml_mean.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_oml_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml_std.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_oml, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml.svg', opts.output_dir, center=0)

    # ICA error only
    if opts.asymptotic_var is False:
        ica_bias_matrix_dim_mean, ica_bias_matrix_dim_std, ica_bias_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', beta_filter=1)
        plot_heatmap(ica_bias_matrix_dim_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_dim_mean.svg', opts.output_dir, center=None)
        plot_heatmap(ica_bias_matrix_dim_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_dim_std.svg', opts.output_dir, center=None)
        # plot_heatmap(ica_bias_matrix_dim, support_sizes, sample_sizes, r'$\dim X$', r'$n$', 'ica_bias_heatmap_sample_size_vs_dim.svg', opts.output_dir, center=None)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        ica_bias_matrix_beta_mean, ica_bias_matrix_beta_std, ica_bias_matrix_beta, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', support_size_filter=10)
        plot_heatmap(ica_bias_matrix_beta_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_beta_mean.svg', opts.output_dir, center=None)
        plot_heatmap(ica_bias_matrix_beta_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_beta_std.svg', opts.output_dir, center=None)
        # plot_heatmap(ica_bias_matrix_beta, betas, sample_sizes, r'$\beta$', r'$n$', 'ica_bias_heatmap_sample_size_vs_beta.svg', opts.output_dir, center=None)


    if opts.asymptotic_var:
        # Prepare data for the new scatter plot
        x_values_var_diff = [res['ica_asymptotic_var'] - res['homl_asymptotic_var'] for res in all_results]
        x_values_var__hyvarinen_diff = [res['ica_asymptotic_var_hyvarinen'] - res['homl_asymptotic_var'] for res in
                                        all_results]
        y_values_bias_diff = [res['biases'][-1] - res['biases'][3] for res in all_results]
        y_values_sigma_diff = [res['sigmas'][-1] - res['sigmas'][3] for res in all_results]
        colors = [res['beta'] for res in all_results]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var_diff, y_values_bias_diff, c=colors, cmap='viridis', alpha=0.75)
        plt.colorbar(scatter, label='Beta')
        plt.xlabel('Difference in Asymptotic Variance (ICA - HOML)')
        plt.ylabel('Difference in Bias (ICA - HOML)')
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_var_vs_bias_diff.svg'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var_diff, y_values_sigma_diff, c=colors, cmap='viridis', alpha=0.75)
        plt.colorbar(scatter, label='Beta')
        plt.xlabel('Difference in Asymptotic Variance (ICA - HOML)')
        plt.ylabel('Difference in Variance (ICA - HOML)')
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_var_vs_asy_var_diff.svg'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var__hyvarinen_diff, y_values_sigma_diff, c=colors, cmap='viridis', alpha=0.75)
        plt.colorbar(scatter, label='Beta')
        plt.xlabel('Difference in Asymptotic Variance (ICA - HOML) Hyvarinen')
        plt.ylabel('Difference in Variance (ICA - HOML)')
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_var_vs_asy_var_diff_hyvarinen.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Prepare data for the scatter plots
        x_values_sample_size = [res['n_samples'] for res in all_results]
        y_values_ica_asymptotic_var = [res['ica_asymptotic_var'] for res in all_results]
        y_values_ica_asymptotic_var_hyvarinen = [res['ica_asymptotic_var_hyvarinen'] for res in all_results]
        y_values_homl_asymptotic_var = [res['homl_asymptotic_var'] for res in all_results]
        y_values_actual_variance_ica = [res['sigmas'][-1] ** 2 * res['n_samples'] for res in all_results]
        y_values_actual_variance_homl = [res['sigmas'][3] ** 2 * res['n_samples'] for res in all_results]

        # colors = [res['beta'] for res in all_results]

        # Scatter plot for ICA asymptotic and actual variance
        plt.figure(figsize=(10, 8))
        plt.scatter(x_values_sample_size, y_values_ica_asymptotic_var, c='blue', alpha=0.75,
                    label='ICA Asymptotic Variance')
        plt.scatter(x_values_sample_size, y_values_actual_variance_ica, c='red', alpha=0.75, label='ICA Actual Variance')
        plt.xlabel('Sample Size')
        plt.ylabel('Variance')
        plt.legend()
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_sample_size_vs_ica_variance.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Scatter plot for ICA Hyvarinen asymptotic and actual variance
        plt.figure(figsize=(10, 8))
        plt.scatter(x_values_sample_size, y_values_ica_asymptotic_var_hyvarinen, c='blue', alpha=0.75,
                    label='ICA Hyvarinen Asymptotic Variance')
        plt.scatter(x_values_sample_size, y_values_actual_variance_ica, c='red', alpha=0.75, label='ICA Actual Variance')
        plt.xlabel('Sample Size')
        plt.ylabel('Variance')
        plt.legend()
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_sample_size_vs_ica_hyvarinen_variance.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Scatter plot for HOML asymptotic and actual variance
        plt.figure(figsize=(10, 8))
        plt.scatter(x_values_sample_size, y_values_homl_asymptotic_var, c='blue', alpha=0.75,
                    label='HOML Asymptotic Variance')
        plt.scatter(x_values_sample_size, y_values_actual_variance_homl, c='red', alpha=0.75, label='HOML Actual Variance')
        plt.xlabel('Sample Size')
        plt.ylabel('Variance')
        plt.legend()
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_sample_size_vs_homl_variance.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Prepare data for the scatter plot
        x_values_ica_var_coeff = [res['ica_var_coeff'] for res in all_results]
        y_values_ica_biases = [res['biases'][-1] for res in all_results]
        y_values_homl_biases = [res['biases'][3] for res in all_results]
        y_errors_ica = [res['sigmas'][-1] / np.sqrt(res['n_samples']) for res in all_results]
        y_errors_homl = [res['sigmas'][3] / np.sqrt(res['n_samples']) for res in all_results]

        # Scatter plot for ICA and HOML biases with sigmas as error bars

        plt.figure(figsize=(10, 8))
        plt.errorbar(x_values_ica_var_coeff, y_values_ica_biases, yerr=y_errors_ica, fmt='o', color='blue', alpha=0.75,
                     label='ICA')
        plt.errorbar(x_values_ica_var_coeff, y_values_homl_biases, yerr=y_errors_homl, fmt='o', color='red', alpha=0.75,
                     label='HOML')
        plt.xlabel(r'$1+(b+a\theta)^2$')
        plt.xscale('log')
        plt.ylabel(r'$\Vert \theta - \hat{\theta} \Vert_2$')
        plt.legend()
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_ica_var_coeff_vs_biases.svg'), dpi=300, bbox_inches='tight')
        plt.close()

    if len(treatment_effects) > 1:
        # Prepare data for the scatter plot
        x_values_true_theta = [res['treatment_effect'] for res in
                               all_results]  # Assuming the true value of theta is the first element of true_coef_outcome
        y_values_ica_asymptotic_var = [res['ica_asymptotic_var'] for res in all_results]
        y_values_homl_asymptotic_var = [res['homl_asymptotic_var'] for res in all_results]
        y_values_actual_variance_ica = [res['sigmas'][-1] ** 2 * res['n_samples'] for res in all_results]
        y_values_actual_variance_homl = [res['sigmas'][3] ** 2 * res['n_samples'] for res in all_results]

        # Scatter plot for true theta vs variances
        plt.figure(figsize=(10, 8))
        plt.scatter(x_values_true_theta, y_values_ica_asymptotic_var, c='blue', alpha=0.75,
                    label='ICA Asymptotic Variance')
        plt.scatter(x_values_true_theta, y_values_actual_variance_ica, c='green', alpha=0.75,
                    label='ICA Actual Variance')
        plt.scatter(x_values_true_theta, y_values_homl_asymptotic_var, c='red', alpha=0.75,
                    label='HOML Asymptotic Variance')
        plt.scatter(x_values_true_theta, y_values_actual_variance_homl, c='orange', alpha=0.75,
                    label='HOML Actual Variance')
        plt.xlabel('True Value of Theta')
        plt.ylabel('Variance')
        plt.legend()
        plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_true_theta_vs_variances.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Prepare data for the scatter plot of MSEs against theta   
        x_values_true_theta = [res['treatment_effect'] for res in all_results]
        y_values_ica_mse = [res['biases'][-1] for res in all_results]
        y_values_homl_mse = [res['biases'][3] for res in all_results]
        y_errors_ica = [res['sigmas'][-1] for res in all_results]
        y_errors_homl = [res['sigmas'][3] for res in all_results]

        # Bar plot for true theta vs MSEs with variances
        plt.figure(figsize=(10, 8))
        bar_width = 0.35
        index = np.arange(len(x_values_true_theta))

        plt.bar(index, y_values_ica_mse, bar_width, color='blue', alpha=0.75, label='ICA MSE', yerr=y_errors_ica,
                capsize=5)
        plt.bar(index + bar_width, y_values_homl_mse, bar_width, color='red', alpha=0.75, label='HOML MSE',
                yerr=y_errors_homl, capsize=5)

        plt.xticks(ticks=index + bar_width / 2, labels=[f"{theta:.2f}" for theta in x_values_true_theta], rotation=45)
        plt.xlabel('ICA var coeff')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(opts.output_dir, 'bar_plot_true_theta_vs_mses_with_variances.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()


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


def setup_treatment_noise():
    discounts = np.array([0, -.5, -2., -4.])
    probs = np.array([.65, .2, .1, .05])
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
