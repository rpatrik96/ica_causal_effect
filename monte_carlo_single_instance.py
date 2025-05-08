import os
import sys

import matplotlib
import matplotlib.pyplot as plt

from plot_utils import plot_method_comparison, plot_and_save_model_errors, plot_error_bar_stats, plot_typography

matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import Lasso
from main_estimation import all_together_cross_fitting
import argparse
from joblib import delayed, Parallel

from ica import ica_treatment_effect_estimation

from tueplots import bundles, figsizes


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
                        type=int, help='sigma_outcome', default=1)
    parser.add_argument("--covariate_pdf", dest="covariate_pdf",
                        type=str, help='pdf of covariates', default="gennorm")

    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./figures")
    parser.add_argument("--check_convergence", dest="check_convergence",
                        type=bool, help='check convergence', default=False)

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

    # Run experiments for different support sizes and beta values
    support_sizes = [2, 5, 10, 20, 50]
    data_samples = [100, 200, 500, 1000, 2000, 5000]
    beta_values = [1.0] if opts.covariate_pdf != "gennorm" else [0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 4.5, 5]

    import os

    # Define the output file path
    results_file_path = os.path.join(opts.output_dir, f'all_results_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}.npy')

    all_results = []
    for n_samples in data_samples:

        # Check if the results file already exists
        if os.path.exists(results_file_path):
            print(f"Results file {results_file_path} already exists. Loading data instead of rerunning experiments.")
            structured_results = np.load(results_file_path, allow_pickle=True)#.item()
            all_results = structured_results
            break

        print(f"\nRunning experiments with sample size: {n_samples}")

        for beta in beta_values:
            print(f"\nRunning experiments with beta: {beta}")

            for support_size in support_sizes:
                print(f"\nRunning experiments with support size: {support_size}")
                print("Support size of sparse functions: {}".format(support_size))

                '''
                True parameters
                '''

                # Support and coefficients for treatment as function of co-variates
                treatment_support = np.random.choice(range(n_dim), size=support_size, replace=False)
                treatment_coef = np.random.uniform(0, 5, size=support_size)
                print("Support of treatment as function of co-variates: {}".format(treatment_support))
                print("Coefficients of treatment as function of co-variates: {}".format(treatment_coef))

                # Distribution of residuals of treatment
                discounts = np.array([0, -.5, -2., -4.])
                probs = np.array([.65, .2, .1, .05])
                mean_discount = np.dot(discounts, probs)
                eta_sample = lambda x: np.array(
                    [discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)])
                # Calculate moments of the residual distribution
                eta_second_moment = np.dot(probs, (discounts - mean_discount) ** 2)
                eta_third_moment = np.dot(probs, (discounts - mean_discount) ** 3)
                eta_fourth_moment = np.dot(probs, (discounts - mean_discount) ** 4)
                print("Second Moment of Eta: {:.2f}".format(eta_second_moment))
                print("Third Moment of Eta: {:.2f}".format(eta_third_moment))
                non_gauss_cond = eta_fourth_moment - 3 * eta_second_moment ** 2
                print("Non-Gaussianity Criterion, E[eta^4] - 3 E[eta^2]^2: {:.2f}".format(
                    non_gauss_cond))

                eta_skewness_squared = eta_third_moment **2

                # Support and coefficients for outcome as function of co-variates
                outcome_support = treatment_support  # np.random.choice(range(n_dim), size=support_size, replace=False)
                outcome_coef = np.random.uniform(0, 5, size=support_size)
                print("Support of outcome as function of co-variates: {}".format(outcome_support))
                print("Coefficients of outcome as function of co-variates: {}".format(outcome_coef))

                # Distribution of outcome residuals
                sigma_outcome = opts.sigma_outcome
                epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

                treatment_effect = 3.0

                true_coef_treatment = np.zeros(n_dim)
                true_coef_treatment[treatment_support] = treatment_coef
                true_coef_outcome = np.zeros(n_dim)
                true_coef_outcome[outcome_support] = outcome_coef
                true_coef_outcome[treatment_support] += treatment_effect * treatment_coef
                print(true_coef_outcome[outcome_support])
                '''
                Run the experiments.
                '''

                if opts.covariate_pdf == "gauss":
                    x_sample = lambda n_samples, n_dim: np.random.normal(size=(n_samples, n_dim))
                elif opts.covariate_pdf == "uniform":
                    x_sample = lambda n_samples, n_dim: np.random.uniform(size=(n_samples, n_dim))
                if opts.covariate_pdf == "gennorm":
                    from scipy.stats import gennorm
                    loc = 0
                    scale = 1
                    x_sample = lambda n_samples, n_dim: gennorm.rvs(beta, size=(n_samples, n_dim))

                # Coefficients recovered by orthogonal ML
                lambda_reg = np.sqrt(np.log(n_dim) / (n_samples))
                results = [r for r in Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(
                    x_sample(n_samples, n_dim),
                    eta_sample(n_samples),
                    epsilon_sample(n_samples),
                    treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef, eta_second_moment,
                    eta_third_moment, lambda_reg
                ) for _ in range(n_experiments)) if (opts.check_convergence is False or r[-1] is not None)]

                ortho_rec_tau = [[ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml,
                                  ica_treatment_effect_estimate] for
                                 ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_treatment_effect_estimate, _
                                 in results]
                first_stage_mse = [
                    [np.linalg.norm(true_coef_treatment - coef_treatment), np.linalg.norm(true_coef_outcome - coef_outcome),
                     np.linalg.norm(ica_treatment_effect_estimate - treatment_effect), ica_mcc] for
                    _, _, _, _, coef_treatment, coef_outcome, ica_treatment_effect_estimate, ica_mcc in results]

                all_results.append({
                    'n_samples': n_samples,
                    'support_size': support_size,
                    'beta': beta,
                    'ortho_rec_tau': ortho_rec_tau,
                    'first_stage_mse': first_stage_mse,
                    'non_gauss_cond': non_gauss_cond,
                    'eta_skewness_squared' : eta_skewness_squared,
                    'eta_second_moment': eta_second_moment,
                    'eta_third_moment': eta_third_moment,
                    'sigma_outcome': sigma_outcome,
                })

                print(f"Done with experiments for support size {support_size} and beta {beta}!")

                '''
                Plotting histograms
                '''

                biases, sigmas = plot_method_comparison(ortho_rec_tau, treatment_effect, opts.output_dir, n_samples, n_dim,
                                                        n_experiments, support_size,
                                                        sigma_outcome, opts.covariate_pdf, beta)
                all_results[-1]['biases'] = biases
                all_results[-1]['sigmas'] = sigmas

                plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, opts.output_dir, n_samples, n_dim, n_experiments,
                                           support_size,
                                           sigma_outcome, opts.covariate_pdf, beta)

        plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts, beta)


    import seaborn as sns
    import matplotlib.pyplot as plt

    def plot_heatmap(data_matrix, x_labels, y_labels, xlabel, ylabel, filename, output_dir, cmap="coolwarm"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, annot=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def prepare_heatmap_data(all_results, x_key, y_key, value_key, diff_index=None):
        x_values = sorted(set([res[x_key] for res in all_results]))
        y_values = sorted(set([res[y_key] for res in all_results]), reverse=True)
        data_matrix = np.zeros((len(y_values), len(x_values)))

        for i, x_val in enumerate(x_values):
            for j, y_val in enumerate(y_values):
                if diff_index is not None:
                    diffs = [res[value_key][-1] - res[value_key][diff_index] for res in all_results if res[x_key] == x_val and res[y_key] == y_val]
                else:
                    diffs = [res[value_key][-1] for res in all_results if res[x_key] == x_val and res[y_key] == y_val]
                if diffs:
                    data_matrix[j, i] = np.mean(diffs)

        return data_matrix, x_values, y_values

    # Plot heatmaps
    bias_diff_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(all_results,'support_size', 'n_samples',  'biases', diff_index=3)
    plot_heatmap(bias_diff_matrix_dim, support_sizes, sample_sizes, r'$\dim X$', r'$n$', 'bias_diff_heatmap_sample_size_vs_dim.svg', opts.output_dir)

    bias_diff_matrix_beta, betas, sample_sizes = prepare_heatmap_data(all_results, 'beta','n_samples',  'biases', diff_index=3)
    plot_heatmap(bias_diff_matrix_beta, betas, sample_sizes, r'$\beta$', r'$n$', 'bias_diff_heatmap_sample_size_vs_beta.svg', opts.output_dir)

    ica_bias_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(all_results,  'support_size', 'n_samples', 'biases')
    plot_heatmap(ica_bias_matrix_dim, support_sizes, sample_sizes, r'$\dim X$', r'$n$', 'ica_bias_heatmap_sample_size_vs_dim.svg', opts.output_dir)

    ica_bias_matrix_beta, betas, sample_sizes = prepare_heatmap_data(all_results,  'beta', 'n_samples', 'biases')
    plot_heatmap(ica_bias_matrix_beta, betas, sample_sizes, r'$\beta$', r'$n$', 'ica_bias_heatmap_sample_size_vs_beta.svg', opts.output_dir)

    # Prepare data for scatter plot
    filtered_results = all_results
    x_values = [abs(res['non_gauss_cond']) - abs(res['eta_skewness_squared']) for res in filtered_results]
    y_values = [res['sigmas'][-1] - res['sigmas'][3] for res in filtered_results]
    colors = [res['beta'] for res in filtered_results]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_values, y_values, c=colors, cmap='viridis', alpha=0.75)
    plt.colorbar(scatter, label='Beta')
    plt.xlabel('Non-Gaussian Condition')
    plt.ylabel('Sigma Difference (ICA - HOML Split)')
    # plt.title('Scatter Plot: Non-Gaussian Condition vs Sigma Difference')
    plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_non_gauss_vs_sigma_diff.svg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save all results in a structured format using numpy, including noise parameters
    import numpy as np
    import os

    # Define the output file path
    results_file_path = os.path.join(opts.output_dir, f'all_results_n_exp_{n_experiments}_sigma_outcome_{opts.sigma_outcome}_pdf_{opts.covariate_pdf}.npy')




    # Save the results as a numpy file
    np.save(results_file_path, all_results)

    print(f"All results with noise parameters have been saved to {results_file_path}")

    print("\nDone with all experiments!")


if __name__ == "__main__":
    main(sys.argv[1:])
