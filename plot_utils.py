import os

import joblib
import numpy as np
from matplotlib import pyplot as plt


def plot_estimates(estimate_list, true_tau, treatment_effect, title="Histogram of estimates"):
    # the histogram of the data
    n, bins, patches = plt.hist(estimate_list, 40, facecolor='green', alpha=0.75)
    sigma = float(np.std(estimate_list))
    mu = float(np.mean(estimate_list))
    # add a 'best fit' line
    from scipy.stats import norm
    y = norm.pdf(bins.astype(float), mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.plot([treatment_effect, treatment_effect], [0, np.max(y)], 'b--', label='true effect')
    plt.title("{}. mean: {:.2f}, sigma: {:.2f}".format(title, mu, sigma))
    plt.legend()
    return np.abs(true_tau - mu), sigma


def plot_method_comparison(ortho_rec_tau, treatment_effect, output_dir, n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf):
    # First figure - histograms
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    bias_ortho, sigma_ortho = plot_estimates(np.array(ortho_rec_tau)[:, 0].flatten(), treatment_effect, treatment_effect,
                                             title="Orthogonal estimates")
    plt.subplot(1, 5, 2)
    bias_robust, sigma_robust = plot_estimates(np.array(ortho_rec_tau)[:, 1].flatten(), treatment_effect, treatment_effect, 
                                             title="Second order orthogonal")
    plt.subplot(1, 5, 3)
    bias_est, sigma_est = plot_estimates(np.array(ortho_rec_tau)[:, 2].flatten(), treatment_effect, treatment_effect,
                                       title="Second order orthogonal with estimates")
    plt.subplot(1, 5, 4)
    bias_second, sigma_second = plot_estimates(np.array(ortho_rec_tau)[:, 3].flatten(), treatment_effect, treatment_effect,
                                             title="Second order orthogonal with estimates on third sample")
    plt.subplot(1, 5, 5)
    bias_ica, sigma_ica = plot_estimates(np.array(ortho_rec_tau)[:, 4].flatten(), treatment_effect, treatment_effect,
                                       title="ICA estimate")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_outcome,covariate_pdf)), dpi=300, bbox_inches='tight')

    print("Ortho ML MSE: {}".format(bias_ortho ** 2 + sigma_ortho ** 2))
    print("Second Order ML MSE: {}".format(bias_second ** 2 + sigma_ortho ** 2))
    print(f"ICA: {bias_ica=}, {sigma_ica=}")

    # Return lists of biases and standard deviations for each method
    biases = [bias_ortho, bias_robust, bias_est, bias_second, bias_ica]
    sigmas = [sigma_ortho, sigma_robust, sigma_est, sigma_second, sigma_ica]
    
    return biases, sigmas


def plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, output_dir, n_samples, n_dim, n_experiments, support_size,
                               sigma_outcome, covariate_pdf):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title("Model_treatment error")
    plt.hist(np.array(first_stage_mse)[:, 0].flatten())
    plt.subplot(1, 4, 2)
    plt.hist(np.array(first_stage_mse)[:, 1].flatten())
    plt.title("Model_outcome error")
    plt.subplot(1, 4, 3)
    plt.hist(np.array(first_stage_mse)[:, 2].flatten())
    plt.title("ICA error")
    plt.subplot(1, 4, 4)
    plt.hist(np.array(first_stage_mse)[:, 3].flatten())
    plt.title("ICA MCC")

    filename_base = 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome,covariate_pdf)

    plt.savefig(os.path.join(output_dir, filename_base + '.svg'),
                dpi=300, bbox_inches='tight')

    # Save the data
    coef_filename = 'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf)
    joblib.dump(ortho_rec_tau, os.path.join(output_dir, coef_filename))
    joblib.dump(first_stage_mse, os.path.join(output_dir, filename_base))

    



def plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts):
    # Create error bar plot comparing errors across dimensions
    plt.figure(figsize=(10, 5))
    # plt.xscale('log')
    # plt.yscale('log')
    methods = ['Ortho ML', 'Robust Ortho', 'Robust Est', 'Robust Split', 'ICA']
    # Extract biases and sigmas for each method across all experiments
    method_biases = {method: [] for method in methods}
    method_sigmas = {method: [] for method in methods}
    dimensions = []
    for result in all_results:
        dimensions.append(result['support_size'])
        for i, method in enumerate(methods):
            method_biases[method].append(result['biases'][i])
            method_sigmas[method].append(result['sigmas'][i])
    # Plot error bars for each method
    for method, color in zip(methods, ['b', 'g', 'r', 'c', 'm']):
        plt.errorbar(dimensions, method_biases[method],
                     yerr=method_sigmas[method],
                     fmt='o-', label=method, color=color)
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.title('Method Errors vs Dimension')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.output_dir,
                             'error_by_dimension_n_samples_{}_n_dim_{}_n_exp_{}_pdf_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, opts.covariate_pdf)), dpi=300, bbox_inches='tight')
    plt.close()
