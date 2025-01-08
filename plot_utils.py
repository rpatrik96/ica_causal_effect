import os

import joblib
import numpy as np
from matplotlib import pyplot as plt


def plot_estimates(estimate_list, true_tau, treatment_effect, title="Histogram of estimates"):
    # the histogram of the data
    n, bins, patches = plt.hist(estimate_list, 40, facecolor='green', alpha=0.75)
    sigma = np.std(estimate_list)
    mu = np.mean(estimate_list)
    # add a 'best fit' line
    from scipy.stats import norm
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.plot([treatment_effect, treatment_effect], [0, np.max(y)], 'b--', label='true effect')
    plt.title("{}. mean: {:.2f}, sigma: {:.2f}".format(title, mu, sigma))
    plt.legend()
    return np.abs(true_tau - mu), sigma


def plot_method_comparison(ortho_rec_tau, treatment_effect, output_dir, n_samples, n_dim, n_experiments, support_size, sigma_outcome):
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
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_outcome)), dpi=300, bbox_inches='tight')

    print("Ortho ML MSE: {}".format(bias_ortho ** 2 + sigma_ortho ** 2))
    print("Second Order ML MSE: {}".format(bias_second ** 2 + sigma_ortho ** 2))

    # Return lists of biases and standard deviations for each method
    biases = [bias_ortho, bias_robust, bias_est, bias_second, bias_ica]
    sigmas = [sigma_ortho, sigma_robust, sigma_est, sigma_second, sigma_ica]
    
    return biases, sigmas


def plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, output_dir, n_samples, n_dim, n_experiments, support_size,
                               sigma_outcome):
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

    filename_base = 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome)

    plt.savefig(os.path.join(output_dir, filename_base + '.svg'),
                dpi=300, bbox_inches='tight')

    # Save the data
    coef_filename = 'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome)
    joblib.dump(ortho_rec_tau, os.path.join(output_dir, coef_filename))
    joblib.dump(first_stage_mse, os.path.join(output_dir, filename_base))

    


def plot_error_vs_support(all_results, n_dim, n_samples, opts, treatment_effect, n_experiments):
    # Extract data for plotting
    support_sizes = [result['support_size'] for result in all_results]
    
    # Calculate mean MSE and std dev for each method across experiments
    def get_mse_stats(method_idx):
        mses = []
        std_devs = []
        for result in all_results:
            errors = [(tau[method_idx] - treatment_effect)**2 for tau in result['ortho_rec_tau']]
            mses.append(np.mean(errors))
            std_devs.append(np.std(errors))
        return np.array(mses), np.array(std_devs)

    ortho_ml_mse, ortho_ml_std = get_mse_stats(0)
    robust_ortho_mse, robust_ortho_std = get_mse_stats(1) 
    robust_est_mse, robust_est_std = get_mse_stats(2)
    robust_split_mse, robust_split_std = get_mse_stats(3)
    ica_mse, ica_std = get_mse_stats(4)

    plt.figure(figsize=(10, 6))
    # Plot MSE with error bars showing ±1 std dev
    plt.xscale('log')
    plt.yscale('log')
    plt.errorbar(support_sizes, ortho_ml_mse, yerr=ortho_ml_std, fmt='o-', label='Orthogonal ML')
    plt.errorbar(support_sizes, robust_ortho_mse, yerr=robust_ortho_std, fmt='s-', label='Robust Orthogonal ML')
    plt.errorbar(support_sizes, robust_est_mse, yerr=robust_est_std, fmt='^-', label='Robust Est ML')
    plt.errorbar(support_sizes, robust_split_mse, yerr=robust_split_std, fmt='v-', label='Robust Split ML')
    plt.errorbar(support_sizes, ica_mse, yerr=ica_std, fmt='D-', label='ICA')
    
    plt.xlabel('Support Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Method MSE vs Support Size')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(opts.output_dir, f'mse_vs_support_size_n{n_samples}_d{n_dim}_exp{n_experiments}.svg'))
    plt.close()


def plot_error_bars_from_density_estimate(all_results, n_dim, n_experiments, n_samples, opts):
    # Create error bar plot comparing errors across dimensions
    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')
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
                             'error_by_dimension_n_samples_{}_n_dim_{}_n_exp_{}.svg'.format(
                                 n_samples, n_dim, n_experiments)), dpi=300, bbox_inches='tight')
    plt.close()
