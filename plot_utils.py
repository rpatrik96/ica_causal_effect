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
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    bias_ortho, sigma_ortho = plot_estimates(np.array(ortho_rec_tau)[:, 0].flatten(), treatment_effect, treatment_effect,
                                             title="Orthogonal estimates")
    plt.subplot(1, 5, 2)
    plot_estimates(np.array(ortho_rec_tau)[:, 1].flatten(), treatment_effect, treatment_effect, title="Second order orthogonal")
    plt.subplot(1, 5, 3)
    plot_estimates(np.array(ortho_rec_tau)[:, 2].flatten(), treatment_effect, treatment_effect,
                   title="Second order orthogonal with estimates")
    plt.subplot(1, 5, 4)
    bias_second, sigma_second = plot_estimates(np.array(ortho_rec_tau)[:, 3].flatten(), treatment_effect, treatment_effect,
                                               title="Second order orthogonal with estimates on third sample")

    plt.subplot(1, 5, 5)
    bias_ica, sigma_ica = plot_estimates(np.array(ortho_rec_tau)[:, 4].flatten(), treatment_effect,treatment_effect,
                                               title="ICA estimate")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}.png'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_outcome)), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir,
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_outcome)), dpi=300, bbox_inches='tight')

    print("Ortho ML MSE: {}".format(bias_ortho ** 2 + sigma_ortho ** 2))
    print("Second Order ML MSE: {}".format(bias_second ** 2 + sigma_ortho ** 2))


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

    plt.savefig(os.path.join(output_dir, filename_base + '.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, filename_base + '.svg'),
                dpi=300, bbox_inches='tight')

    # Save the data
    coef_filename = 'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome)
    joblib.dump(ortho_rec_tau, os.path.join(output_dir, coef_filename))
    joblib.dump(first_stage_mse, os.path.join(output_dir, filename_base))


def plot_error_vs_support(all_results, n_dim, n_samples, opts, treatment_effect):
    # Extract data for plotting
    support_sizes = [result['support_size'] for result in all_results]
    # Calculate mean errors for each method across experiments
    ortho_ml_errors = [np.mean([abs(tau[0] - treatment_effect) for tau in result['ortho_rec_tau']]) for result in
                       all_results]
    robust_ortho_errors = [np.mean([abs(tau[1] - treatment_effect) for tau in result['ortho_rec_tau']]) for result in
                           all_results]
    robust_est_errors = [np.mean([abs(tau[2] - treatment_effect) for tau in result['ortho_rec_tau']]) for result in
                         all_results]
    robust_split_errors = [np.mean([abs(tau[3] - treatment_effect) for tau in result['ortho_rec_tau']]) for result in
                           all_results]
    ica_errors = [np.mean([abs(tau[4] - treatment_effect) for tau in result['ortho_rec_tau']]) for result in
                  all_results]
    plt.figure(figsize=(10, 6))
    plt.plot(support_sizes, ortho_ml_errors, 'o-', label='Orthogonal ML')
    plt.plot(support_sizes, robust_ortho_errors, 's-', label='Robust Orthogonal ML')
    plt.plot(support_sizes, robust_est_errors, '^-', label='Robust Est ML')
    plt.plot(support_sizes, robust_split_errors, 'v-', label='Robust Split ML')
    plt.plot(support_sizes, ica_errors, 'D-', label='ICA')
    plt.xlabel('Support Size')
    plt.ylabel('Absolute Error')
    plt.title('Method Errors vs Support Size')
    plt.legend()
    plt.grid(True)
    # Save plot
    plt.savefig(os.path.join(opts.output_dir, f'errors_vs_support_size_n{n_samples}_d{n_dim}.svg'))
    plt.close()
