import os

import joblib
import numpy as np
from matplotlib import pyplot as plt


from matplotlib import rc


def plot_typography(
    usetex: bool = False, small: int = 20, medium: int = 24, big: int = 28
):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).
    :param usetex: flag to indicate the usage of LaTeX (needs LaTeX indstalled)
    :param small: small font size in pt (for legends and axes' ticks)
    :param medium: medium font size in pt (for axes' labels)
    :param big: big font size in pt (for titles)
    :return:
    """

    # font family
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    # backend
    rc("text", usetex=usetex)
    rc("font", family="serif")

    # font sizes
    rc("font", size=small)  # controls default text sizes
    rc("axes", titlesize=big)  # fontsize of the axes title
    rc("axes", labelsize=medium)  # fontsize of the x and y labels
    rc("xtick", labelsize=small)  # fontsize of the tick labels
    rc("ytick", labelsize=small)  # fontsize of the tick labels
    rc("legend", fontsize=small)  # legend fontsize
    rc("figure", titlesize=big)  # fontsize of the figure title


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
    mses = [np.linalg.norm(true_tau - estimate) for estimate in estimate_list]
    return np.mean(mses), np.std(mses)


def plot_method_comparison(ortho_rec_tau, treatment_effect, output_dir, n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta):
    # First figure - histograms
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    bias_ortho, sigma_ortho = plot_estimates(np.array(ortho_rec_tau)[:, 0].flatten(), treatment_effect, treatment_effect,
                                             title="OML")
    plt.subplot(1, 5, 2)
    bias_robust, sigma_robust = plot_estimates(np.array(ortho_rec_tau)[:, 1].flatten(), treatment_effect, treatment_effect, 
                                             title="HOML")
    plt.subplot(1, 5, 3)
    bias_est, sigma_est = plot_estimates(np.array(ortho_rec_tau)[:, 2].flatten(), treatment_effect, treatment_effect,
                                       title="HOML (Est.)")
    plt.subplot(1, 5, 4)
    bias_second, sigma_second = plot_estimates(np.array(ortho_rec_tau)[:, 3].flatten(), treatment_effect, treatment_effect,
                                             title="HOML (Split)")
    plt.subplot(1, 5, 5)
    bias_ica, sigma_ica = plot_estimates(np.array(ortho_rec_tau)[:, 4].flatten(), treatment_effect, treatment_effect,
                                       title="ICA")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}_beta_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta)), dpi=300, bbox_inches='tight')

    print("Ortho ML MSE: {}".format(bias_ortho ** 2 + sigma_ortho ** 2))
    print("Second Order ML MSE: {}".format(bias_second ** 2 + sigma_ortho ** 2))
    print(f"ICA: {bias_ica=}, {sigma_ica=}")

    # Return lists of biases and standard deviations for each method
    biases = [bias_ortho, bias_robust, bias_est, bias_second, bias_ica]
    sigmas = [sigma_ortho, sigma_robust, sigma_est, sigma_second, sigma_ica]
    
    return biases, sigmas

def plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, output_dir, n_samples, n_dim, n_experiments, support_size,
                               sigma_outcome, covariate_pdf, beta):
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

    filename_base = 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}_beta_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta)

    plt.savefig(os.path.join(output_dir, filename_base + '.svg'),
                dpi=300, bbox_inches='tight')

    # Save the data
    coef_filename = 'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}_beta_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta)
    joblib.dump(ortho_rec_tau, os.path.join(output_dir, coef_filename))
    joblib.dump(first_stage_mse, os.path.join(output_dir, filename_base))
    



def plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts, beta):
    # Create a high-quality error bar plot comparing errors across dimensions
    plt.figure(figsize=(10, 6))
    methods = ['OML', 'HOML', 'HOML (Est.)', 'HOML (Split)', 'ICA']
    method_biases = {method: [] for method in methods}
    method_sigmas = {method: [] for method in methods}
    dimensions = []

    for result in all_results:
        dimensions.append(result['support_size'])
        for i, method in enumerate(methods):
            method_biases[method].append(result['biases'][i])
            method_sigmas[method].append(result['sigmas'][i])

    # Define a color palette for better visual distinction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot error bars for each method with enhanced styling
    for method, color in zip(methods, colors):
        plt.errorbar(dimensions, method_biases[method],
                     yerr=method_sigmas[method],
                     fmt='o-', label=method, color=color, capsize=4, elinewidth=2, markeredgewidth=2)

    # Use a logarithmic scale for the y-axis to handle large error magnitudes and variances
    plt.yscale('log')

    plt.xlabel(r'$\dim X$')
    plt.ylabel(r'$\Vert\theta-\hat{\theta} \Vert_2$')
    # plt.title('Method Errors vs Dimension', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", linestyle='-.', linewidth=0.5)
    plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])
    plt.yticks()
    plt.tight_layout()

    # Save the plot with a high resolution suitable for conferences
    plt.savefig(os.path.join(opts.output_dir,
                             'error_by_dimension_n_samples_{}_n_dim_{}_n_exp_{}_pdf_{}_beta_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, opts.covariate_pdf, beta)), dpi=600, bbox_inches='tight')
    plt.close()