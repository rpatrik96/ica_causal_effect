import os

import joblib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc


def plot_typography(
        usetex: bool = False, small: int = 28, medium: int = 34, big: int = 40
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


def plot_estimates(estimate_list, true_tau, treatment_effect, title="Histogram of estimates", plot=False,
                   relative_error=False):
    # the histogram of the data
    n, bins, patches = plt.hist(estimate_list, 40, facecolor='green', alpha=0.75)
    sigma = float(np.nanstd(estimate_list))
    mu = float(np.nanmean(estimate_list))
    # add a 'best fit' line
    from scipy.stats import norm
    y = norm.pdf(bins.astype(float), mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    if plot:
        plt.plot([treatment_effect, treatment_effect], [0, np.max(y)], 'b--', label='true effect')
        plt.title("{}. mean: {:.2f}, sigma: {:.2f}".format(title, mu, sigma))
        plt.legend()

    if relative_error:
        mses = [np.linalg.norm((true_tau - estimate) / true_tau) for estimate in estimate_list]
    else:
        mses = [np.linalg.norm(true_tau - estimate) for estimate in estimate_list]

    return np.nanmean(mses), np.nanstd(mses)


def plot_method_comparison(ortho_rec_tau, treatment_effect, output_dir, n_samples, n_dim, n_experiments, support_size,
                           sigma_outcome, covariate_pdf, beta, plot=False, relative_error=False, verbose=False):
    # Create subfolder for the experiment
    experiment_dir = os.path.join(output_dir, f"recovered_coefficients")
    os.makedirs(experiment_dir, exist_ok=True)

    # First figure - histograms
    if plot:
        plt.figure(figsize=(25, 5))
        plt.subplot(1, 5, 1)

    array = np.array(ortho_rec_tau)

    if np.isnan(array).any():
        if verbose:
            print("There is a NaN in the array.")

    bias_ortho, sigma_ortho = plot_estimates(array[:, 0].flatten(), treatment_effect, treatment_effect,
                                             title="OML", plot=plot, relative_error=relative_error)
    if plot:
        plt.subplot(1, 5, 2)
    bias_robust, sigma_robust = plot_estimates(array[:, 1].flatten(), treatment_effect, treatment_effect,
                                               title="HOML", plot=plot, relative_error=relative_error)
    if plot:
        plt.subplot(1, 5, 3)
    bias_est, sigma_est = plot_estimates(array[:, 2].flatten(), treatment_effect, treatment_effect,
                                         title="HOML (Est.)", plot=plot, relative_error=relative_error)
    if plot:
        plt.subplot(1, 5, 4)
    bias_second, sigma_second = plot_estimates(array[:, 3].flatten(), treatment_effect, treatment_effect,
                                               title="HOML (Split)", plot=plot, relative_error=relative_error)
    if plot:
        plt.subplot(1, 5, 5)
    bias_ica, sigma_ica = plot_estimates(array[:, 4].flatten(), treatment_effect, treatment_effect,
                                         title="ICA", plot=plot, relative_error=relative_error)

    if verbose:
        print(f"ICA estimates{array[:, 4].flatten()}")

    if plot:
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir,
                                 'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}_beta_{}.svg'.format(
                                     n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf,
                                     beta)), dpi=300, bbox_inches='tight')

    if verbose:
        print("Ortho ML MSE: {}".format(bias_ortho ** 2 + sigma_ortho ** 2))
        print("Second Order ML MSE: {}".format(bias_second ** 2 + sigma_ortho ** 2))
        print(f"ICA: {bias_ica=}, {sigma_ica=}")

    # Return lists of biases and standard deviations for each method
    biases = [bias_ortho, bias_robust, bias_est, bias_second, bias_ica]
    sigmas = [sigma_ortho, sigma_robust, sigma_est, sigma_second, sigma_ica]

    return biases, sigmas


def plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, output_dir, n_samples, n_dim, n_experiments,
                               support_size,
                               sigma_outcome, covariate_pdf, beta, plot=False, save=False):
    # Create subfolder for the experiment
    experiment_dir = os.path.join(output_dir, f"model_errors")
    os.makedirs(experiment_dir, exist_ok=True)

    filename_base = 'model_errors'

    if plot:
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

        plt.savefig(os.path.join(experiment_dir, filename_base + '.svg'),
                    dpi=300, bbox_inches='tight')

    # Save the data
    if save:
        coef_filename = 'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}_beta_{}'.format(
            n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta)
        joblib.dump(ortho_rec_tau, os.path.join(experiment_dir, coef_filename))
        joblib.dump(first_stage_mse, os.path.join(experiment_dir, filename_base))


def plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts, beta):
    # Create subfolder for the experiment
    experiment_dir = os.path.join(opts.output_dir, f"error_bars")
    os.makedirs(experiment_dir, exist_ok=True)

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
    plt.savefig(os.path.join(experiment_dir,
                             'error_by_dimension_n_samples_{}_n_dim_{}_n_exp_{}_pdf_{}_beta_{}.svg'.format(
                                 n_samples, n_dim, n_experiments, opts.covariate_pdf, beta)), dpi=600,
                bbox_inches='tight')
    plt.close()


def plot_ica_gennorm_beta_filter_bias(all_results, opts, ):

    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        ica_bias_matrix_beta_mean, ica_bias_matrix_beta_std, ica_bias_matrix_beta, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', support_size_filter=10)
        plot_heatmap(ica_bias_matrix_beta_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_beta_mean.svg', experiment_dir, center=None)
        plot_heatmap(ica_bias_matrix_beta_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_beta_std.svg', experiment_dir, center=None)
        # plot_heatmap(ica_bias_matrix_beta, betas, sample_sizes, r'$\beta$', r'$n$', 'ica_bias_heatmap_sample_size_vs_beta.svg', experiment_dir, center=None)


def plot_ica_gennorm_support_filter_mcc(all_results, opts, ):

    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.asymptotic_var is False:
        ica_mcc_matrix_dim_mean, ica_mcc_matrix_dim_std, ica_mcc_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'first_stage_mse', support_size_filter=10)
        plot_heatmap(ica_mcc_matrix_dim_mean, support_sizes, sample_sizes, r'$\beta$', r'$n$',
                     'ica_mcc_heatmap_sample_size_vs_beta_mean.svg', experiment_dir, center=None)
        plot_heatmap(ica_mcc_matrix_dim_std, support_sizes, sample_sizes, r'$\beta$', r'$n$',
                     'ica_mcc_heatmap_sample_size_vs_beta_std.svg', experiment_dir, center=None)
        plot_heatmap(ica_mcc_matrix_dim, support_sizes, sample_sizes, r'$\beta$', r'$n$',
                     'ica_mcc_heatmap_sample_size_vs_beta.svg', experiment_dir, center=None)


def plot_ica_gennorm_beta_filter(all_results, opts, ):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    # ICA error only
    if opts.asymptotic_var is False:
        ica_bias_matrix_dim_mean, ica_bias_matrix_dim_std, ica_bias_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', beta_filter=1)
        plot_heatmap(ica_bias_matrix_dim_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_dim_mean.svg', experiment_dir, center=None)
        plot_heatmap(ica_bias_matrix_dim_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_dim_std.svg', experiment_dir, center=None)
        # plot_heatmap(ica_bias_matrix_dim, support_sizes, sample_sizes, r'$\dim X$', r'$n$', 'ica_bias_heatmap_sample_size_vs_dim.svg', experiment_dir, center=None)


def plot_oml_ica_comparison_gennorm_support_filter(all_results, opts, ):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        bias_diff_matrix_beta_oml_mean, bias_diff_matrix_beta_oml_std, bias_diff_matrix_beta_oml, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', diff_index=0, support_size_filter=10)
        plot_heatmap(bias_diff_matrix_beta_oml_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml_mean.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_oml_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml_std.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_oml, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml.svg', experiment_dir, center=0)


def plot_oml_ica_comparison_gennorm_beta_filter(all_results, opts, ):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Plot heatmaps for comparison with OML, filtered for beta=1
    if opts.asymptotic_var is False:
        bias_diff_matrix_dim_oml_mean, bias_diff_matrix_dim_oml_std, bias_diff_matrix_dim_oml, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=0, beta_filter=1)
        plot_heatmap(bias_diff_matrix_dim_oml_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_oml_mean.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_oml_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_oml_std.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_oml, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_oml.svg', experiment_dir, center=0)


def plot_homl_ica_comparison_gennorm_support_filter(all_results, opts, ):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        bias_diff_matrix_beta_homl_mean, bias_diff_matrix_beta_homl_std, bias_diff_matrix_beta_homl, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', diff_index=3, support_size_filter=10)
        plot_heatmap(bias_diff_matrix_beta_homl_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_mean.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_homl_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_std.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_homl, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl.svg', experiment_dir, center=0)


def plot_homl_ica_comparison_gennorm_beta_filter(all_results, opts, ):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Plot heatmaps for comparison with HOML Split, filtered for beta=1
    if opts.asymptotic_var is False:
        bias_diff_matrix_dim_homl_mean, bias_diff_matrix_dim_homl_std, bias_diff_matrix_dim_homl, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', diff_index=3, beta_filter=1)
        plot_heatmap(bias_diff_matrix_dim_homl_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl_mean.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_homl_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl_std.svg', experiment_dir, center=0)
        plot_heatmap(bias_diff_matrix_dim_homl, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl.svg', experiment_dir, center=0)


def plot_multi_treatment(all_results, opts, treatment_effects):
    experiment_dir = os.path.join(opts.output_dir, "multi_treatment")
    os.makedirs(experiment_dir, exist_ok=True)

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
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_true_theta_vs_variances.svg'), dpi=300,
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
        plt.savefig(os.path.join(experiment_dir, 'bar_plot_true_theta_vs_mses_with_variances.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()


def plot_asymptotic_var_comparison(all_results, opts, asymptotic_var_versions=False):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "asymptotic_var_comparison", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        # Prepare data for the scatter plot
        x_values_ica_var_coeff = [res['ica_var_coeff'] for res in all_results]
        treatment_coef = [res['treatment_coefficient'] for res in all_results]
        outcome_coef = [res['outcome_coefficient'] for res in all_results]
        y_values_ica_biases = [res['biases'][-1] for res in all_results]
        y_values_homl_biases = [res['biases'][3] for res in all_results]
        y_errors_ica = [res['sigmas'][-1] / np.sqrt(res['n_samples']) for res in all_results]
        y_errors_homl = [res['sigmas'][3] / np.sqrt(res['n_samples']) for res in all_results]

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))

        # Subplot 1: x-axis is x_values_ica_var_coeff
        axs.errorbar(x_values_ica_var_coeff, y_values_ica_biases, yerr=y_errors_ica, fmt='o', color='blue',
                        alpha=0.75,
                        label='ICA')
        axs.errorbar(x_values_ica_var_coeff, y_values_homl_biases, yerr=y_errors_homl, fmt='o', color='red',
                        alpha=0.75,
                        label='HOML')
        axs.set_xlabel(r'$1+\Vert b+a\theta\Vert_2^2$')
        axs.set_xscale('log')
        axs.set_ylabel(r'$|\theta-\hat{\theta}|$')
        axs.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'gennorm_asymp_var.svg'), dpi=300, bbox_inches='tight')
        plt.close()

        return 0

    if asymptotic_var_versions is True:

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
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_var_vs_bias_diff.svg'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var_diff, y_values_sigma_diff, c=colors, cmap='viridis', alpha=0.75)
        plt.colorbar(scatter, label='Beta')
        plt.xlabel('Difference in Asymptotic Variance (ICA - HOML)')
        plt.ylabel('Difference in Variance (ICA - HOML)')
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_var_vs_asy_var_diff.svg'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var__hyvarinen_diff, y_values_sigma_diff, c=colors, cmap='viridis', alpha=0.75)
        plt.colorbar(scatter, label='Beta')
        plt.xlabel('Difference in Asymptotic Variance (ICA - HOML) Hyvarinen')
        plt.ylabel('Difference in Variance (ICA - HOML)')
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_var_vs_asy_var_diff_hyvarinen.svg'), dpi=300,
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
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_sample_size_vs_ica_variance.svg'), dpi=300,
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
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_sample_size_vs_ica_hyvarinen_variance.svg'), dpi=300,
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
        plt.savefig(os.path.join(experiment_dir, 'scatter_plot_sample_size_vs_homl_variance.svg'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    # Prepare data for the scatter plot
    x_values_ica_var_coeff = [res['ica_var_coeff'] for res in all_results]
    treatment_coef = [res['treatment_coefficient'] for res in all_results]
    outcome_coef = [res['outcome_coefficient'] for res in all_results]
    y_values_ica_biases = [res['biases'][-1] for res in all_results]
    y_values_homl_biases = [res['biases'][3] for res in all_results]
    y_errors_ica = [res['sigmas'][-1] / np.sqrt(res['n_samples']) for res in all_results]
    y_errors_homl = [res['sigmas'][3] / np.sqrt(res['n_samples']) for res in all_results]

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: x-axis is x_values_ica_var_coeff
    axs[0].errorbar(x_values_ica_var_coeff, y_values_ica_biases, yerr=y_errors_ica, fmt='o', color='blue', alpha=0.75,
                    label='ICA')
    axs[0].errorbar(x_values_ica_var_coeff, y_values_homl_biases, yerr=y_errors_homl, fmt='o', color='red', alpha=0.75,
                    label='HOML')
    axs[0].set_xlabel(r'$1+(b+a\theta)^2$')
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$|\theta-\hat{\theta}|$')
    axs[0].legend()

    # Subplot 2: x-axis is treatment_coef
    avg_y_values_ica_biases_treatment = [
        np.mean([res['biases'][-1] for res in all_results if res['treatment_coefficient'] == t_coef]) for t_coef in
        treatment_coef]
    avg_y_values_homl_biases_treatment = [
        np.mean([res['biases'][3] for res in all_results if res['treatment_coefficient'] == t_coef]) for t_coef in
        treatment_coef]
    avg_y_errors_ica_treatment = [np.mean([res['sigmas'][-1] / np.sqrt(res['n_samples']) for res in all_results if
                                           res['treatment_coefficient'] == t_coef]) for t_coef in treatment_coef]
    avg_y_errors_homl_treatment = [np.mean(
        [res['sigmas'][3] / np.sqrt(res['n_samples']) for res in all_results if res['treatment_coefficient'] == t_coef])
                                   for t_coef in treatment_coef]

    axs[1].errorbar(treatment_coef, y_values_ica_biases, yerr=y_errors_ica, fmt='o', color='blue', alpha=0.75,
                    label='ICA')
    # axs[1].errorbar(treatment_coef, avg_y_values_ica_biases_treatment, yerr=avg_y_errors_ica_treatment, fmt='o', color='blue', alpha=0.75, label='ICA')
    axs[1].errorbar(treatment_coef, y_values_homl_biases, yerr=y_errors_homl, fmt='o', color='red', alpha=0.75,
                    label='HOML')
    # axs[1].errorbar(treatment_coef, avg_y_values_homl_biases_treatment, yerr=avg_y_errors_homl_treatment, fmt='o', color='red', alpha=0.75, label='HOML')
    axs[1].set_xlabel('Treatment Coefficient')
    axs[1].set_ylabel(r'$|\theta-\hat{\theta}|$')
    axs[1].legend()

    # Subplot 3: x-axis is outcome_coef
    avg_y_values_ica_biases_outcome = [
        np.mean([res['biases'][-1] for res in all_results if res['outcome_coefficient'] == o_coef]) for o_coef in
        outcome_coef]
    avg_y_values_homl_biases_outcome = [
        np.mean([res['biases'][3] for res in all_results if res['outcome_coefficient'] == o_coef]) for o_coef in
        outcome_coef]
    avg_y_errors_ica_outcome = [np.mean(
        [res['sigmas'][-1] / np.sqrt(res['n_samples']) for res in all_results if res['outcome_coefficient'] == o_coef])
                                for o_coef in outcome_coef]
    avg_y_errors_homl_outcome = [np.mean(
        [res['sigmas'][3] / np.sqrt(res['n_samples']) for res in all_results if res['outcome_coefficient'] == o_coef])
                                 for o_coef in outcome_coef]

    axs[2].errorbar(outcome_coef, y_values_ica_biases, yerr=y_errors_ica, fmt='o', color='blue', alpha=0.75,
                    label='ICA')
    # axs[2].errorbar(outcome_coef, avg_y_values_ica_biases_outcome, yerr=avg_y_errors_ica_outcome, fmt='o', color='blue', alpha=0.75, label='ICA')
    axs[2].errorbar(outcome_coef, y_values_homl_biases, yerr=y_errors_homl, fmt='o', color='red', alpha=0.75,
                    label='HOML')
    # axs[2].errorbar(outcome_coef, avg_y_values_homl_biases_outcome, yerr=avg_y_errors_homl_outcome, fmt='o', color='red', alpha=0.75, label='HOML')
    axs[2].set_xlabel('Outcome Coefficient')
    axs[2].set_ylabel(r'$|\theta-\hat{\theta}|$')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'errorbar_subplots_biases.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Prepare data for the heatmap plots
    treatment_coef = [res['treatment_coefficient'] for res in all_results]
    outcome_coef = [res['outcome_coefficient'] for res in all_results]

    # Create a grid for the heatmap
    treatment_coef_unique = np.unique(treatment_coef)
    outcome_coef_unique = np.unique(outcome_coef)

    # Initialize matrices for biases and their differences
    ica_bias_matrix = np.zeros((len(treatment_coef_unique), len(outcome_coef_unique)))
    homl_bias_matrix = np.zeros((len(treatment_coef_unique), len(outcome_coef_unique)))
    bias_difference_matrix = np.zeros((len(treatment_coef_unique), len(outcome_coef_unique)))

    # Fill the matrices with bias values and their differences
    for i, t_coef in enumerate(treatment_coef_unique):
        for j, o_coef in enumerate(outcome_coef_unique):
            indices = [k for k, (t, o) in enumerate(zip(treatment_coef, outcome_coef)) if t == t_coef and o == o_coef]
            if indices:
                ica_bias_matrix[i, j] = np.mean([y_values_ica_biases[k] for k in indices])
                homl_bias_matrix[i, j] = np.mean([y_values_homl_biases[k] for k in indices])
                bias_difference_matrix[i, j] = homl_bias_matrix[i, j] - ica_bias_matrix[i, j]

    # Plot heatmap for ICA biases
    plt.figure(figsize=(10, 8))
    plt.imshow(ica_bias_matrix, aspect='auto', cmap='viridis', origin='lower',
               extent=[outcome_coef_unique.min(), outcome_coef_unique.max(), treatment_coef_unique.min(),
                       treatment_coef_unique.max()])
    plt.colorbar(label='ICA Error')
    plt.xlabel('Outcome Coefficient')
    plt.ylabel('Treatment Coefficient')
    plt.savefig(os.path.join(experiment_dir, 'heatmap_ica_biases.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot heatmap for HOML biases
    plt.figure(figsize=(10, 8))
    plt.imshow(homl_bias_matrix, aspect='auto', cmap='viridis', origin='lower',
               extent=[outcome_coef_unique.min(), outcome_coef_unique.max(), treatment_coef_unique.min(),
                       treatment_coef_unique.max()])
    plt.colorbar(label='HOML Bias')
    plt.xlabel('Outcome Coefficient')
    plt.ylabel('Treatment Coefficient')
    plt.title('Heatmap of HOML Biases')
    plt.savefig(os.path.join(experiment_dir, 'heatmap_homl_biases.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot heatmap for the difference in biases (HOML - ICA)
    plt.figure(figsize=(10, 8))
    plt.imshow(bias_difference_matrix, aspect='auto', cmap='coolwarm', origin='lower',
               extent=[outcome_coef_unique.min(), outcome_coef_unique.max(), treatment_coef_unique.min(),
                       treatment_coef_unique.max()])
    plt.colorbar(label='Error Difference (HOML - ICA)')
    plt.xlabel('Outcome Coefficient')
    plt.ylabel('Treatment Coefficient')
    plt.savefig(os.path.join(experiment_dir, 'heatmap_bias_differences.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create a discrete heatmap for bias differences
    discrete_bias_difference_matrix = np.zeros((len(treatment_coef_unique), len(outcome_coef_unique)))

    # todo: check whether this is the correct STD

    for i, t_coef in enumerate(treatment_coef_unique):
        for j, o_coef in enumerate(outcome_coef_unique):
            indices = [k for k, (t, o) in enumerate(zip(treatment_coef, outcome_coef)) if t == t_coef and o == o_coef]
            if indices:
                ica_mean = np.mean([y_values_ica_biases[k] for k in indices])
                ica_std = np.mean([y_errors_ica[k] for k in indices])
                compare_mean = np.mean([y_values_homl_biases[k] for k in indices])
                compare_std = np.mean([y_errors_homl[k] for k in indices])

                if (ica_mean + ica_std) < (compare_mean - compare_std):
                    discrete_bias_difference_matrix[i, j] = 1
                elif (ica_mean - ica_std) > (compare_mean + compare_std):
                    discrete_bias_difference_matrix[i, j] = -1
                else:
                    discrete_bias_difference_matrix[i, j] = 0

    # Plot discrete heatmap for the difference in biases
    plt.figure(figsize=(10, 8))
    plt.imshow(discrete_bias_difference_matrix, aspect='auto', cmap='bwr', origin='lower',
               extent=[outcome_coef_unique.min(), outcome_coef_unique.max(), treatment_coef_unique.min(),
                       treatment_coef_unique.max()])
    plt.colorbar(ticks=[-1, 0, 1], label='Error Difference (HOML - ICA)')
    plt.xlabel('Outcome Coefficient')
    plt.ylabel('Treatment Coefficient')
    plt.savefig(os.path.join(experiment_dir, 'discrete_heatmap_bias_differences.svg'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_mse(all_results, data_samples, opts, support_sizes, beta_values):
    treatment_effect_value = all_results[0]['treatment_effect']
    experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:


        homl_bias_matrix, _, _, support_sizes, sample_sizes = prepare_heatmap_data(all_results, 'support_size', 'n_samples', 'biases',
                                                            diff_index=3, beta_filter=1, relative_error=True)

        plot_heatmap(homl_bias_matrix, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_dim_homl_mean_rel.svg', experiment_dir, center=0)



        homl_bias_matrix, _, _, betas, sample_sizes = prepare_heatmap_data(all_results, 'beta', 'n_samples', 'biases',
                                                            diff_index=3, support_size_filter=10, relative_error=True)


        plot_heatmap(homl_bias_matrix, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_mean_rel.svg', experiment_dir, center=0)



        plt.close()


def plot_heatmap(data_matrix, x_labels, y_labels, xlabel, ylabel, filename, output_dir, cmap="coolwarm",
                 center=None):
    plot_typography()
    # plot_typography(small=20, medium=24, big=30)
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
            ica_mean = [res[value_key + value_key_suffix][-1] if value_key != "first_stage_mse" else np.mean(
                [z[-1] for z in res[value_key]]) for res in all_results if (
                                res[x_key] == x_val and res[y_key] == y_val and (
                                beta_filter is None or res['beta'] == beta_filter) and (
                                        support_size_filter is None or res[
                                    'support_size'] == support_size_filter))][0]
            ica_std = \
                [res[sigmas_key][-1] if value_key != "first_stage_mse" else np.std([z[-1] for z in res[value_key]]) for
                 res in all_results if (res[x_key] == x_val and res[y_key] == y_val and (
                        beta_filter is None or res['beta'] == beta_filter) and (support_size_filter is None or res[
                    'support_size'] == support_size_filter))][0]

            if diff_index is not None:
                compare_mean = [res[value_key + value_key_suffix][diff_index] for res in all_results if (
                        res[x_key] == x_val and res[y_key] == y_val and (
                        beta_filter is None or res['beta'] == beta_filter) and (
                                support_size_filter is None or res['support_size'] == support_size_filter))][0]
                compare_std = [res[sigmas_key][diff_index] for res in all_results if (
                        res[x_key] == x_val and res[y_key] == y_val and (
                        beta_filter is None or res['beta'] == beta_filter) and (
                                support_size_filter is None or res['support_size'] == support_size_filter))][0]
                diffs = [res[value_key + value_key_suffix][-1] - res[value_key + value_key_suffix][diff_index] for
                         res in all_results if (res[x_key] == x_val and res[y_key] == y_val and (
                            beta_filter is None or res['beta'] == beta_filter) and (
                                                        support_size_filter is None or res[
                                                    'support_size'] == support_size_filter))]

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
