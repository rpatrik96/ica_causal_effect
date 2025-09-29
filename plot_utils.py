import os

import joblib
import numpy as np
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


def plot_estimates(estimate_list, true_tau, treatment_effect, title="Histogram of estimates", plot=False, relative_error=False):
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


def plot_method_comparison(ortho_rec_tau, treatment_effect, output_dir, n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta, plot=False, relative_error=False):
    # First figure - histograms
    if plot:
        plt.figure(figsize=(25, 5))
        plt.subplot(1, 5, 1)

    array = np.array(ortho_rec_tau)

    if np.isnan(array).any():
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

    print(f"ICA estimates{array[:, 4].flatten()}")

    if plot:
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
                               sigma_outcome, covariate_pdf, beta, plot=False, save=False):
    filename_base = 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_outcome_{}_pdf_{}_beta_{}'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_outcome, covariate_pdf, beta)

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



        plt.savefig(os.path.join(output_dir, filename_base + '.svg'),
                    dpi=300, bbox_inches='tight')

    # Save the data
    if save:
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


def plot_ica_gennorm_beta_filter_bias(all_results, opts, plot_heatmap, prepare_heatmap_data):
    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        ica_bias_matrix_beta_mean, ica_bias_matrix_beta_std, ica_bias_matrix_beta, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', support_size_filter=10)
        plot_heatmap(ica_bias_matrix_beta_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_beta_mean.svg', opts.output_dir, center=None)
        plot_heatmap(ica_bias_matrix_beta_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_beta_std.svg', opts.output_dir, center=None)
        # plot_heatmap(ica_bias_matrix_beta, betas, sample_sizes, r'$\beta$', r'$n$', 'ica_bias_heatmap_sample_size_vs_beta.svg', opts.output_dir, center=None)


def plot_ica_gennorm_support_filter_mcc(all_results, opts, plot_heatmap, prepare_heatmap_data):
    if opts.asymptotic_var is False:
        ica_mcc_matrix_dim_mean, ica_mcc_matrix_dim_std, ica_mcc_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'first_stage_mse', support_size_filter=10)
        plot_heatmap(ica_mcc_matrix_dim_mean, support_sizes, sample_sizes, r'$\beta$', r'$n$',
                     'ica_mcc_heatmap_sample_size_vs_beta_mean.svg', opts.output_dir, center=None)
        plot_heatmap(ica_mcc_matrix_dim_std, support_sizes, sample_sizes, r'$\beta$', r'$n$',
                     'ica_mcc_heatmap_sample_size_vs_beta_std.svg', opts.output_dir, center=None)
        plot_heatmap(ica_mcc_matrix_dim, support_sizes, sample_sizes, r'$\beta$', r'$n$',
                     'ica_mcc_heatmap_sample_size_vs_beta.svg', opts.output_dir, center=None)


def plot_ica_gennorm_beta_filter(all_results, opts, plot_heatmap, prepare_heatmap_data):
    # ICA error only
    if opts.asymptotic_var is False:
        ica_bias_matrix_dim_mean, ica_bias_matrix_dim_std, ica_bias_matrix_dim, support_sizes, sample_sizes = prepare_heatmap_data(
            all_results, 'support_size', 'n_samples', 'biases', beta_filter=1)
        plot_heatmap(ica_bias_matrix_dim_mean, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_dim_mean.svg', opts.output_dir, center=None)
        plot_heatmap(ica_bias_matrix_dim_std, support_sizes, sample_sizes, r'$\dim X$', r'$n$',
                     'ica_bias_heatmap_sample_size_vs_dim_std.svg', opts.output_dir, center=None)
        # plot_heatmap(ica_bias_matrix_dim, support_sizes, sample_sizes, r'$\dim X$', r'$n$', 'ica_bias_heatmap_sample_size_vs_dim.svg', opts.output_dir, center=None)


def plot_oml_ica_comparison_gennorm_support_filter(all_results, opts, plot_heatmap, prepare_heatmap_data):
    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        bias_diff_matrix_beta_oml_mean, bias_diff_matrix_beta_oml_std, bias_diff_matrix_beta_oml, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', diff_index=0, support_size_filter=10)
        plot_heatmap(bias_diff_matrix_beta_oml_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml_mean.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_oml_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml_std.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_oml, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_oml.svg', opts.output_dir, center=0)


def plot_oml_ica_comparison_gennorm_beta_filter(all_results, opts, plot_heatmap, prepare_heatmap_data):
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


def plot_homl_ica_comparison_gennorm_support_filter(all_results, opts, plot_heatmap, prepare_heatmap_data):
    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        bias_diff_matrix_beta_homl_mean, bias_diff_matrix_beta_homl_std, bias_diff_matrix_beta_homl, betas, sample_sizes = prepare_heatmap_data(
            all_results, 'beta', 'n_samples', 'biases', diff_index=3, support_size_filter=10)
        plot_heatmap(bias_diff_matrix_beta_homl_mean, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_mean.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_homl_std, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl_std.svg', opts.output_dir, center=0)
        plot_heatmap(bias_diff_matrix_beta_homl, betas, sample_sizes, r'$\beta$', r'$n$',
                     'bias_diff_heatmap_sample_size_vs_beta_homl.svg', opts.output_dir, center=0)


def plot_homl_ica_comparison_gennorm_beta_filter(all_results, opts, plot_heatmap, prepare_heatmap_data):
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


def plot_multi_treatment(all_results, opts, treatment_effects):
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


def plot_asymptotic_var_comparison(all_results, opts):
    if opts.asymptotic_var:
        return 0


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
    plt.ylabel(r'$\mathrm{Var}(\hat{\theta})$')
    plt.legend()
    plt.savefig(os.path.join(opts.output_dir, 'scatter_plot_ica_var_coeff_vs_biases.svg'), dpi=300, bbox_inches='tight')
    plt.close()
