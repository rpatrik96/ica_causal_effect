import os

import joblib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_typography(usetex: bool = False, small: int = 28, medium: int = 34, big: int = 40, preset: str = None):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).

    Args:
        usetex: flag to indicate the usage of LaTeX (needs LaTeX installed)
        small: small font size in pt (for legends and axes' ticks)
        medium: medium font size in pt (for axes' labels)
        big: big font size in pt (for titles)
        preset: If "publication", uses smaller fonts suitable for multi-panel figures:
                small=9, medium=10, big=11. Overrides small/medium/big if set.
    """
    # Apply preset if specified
    if preset == "publication":
        small, medium, big = 9, 10, 11

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


def add_legend_outside(ax, loc="right", **kwargs):
    """
    Add legend outside the plot area to avoid occlusion.

    Args:
        ax: Matplotlib axes object
        loc: Position for legend - "right" (default), "top", or "bottom"
        **kwargs: Additional arguments passed to ax.legend()

    Returns:
        The axes object for chaining
    """
    if loc == "right":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), **kwargs)
    elif loc == "top":
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=99, **kwargs)
    elif loc == "bottom":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=99, **kwargs)
    else:
        # Fallback to standard legend if unknown loc
        ax.legend(loc=loc, **kwargs)
    return ax


def plot_estimates(
    estimate_list, true_tau, treatment_effect, title="Histogram of estimates", plot=False, relative_error=False
):
    # the histogram of the data
    _, bins, _ = plt.hist(estimate_list, 40, facecolor="green", alpha=0.75)
    sigma = float(np.nanstd(estimate_list))
    mu = float(np.nanmean(estimate_list))
    # add a 'best fit' line
    from scipy.stats import norm

    y = norm.pdf(bins.astype(float), mu, sigma)
    plt.plot(bins, y, "r--", linewidth=1)
    if plot:
        plt.plot([treatment_effect, treatment_effect], [0, np.max(y)], "b--", label="true effect")
        plt.title(f"{title}. mean: {mu:.2f}, sigma: {sigma:.2f}")
        plt.legend()

    if relative_error:
        mses = [np.linalg.norm((true_tau - estimate) / true_tau) for estimate in estimate_list]
    else:
        mses = [np.linalg.norm(true_tau - estimate) for estimate in estimate_list]

    return np.nanmean(mses), np.nanstd(mses)


def plot_method_comparison(
    ortho_rec_tau,
    treatment_effect,
    output_dir,
    n_samples,
    n_dim,
    n_experiments,
    support_size,
    sigma_outcome,
    covariate_pdf,
    beta,
    plot=False,
    relative_error=False,
    verbose=False,
):
    # Create subfolder for the experiment
    experiment_dir = os.path.join(output_dir, "recovered_coefficients")
    os.makedirs(experiment_dir, exist_ok=True)

    # First figure - histograms
    if plot:
        plt.figure(figsize=(25, 5))
        plt.subplot(1, 5, 1)

    array = np.array(ortho_rec_tau)

    if np.isnan(array).any():
        if verbose:
            print("There is a NaN in the array.")

    bias_ortho, sigma_ortho = plot_estimates(
        array[:, 0].flatten(), treatment_effect, treatment_effect, title="OML", plot=plot, relative_error=relative_error
    )
    if plot:
        plt.subplot(1, 5, 2)
    bias_robust, sigma_robust = plot_estimates(
        array[:, 1].flatten(),
        treatment_effect,
        treatment_effect,
        title="Higher-order OML",
        plot=plot,
        relative_error=relative_error,
    )
    if plot:
        plt.subplot(1, 5, 3)
    bias_est, sigma_est = plot_estimates(
        array[:, 2].flatten(),
        treatment_effect,
        treatment_effect,
        title="Higher-order OML (Est.)",
        plot=plot,
        relative_error=relative_error,
    )
    if plot:
        plt.subplot(1, 5, 4)
    bias_second, sigma_second = plot_estimates(
        array[:, 3].flatten(),
        treatment_effect,
        treatment_effect,
        title="Higher-order OML (Split)",
        plot=plot,
        relative_error=relative_error,
    )
    if plot:
        plt.subplot(1, 5, 5)
    bias_ica, sigma_ica = plot_estimates(
        array[:, 4].flatten(), treatment_effect, treatment_effect, title="ICA", plot=plot, relative_error=relative_error
    )

    if verbose:
        print(f"ICA estimates{array[:, 4].flatten()}")

    if plot:
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                experiment_dir,
                f"recovered_coefficients_from_each_method_n_samples_{n_samples}_n_dim_{n_dim}"
                f"_n_exp_{n_experiments}_support_{support_size}_sigma_outcome_{sigma_outcome}"
                f"_pdf_{covariate_pdf}_beta_{beta}.svg",
            ),
            dpi=300,
            bbox_inches="tight",
        )

    if verbose:
        print(f"Ortho ML MSE: {bias_ortho**2 + sigma_ortho**2}")
        print(f"Second Order ML MSE: {bias_second**2 + sigma_ortho**2}")
        print(f"ICA: {bias_ica=}, {sigma_ica=}")

    # Return lists of biases and standard deviations for each method
    biases = [bias_ortho, bias_robust, bias_est, bias_second, bias_ica]
    sigmas = [sigma_ortho, sigma_robust, sigma_est, sigma_second, sigma_ica]

    return biases, sigmas


def plot_and_save_model_errors(
    first_stage_mse,
    ortho_rec_tau,
    output_dir,
    n_samples,
    n_dim,
    n_experiments,
    support_size,
    sigma_outcome,
    covariate_pdf,
    beta,
    plot=False,
    save=False,
):
    # Create subfolder for the experiment
    experiment_dir = os.path.join(output_dir, "model_errors")
    os.makedirs(experiment_dir, exist_ok=True)

    filename_base = "model_errors"

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

        plt.savefig(os.path.join(experiment_dir, filename_base + ".svg"), dpi=300, bbox_inches="tight")

    # Save the data
    if save:
        coef_filename = (
            f"recovered_coefficients_from_each_method_n_samples_{n_samples}_n_dim_{n_dim}"
            f"_n_exp_{n_experiments}_support_{support_size}_sigma_outcome_{sigma_outcome}"
            f"_pdf_{covariate_pdf}_beta_{beta}"
        )
        joblib.dump(ortho_rec_tau, os.path.join(experiment_dir, coef_filename))
        joblib.dump(first_stage_mse, os.path.join(experiment_dir, filename_base))


def plot_error_bar_stats(all_results, n_dim, n_experiments, n_samples, opts, beta):
    # Create subfolder for the experiment
    experiment_dir = os.path.join(opts.output_dir, "error_bars")
    os.makedirs(experiment_dir, exist_ok=True)

    # Create a high-quality error bar plot comparing errors across dimensions
    plt.figure(figsize=(10, 6))
    methods = ["OML", "Higher-order OML", "Higher-order OML (Est.)", "Higher-order OML (Split)", "ICA"]
    method_biases = {method: [] for method in methods}
    method_sigmas = {method: [] for method in methods}
    dimensions = []

    for result in all_results:
        dimensions.append(result["support_size"])
        for i, method in enumerate(methods):
            method_biases[method].append(result["biases"][i])
            method_sigmas[method].append(result["sigmas"][i])

    # Define a color palette for better visual distinction
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot error bars for each method with enhanced styling
    for method, color in zip(methods, colors):
        plt.errorbar(
            dimensions,
            method_biases[method],
            yerr=method_sigmas[method],
            fmt="o-",
            label=method,
            color=color,
            capsize=4,
            elinewidth=2,
            markeredgewidth=2,
        )

    # Use a logarithmic scale for the y-axis to handle large error magnitudes and variances
    plt.yscale("log")

    plt.xlabel(r"Covariate dimension $d$")
    plt.ylabel(r"$\Vert\theta-\hat{\theta} \Vert_2$")
    # plt.title('Method Errors vs Dimension', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", linestyle="-.", linewidth=0.5)
    plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])
    plt.yticks()
    # plt.tight_layout()

    # Save the plot with a high resolution suitable for conferences
    plt.savefig(
        os.path.join(
            experiment_dir,
            f"error_by_dimension_n_samples_{n_samples}_n_dim_{n_dim}_n_exp_{n_experiments}"
            f"_pdf_{opts.covariate_pdf}_beta_{beta}.svg",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def plot_gennorm(
    all_results,
    opts,
    filter_type="support",
    filter_value=10,
    compare_method="oml",
    plot_type="bias",
    plot_binary=False,
    save_subfolder=True,
    ica_var_threshold=100,
    filter_below=True,
    filter_ica_var_coeff=False,
):
    if save_subfolder:
        treatment_effect_value = all_results[0]["treatment_effect"]
        experiment_dir = os.path.join(opts.output_dir, "gennorm", f"treatment_effect_{treatment_effect_value}")
    else:
        experiment_dir = os.path.join(opts.output_dir, "gennorm")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.asymptotic_var is False:

        if filter_ica_var_coeff and compare_method is not None:
            # Filter results based on the ica_var_threshold and filter_below
            if filter_below:
                filtered_results = [res for res in all_results if res["ica_var_coeff"] <= ica_var_threshold]
            else:
                filtered_results = [res for res in all_results if res["ica_var_coeff"] > ica_var_threshold]
            all_results = filtered_results

        # Determine the diff_index based on the compare_method
        if compare_method == "oml":
            diff_index = 0
        elif compare_method == "homl":
            diff_index = 3
        elif compare_method is None:
            diff_index = None
        else:
            raise ValueError("Invalid compare_method. Use 'oml', 'homl', or None for no comparison.")

        # Determine the value_key based on the plot_type
        if plot_type == "bias":
            value_key = "biases"
            filename_prefix = "bias"
        elif plot_type == "rmse":
            value_key = "biases"
            filename_prefix = "rmse"
        elif plot_type == "mcc":
            value_key = "first_stage_mse"
            filename_prefix = "mcc"
        else:
            raise ValueError("Invalid plot_type. Use 'bias', 'rmse', or 'mcc'.")

        compute_rmse = plot_type == "rmse"

        if filter_type == "support":
            data_matrix_mean, data_matrix_std, data_matrix, x_labels, sample_sizes, data_matrix_rmse = (
                prepare_heatmap_data(
                    all_results,
                    "beta",
                    "n_samples",
                    value_key,
                    diff_index=diff_index,
                    support_size_filter=filter_value,
                    compute_rmse=compute_rmse,
                )
            )
            x_label = r"Gen. normal param. $\beta$"
            filename_suffix = f'beta_{compare_method if compare_method else "ica"}'
        elif filter_type == "beta":
            data_matrix_mean, data_matrix_std, data_matrix, x_labels, sample_sizes, data_matrix_rmse = (
                prepare_heatmap_data(
                    all_results,
                    "support_size",
                    "n_samples",
                    value_key,
                    diff_index=diff_index,
                    beta_filter=filter_value,
                    compute_rmse=compute_rmse,
                )
            )
            x_label = r"Covariate dimension $d$"
            filename_suffix = f'dim_{compare_method if compare_method else "ica"}'
        else:
            raise ValueError("Invalid filter_type. Use 'support' or 'beta'.")

        if filter_ica_var_coeff:
            filename_suffix = f'{filename_suffix}_filtered_{ica_var_threshold}_{"below" if filter_below else "above"}'

        # For RMSE plots, use the RMSE data matrix; otherwise use mean
        if plot_type == "rmse" and data_matrix_rmse is not None:
            plot_heatmap(
                data_matrix_rmse,
                x_labels,
                sample_sizes,
                x_label,
                r"Sample size $n$",
                f"{filename_prefix}_sample_size_vs_{filename_suffix}.svg",
                experiment_dir,
                center=0,
            )
        else:
            plot_heatmap(
                data_matrix_mean,
                x_labels,
                sample_sizes,
                x_label,
                r"Sample size $n$",
                f"{filename_prefix}_sample_size_vs_{filename_suffix}_mean.svg",
                experiment_dir,
                center=0,
            )

            if plot_binary:
                plot_heatmap(
                    data_matrix,
                    x_labels,
                    sample_sizes,
                    x_label,
                    r"Sample size $n$",
                    f"{filename_prefix}_sample_size_vs_{filename_suffix}.svg",
                    experiment_dir,
                    center=0,
                )

            if compare_method is None:
                plot_heatmap(
                    data_matrix_std,
                    x_labels,
                    sample_sizes,
                    x_label,
                    r"Sample size $n$",
                    f"{filename_prefix}_sample_size_vs_{filename_suffix}_std.svg",
                    experiment_dir,
                    center=0,
                )


def plot_multi_treatment(all_results, opts, treatment_effects):
    experiment_dir = os.path.join(opts.output_dir, "multi_treatment")
    os.makedirs(experiment_dir, exist_ok=True)

    if len(treatment_effects) > 1:
        # Prepare data for the scatter plot
        x_values_true_theta = [
            res["treatment_effect"] for res in all_results
        ]  # Assuming the true value of theta is the first element of true_coef_outcome
        y_values_ica_asymptotic_var = [res["ica_asymptotic_var"] for res in all_results]
        y_values_homl_asymptotic_var = [res["homl_asymptotic_var"] for res in all_results]
        y_values_actual_variance_ica = [res["sigmas"][-1] ** 2 * res["n_samples"] for res in all_results]
        y_values_actual_variance_homl = [res["sigmas"][3] ** 2 * res["n_samples"] for res in all_results]

        # Scatter plot for true theta vs variances
        plt.figure(figsize=(10, 8))
        plt.scatter(
            x_values_true_theta, y_values_ica_asymptotic_var, c="blue", alpha=0.75, label="ICA Asymptotic Variance"
        )
        plt.scatter(
            x_values_true_theta, y_values_actual_variance_ica, c="green", alpha=0.75, label="ICA Actual Variance"
        )
        plt.scatter(
            x_values_true_theta,
            y_values_homl_asymptotic_var,
            c="red",
            alpha=0.75,
            label="Higher-order OML Asymptotic Variance",
        )
        plt.scatter(
            x_values_true_theta,
            y_values_actual_variance_homl,
            c="orange",
            alpha=0.75,
            label="Higher-order OML Actual Variance",
        )
        plt.xlabel("True Value of Theta")
        plt.ylabel("Variance")
        plt.legend()
        plt.savefig(
            os.path.join(experiment_dir, "scatter_plot_true_theta_vs_variances.svg"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Prepare data for the scatter plot of MSEs against theta
        x_values_true_theta = [res["treatment_effect"] for res in all_results]
        y_values_ica_mse = [res["biases"][-1] for res in all_results]
        y_values_homl_mse = [res["biases"][3] for res in all_results]
        y_errors_ica = [res["sigmas"][-1] for res in all_results]
        y_errors_homl = [res["sigmas"][3] for res in all_results]

        # Bar plot for true theta vs MSEs with variances
        plt.figure(figsize=(10, 8))
        bar_width = 0.35
        index = np.arange(len(x_values_true_theta))

        plt.bar(
            index, y_values_ica_mse, bar_width, color="blue", alpha=0.75, label="ICA MSE", yerr=y_errors_ica, capsize=5
        )
        plt.bar(
            index + bar_width,
            y_values_homl_mse,
            bar_width,
            color="red",
            alpha=0.75,
            label="Higher-order OML MSE",
            yerr=y_errors_homl,
            capsize=5,
        )

        plt.xticks(ticks=index + bar_width / 2, labels=[f"{theta:.2f}" for theta in x_values_true_theta], rotation=45)
        plt.xlabel("ICA var coeff")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(experiment_dir, "bar_plot_true_theta_vs_mses_with_variances.svg"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def plot_asymptotic_var_comparison(
    all_results, opts, asymptotic_var_versions=False, save_subfolder=True, coeff_plots=False
):
    treatment_effect_value = all_results[0]["treatment_effect"]
    if save_subfolder:
        experiment_dir = os.path.join(
            opts.output_dir, "asymptotic_var_comparison", f"treatment_effect_{treatment_effect_value}"
        )
    else:
        experiment_dir = os.path.join(opts.output_dir, "asymptotic_var_comparison")
    os.makedirs(experiment_dir, exist_ok=True)

    if opts.covariate_pdf == "gennorm" and opts.asymptotic_var is False:
        # Prepare data for the scatter plot
        # filtered_results = [res for res in all_results if res['ica_var_coeff'] <= 50]

        x_values_ica_var_coeff = [res["ica_var_coeff"] for res in all_results]
        treatment_coef = [res["treatment_coefficient"] for res in all_results]
        outcome_coef = [res["outcome_coefficient"] for res in all_results]
        y_values_ica_biases = [res["biases"][-1] for res in all_results]
        y_values_homl_biases = [res["biases"][3] for res in all_results]
        y_errors_ica = [res["sigmas"][-1] / np.sqrt(res["n_samples"]) for res in all_results]
        y_errors_homl = [res["sigmas"][3] / np.sqrt(res["n_samples"]) for res in all_results]

        # Create a figure with 3 subplots
        _, axs = plt.subplots(1, 1, figsize=(10, 8))

        # Subplot 1: x-axis is x_values_ica_var_coeff
        axs.errorbar(
            x_values_ica_var_coeff,
            y_values_ica_biases,
            yerr=y_errors_ica,
            fmt="o",
            color="blue",
            alpha=0.75,
            label="ICA",
        )
        axs.errorbar(
            x_values_ica_var_coeff,
            y_values_homl_biases,
            yerr=y_errors_homl,
            fmt="o",
            color="red",
            alpha=0.75,
            label="Higher-order OML",
        )
        axs.set_xlabel(r"$1+\Vert b+a\theta\Vert_2^2$")
        axs.set_xscale("log")
        axs.set_ylabel(r"Mean squared $|\theta-\hat{\theta}|$")
        axs.legend()

        # plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "gennorm_asymp_var.svg"), dpi=300, bbox_inches="tight")
        plt.close()

    if asymptotic_var_versions is True:
        # Prepare data for the new scatter plot
        x_values_var_diff = [res["ica_asymptotic_var"] - res["homl_asymptotic_var"] for res in all_results]
        x_values_var__hyvarinen_diff = [
            res["ica_asymptotic_var_hyvarinen"] - res["homl_asymptotic_var"] for res in all_results
        ]
        y_values_bias_diff = [res["biases"][-1] - res["biases"][3] for res in all_results]
        y_values_sigma_diff = [res["sigmas"][-1] - res["sigmas"][3] for res in all_results]
        colors = [res["beta"] for res in all_results]
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var_diff, y_values_bias_diff, c=colors, cmap="viridis", alpha=0.75)
        plt.colorbar(scatter, label="Beta")
        plt.xlabel("Difference in Asymptotic Variance (ICA - HOML)")
        plt.ylabel("Difference in Bias (ICA - HOML)")
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(experiment_dir, "scatter_plot_var_vs_bias_diff.svg"), dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var_diff, y_values_sigma_diff, c=colors, cmap="viridis", alpha=0.75)
        plt.colorbar(scatter, label="Beta")
        plt.xlabel("Difference in Asymptotic Variance (ICA - HOML)")
        plt.ylabel("Difference in Variance (ICA - HOML)")
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(os.path.join(experiment_dir, "scatter_plot_var_vs_asy_var_diff.svg"), dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_values_var__hyvarinen_diff, y_values_sigma_diff, c=colors, cmap="viridis", alpha=0.75)
        plt.colorbar(scatter, label="Beta")
        plt.xlabel("Difference in Asymptotic Variance (ICA - HOML) Hyvarinen")
        plt.ylabel("Difference in Variance (ICA - HOML)")
        # plt.title('Scatter Plot: Asymptotic Variance vs Bias Difference')
        plt.savefig(
            os.path.join(experiment_dir, "scatter_plot_var_vs_asy_var_diff_hyvarinen.svg"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        # Prepare data for the scatter plots
        x_values_sample_size = [res["n_samples"] for res in all_results]
        y_values_ica_asymptotic_var = [res["ica_asymptotic_var"] for res in all_results]
        y_values_ica_asymptotic_var_hyvarinen = [res["ica_asymptotic_var_hyvarinen"] for res in all_results]
        y_values_homl_asymptotic_var = [res["homl_asymptotic_var"] for res in all_results]
        y_values_actual_variance_ica = [res["sigmas"][-1] ** 2 * res["n_samples"] for res in all_results]
        y_values_actual_variance_homl = [res["sigmas"][3] ** 2 * res["n_samples"] for res in all_results]
        # colors = [res['beta'] for res in all_results]
        # Scatter plot for ICA asymptotic and actual variance
        plt.figure(figsize=(10, 8))
        plt.scatter(
            x_values_sample_size, y_values_ica_asymptotic_var, c="blue", alpha=0.75, label="ICA Asymptotic Variance"
        )
        plt.scatter(
            x_values_sample_size, y_values_actual_variance_ica, c="red", alpha=0.75, label="ICA Actual Variance"
        )
        plt.xlabel("Sample Size")
        plt.ylabel("Variance")
        plt.legend()
        plt.savefig(
            os.path.join(experiment_dir, "scatter_plot_sample_size_vs_ica_variance.svg"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        # Scatter plot for ICA Hyvarinen asymptotic and actual variance
        plt.figure(figsize=(10, 8))
        plt.scatter(
            x_values_sample_size,
            y_values_ica_asymptotic_var_hyvarinen,
            c="blue",
            alpha=0.75,
            label="ICA Hyvarinen Asymptotic Variance",
        )
        plt.scatter(
            x_values_sample_size, y_values_actual_variance_ica, c="red", alpha=0.75, label="ICA Actual Variance"
        )
        plt.xlabel("Sample Size")
        plt.ylabel("Variance")
        plt.legend()
        plt.savefig(
            os.path.join(experiment_dir, "scatter_plot_sample_size_vs_ica_hyvarinen_variance.svg"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        # Scatter plot for HOML asymptotic and actual variance
        plt.figure(figsize=(10, 8))
        plt.scatter(
            x_values_sample_size, y_values_homl_asymptotic_var, c="blue", alpha=0.75, label="HOML Asymptotic Variance"
        )
        plt.scatter(
            x_values_sample_size, y_values_actual_variance_homl, c="red", alpha=0.75, label="HOML Actual Variance"
        )
        plt.xlabel("Sample Size")
        plt.ylabel("Variance")
        plt.legend()
        plt.savefig(
            os.path.join(experiment_dir, "scatter_plot_sample_size_vs_homl_variance.svg"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    if opts.asymptotic_var is True or opts.scalar_coeffs is True:
        # Prepare data for the scatter plot
        x_values_ica_var_coeff = [res["ica_var_coeff"] for res in all_results]
        treatment_coef = [res["treatment_coefficient"] for res in all_results]
        outcome_coef = [res["outcome_coefficient"] for res in all_results]
        y_values_ica_biases = [res["biases"][-1] for res in all_results]
        y_values_homl_biases = [res["biases"][3] for res in all_results]
        y_errors_ica = [res["sigmas"][-1] / np.sqrt(res["n_samples"]) for res in all_results]
        y_errors_homl = [res["sigmas"][3] / np.sqrt(res["n_samples"]) for res in all_results]

        # Create a separate plot for Subplot 1
        plt.figure(figsize=(8, 6))
        plt.errorbar(
            x_values_ica_var_coeff,
            y_values_ica_biases,
            yerr=y_errors_ica,
            fmt="o",
            color="blue",
            alpha=0.75,
            label="ICA",
        )
        plt.errorbar(
            x_values_ica_var_coeff,
            y_values_homl_biases,
            yerr=y_errors_homl,
            fmt="o",
            color="red",
            alpha=0.75,
            label="HOML",
        )
        plt.xlabel(r"$1+(b+a\theta)^2$")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel(r"Mean squared $|\theta-\hat{\theta}|$")
        plt.legend(
            # loc='upper center',ncol=2, bbox_to_anchor=(0., -0.15)
        )
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "errorbar_plot_ica_vs_homl.svg"), dpi=300, bbox_inches="tight")
        plt.close()

        # Create a violin plot grouped by ica_var_coeff
        plt.figure(figsize=(8, 6))
        bins = np.linspace(min(x_values_ica_var_coeff), max(x_values_ica_var_coeff), 11)
        data_grouped = {i: ([], []) for i in range(10)}

        # Bin the data into separate structures
        data_grouped = {i: {"ica": [], "homl": []} for i in range(10)}

        for i, coeff in enumerate(x_values_ica_var_coeff):
            bin_index = min(np.digitize(coeff, bins) - 1, 9)  # Ensure bin_index is within range
            data_grouped[bin_index]["ica"].append(y_values_ica_biases[i])
            data_grouped[bin_index]["homl"].append(y_values_homl_biases[i])

        # Prepare positions and data for plotting
        positions_ica = [i * 3 for i in range(10) if data_grouped[i]["ica"]]
        positions_homl = [i * 3 + 1 for i in range(10) if data_grouped[i]["homl"]]
        data_ica = [data_grouped[i]["ica"] for i in range(10) if data_grouped[i]["ica"]]
        data_homl = [data_grouped[i]["homl"] for i in range(10) if data_grouped[i]["homl"]]
        # labels = [f'ICA {bins[i]:.2f}' for i in range(10) if data_grouped[i]['ica']] + \
        #  [f'HOML {bins[i]:.2f}' for i in range(10) if data_grouped[i]['homl']]

        if data_ica or data_homl:  # Check if there is any data to plot
            if data_ica:
                parts_ica = plt.violinplot(data_ica, positions=positions_ica, showmeans=True, showmedians=True)
                for pc in parts_ica["bodies"]:
                    pc.set_facecolor("blue")  # ICA color
            if data_homl:
                parts_homl = plt.violinplot(data_homl, positions=positions_homl, showmeans=True, showmedians=True)
                for pc in parts_homl["bodies"]:
                    pc.set_facecolor("red")  # HOML color

            plt.yscale("log")
            plt.ylabel(r"Mean squared $|\theta-\hat{\theta}|$")
            plt.title("Violin plot of ICA vs HOML biases grouped by ICA Var Coeff")
            plt.legend(
                handles=[
                    plt.Line2D([0], [0], color="blue", lw=4, label="ICA"),
                    plt.Line2D([0], [0], color="red", lw=4, label="HOML"),
                ],
                title="Method",
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(experiment_dir, "violin_plot_ica_vs_homl_grouped.svg"), dpi=300, bbox_inches="tight"
            )
        plt.close()

        # Create subplots for treatment and outcome coefficients if coeff_plots is true
        if coeff_plots:
            _, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Subplot 2: x-axis is treatment_coef
            axs[0].errorbar(
                treatment_coef, y_values_ica_biases, yerr=y_errors_ica, fmt="o", color="blue", alpha=0.75, label="ICA"
            )
            axs[0].errorbar(
                treatment_coef,
                y_values_homl_biases,
                yerr=y_errors_homl,
                fmt="o",
                color="red",
                alpha=0.75,
                label="Higher-order OML",
            )
            axs[0].set_xlabel("Treatment Coefficient")
            axs[0].set_ylabel(r"$|\theta-\hat{\theta}|$")
            axs[0].legend()

            # Subplot 3: x-axis is outcome_coef
            axs[1].errorbar(
                outcome_coef, y_values_ica_biases, yerr=y_errors_ica, fmt="o", color="blue", alpha=0.75, label="ICA"
            )
            axs[1].errorbar(
                outcome_coef,
                y_values_homl_biases,
                yerr=y_errors_homl,
                fmt="o",
                color="red",
                alpha=0.75,
                label="Higher-order OML",
            )
            axs[1].set_xlabel("Outcome Coefficient")
            axs[1].set_ylabel(r"$|\theta-\hat{\theta}|$")
            axs[1].legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(experiment_dir, "errorbar_subplots_coefficients.svg"), dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Prepare data for the heatmap plots
        treatment_coef = [res["treatment_coefficient"] for res in all_results]
        outcome_coef = [res["outcome_coefficient"] for res in all_results]

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
                indices = [
                    k for k, (t, o) in enumerate(zip(treatment_coef, outcome_coef)) if t == t_coef and o == o_coef
                ]
                if indices:
                    ica_bias_matrix[i, j] = np.mean([y_values_ica_biases[k] for k in indices])
                    homl_bias_matrix[i, j] = np.mean([y_values_homl_biases[k] for k in indices])
                    bias_difference_matrix[i, j] = homl_bias_matrix[i, j] - ica_bias_matrix[i, j]

        # Plot heatmap for ICA biases
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            ica_bias_matrix,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[
                outcome_coef_unique.min(),
                outcome_coef_unique.max(),
                treatment_coef_unique.min(),
                treatment_coef_unique.max(),
            ],
        )
        # Add value annotations
        for i, _ in enumerate(treatment_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
            for j, _ in enumerate(outcome_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
                ax.text(
                    outcome_coef_unique[j],
                    treatment_coef_unique[i],
                    f"{ica_bias_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if ica_bias_matrix[i, j] > ica_bias_matrix.max() / 2 else "black",
                    fontsize=10,
                )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label="ICA Error")
        ax.set_xlabel("Outcome Coefficient")
        ax.set_ylabel("Treatment Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "heatmap_ica_biases.svg"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot heatmap for HOML biases
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            homl_bias_matrix,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[
                outcome_coef_unique.min(),
                outcome_coef_unique.max(),
                treatment_coef_unique.min(),
                treatment_coef_unique.max(),
            ],
        )
        # Add value annotations
        for i, _ in enumerate(treatment_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
            for j, _ in enumerate(outcome_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
                ax.text(
                    outcome_coef_unique[j],
                    treatment_coef_unique[i],
                    f"{homl_bias_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if homl_bias_matrix[i, j] > homl_bias_matrix.max() / 2 else "black",
                    fontsize=10,
                )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label="HOML Bias")
        ax.set_xlabel("Outcome Coefficient")
        ax.set_ylabel("Treatment Coefficient")
        ax.set_title("Heatmap of HOML Biases")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "heatmap_homl_biases.svg"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot heatmap for the difference in biases (HOML - ICA)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            bias_difference_matrix,
            aspect="auto",
            cmap="coolwarm",
            origin="lower",
            extent=[
                outcome_coef_unique.min(),
                outcome_coef_unique.max(),
                treatment_coef_unique.min(),
                treatment_coef_unique.max(),
            ],
        )
        # Add value annotations
        vmax = max(abs(bias_difference_matrix.min()), abs(bias_difference_matrix.max()))
        for i, _ in enumerate(treatment_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
            for j, _ in enumerate(outcome_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
                ax.text(
                    outcome_coef_unique[j],
                    treatment_coef_unique[i],
                    f"{bias_difference_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(bias_difference_matrix[i, j]) > vmax / 2 else "black",
                    fontsize=10,
                )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label="Error Difference (HOML - ICA)")
        ax.set_xlabel("Outcome Coefficient")
        ax.set_ylabel("Treatment Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "heatmap_bias_differences.svg"), dpi=300, bbox_inches="tight")
        plt.close()

        # Create a discrete heatmap for bias differences
        discrete_bias_difference_matrix = np.zeros((len(treatment_coef_unique), len(outcome_coef_unique)))

        # todo: check whether this is the correct STD

        for i, t_coef in enumerate(treatment_coef_unique):
            for j, o_coef in enumerate(outcome_coef_unique):
                indices = [
                    k for k, (t, o) in enumerate(zip(treatment_coef, outcome_coef)) if t == t_coef and o == o_coef
                ]
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
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            discrete_bias_difference_matrix,
            aspect="auto",
            cmap="bwr",
            origin="lower",
            extent=[
                outcome_coef_unique.min(),
                outcome_coef_unique.max(),
                treatment_coef_unique.min(),
                treatment_coef_unique.max(),
            ],
        )
        # Add value annotations
        for i, _ in enumerate(treatment_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
            for j, _ in enumerate(outcome_coef_unique):  # pylint: disable=unnecessary-list-index-lookup
                val = discrete_bias_difference_matrix[i, j]
                label = "HOML" if val > 0 else "ICA" if val < 0 else "="
                ax.text(
                    outcome_coef_unique[j],
                    treatment_coef_unique[i],
                    label,
                    ha="center",
                    va="center",
                    color="white" if abs(val) > 0.5 else "black",
                    fontsize=10,
                )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, ticks=[-1, 0, 1], label="Error Difference (HOML - ICA)")
        ax.set_xlabel("Outcome Coefficient")
        ax.set_ylabel("Treatment Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "discrete_heatmap_bias_differences.svg"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_heatmap(data_matrix, x_labels, y_labels, xlabel, ylabel, filename, output_dir, cmap="coolwarm", center=None):
    plot_typography()
    # plot_typography(small=20, medium=24, big=30)
    plt.figure(figsize=(10, 8))
    # Set the midpoint of the color scale to the specified center if provided
    sns.heatmap(
        data_matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        center=center,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def prepare_heatmap_data(
    all_results,
    x_key,
    y_key,
    value_key,
    diff_index=None,
    beta_filter=None,
    support_size_filter=None,
    relative_error=False,
    compute_rmse=False,
):
    x_values = sorted(
        {
            res[x_key]
            for res in all_results
            if (beta_filter is None or res["beta"] == beta_filter)
            and (support_size_filter is None or res["support_size"] == support_size_filter)
        }
    )
    y_values = sorted(
        {
            res[y_key]
            for res in all_results
            if (beta_filter is None or res["beta"] == beta_filter)
            and (support_size_filter is None or res["support_size"] == support_size_filter)
        },
        reverse=True,
    )
    data_matrix = np.zeros((len(y_values), len(x_values)))
    data_matrix_mean = np.zeros((len(y_values), len(x_values)))
    data_matrix_std = np.zeros((len(y_values), len(x_values)))
    data_matrix_rmse = np.zeros((len(y_values), len(x_values))) if compute_rmse else None

    # Determine the keys based on whether relative error is considered
    value_key_suffix = "_rel" if relative_error else ""
    sigmas_key = "sigmas" + value_key_suffix

    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            # Filter results matching current (x_val, y_val) and filter criteria
            matching_results = [
                res
                for res in all_results
                if (
                    res[x_key] == x_val
                    and res[y_key] == y_val
                    and (beta_filter is None or res["beta"] == beta_filter)
                    and (support_size_filter is None or res["support_size"] == support_size_filter)
                )
            ]

            if not matching_results:
                # No matching results for this combination, use NaN
                data_matrix_mean[j, i] = np.nan
                data_matrix_std[j, i] = np.nan
                data_matrix[j, i] = 0
                if compute_rmse:
                    assert data_matrix_rmse is not None
                    data_matrix_rmse[j, i] = np.nan  # pylint: disable=unsupported-assignment-operation
                continue

            res = matching_results[0]
            rmse_diffs = []  # Initialize for pylint

            ica_mean = (
                res[value_key + value_key_suffix][-1]
                if value_key != "first_stage_mse"
                else np.mean([z[-1] for z in res[value_key]])
            )
            ica_std = res[sigmas_key][-1] if value_key != "first_stage_mse" else np.std([z[-1] for z in res[value_key]])

            if diff_index is not None:
                compare_mean = res[value_key + value_key_suffix][diff_index]
                compare_std = res[sigmas_key][diff_index]
                diffs = [
                    r[value_key + value_key_suffix][-1] - r[value_key + value_key_suffix][diff_index]
                    for r in matching_results
                ]

                if compute_rmse:
                    # RMSE differences: RMSE_ICA - RMSE_compare for each result
                    rmse_diffs = [
                        np.sqrt(r[value_key + value_key_suffix][-1] ** 2 + r[sigmas_key][-1] ** 2)
                        - np.sqrt(r[value_key + value_key_suffix][diff_index] ** 2 + r[sigmas_key][diff_index] ** 2)
                        for r in matching_results
                    ]

                if (ica_mean + ica_std) < (compare_mean - compare_std):
                    data_matrix[j, i] = -1
                elif (ica_mean - ica_std) > (compare_mean + compare_std):
                    data_matrix[j, i] = 1
                else:
                    data_matrix[j, i] = 0

            else:
                diffs = ica_mean
                if compute_rmse:
                    # RMSE for each result (no comparison)
                    rmse_diffs = [
                        np.sqrt(r[value_key + value_key_suffix][-1] ** 2 + r[sigmas_key][-1] ** 2)
                        for r in matching_results
                    ]

            if diffs:
                data_matrix_mean[j, i] = np.nanmean(diffs)
                data_matrix_std[j, i] = ica_std
                if compute_rmse:
                    assert data_matrix_rmse is not None
                    data_matrix_rmse[j, i] = np.nanmean(rmse_diffs)  # pylint: disable=unsupported-assignment-operation

    return (
        data_matrix_mean,
        data_matrix_std,
        data_matrix,
        x_values,
        y_values,
        data_matrix_rmse,
    )
