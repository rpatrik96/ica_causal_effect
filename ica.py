import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import gennorm
from sklearn.decomposition import FastICA
from tueplots import bundles

from ica_utils import calculate_mse
from mcc import calc_disent_metrics
from plot_utils import plot_typography


def generate_ica_data(
    n_covariates=1,
    n_treatments=1,
    batch_size=4096,
    slope=1.0,
    sparse_prob=0.3,
    beta=1.0,
    loc=0,
    scale=1,
    nonlinearity="leaky_relu",
    theta_choice="fixed",
    split_noise_dist=False,
):
    # Create sparse matrix of shape (n_treatments x n_covariates)
    binary_mask = torch.bernoulli(torch.ones(n_treatments, n_covariates) * sparse_prob)
    random_coeffs = torch.randn(n_treatments, n_covariates)
    A_covariates = binary_mask * random_coeffs

    if theta_choice == "fixed":
        theta = torch.tensor([1.55, 0.65, -2.45, 1.75, -1.35])[:n_treatments]  # Fixed vector of thetas
    elif theta_choice == "uniform":
        theta = torch.rand(n_treatments)  # Draw theta from a uniform distribution
    elif theta_choice == "gaussian":
        theta = torch.randn(n_treatments)  # Draw theta from a Gaussian distribution
    else:
        raise ValueError(f"Unsupported theta_choice for theta generation: {theta_choice}")

    B = torch.randn(n_covariates)  # Base effects on outcome per covariate

    distribution = gennorm(beta, loc=loc, scale=scale)

    source_dim = n_covariates + n_treatments + 1  # +1 for outcome
    if split_noise_dist is False:
        S = torch.tensor(distribution.rvs(size=(batch_size, source_dim))).float()
    else:
        S_X = torch.tensor(gennorm(beta=2.0, loc=loc, scale=scale).rvs(size=(batch_size, n_covariates))).float()
        S_TY = torch.tensor(distribution.rvs(size=(batch_size, n_treatments + 1))).float()
        S = torch.hstack((S_X, S_TY))
    X = S.clone()

    # Define activation function based on the nonlinearity parameter
    activation_functions = {
        "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=slope),
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }

    if nonlinearity not in activation_functions:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    activation = activation_functions[nonlinearity]

    # Covariates remain independent
    X[:, :n_covariates] = S[:, :n_covariates]

    # Treatments depend on all covariates
    treatment_indices = torch.arange(n_covariates, n_covariates + n_treatments)
    X[:, treatment_indices] = S[:, treatment_indices]
    # Modified to use the sparse matrix multiplication
    covariate_effects = activation(S[:, :n_covariates]) @ A_covariates.t()
    X[:, treatment_indices] += covariate_effects

    # Outcome depends on all covariates and treatments
    X[:, -1] = S[:, -1]

    # Add treatment effects
    X[:, -1] += (theta * X[:, treatment_indices]).sum(dim=1)

    # Add covariate effects

    # Use the sum of all sparse connections for the outcome
    X[:, -1] += (B * activation(S[:, :n_covariates])).sum(dim=1)

    return S, X, theta


def ica_treatment_effect_estimation(
    X, S, random_state=0, whiten="unit-variance", check_convergence=True, n_treatments=1, verbose=True, fun="logcosh"
):
    from warnings import catch_warnings  # pylint: disable=import-outside-toplevel

    tol = 1e-4  # Initial tolerance
    # Maximum tolerance to try (unused but kept for future enhancement)
    # _max_tol = 1e-2
    # random_state = 12143

    for attempt in range(5):
        with catch_warnings(record=True) as w:
            # filterwarnings('error')

            ica = FastICA(
                n_components=X.shape[1],
                random_state=random_state + attempt,
                max_iter=1000,
                whiten=whiten,
                tol=tol,
                fun=fun,
            )
            S_hat = ica.fit_transform(X)

            if len(w) > 0 and check_convergence is True:
                if verbose:
                    print(f"warning at {attempt=}")
                # Increase tolerance for next attempt
                # tol = min(tol * 2, _max_tol)
                # if tol >= _max_tol:  # Stop if max tolerance reached
                print("Max tolerance reached without convergence")
                return (
                    np.nan
                    * np.ones(
                        n_treatments,
                    ),
                    None,
                )
            if verbose:
                print(f"success at {attempt=}")
            break

    results = calc_disent_metrics(S, S_hat)
    # resolve the permutations
    permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
    # normalize to get 1 at epsilon -> Y
    permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

    n_covariates = X.shape[1] - 1 - n_treatments  # Assuming  and 1 outcome
    treatment_effect_estimate = permuted_scaled_mixing[-1, n_covariates:-1]

    return treatment_effect_estimate, results["permutation_disentanglement_score"]


def main_multi():
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_dims = [2, 5, 10, 20, 50]
    n_treatments = [1, 2, 5]
    n_seeds = 20

    # Initialize dictionary to store results
    results_dict = {
        "sample_sizes": [],
        "n_covariates": [],
        "n_treatments": [],
        "true_params": [],
        "treatment_effects": [],
        "treatment_effects_iv": [],
        "mccs": [],
    }

    results_file = "results_multi_treatment.npy"
    if os.path.exists(results_file):
        print(f"Results file '{results_file}' already exists. Loading data.")
        loaded_results = np.load(results_file, allow_pickle=True).item()
        results_dict["sample_sizes"].extend(loaded_results["sample_sizes"])
        results_dict["n_covariates"].extend(loaded_results["n_covariates"])
        results_dict["n_treatments"].extend(loaded_results["n_treatments"])
        results_dict["true_params"].extend(loaded_results["true_params"])
        results_dict["treatment_effects"].extend(loaded_results["treatment_effects"])
        results_dict["treatment_effects_iv"].extend(loaded_results["treatment_effects_iv"])
        results_dict["mccs"].extend(loaded_results["mccs"])
    else:
        from linearmodels.iv import IV2SLS

        for n_samples in sample_sizes:
            for n_covariates in n_dims:
                for n_treatment in n_treatments:
                    S, X, true_params = generate_ica_data(
                        batch_size=n_samples,
                        n_covariates=n_covariates,
                        n_treatments=n_treatment,
                        slope=1.0,
                        sparse_prob=0.3,
                    )

                    treatment_indices = torch.arange(n_covariates, n_covariates + n_treatment).numpy()
                    T = X[:, treatment_indices]
                    X_cov = X[:, :n_covariates]
                    Y = X[:, -1].reshape(
                        -1,
                    )

                    # Create a pandas dataframe with separate columns for Y, each column of T, and each column of X
                    data = {
                        "Y": Y,
                        **{f"T{k}": T[:, k] for k in range(T.shape[1])},
                        **{f"X{l}": X_cov[:, l] for l in range(X_cov.shape[1])},
                    }

                    T_names = [f"T{k}" for k in range(T.shape[1])]

                    iv_df = pd.DataFrame(data)

                    formula = (
                        "Y ~ 1 + "
                        + " + ".join([f"T{k}" for k in range(T.shape[1])])
                        + " + "
                        + " + ".join([f"X{l}" for l in range(X_cov.shape[1])])
                    )

                    for seed in range(n_seeds):
                        treatment_effects, mcc = ica_treatment_effect_estimation(
                            X, S, random_state=seed, check_convergence=False, n_treatments=n_treatment, verbose=False
                        )

                        # Fit the IV regression model

                        iv_model = IV2SLS.from_formula(formula, iv_df).fit()
                        treatment_effects_iv = iv_model.params[T_names]

                        # Store results in dictionary
                        results_dict["sample_sizes"].append(n_samples)
                        results_dict["n_covariates"].append(n_covariates)
                        results_dict["n_treatments"].append(n_treatment)
                        results_dict["true_params"].append(true_params)
                        results_dict["treatment_effects"].append(treatment_effects)
                        results_dict["treatment_effects_iv"].append(treatment_effects_iv)
                        results_dict["mccs"].append(mcc)

        # Save results dictionary
        np.save(results_file, results_dict)

    def plot_heatmap(data, x_labels, y_labels, x_label, y_label, _title, filename, center=0):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            data, xticklabels=x_labels, yticklabels=y_labels, cmap="coolwarm", annot=True, fmt=".2f", center=center
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.title(title)
        # plt.tight_layout()
        save_figure(filename)
        plt.close()

    # Refactored data filtering for heatmap preparation

    def filter_indices(results_dict, sample_size, treatment_count=None, covariate_dim=None):
        return [
            i
            for i, (s, t, d) in enumerate(
                zip(results_dict["sample_sizes"], results_dict["n_treatments"], results_dict["n_covariates"])
            )
            if s == sample_size
            and (treatment_count is None or t == treatment_count)
            and (covariate_dim is None or d == covariate_dim)
        ]

    def calculate_treatment_effect_diff(results_dict, indices):
        est_params_ica = [results_dict["treatment_effects"][i] for i in indices]
        est_params_iv = [results_dict["treatment_effects_iv"][i] for i in indices]
        return np.nanmean([np.linalg.norm(est_ica - est_iv) for est_ica, est_iv in zip(est_params_ica, est_params_iv)])

    # Prepare data for heatmap: x-axis is number of treatments, y-axis is sample size, covariate dimension is 10
    covariate_dimension = 10
    treatment_effect_diff = {}
    treatment_effect_ica = {}
    treatment_effect_ica_std = {}
    for n_samples in set(results_dict["sample_sizes"]):
        for n_treatment in set(results_dict["n_treatments"]):

            indices = filter_indices(results_dict, n_samples, n_treatment, covariate_dimension)
            if indices:
                diff = calculate_treatment_effect_diff(results_dict, indices)
                treatment_effect_diff[(n_samples, n_treatment)] = diff
                # Calculate ICA error only
                est_params_ica = [results_dict["treatment_effects"][i] for i in indices]
                true_params = [results_dict["true_params"][i].numpy() for i in indices]

                ica_error = calculate_mse(true_params, est_params_ica, relative_error=True)
                # ica_error = [np.linalg.norm(est_ica - true.numpy()) for est_ica, true in
                #              zip(est_params_ica, true_params)]
                treatment_effect_ica[(n_samples, n_treatment)] = np.nanmean(ica_error)
                treatment_effect_ica_std[(n_samples, n_treatment)] = np.nanstd(ica_error)

    # Create heatmap data for difference
    sample_sizes = sorted(set(results_dict["sample_sizes"]), reverse=True)
    num_treatments = sorted(set(results_dict["n_treatments"]))
    heatmap_data = np.array([[treatment_effect_diff.get((s, t), np.nan) for t in num_treatments] for s in sample_sizes])

    plot_heatmap(
        heatmap_data,
        num_treatments,
        sample_sizes,
        r"Number of treatments $m$",
        r"Sample size $n$",
        "Difference in Treatment Effects (Covariate Dimension = 10)",
        "heatmap_multi_treatments_vs_samples.svg",
        center=0,
    )

    # Create heatmap data for ICA error only
    heatmap_data_ica = np.array(
        [[treatment_effect_ica.get((s, t), np.nan) for t in num_treatments] for s in sample_sizes]
    )
    heatmap_data_ica_std = np.array(
        [[treatment_effect_ica_std.get((s, t), np.nan) for t in num_treatments] for s in sample_sizes]
    )

    plot_heatmap(
        heatmap_data_ica,
        num_treatments,
        sample_sizes,
        r"Number of treatments $m$",
        r"Sample size $n$",
        "ICA Rel. Error Mean (Covariate Dimension = 10)",
        "heatmap_ica_treatments_vs_samples_rel.svg",
        center=None,
    )
    plot_heatmap(
        heatmap_data_ica_std,
        num_treatments,
        sample_sizes,
        r"Number of treatments $m$",
        r"Sample size $n$",
        "ICA Rel. Error Std (Covariate Dimension = 10)",
        "heatmap_ica_treatments_vs_samples_rel_std.svg",
        center=None,
    )

    # Prepare data for heatmap: x-axis is dimension, y-axis is sample size, number of treatments is 2
    num_treatments_fixed = 2
    treatment_effect_diff_dim = {}
    treatment_effect_ica_dim = {}
    treatment_effect_ica_dim_std = {}
    for n_samples in set(results_dict["sample_sizes"]):
        for dimension in set(results_dict["n_covariates"]):
            indices = filter_indices(results_dict, n_samples, num_treatments_fixed, dimension)
            if indices:
                diff = calculate_treatment_effect_diff(results_dict, indices)
                treatment_effect_diff_dim[(n_samples, dimension)] = diff
                # Calculate ICA error only
                est_params_ica = [results_dict["treatment_effects"][i] for i in indices]
                true_params = [results_dict["true_params"][i].numpy() for i in indices]
                ica_error = calculate_mse(true_params, est_params_ica, relative_error=True)
                # Alternative: ica_error = [np.linalg.norm(est_ica - true.numpy())
                #              for est_ica, true in zip(est_params_ica, true_params)]
                treatment_effect_ica_dim[(n_samples, dimension)] = np.nanmean(ica_error)
                treatment_effect_ica_dim_std[(n_samples, dimension)] = np.nanstd(ica_error)

    # Create heatmap data for difference
    dimensions = sorted(set(results_dict["n_covariates"]))
    heatmap_data_dim = np.array(
        [[treatment_effect_diff_dim.get((s, d), np.nan) for d in dimensions] for s in sample_sizes]
    )

    plot_heatmap(
        heatmap_data_dim,
        dimensions,
        sample_sizes,
        r"Covariate dimension $d$",
        r"Sample size $n$",
        "Difference in Treatment Effects (Number of Treatments = 2)",
        "heatmap_multi_dimensions_vs_samples_rel.svg",
        center=0,
    )

    # Create heatmap data for ICA error only
    heatmap_data_ica_dim = np.array(
        [[treatment_effect_ica_dim.get((s, d), np.nan) for d in dimensions] for s in sample_sizes]
    )
    heatmap_data_ica_dim_std = np.array(
        [[treatment_effect_ica_dim_std.get((s, d), np.nan) for d in dimensions] for s in sample_sizes]
    )

    plot_heatmap(
        heatmap_data_ica_dim,
        dimensions,
        sample_sizes,
        r"Covariate dimension $d$",
        r"Sample size $n$",
        "ICA Rel. Error Mean (Number of Treatments = 2)",
        "heatmap_ica_multi_dimensions_vs_samples_rel.svg",
        center=None,
    )
    plot_heatmap(
        heatmap_data_ica_dim_std,
        dimensions,
        sample_sizes,
        r"Covariate dimension $d$",
        r"Sample size $n$",
        "ICA Rel. Error Std (Number of Treatments = 2)",
        "heatmap_ica_multi_dimensions_vs_samples_std.svg",
        center=None,
    )


def main_nonlinear():
    # import matplotlib.pyplot as plt
    # import numpy as np

    sample_sizes = [5000]
    n_dims = [2, 5, 10, 20, 50]
    slopes = [0, 0.1, 0.2, 0.5, 1.0]
    n_seeds = 20
    nonlinearities = ["leaky_relu", "relu", "sigmoid", "tanh"]
    results_file = "results_main_nonlinear.npy"

    # Initialize dictionary to store results
    results_dict = {
        "sample_sizes": [],
        "n_covariates": [],
        "true_params": [],
        "treatment_effects": [],
        "slopes": [],
        "mccs": [],
        "nonlinearities": [],
    }
    # import os
    if os.path.exists(results_file):
        print(f"Results file '{results_file}' already exists. Loading data.")
        results_dict = np.load(results_file, allow_pickle=True).item()
    else:

        plt.rcParams.update(bundles.icml2022(usetex=True))
        plot_typography()

        for n_samples in sample_sizes:
            for n_covariates in n_dims:
                for nonlinearity in nonlinearities:
                    if nonlinearity == "leaky_relu":
                        for slope in slopes:
                            S, X, true_params = generate_ica_data(
                                batch_size=n_samples,
                                n_covariates=n_covariates,
                                n_treatments=1,
                                slope=slope,
                                sparse_prob=0.3,
                                nonlinearity=nonlinearity,
                            )

                            for seed in range(n_seeds):
                                treatment_effects, mcc = ica_treatment_effect_estimation(
                                    X, S, random_state=seed, check_convergence=False, n_treatments=1
                                )

                                # Store results in dictionary

                                results_dict["slopes"].append(slope)

                                results_dict["sample_sizes"].append(n_samples)
                                results_dict["n_covariates"].append(n_covariates)
                                results_dict["true_params"].append(true_params)
                                results_dict["treatment_effects"].append(treatment_effects)
                                results_dict["mccs"].append(mcc)
                                results_dict["nonlinearities"].append(nonlinearity)
                    else:
                        S, X, true_params = generate_ica_data(
                            batch_size=n_samples,
                            n_covariates=n_covariates,
                            n_treatments=1,
                            slope=1.0,
                            # Default slope for other nonlinearities
                            sparse_prob=0.3,
                            nonlinearity=nonlinearity,
                        )

                        for seed in range(n_seeds):
                            treatment_effects, mcc = ica_treatment_effect_estimation(
                                X, S, random_state=seed, check_convergence=False, n_treatments=1
                            )

                            # Store results in dictionary
                            results_dict["slopes"].append(1.0)  # Default slope for other nonlinearitiesmcc)

                            results_dict["sample_sizes"].append(n_samples)
                            results_dict["n_covariates"].append(n_covariates)
                            results_dict["true_params"].append(true_params)
                            results_dict["treatment_effects"].append(treatment_effects)
                            results_dict["mccs"].append(mcc)
                            results_dict["nonlinearities"].append(nonlinearity)

        # Save results dictionary
        np.save(results_file, results_dict)
    # Filter the data
    filtered_indices = [
        i
        for i, nonlinearity in enumerate(results_dict["nonlinearities"])
        if (nonlinearity == "leaky_relu" and results_dict["slopes"][i] == 0.2) or (nonlinearity != "leaky_relu")
    ]
    filtered_results = {key: [results_dict[key][i] for i in filtered_indices] for key in results_dict}

    # Prepare data for heatmap
    dimensions = sorted(set(filtered_results["n_covariates"]), reverse=True)
    nonlinearities = sorted(set(filtered_results["nonlinearities"]))
    heatmap_data = np.zeros((len(dimensions), len(nonlinearities)))
    heatmap_data_std = np.zeros((len(dimensions), len(nonlinearities)))

    for i, dim in enumerate(dimensions):
        for j, nonlinearity in enumerate(nonlinearities):
            relevant_indices = [
                index
                for index, (d, n) in enumerate(
                    zip(filtered_results["n_covariates"], filtered_results["nonlinearities"])
                )
                if d == dim and n == nonlinearity
            ]
            if relevant_indices:
                heatmap_data[i, j] = np.mean(
                    [
                        calculate_mse(
                            filtered_results["true_params"][index], filtered_results["treatment_effects"][index]
                        )
                        for index in relevant_indices
                    ]
                )
                heatmap_data_std[i, j] = np.std(
                    [
                        calculate_mse(
                            filtered_results["true_params"][index], filtered_results["treatment_effects"][index]
                        )
                        for index in relevant_indices
                    ]
                )

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=nonlinearities, yticklabels=dimensions, cmap="coolwarm", annot=True)
    plt.xlabel("Nonlinearity")
    plt.ylabel(r"Covariate dimension $d$")
    # plt.title('Heatmap of MSEs: Dimension vs Nonlinearity')
    save_figure("heatmap_dimension_vs_nonlinearity.svg")  # , dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data_std, xticklabels=nonlinearities, yticklabels=dimensions, cmap="coolwarm", annot=True)
    plt.xlabel("Nonlinearity")
    plt.ylabel(r"Covariate dimension $d$")
    # plt.title('Heatmap of MSEs: Dimension vs Nonlinearity')
    save_figure("heatmap_dimension_vs_nonlinearity_std.svg")  # , dpi=300, bbox_inches='tight')
    plt.close()

    # Filter the data to only include 'leaky_relu' nonlinearity
    filtered_indices_leaky_relu = [
        index for index, nonlinearity in enumerate(results_dict["nonlinearities"]) if nonlinearity == "leaky_relu"
    ]
    filtered_results_leaky_relu = {
        key: [results_dict[key][i] for i in filtered_indices_leaky_relu] for key in results_dict
    }

    # Prepare data for heatmap
    dimensions = sorted(set(filtered_results_leaky_relu["n_covariates"]), reverse=True)
    slopes = sorted(set(filtered_results_leaky_relu["slopes"]))
    heatmap_data = np.zeros((len(dimensions), len(slopes)))

    for i, dim in enumerate(dimensions):
        for j, slope in enumerate(slopes):
            relevant_indices = [
                index
                for index, (d, s) in enumerate(
                    zip(filtered_results_leaky_relu["n_covariates"], filtered_results_leaky_relu["slopes"])
                )
                if d == dim and s == slope
            ]
            if relevant_indices:
                heatmap_data[i, j] = np.mean(
                    [
                        calculate_mse(
                            filtered_results_leaky_relu["true_params"][index],
                            filtered_results_leaky_relu["treatment_effects"][index],
                        )
                        for index in relevant_indices
                    ]
                )

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=slopes, yticklabels=dimensions, cmap="coolwarm", annot=True)
    plt.xlabel("Slope")
    plt.ylabel(r"Covariate dimension $d$")
    # plt.title('Heatmap of MCCs: Dimension vs Slope for Leaky ReLU')
    save_figure("heatmap_dimension_vs_slope_leaky_relu.svg")
    plt.close()


def main_fun():
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    n_samples = 5000
    n_covariates = 50
    n_treatment = 1
    n_seeds = 20

    # Initialize dictionary to store results
    results_dict = {
        "sample_sizes": [],
        "n_covariates": [],
        "n_treatments": [],
        "true_params": [],
        "treatment_effects": [],
        "treatment_effects_iv": [],
        "mccs": [],
        "fun_options": [],
    }

    S, X, true_params = generate_ica_data(
        batch_size=n_samples, n_covariates=n_covariates, n_treatments=n_treatment, slope=1.0, sparse_prob=0.3
    )
    fun_options = ["logcosh", "exp", "cube"]

    for fun in fun_options:
        for seed in range(n_seeds):
            treatment_effects, mcc = ica_treatment_effect_estimation(
                X, S, random_state=seed, check_convergence=False, n_treatments=n_treatment, fun=fun
            )

            # Store results in dictionary
            results_dict["sample_sizes"].append(n_samples)
            results_dict["n_covariates"].append(n_covariates)
            results_dict["n_treatments"].append(n_treatment)
            results_dict["true_params"].append(true_params)
            results_dict["treatment_effects"].append(treatment_effects)
            results_dict["mccs"].append(mcc)
            results_dict["fun_options"].append(fun)

    # Save results dictionary
    np.save("results_main_fun.npy", results_dict)

    plt.figure(figsize=(10, 6))

    # Process data based on 'fun' and calculate MSE
    mse_by_fun = {fun: [] for fun in set(fun_options)}
    for true_param, est_param, fun_label in zip(
        results_dict["true_params"], results_dict["treatment_effects"], results_dict["fun_options"]
    ):
        if est_param is not None:  # Handle cases where estimation failed
            errors = [np.linalg.norm(est - true) for est, true in zip(est_param, true_param)]
            mse_by_fun[fun_label].append(np.mean(errors))
        else:
            mse_by_fun[fun_label].append(np.nan)

    # Create a bar plot for each 'fun' on the x-axis
    bar_width = 0.2
    bar_positions = np.arange(len(fun_options))

    # Calculate mean and standard deviation for each 'fun'
    means = [np.mean(mse_by_fun[fun]) for fun in fun_options]
    std_devs = [np.std(mse_by_fun[fun]) for fun in fun_options]

    plt.bar(
        bar_positions,
        means,
        yerr=std_devs,
        width=bar_width,
        capsize=5,
        tick_label=fun_options,
        label=f"{n_treatment} (ICA)",
    )
    plt.xlabel("FastICA objective function")

    # plt.legend(loc='lower center', ncol=int(n_treatment/2), bbox_to_anchor=(0.5, -0.15))

    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'Covariate dimension $d$')
    plt.ylabel(r"Mean squared $|(\theta-\hat{\theta})/\theta|$")
    # plt.grid(True, which="both", linestyle='-.', linewidth=0.5)
    # plt.legend()
    # plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])

    save_figure("ica_mse_fun.svg")
    plt.close()


def setup_plot():
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()


def initialize_results_dict(keys):
    return {key: [] for key in keys}


def save_results(filename, results_dict):
    np.save(filename, results_dict)


def plot_error_bars(x_values, means, std_devs, xlabel, ylabel, filename, _x_ticks=None):
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(x_values))
    plt.errorbar(bar_positions, means, yerr=std_devs, fmt="o", capsize=5)
    plt.xticks(bar_positions, [f"{x:.2f}" for x in x_values], fontsize=18)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="-.", linewidth=0.5)
    save_figure(filename)
    plt.close()


def plot_heatmap(data_matrix, x_labels, y_labels, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        data_matrix, xticklabels=x_labels, yticklabels=y_labels, cmap="coolwarm", annot=True, annot_kws={"size": 18}
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(ticks=plt.yticks()[0], labels=[f"{x:.1f}" for x in plt.yticks()[0]])
    save_figure(filename)
    plt.close()


def main_sparsity():
    """Run sparsity ablation experiment for ICA treatment effect estimation."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    # Setup experiment configuration
    results_manager = ExperimentResultsManager("results_main_sparsity.npy")
    n_seeds = 20

    # Load or create results
    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "sparsities"]
    )

    # Run experiments if not already done
    if not results_manager.exists():
        # Configure experiment
        base_config = DataGenerationConfig(batch_size=5000, n_covariates=50, n_treatments=1, slope=1.0)
        estimation_config = EstimationConfig(check_convergence=False)

        # Define parameter grid
        sparsities = np.linspace(0, 1.0, num=11)[:-1]
        param_grid = {"sparse_prob": sparsities}

        # Run parameter sweep
        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_parameter_sweep(
            param_grid=param_grid, n_seeds=n_seeds, data_gen_config=base_config, results_dict=results_dict
        )

        # Rename sparse_prob to sparsities for consistency with original
        results_dict["sparsities"] = results_dict.pop("sparse_prob")

        results_manager.save(results_dict)

    # Analyze results
    sorted_sparsities, means, stds = ResultsAnalyzer.prepare_error_bar_data(
        results_dict, parameter_key="sparsities", relative_error=False
    )

    # Create plot
    plot_error_bars(
        sorted_sparsities,
        means,
        stds,
        r"Sparsity of $\mathrm{\mathbf{A}}$",
        r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        "ica_mse_vs_dim_sparsity.svg",
    )


def main_gennorm():
    """Run generalized normal distribution parameter ablation (linear PLR)."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm.npy")
    n_seeds = 20
    n_samples = 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "beta"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples, n_covariates=50, n_treatments=1, slope=1.0, sparse_prob=0.3
        )
        estimation_config = EstimationConfig(check_convergence=False)

        beta_values = np.linspace(0.5, 5, num=10)
        param_grid = {"beta": beta_values}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_parameter_sweep(
            param_grid=param_grid, n_seeds=n_seeds, data_gen_config=base_config, results_dict=results_dict
        )

        results_manager.save(results_dict)

    # Prepare and plot data
    beta_values, means, stds = ResultsAnalyzer.prepare_error_bar_data(
        results_dict, parameter_key="beta", relative_error=True
    )

    plot_error_bars_new(
        x_values=beta_values,
        means=means,
        std_devs=stds,
        xlabel=r"Gen. normal param. $\beta$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_mse_vs_beta_n{n_samples}.svg",
        use_log_scale=True,
    )


def main_gennorm_nonlinear():
    """Run generalized normal distribution parameter ablation (nonlinear PLR)."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm_nonlinear.npy")
    n_seeds = 20
    n_samples = 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "beta"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=50,
            n_treatments=1,
            slope=0.2,
            sparse_prob=0.3,
            nonlinearity="leaky_relu",
        )
        estimation_config = EstimationConfig(check_convergence=False)

        beta_values = np.linspace(0.5, 5, num=10)
        param_grid = {"beta": beta_values}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_parameter_sweep(
            param_grid=param_grid, n_seeds=n_seeds, data_gen_config=base_config, results_dict=results_dict
        )

        results_manager.save(results_dict)

    # Prepare and plot data
    beta_values, means, stds = ResultsAnalyzer.prepare_error_bar_data(
        results_dict, parameter_key="beta", relative_error=True
    )

    # Print statistics
    for beta, mean_mse, std_mse in zip(beta_values, means, stds):
        print(rf"beta={beta:.2f}: {mean_mse:.4f}±{std_mse:.4f}")

    plot_error_bars_new(
        x_values=beta_values,
        means=means,
        std_devs=stds,
        xlabel=r"Gen. normal param. $\beta$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_mse_vs_beta_nonlinear_n{n_samples}.svg",
        use_log_scale=True,
    )


def main_nonlinear_theta():
    """Run theta distribution ablation experiment."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm_nonlinear_theta.npy")
    n_seeds = 20
    n_samples = 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "theta_choice"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=10,
            n_treatments=1,
            slope=0.2,
            sparse_prob=0.3,
            beta=1.0,
            nonlinearity="leaky_relu",
        )
        estimation_config = EstimationConfig(check_convergence=False)

        theta_choices = ["fixed", "uniform", "gaussian"]
        param_grid = {"theta_choice": theta_choices}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_parameter_sweep(
            param_grid=param_grid, n_seeds=n_seeds, data_gen_config=base_config, results_dict=results_dict
        )

        results_manager.save(results_dict)

    # Prepare data for plotting
    theta_choices = sorted(set(results_dict["theta_choice"]))
    means = []
    stds = []

    for theta_choice in theta_choices:
        mean_stat, std_stat = ResultsAnalyzer.compute_method_statistics(
            results_dict,
            ResultsAnalyzer.filter_by_parameter(results_dict, "theta_choice", theta_choice),
            relative_error=True,
        )
        means.append(mean_stat)
        stds.append(std_stat)
        print(rf"{theta_choice}: {mean_stat:.4f}±{std_stat:.4f}")

    plot_error_bars_new(
        x_values=list(range(len(theta_choices))),
        means=means,
        std_devs=stds,
        xlabel=r"$p(\theta)$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_mse_vs_theta_choice_nonlinear_n{n_samples}.svg",
        x_ticks=theta_choices,
        use_log_scale=True,
    )


def main_nonlinear_noise_split():
    """Run noise distribution split ablation experiment."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm_nonlinear_noise_split.npy")
    n_seeds = 20
    n_samples = 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "split_noise_dist"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=50,
            n_treatments=1,
            slope=0.2,
            sparse_prob=0.3,
            beta=1.0,
            nonlinearity="leaky_relu",
        )
        estimation_config = EstimationConfig(check_convergence=False)

        noise_splits = [True, False]
        param_grid = {"split_noise_dist": noise_splits}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_parameter_sweep(
            param_grid=param_grid, n_seeds=n_seeds, data_gen_config=base_config, results_dict=results_dict
        )

        results_manager.save(results_dict)

    # Prepare data for plotting
    noise_splits = sorted(set(results_dict["split_noise_dist"]))
    means = []
    stds = []

    for noise_split in noise_splits:
        mean_stat, std_stat = ResultsAnalyzer.compute_method_statistics(
            results_dict,
            ResultsAnalyzer.filter_by_parameter(results_dict, "split_noise_dist", noise_split),
            relative_error=True,
        )
        means.append(mean_stat)
        stds.append(std_stat)
        print(rf"{noise_split}: {mean_stat:.4f}±{std_stat:.4f}")

    plot_error_bars_new(
        x_values=list(range(len(noise_splits))),
        means=means,
        std_devs=stds,
        xlabel=r"Gaussian $X$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_mse_vs_noise_split_nonlinear_n{n_samples}.svg",
        x_ticks=noise_splits,
        use_log_scale=True,
    )


def main_loc_scale():
    """Run location and scale parameter ablation experiment."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    # Setup experiment configuration
    results_manager = ExperimentResultsManager("results_main_loc_scale.npy")
    n_seeds = 20
    n_samples = 5000

    # Load or create results
    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "loc", "scale"]
    )

    # Run experiments if not already done
    if not results_manager.exists():
        # Configure experiment
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=50,
            n_treatments=1,
            slope=1.0,
            sparse_prob=0.3,
            beta=1.0,
        )
        estimation_config = EstimationConfig(check_convergence=False)

        # Define parameter grid
        loc_values = np.linspace(-5, 5, num=10)
        scale_values = np.linspace(0.5, 5, num=10)
        param_grid = {"loc": loc_values, "scale": scale_values}

        # Run parameter sweep
        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_parameter_sweep(
            param_grid=param_grid, n_seeds=n_seeds, data_gen_config=base_config, results_dict=results_dict
        )

        results_manager.save(results_dict)

    # Prepare data for heatmap
    loc_values = sorted(set(results_dict["loc"]))
    scale_values = sorted(set(results_dict["scale"]))
    mse_matrix = np.zeros((len(loc_values), len(scale_values)))

    for i, loc in enumerate(loc_values):
        for j, scale in enumerate(scale_values):
            indices = [
                idx
                for idx, (l, s) in enumerate(zip(results_dict["loc"], results_dict["scale"]))
                if l == loc and s == scale
            ]
            if indices:
                mse_values = [
                    calculate_mse(results_dict["true_params"][idx], results_dict["treatment_effects"][idx])
                    for idx in indices
                ]
                mse_matrix[i, j] = np.nanmean(mse_values)

    # Plot heatmap
    plot_heatmap(
        mse_matrix, scale_values, loc_values, "Scale", "Location", f"ica_mse_heatmap_loc_scale_n{n_samples}.svg"
    )


def save_figure(filename):

    # Ensure the directory exists
    figures_dir = "figures/ica"
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_typography()

    #
    #
    # print("Running multiple treatment effect estimation with ICA...")
    # main_multi()
    #
    # # print("Running treatment effect estimation with ICA in nonlinear PLR...")
    # main_nonlinear()
    # #
    # print("Running treatment effect estimation with ICA in nonlinear PLR with gennorm noise...")
    # main_gennorm_nonlinear()
    # #
    # print("Running treatment effect estimation with ICA in nonlinear PLR with different theta choices...")
    # main_nonlinear_theta()
    # #
    # print("Running treatment effect estimation with ICA in nonlinear PLR with different noises...")
    # main_nonlinear_noise_split()

    print("Running the sparsity ablation for treatment effect estimation with ICA in linear PLR...")
    main_sparsity()

    # print("Running the loss function ablation for treatment effect estimation with ICA in linear PLR...")
    # main_fun()
    #
    # print("Running the gennorm ablation for treatment effect estimation with ICA...")
    # main_gennorm()
    #
    # print("Running the loc scale ablation for treatment effect estimation with ICA...")
    # main_loc_scale()
