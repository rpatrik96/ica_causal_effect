"""ICA-only experiment drivers restored from the companion-paper code."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tueplots import bundles

from ica import (
    generate_ica_data,
    ica_treatment_effect_estimation,
    ica_treatment_effect_estimation_eps_row,
)
from ica_utils import calculate_mse
from plot_utils import plot_typography

__all__ = [
    "generate_ica_data",
    "ica_treatment_effect_estimation",
    "ica_treatment_effect_estimation_eps_row",
]


# NOTE(eval 2026-09-26): The nonlinear DGP applies phi element-wise and then
# sums the transformed covariates, whereas the paper states phi(<w, x>).
# NOTE(eval 2026-09-26): ica_treatment_effect_estimation resolves the ICA
# permutation against the true sources, so its component matching is oracle.
# NOTE(eval 2026-09-26): sparse_prob is the probability of a NON-zero DGP
# coefficient, including in the experiments compared with DirectLiNGAM.


def _seed_count(n_seeds):
    return 20 if n_seeds is None else n_seeds


def directlingam_treatment_effect_estimation(X, n_treatments=1, random_state=0, verbose=True, prior_knowledge=None):
    """Estimate treatment effects using DirectLiNGAM.

    DirectLiNGAM is a causal discovery method that identifies the causal structure
    and estimates causal effects in linear non-Gaussian acyclic models.

    Args:
        X: Observed data matrix (n_samples x n_variables).
            Variables should be ordered as [covariates, treatments, outcome].
        n_treatments: Number of treatment variables.
        random_state: Random seed for reproducibility.
        verbose: Whether to print progress information.
        prior_knowledge: Optional prior knowledge matrix for DirectLiNGAM.
            -1: no prior knowledge (default)
             0: no causal relation from j to i
             1: causal relation from j to i

    Returns:
        treatment_effect_estimate: Estimated treatment effects (array of shape n_treatments).
        adjacency_matrix: Estimated adjacency matrix from DirectLiNGAM.
    """
    from lingam import DirectLiNGAM

    np.random.seed(random_state)

    n_variables = X.shape[1]
    n_covariates = n_variables - n_treatments - 1  # Subtract treatments and outcome

    # Convert to numpy if tensor
    if hasattr(X, "numpy"):
        X_np = X.numpy()
    else:
        X_np = np.array(X)

    # Fit DirectLiNGAM model
    model = DirectLiNGAM(random_state=random_state, prior_knowledge=prior_knowledge)

    try:
        model.fit(X_np)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        if verbose:
            print(f"DirectLiNGAM fitting failed: {e}")
        return np.nan * np.ones(n_treatments), None

    # Extract adjacency matrix (causal effects)
    adjacency_matrix = model.adjacency_matrix_

    # Treatment effect is the coefficient from treatment(s) to outcome
    # In our setup: outcome is the last variable (index -1)
    # Treatments are at indices [n_covariates, n_covariates + n_treatments)
    outcome_idx = n_variables - 1
    treatment_indices = list(range(n_covariates, n_covariates + n_treatments))

    # The adjacency matrix B has B[i,j] = effect of j on i
    # So treatment effect on outcome is B[outcome_idx, treatment_idx]
    treatment_effect_estimate = adjacency_matrix[outcome_idx, treatment_indices]

    if verbose:
        print(f"DirectLiNGAM estimated causal order: {model.causal_order_}")

    return treatment_effect_estimate, adjacency_matrix


def main_multi(n_seeds=None, quick=False):
    """Run multi-treatment ICA experiment.

    Produces heatmaps of relative error across sample sizes, dimensions,
    and treatment counts.
    """
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    sample_sizes = [100] if quick else [100, 200, 500, 1000, 2000, 5000]
    n_dims = [10] if quick else [2, 5, 10, 20, 50]
    n_treatments = [2] if quick else [1, 2, 5]
    n_seeds = _seed_count(n_seeds)

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
                    # NOTE(eval 2026-09-26): Data are generated once per grid
                    # cell; the seeds below only restart FastICA.
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
                        **{f"X{j}": X_cov[:, j] for j in range(X_cov.shape[1])},
                    }

                    T_names = [f"T{k}" for k in range(T.shape[1])]

                    iv_df = pd.DataFrame(data)

                    formula = (
                        "Y ~ 1 + "
                        + " + ".join([f"T{k}" for k in range(T.shape[1])])
                        + " + "
                        + " + ".join([f"X{j}" for j in range(X_cov.shape[1])])
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
            data,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 10},
            center=center,
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

    # Create heatmap data for ICA error std (m=2 only)
    heatmap_data_ica_dim_std = np.array(
        [[treatment_effect_ica_dim_std.get((s, d), np.nan) for d in dimensions] for s in sample_sizes]
    )

    # Multi-panel heatmap: ICA relative error, dimensions vs samples, for each treatment count
    num_treatments_list = sorted(set(results_dict["n_treatments"]))
    heatmap_panels = {}
    for n_treatment in num_treatments_list:
        panel_data = {}
        for n_samples_val in set(results_dict["sample_sizes"]):
            for dimension in set(results_dict["n_covariates"]):
                indices = filter_indices(results_dict, n_samples_val, n_treatment, dimension)
                if indices:
                    est_ica = [results_dict["treatment_effects"][i] for i in indices]
                    true_p = [results_dict["true_params"][i].numpy() for i in indices]
                    ica_err = calculate_mse(true_p, est_ica, relative_error=True)
                    panel_data[(n_samples_val, dimension)] = np.nanmean(ica_err)
        heatmap_panels[n_treatment] = np.array(
            [[panel_data.get((s, d), np.nan) for d in dimensions] for s in sample_sizes]
        )

    # Shared robust colorscale (2nd-98th percentile) for comparability across panels
    all_vals = np.concatenate([p.ravel() for p in heatmap_panels.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin_robust = np.percentile(all_vals, 2)
    vmax_robust = np.percentile(all_vals, 98)

    _, axes = plt.subplots(
        1, len(num_treatments_list), figsize=(6 * len(num_treatments_list), 8), constrained_layout=True
    )
    if len(num_treatments_list) == 1:
        axes = [axes]
    for ax, n_treatment in zip(axes, num_treatments_list):
        sns.heatmap(
            heatmap_panels[n_treatment],
            xticklabels=dimensions,
            yticklabels=sample_sizes,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 10},
            vmin=vmin_robust,
            vmax=vmax_robust,
            ax=ax,
        )
        ax.set_xlabel(r"Covariate dimension $d$")
        ax.set_ylabel(r"Sample size $n$")
        ax.set_title(rf"$m = {n_treatment}$")
    save_figure("heatmap_ica_multi_dimensions_vs_samples_rel.svg")

    # Single-panel heatmap for m=2 (main text version)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_panels[2],
        xticklabels=dimensions,
        yticklabels=sample_sizes,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        vmin=vmin_robust,
        vmax=vmax_robust,
    )
    plt.xlabel(r"Covariate dimension $d$")
    plt.ylabel(r"Sample size $n$")
    save_figure("heatmap_ica_multi_dimensions_vs_samples_rel_m2.svg")

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


def main_nonlinear(n_seeds=None, quick=False):
    """Run nonlinearity ablation: heatmaps of MSE across covariate dimensions and activation functions/slopes."""
    # import matplotlib.pyplot as plt
    # import numpy as np

    sample_sizes = [100] if quick else [5000]
    n_dims = [2] if quick else [2, 5, 10, 20, 50]
    slopes = [0.2] if quick else [0, 0.1, 0.2, 0.5, 1.0]
    n_seeds = _seed_count(n_seeds)
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
                            # NOTE(eval 2026-09-26): Data are generated once
                            # per grid cell; seeds only restart FastICA.
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
                        # NOTE(eval 2026-09-26): Data are generated once per
                        # grid cell; seeds only restart FastICA.
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
    sns.heatmap(
        heatmap_data,
        xticklabels=nonlinearities,
        yticklabels=dimensions,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )
    plt.xlabel("Nonlinearity")
    plt.ylabel(r"Covariate dimension $d$")
    # plt.title('Heatmap of MSEs: Dimension vs Nonlinearity')
    save_figure("heatmap_dimension_vs_nonlinearity.svg")  # , dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data_std,
        xticklabels=nonlinearities,
        yticklabels=dimensions,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )
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
    sns.heatmap(
        heatmap_data,
        xticklabels=slopes,
        yticklabels=dimensions,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )
    plt.xlabel("Slope")
    plt.ylabel(r"Covariate dimension $d$")
    # plt.title('Heatmap of MCCs: Dimension vs Slope for Leaky ReLU')
    save_figure("heatmap_dimension_vs_slope_leaky_relu.svg")
    plt.close()


def main_fun(n_seeds=None, quick=False):
    """Run FastICA contrast function ablation: compare logcosh, exp, and cube objectives."""
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    n_samples = 100 if quick else 5000
    n_covariates = 2 if quick else 50
    n_treatment = 1
    n_seeds = _seed_count(n_seeds)

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

    # NOTE(eval 2026-09-26): Data are generated once here; seeds only restart
    # FastICA for each contrast function.
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
    """Configure matplotlib with ICML 2022 style and custom typography."""
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()


def initialize_results_dict(keys):
    """Create an empty results dictionary with the given keys mapped to empty lists."""
    return {key: [] for key in keys}


def save_results(filename, results_dict):
    """Persist results dictionary to a .npy file."""
    np.save(filename, results_dict)


def plot_error_bars(x_values, means, std_devs, xlabel, ylabel, filename, _x_ticks=None):
    """Plot error bars with log-scale y-axis and save to figures/ica/."""
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
    """Plot an annotated seaborn heatmap with coolwarm colormap and save to figures/ica/."""
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        data_matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(ticks=plt.yticks()[0], labels=[f"{x:.1f}" for x in plt.yticks()[0]])
    save_figure(filename)
    plt.close()


def main_sparsity(n_seeds=None, quick=False):
    """Run sparsity ablation experiment for ICA treatment effect estimation."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    # Setup experiment configuration
    results_manager = ExperimentResultsManager("results_main_sparsity.npy")
    n_seeds = _seed_count(n_seeds)

    # Load or create results
    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "sparsities"]
    )

    # Run experiments if not already done
    if not results_manager.exists():
        # Configure experiment
        base_config = DataGenerationConfig(
            batch_size=100 if quick else 5000,
            n_covariates=2 if quick else 50,
            n_treatments=1,
            slope=1.0,
        )
        estimation_config = EstimationConfig(check_convergence=False)

        # Define parameter grid
        sparsities = np.array([0.2, 0.8]) if quick else np.linspace(0, 1.0, num=11)[:-1]
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


def main_gennorm(n_seeds=None, quick=False):
    """Run generalized normal distribution parameter ablation (linear PLR)."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm.npy")
    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "beta"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples, n_covariates=2 if quick else 50, n_treatments=1, slope=1.0, sparse_prob=0.3
        )
        estimation_config = EstimationConfig(check_convergence=False)

        beta_values = np.array([1.0, 2.0]) if quick else np.linspace(0.5, 5, num=10)
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


def main_gennorm_nonlinear(n_seeds=None, quick=False):
    """Run generalized normal distribution parameter ablation (nonlinear PLR)."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm_nonlinear.npy")
    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "beta"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=2 if quick else 50,
            n_treatments=1,
            slope=0.2,
            sparse_prob=0.3,
            nonlinearity="leaky_relu",
        )
        estimation_config = EstimationConfig(check_convergence=False)

        beta_values = np.array([1.0, 2.0]) if quick else np.linspace(0.5, 5, num=10)
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


def main_nonlinear_theta(n_seeds=None, quick=False):
    """Run theta distribution ablation experiment."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm_nonlinear_theta.npy")
    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "theta_choice"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=2 if quick else 10,
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


def main_nonlinear_noise_split(n_seeds=None, quick=False):
    """Run noise distribution split ablation experiment."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_error_bars as plot_error_bars_new
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager, ResultsAnalyzer

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_gennorm_nonlinear_noise_split.npy")
    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 5000

    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "split_noise_dist"]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=2 if quick else 50,
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


def main_loc_scale(n_seeds=None, quick=False):
    """Run location and scale parameter ablation experiment."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    # Setup experiment configuration
    results_manager = ExperimentResultsManager("results_main_loc_scale.npy")
    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 5000

    # Load or create results
    results_dict = results_manager.load_or_create(
        ["sample_sizes", "n_covariates", "n_treatments", "true_params", "treatment_effects", "mccs", "loc", "scale"]
    )

    # Run experiments if not already done
    if not results_manager.exists():
        # Configure experiment
        base_config = DataGenerationConfig(
            batch_size=n_samples,
            n_covariates=2 if quick else 50,
            n_treatments=1,
            slope=1.0,
            sparse_prob=0.3,
            beta=1.0,
        )
        estimation_config = EstimationConfig(check_convergence=False)

        # Define parameter grid
        loc_values = np.array([-1.0, 1.0]) if quick else np.linspace(-5, 5, num=10)
        scale_values = np.array([0.5, 1.0]) if quick else np.linspace(0.5, 5, num=10)
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
                for idx, (loc_val, s) in enumerate(zip(results_dict["loc"], results_dict["scale"]))
                if loc_val == loc and s == scale
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


def save_comparison_summary_markdown(
    parameter_name: str,
    parameter_values: list,
    series_data: dict,
    runtime_data: dict,
    output_path: str,
    title: str,
    n_samples: int,
    n_seeds: int,
):
    """Save comparison results summary as a markdown file.

    Args:
        parameter_name: Name of the parameter being varied (e.g., "Sparsity", "Beta")
        parameter_values: List of parameter values
        series_data: Dict mapping method name to (means, stds) tuple
        runtime_data: Dict mapping method name to (runtime_means, runtime_stds) tuple
        output_path: Path to save the markdown file
        title: Title for the summary
        n_samples: Number of samples used
        n_seeds: Number of seeds/experiments
    """
    md_lines = []
    md_lines.append(f"# {title}\n")
    md_lines.append("## Experiment Settings\n")
    md_lines.append(f"- **n_samples**: {n_samples}")
    md_lines.append(f"- **n_seeds**: {n_seeds}")
    md_lines.append(f"- **Parameter varied**: {parameter_name}")
    md_lines.append("")

    # Get method names
    methods = list(series_data.keys())

    # Build results table
    md_lines.append("## Results Summary\n")

    # Header row
    header = f"| {parameter_name} |"
    separator = "|:------|"
    for method in methods:
        header += f" {method} MSE | {method} Std |"
        separator += " ---:| ---:|"
    if runtime_data:
        for method in methods:
            header += f" {method} Time (s) | {method} Time Std |"
            separator += " ---:| ---:|"
    header += " Winner |"
    separator += ":------:|"

    md_lines.append(header)
    md_lines.append(separator)

    # Data rows
    for i, param_val in enumerate(parameter_values):
        row = f"| {param_val} |"
        mse_vals = {}
        for method in methods:
            means, stds = series_data[method]
            mse_val = means[i] if i < len(means) else np.nan
            std_val = stds[i] if i < len(stds) else np.nan
            mse_vals[method] = mse_val
            row += f" {mse_val:.4f} | {std_val:.4f} |" if not np.isnan(mse_val) else " N/A | N/A |"

        if runtime_data:
            for method in methods:
                if method in runtime_data:
                    rt_means, rt_stds = runtime_data[method]
                    rt_val = rt_means[i] if i < len(rt_means) else np.nan
                    rt_std = rt_stds[i] if i < len(rt_stds) else np.nan
                    row += f" {rt_val:.4f} | {rt_std:.4f} |" if not np.isnan(rt_val) else " N/A | N/A |"
                else:
                    row += " N/A | N/A |"

        # Determine winner
        valid_mse = {m: v for m, v in mse_vals.items() if not np.isnan(v)}
        if valid_mse:
            winner = min(valid_mse, key=valid_mse.get)
            row += f" **{winner}** |"
        else:
            row += " N/A |"

        md_lines.append(row)

    # Add runtime summary if available
    if runtime_data:
        md_lines.append("")
        md_lines.append("## Runtime Summary\n")
        for method in methods:
            if method in runtime_data:
                rt_means, _ = runtime_data[method]
                valid_runtimes = [r for r in rt_means if not np.isnan(r)]
                if valid_runtimes:
                    avg_runtime = np.mean(valid_runtimes)
                    md_lines.append(f"- **{method}**: Average runtime = {avg_runtime:.4f}s")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown summary saved to: {output_path}")


def main_sparsity_comparison(n_seeds=None, quick=False):
    """Run sparsity ablation experiment comparing ICA and DirectLiNGAM."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_multiple_error_bars, plot_runtime_comparison, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    # Setup experiment configuration
    results_manager = ExperimentResultsManager("results_main_sparsity_comparison.npy")
    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 5000

    # Load or create results
    results_dict = results_manager.load_or_create(
        [
            "sample_sizes",
            "n_covariates",
            "n_treatments",
            "true_params",
            "treatment_effects_ica",
            "treatment_effects_directlingam",
            "aux_result_ica",
            "aux_result_directlingam",
            "runtime_ica",
            "runtime_directlingam",
            "sparse_prob",
        ]
    )

    # Run experiments if not already done
    if not results_manager.exists():
        # Configure experiment
        # NOTE(eval 2026-09-26): sparse_prob is the probability of a NON-zero
        # entry despite the sparsity label used by the original experiment.
        base_config = DataGenerationConfig(
            batch_size=n_samples, n_covariates=2 if quick else 50, n_treatments=1, slope=1.0
        )
        estimation_config = EstimationConfig(check_convergence=False, verbose=False)

        # Define parameter grid
        sparsities = np.array([0.2, 0.8]) if quick else np.linspace(0, 1.0, num=11)[:-1]
        param_grid = {"sparse_prob": sparsities}

        # Run parameter sweep with both methods
        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_multi_method_sweep(
            param_grid=param_grid,
            n_seeds=n_seeds,
            data_gen_config=base_config,
            results_dict=results_dict,
            methods=["ica", "directlingam"],
        )

        results_manager.save(results_dict)

    # Analyze and plot results
    # Validate that all result lists have the same length
    n_results = len(results_dict["sparse_prob"])
    for key in ["true_params", "treatment_effects_ica", "treatment_effects_directlingam"]:
        if key in results_dict and len(results_dict[key]) != n_results:
            raise ValueError(
                f"List length mismatch: sparse_prob has {n_results} entries, "
                f"but {key} has {len(results_dict[key])} entries. "
                f"Delete the results file and re-run experiments."
            )

    sparsities = sorted(set(results_dict["sparse_prob"]))

    # Compute statistics for each method
    # Each row has both methods' results, so we filter by parameter only
    series_data = {}
    runtime_data = {}
    for method in ["ica", "directlingam"]:
        te_key = f"treatment_effects_{method}"
        runtime_key = f"runtime_{method}"
        means = []
        stds = []
        runtime_means = []
        runtime_stds = []

        for sparsity in sparsities:
            # Filter indices for this sparsity value
            indices = [i for i, sp in enumerate(results_dict["sparse_prob"]) if sp == sparsity]

            if indices:
                errors = []
                runtimes = []
                for idx in indices:
                    true_params = results_dict["true_params"][idx]
                    est_params = results_dict[te_key][idx]
                    if hasattr(true_params, "numpy"):
                        true_params = true_params.numpy()
                    error = calculate_mse(true_params, est_params, relative_error=False)
                    errors.append(error)
                    if runtime_key in results_dict:
                        runtimes.append(results_dict[runtime_key][idx])

                means.append(np.nanmean(errors))
                stds.append(np.nanstd(errors))
                if runtimes:
                    runtime_means.append(np.nanmean(runtimes))
                    runtime_stds.append(np.nanstd(runtimes))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                runtime_means.append(np.nan)
                runtime_stds.append(np.nan)

        method_label = "FastICA" if method == "ica" else "DirectLiNGAM"
        series_data[method_label] = (means, stds)
        if runtime_means:
            runtime_data[method_label] = (runtime_means, runtime_stds)

    # Create comparison plot
    plot_multiple_error_bars(
        parameter_values=sparsities,
        series_data=series_data,
        xlabel=r"Sparsity of $\mathrm{\mathbf{A}}$",
        ylabel=r"Mean squared $|\theta-\hat{\theta}|$",
        filename=f"ica_vs_directlingam_sparsity_n{n_samples}.svg",
        use_log_scale=True,
    )

    # Create runtime comparison plot
    if runtime_data:
        plot_runtime_comparison(
            parameter_values=sparsities,
            series_data=runtime_data,
            xlabel=r"Sparsity of $\mathrm{\mathbf{A}}$",
            filename=f"ica_vs_directlingam_sparsity_runtime_n{n_samples}.svg",
            use_log_scale=True,
        )

    # Save markdown summary
    save_comparison_summary_markdown(
        parameter_name="Sparsity",
        parameter_values=sparsities,
        series_data=series_data,
        runtime_data=runtime_data,
        output_path=f"figures/ica/ica_vs_directlingam_sparsity_n{n_samples}_summary.md",
        title="ICA vs DirectLiNGAM: Sparsity Comparison",
        n_samples=n_samples,
        n_seeds=n_seeds,
    )


def main_gennorm_comparison(n_seeds=None, quick=False):
    """Run generalized normal distribution ablation comparing ICA and DirectLiNGAM."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_multiple_error_bars, plot_runtime_comparison, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    n_seeds = _seed_count(n_seeds)
    n_samples = 100 if quick else 200
    results_manager = ExperimentResultsManager(f"results_main_gennorm_comparison_{n_samples}.npy")

    results_dict = results_manager.load_or_create(
        [
            "sample_sizes",
            "n_covariates",
            "n_treatments",
            "true_params",
            "treatment_effects_ica",
            "treatment_effects_directlingam",
            "aux_result_ica",
            "aux_result_directlingam",
            "runtime_ica",
            "runtime_directlingam",
            "beta",
        ]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples, n_covariates=2 if quick else 10, n_treatments=1, slope=1.0, sparse_prob=0.3
        )
        estimation_config = EstimationConfig(check_convergence=False, verbose=False)

        beta_values = np.array([1.0, 2.0]) if quick else np.linspace(0.5, 5, num=10)
        param_grid = {"beta": beta_values}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_multi_method_sweep(
            param_grid=param_grid,
            n_seeds=n_seeds,
            data_gen_config=base_config,
            results_dict=results_dict,
            methods=["ica", "directlingam"],
        )

        results_manager.save(results_dict)

    # Analyze results
    # Validate that all result lists have the same length
    n_results = len(results_dict["beta"])
    for key in ["true_params", "treatment_effects_ica", "treatment_effects_directlingam"]:
        if key in results_dict and len(results_dict[key]) != n_results:
            raise ValueError(
                f"List length mismatch: beta has {n_results} entries, "
                f"but {key} has {len(results_dict[key])} entries. "
                f"Delete the results file and re-run experiments."
            )

    beta_values = sorted(set(results_dict["beta"]))

    series_data = {}
    runtime_data = {}
    for method in ["ica", "directlingam"]:
        te_key = f"treatment_effects_{method}"
        runtime_key = f"runtime_{method}"
        means = []
        stds = []
        runtime_means = []
        runtime_stds = []

        for beta in beta_values:
            # Filter indices for this beta value
            indices = [i for i, b in enumerate(results_dict["beta"]) if b == beta]

            if indices:
                errors = []
                runtimes = []
                for idx in indices:
                    true_params = results_dict["true_params"][idx]
                    est_params = results_dict[te_key][idx]
                    if hasattr(true_params, "numpy"):
                        true_params = true_params.numpy()
                    error = calculate_mse(true_params, est_params, relative_error=True)
                    errors.append(error)
                    if runtime_key in results_dict:
                        runtimes.append(results_dict[runtime_key][idx])

                means.append(np.nanmean(errors))
                stds.append(np.nanstd(errors))
                if runtimes:
                    runtime_means.append(np.nanmean(runtimes))
                    runtime_stds.append(np.nanstd(runtimes))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                runtime_means.append(np.nan)
                runtime_stds.append(np.nan)

        method_label = "FastICA" if method == "ica" else "DirectLiNGAM"
        series_data[method_label] = (means, stds)
        if runtime_means:
            runtime_data[method_label] = (runtime_means, runtime_stds)

    plot_multiple_error_bars(
        parameter_values=beta_values,
        series_data=series_data,
        xlabel=r"Gen. normal param. $\beta$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_vs_directlingam_gennorm_n{n_samples}.svg",
        use_log_scale=True,
    )

    # Create runtime comparison plot
    if runtime_data:
        plot_runtime_comparison(
            parameter_values=beta_values,
            series_data=runtime_data,
            xlabel=r"Gen. normal param. $\beta$",
            filename=f"ica_vs_directlingam_gennorm_runtime_n{n_samples}.svg",
            use_log_scale=True,
        )

    # Save markdown summary
    save_comparison_summary_markdown(
        parameter_name="Beta",
        parameter_values=beta_values,
        series_data=series_data,
        runtime_data=runtime_data,
        output_path=f"figures/ica/ica_vs_directlingam_gennorm_n{n_samples}_summary.md",
        title="ICA vs DirectLiNGAM: Generalized Normal Distribution Comparison",
        n_samples=n_samples,
        n_seeds=n_seeds,
    )


def main_sample_size_comparison(n_seeds=None, quick=False):
    """Run sample size ablation experiment comparing ICA and DirectLiNGAM."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_multiple_error_bars, plot_runtime_comparison, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    results_manager = ExperimentResultsManager("results_main_sample_size_comparison.npy")
    n_seeds = _seed_count(n_seeds)

    results_dict = results_manager.load_or_create(
        [
            "sample_sizes",
            "n_covariates",
            "n_treatments",
            "true_params",
            "treatment_effects_ica",
            "treatment_effects_directlingam",
            "aux_result_ica",
            "aux_result_directlingam",
            "runtime_ica",
            "runtime_directlingam",
            "batch_size",
        ]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            n_covariates=2 if quick else 10,
            n_treatments=1,
            slope=1.0,
            sparse_prob=0.3,
            beta=1.0,
        )
        estimation_config = EstimationConfig(check_convergence=False, verbose=False)

        sample_sizes = [100, 200] if quick else [100, 200, 500, 1000, 2000, 5000]
        param_grid = {"batch_size": sample_sizes}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_multi_method_sweep(
            param_grid=param_grid,
            n_seeds=n_seeds,
            data_gen_config=base_config,
            results_dict=results_dict,
            methods=["ica", "directlingam"],
        )

        results_manager.save(results_dict)

    # Analyze results
    # Validate that all result lists have the same length
    n_results = len(results_dict["batch_size"])
    for key in ["true_params", "treatment_effects_ica", "treatment_effects_directlingam"]:
        if key in results_dict and len(results_dict[key]) != n_results:
            raise ValueError(
                f"List length mismatch: batch_size has {n_results} entries, "
                f"but {key} has {len(results_dict[key])} entries. "
                f"Delete the results file and re-run experiments."
            )

    sample_sizes = sorted(set(results_dict["batch_size"]))

    series_data = {}
    runtime_data = {}
    for method in ["ica", "directlingam"]:
        te_key = f"treatment_effects_{method}"
        runtime_key = f"runtime_{method}"
        means = []
        stds = []
        runtime_means = []
        runtime_stds = []

        for n in sample_sizes:
            # Filter indices for this batch_size value
            indices = [i for i, bs in enumerate(results_dict["batch_size"]) if bs == n]

            if indices:
                errors = []
                runtimes = []
                for idx in indices:
                    true_params = results_dict["true_params"][idx]
                    est_params = results_dict[te_key][idx]
                    if hasattr(true_params, "numpy"):
                        true_params = true_params.numpy()
                    error = calculate_mse(true_params, est_params, relative_error=True)
                    errors.append(error)
                    if runtime_key in results_dict:
                        runtimes.append(results_dict[runtime_key][idx])

                means.append(np.nanmean(errors))
                stds.append(np.nanstd(errors))
                if runtimes:
                    runtime_means.append(np.nanmean(runtimes))
                    runtime_stds.append(np.nanstd(runtimes))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                runtime_means.append(np.nan)
                runtime_stds.append(np.nan)

        method_label = "FastICA" if method == "ica" else "DirectLiNGAM"
        series_data[method_label] = (means, stds)
        if runtime_means:
            runtime_data[method_label] = (runtime_means, runtime_stds)

    plot_multiple_error_bars(
        parameter_values=sample_sizes,
        series_data=series_data,
        xlabel=r"Sample size $n$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename="ica_vs_directlingam_sample_size.svg",
        use_log_scale=True,
    )

    # Create runtime comparison plot
    if runtime_data:
        plot_runtime_comparison(
            parameter_values=sample_sizes,
            series_data=runtime_data,
            xlabel=r"Sample size $n$",
            filename="ica_vs_directlingam_sample_size_runtime.svg",
            use_log_scale=True,
        )

    # Save markdown summary
    save_comparison_summary_markdown(
        parameter_name="Sample Size",
        parameter_values=sample_sizes,
        series_data=series_data,
        runtime_data=runtime_data,
        output_path="figures/ica/ica_vs_directlingam_sample_size_summary.md",
        title="ICA vs DirectLiNGAM: Sample Size Comparison",
        n_samples="varied",
        n_seeds=n_seeds,
    )


def main_n_covariates_comparison(n_seeds=None, quick=False):
    """Run covariate dimension ablation experiment comparing ICA and DirectLiNGAM."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_multiple_error_bars, plot_runtime_comparison, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    n_samples = 100 if quick else 1000
    results_manager = ExperimentResultsManager(f"results_main_n_covariates_comparison_n{n_samples}.npy")
    n_seeds = _seed_count(n_seeds)

    results_dict = results_manager.load_or_create(
        [
            "sample_sizes",
            "n_covariates",
            "n_treatments",
            "true_params",
            "treatment_effects_ica",
            "treatment_effects_directlingam",
            "aux_result_ica",
            "aux_result_directlingam",
            "runtime_ica",
            "runtime_directlingam",
        ]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(batch_size=n_samples, n_treatments=1, slope=1.0, sparse_prob=0.3, beta=1.0)
        estimation_config = EstimationConfig(check_convergence=False, verbose=False)

        n_covariates_values = [2, 3] if quick else [2, 5, 10, 20, 50]
        param_grid = {"n_covariates": n_covariates_values}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_multi_method_sweep(
            param_grid=param_grid,
            n_seeds=n_seeds,
            data_gen_config=base_config,
            results_dict=results_dict,
            methods=["ica", "directlingam"],
        )

        results_manager.save(results_dict)

    # Analyze results
    # Validate that all result lists have the same length
    n_results = len(results_dict["n_covariates"])
    for key in ["true_params", "treatment_effects_ica", "treatment_effects_directlingam"]:
        if key in results_dict and len(results_dict[key]) != n_results:
            raise ValueError(
                f"List length mismatch: n_covariates has {n_results} entries, "
                f"but {key} has {len(results_dict[key])} entries. "
                f"Delete the results file and re-run experiments."
            )

    n_covariates_values = sorted(set(results_dict["n_covariates"]))

    series_data = {}
    runtime_data = {}
    for method in ["ica", "directlingam"]:
        te_key = f"treatment_effects_{method}"
        runtime_key = f"runtime_{method}"
        means = []
        stds = []
        runtime_means = []
        runtime_stds = []

        for n_cov in n_covariates_values:
            # Filter indices for this n_covariates value
            indices = [i for i, nc in enumerate(results_dict["n_covariates"]) if nc == n_cov]

            if indices:
                errors = []
                runtimes = []
                for idx in indices:
                    true_params = results_dict["true_params"][idx]
                    est_params = results_dict[te_key][idx]
                    if hasattr(true_params, "numpy"):
                        true_params = true_params.numpy()
                    error = calculate_mse(true_params, est_params, relative_error=True)
                    errors.append(error)
                    if runtime_key in results_dict:
                        runtimes.append(results_dict[runtime_key][idx])

                means.append(np.nanmean(errors))
                stds.append(np.nanstd(errors))
                if runtimes:
                    runtime_means.append(np.nanmean(runtimes))
                    runtime_stds.append(np.nanstd(runtimes))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                runtime_means.append(np.nan)
                runtime_stds.append(np.nan)

        method_label = "FastICA" if method == "ica" else "DirectLiNGAM"
        series_data[method_label] = (means, stds)
        if runtime_means:
            runtime_data[method_label] = (runtime_means, runtime_stds)

    plot_multiple_error_bars(
        parameter_values=n_covariates_values,
        series_data=series_data,
        xlabel=r"Covariate dimension $d$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_vs_directlingam_n_covariates_n{n_samples}.svg",
        use_log_scale=True,
    )

    # Create runtime comparison plot
    if runtime_data:
        plot_runtime_comparison(
            parameter_values=n_covariates_values,
            series_data=runtime_data,
            xlabel=r"Covariate dimension $d$",
            filename=f"ica_vs_directlingam_n_covariates_runtime_n{n_samples}.svg",
            use_log_scale=True,
        )

    # Save markdown summary
    save_comparison_summary_markdown(
        parameter_name="n_covariates",
        parameter_values=n_covariates_values,
        series_data=series_data,
        runtime_data=runtime_data,
        output_path=f"figures/ica/ica_vs_directlingam_n_covariates_n{n_samples}_summary.md",
        title="ICA vs DirectLiNGAM: Covariate Dimension Comparison",
        n_samples=n_samples,
        n_seeds=n_seeds,
    )


def main_n_treatments_comparison(n_seeds=None, quick=False):
    """Run number of treatments ablation experiment comparing ICA and DirectLiNGAM."""
    from experiment_runner import ExperimentRunner
    from ica_plotting import plot_multiple_error_bars, plot_runtime_comparison, setup_experiment_environment
    from ica_utils import DataGenerationConfig, EstimationConfig, ExperimentResultsManager

    setup_experiment_environment()

    n_samples = 100 if quick else 1000
    results_manager = ExperimentResultsManager(f"results_main_n_treatments_comparison_n{n_samples}.npy")
    n_seeds = _seed_count(n_seeds)

    results_dict = results_manager.load_or_create(
        [
            "sample_sizes",
            "n_covariates",
            "n_treatments",
            "true_params",
            "treatment_effects_ica",
            "treatment_effects_directlingam",
            "aux_result_ica",
            "aux_result_directlingam",
            "runtime_ica",
            "runtime_directlingam",
        ]
    )

    if not results_manager.exists():
        base_config = DataGenerationConfig(
            batch_size=n_samples, n_covariates=2 if quick else 10, slope=1.0, sparse_prob=0.3, beta=1.0
        )
        estimation_config = EstimationConfig(check_convergence=False, verbose=False)

        n_treatments_values = [1, 2] if quick else [1, 2, 3, 4, 5]
        param_grid = {"n_treatments": n_treatments_values}

        runner = ExperimentRunner(estimation_config)
        results_dict = runner.run_multi_method_sweep(
            param_grid=param_grid,
            n_seeds=n_seeds,
            data_gen_config=base_config,
            results_dict=results_dict,
            methods=["ica", "directlingam"],
        )

        results_manager.save(results_dict)

    # Analyze results
    # Validate that all result lists have the same length
    n_results = len(results_dict["n_treatments"])
    for key in ["true_params", "treatment_effects_ica", "treatment_effects_directlingam"]:
        if key in results_dict and len(results_dict[key]) != n_results:
            raise ValueError(
                f"List length mismatch: n_treatments has {n_results} entries, "
                f"but {key} has {len(results_dict[key])} entries. "
                f"Delete the results file and re-run experiments."
            )

    n_treatments_values = sorted(set(results_dict["n_treatments"]))

    series_data = {}
    runtime_data = {}
    for method in ["ica", "directlingam"]:
        te_key = f"treatment_effects_{method}"
        runtime_key = f"runtime_{method}"
        means = []
        stds = []
        runtime_means = []
        runtime_stds = []

        for n_treat in n_treatments_values:
            # Filter indices for this n_treatments value
            indices = [i for i, nt in enumerate(results_dict["n_treatments"]) if nt == n_treat]

            if indices:
                errors = []
                runtimes = []
                for idx in indices:
                    true_params = results_dict["true_params"][idx]
                    est_params = results_dict[te_key][idx]
                    if hasattr(true_params, "numpy"):
                        true_params = true_params.numpy()
                    error = calculate_mse(true_params, est_params, relative_error=True)
                    errors.append(error)
                    if runtime_key in results_dict:
                        runtimes.append(results_dict[runtime_key][idx])

                means.append(np.nanmean(errors))
                stds.append(np.nanstd(errors))
                if runtimes:
                    runtime_means.append(np.nanmean(runtimes))
                    runtime_stds.append(np.nanstd(runtimes))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                runtime_means.append(np.nan)
                runtime_stds.append(np.nan)

        method_label = "FastICA" if method == "ica" else "DirectLiNGAM"
        series_data[method_label] = (means, stds)
        if runtime_means:
            runtime_data[method_label] = (runtime_means, runtime_stds)

    plot_multiple_error_bars(
        parameter_values=n_treatments_values,
        series_data=series_data,
        xlabel=r"Number of treatments $m$",
        ylabel=r"Mean squared $|(\theta-\hat{\theta})/\theta|$",
        filename=f"ica_vs_directlingam_n_treatments_n{n_samples}.svg",
        use_log_scale=True,
    )

    # Create runtime comparison plot
    if runtime_data:
        plot_runtime_comparison(
            parameter_values=n_treatments_values,
            series_data=runtime_data,
            xlabel=r"Number of treatments $m$",
            filename=f"ica_vs_directlingam_n_treatments_runtime_n{n_samples}.svg",
            use_log_scale=True,
        )

    # Save markdown summary
    save_comparison_summary_markdown(
        parameter_name="n_treatments",
        parameter_values=n_treatments_values,
        series_data=series_data,
        runtime_data=runtime_data,
        output_path=f"figures/ica/ica_vs_directlingam_n_treatments_n{n_samples}_summary.md",
        title="ICA vs DirectLiNGAM: Number of Treatments Comparison",
        n_samples=n_samples,
        n_seeds=n_seeds,
    )


def save_figure(filename):
    """Save the current matplotlib figure to figures/ica/ at 300 DPI and close it."""
    # Ensure the directory exists
    figures_dir = "figures/ica"
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


EXPERIMENTS = {
    "multi": main_multi,
    "nonlinear": main_nonlinear,
    "fun": main_fun,
    "sparsity": main_sparsity,
    "gennorm": main_gennorm,
    "gennorm_nonlinear": main_gennorm_nonlinear,
    "nonlinear_theta": main_nonlinear_theta,
    "nonlinear_noise_split": main_nonlinear_noise_split,
    "loc_scale": main_loc_scale,
    "sparsity_comparison": main_sparsity_comparison,
    "gennorm_comparison": main_gennorm_comparison,
    "sample_size_comparison": main_sample_size_comparison,
    "n_covariates_comparison": main_n_covariates_comparison,
    "n_treatments_comparison": main_n_treatments_comparison,
}


def main(argv=None):
    """Run one restored experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use a tiny grid for smoke testing")
    args = parser.parse_args(argv)
    if args.n_seeds is not None and args.n_seeds < 1:
        parser.error("--n_seeds must be positive")
    EXPERIMENTS[args.experiment](n_seeds=args.n_seeds, quick=args.quick)


if __name__ == "__main__":
    main()
