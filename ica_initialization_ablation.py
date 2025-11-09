"""
ICA Initialization Ablation Study

This script performs an ablation study comparing different initialization strategies
(standard vs random triangular) for triangular, constrained, and regularized ICA methods.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles

from ica import generate_ica_data
from ica_variants import ica_treatment_effect_estimation_variant
from ica_utils import calculate_mse
from plot_utils import plot_typography


def run_initialization_ablation(
    n_covariates=50,
    n_treatments=1,
    batch_size=5000,
    n_seeds=20,
    output_dir="figures/ica/initialization_ablation",
    beta=1.0,
    sparse_prob=0.3,
    nonlinearity="leaky_relu",
):
    """
    Run ablation study comparing initialization strategies across ICA variants.

    Parameters
    ----------
    n_covariates : int
        Number of covariates
    n_treatments : int
        Number of treatment variables
    batch_size : int
        Sample size
    n_seeds : int
        Number of random seeds to test
    output_dir : str
        Directory to save results
    beta : float
        Shape parameter for generalized normal distribution
    sparse_prob : float
        Sparsity probability for treatment effects
    nonlinearity : str
        Nonlinearity type for data generation

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing all results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define experimental conditions
    variants = ["triangular", "constrained", "regularized"]
    initializations = ["random_triangular", "standard", "identity"]

    # Storage for results
    all_results = []

    print(f"Running initialization ablation with {n_seeds} seeds...")
    print(f"Variants: {variants}")
    print(f"Initializations: {initializations}")

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")

        # Generate data
        S, X, theta_true = generate_ica_data(
            n_covariates=n_covariates,
            n_treatments=n_treatments,
            batch_size=batch_size,
            beta=beta,
            sparse_prob=sparse_prob,
            nonlinearity=nonlinearity,
        )

        # Convert to numpy
        X_np = X.numpy()
        S_np = S.numpy()
        theta_true_np = theta_true.numpy()

        # Test each combination of variant and initialization
        for variant in variants:
            for init in initializations:
                print(f"  Testing {variant} with {init} initialization...")

                # Set variant-specific parameters
                variant_kwargs = {}
                if variant == "constrained":
                    variant_kwargs = {"orthogonal": False, "non_negative": False}
                elif variant == "regularized":
                    variant_kwargs = {"l1_penalty": 0.01, "l2_penalty": 0.01}

                try:
                    theta_est, mcc = ica_treatment_effect_estimation_variant(
                        X_np,
                        S_np,
                        variant=variant,
                        random_state=seed,
                        n_treatments=n_treatments,
                        verbose=False,
                        init=init,
                        max_iter=1000,
                        learning_rate=0.01,
                        **variant_kwargs,
                    )

                    # Calculate errors
                    if theta_est is not None and not np.any(np.isnan(theta_est)):
                        abs_error = calculate_mse(theta_est, theta_true_np, relative=False)
                        rel_error = calculate_mse(theta_est, theta_true_np, relative=True)
                        converged = True
                    else:
                        abs_error = np.nan
                        rel_error = np.nan
                        converged = False
                        mcc = np.nan

                    # Store results
                    all_results.append({
                        "seed": seed,
                        "variant": variant,
                        "initialization": init,
                        "theta_true": theta_true_np[0] if n_treatments == 1 else theta_true_np,
                        "theta_est": theta_est[0] if (theta_est is not None and n_treatments == 1) else theta_est,
                        "abs_error": abs_error,
                        "rel_error": rel_error,
                        "mcc": mcc,
                        "converged": converged,
                    })

                except Exception as e:
                    print(f"    Error: {e}")
                    all_results.append({
                        "seed": seed,
                        "variant": variant,
                        "initialization": init,
                        "theta_true": theta_true_np[0] if n_treatments == 1 else theta_true_np,
                        "theta_est": np.nan,
                        "abs_error": np.nan,
                        "rel_error": np.nan,
                        "mcc": np.nan,
                        "converged": False,
                    })

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save raw results
    results_file = os.path.join(output_dir, "initialization_ablation_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    return results_df


def plot_ablation_results(results_df, output_dir="figures/ica/initialization_ablation"):
    """
    Create visualizations for the ablation study results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_initialization_ablation
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Apply plotting style
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    # Filter out non-converged runs
    results_converged = results_df[results_df["converged"] == True].copy()

    if len(results_converged) == 0:
        print("Warning: No converged runs found!")
        return

    print(f"\nConvergence rate: {len(results_converged)}/{len(results_df)} "
          f"({100 * len(results_converged) / len(results_df):.1f}%)")

    # 1. Absolute Error by Variant and Initialization
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=results_converged,
        x="variant",
        y="abs_error",
        hue="initialization",
        ax=ax
    )
    ax.set_xlabel("ICA Variant")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Treatment Effect Estimation Error by Initialization")
    ax.legend(title="Initialization", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "abs_error_by_initialization.pdf"))
    plt.close()

    # 2. Relative Error by Variant and Initialization
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=results_converged,
        x="variant",
        y="rel_error",
        hue="initialization",
        ax=ax
    )
    ax.set_xlabel("ICA Variant")
    ax.set_ylabel("Relative Error")
    ax.set_title("Relative Treatment Effect Error by Initialization")
    ax.legend(title="Initialization", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rel_error_by_initialization.pdf"))
    plt.close()

    # 3. MCC Scores by Variant and Initialization
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=results_converged,
        x="variant",
        y="mcc",
        hue="initialization",
        ax=ax
    )
    ax.set_xlabel("ICA Variant")
    ax.set_ylabel("MCC Score")
    ax.set_title("Disentanglement Quality by Initialization")
    ax.legend(title="Initialization", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcc_by_initialization.pdf"))
    plt.close()

    # 4. Convergence Rate by Variant and Initialization
    convergence_df = results_df.groupby(["variant", "initialization"])["converged"].agg(
        ["sum", "count"]
    ).reset_index()
    convergence_df["convergence_rate"] = convergence_df["sum"] / convergence_df["count"]

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_data = convergence_df.pivot(index="variant", columns="initialization", values="convergence_rate")
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Initialization")
    ax.set_ylabel("ICA Variant")
    ax.set_title("Convergence Rate by Method and Initialization")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_rate_heatmap.pdf"))
    plt.close()

    # 5. Summary statistics table
    summary_stats = results_converged.groupby(["variant", "initialization"]).agg({
        "abs_error": ["mean", "std"],
        "rel_error": ["mean", "std"],
        "mcc": ["mean", "std"],
    }).round(4)

    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_file)
    print(f"\nSummary statistics saved to {summary_file}")
    print("\nSummary Statistics:")
    print(summary_stats)

    # 6. Pairwise comparison plot (initialization comparison for each variant)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, variant in enumerate(["triangular", "constrained", "regularized"]):
        variant_data = results_converged[results_converged["variant"] == variant]

        axes[idx].violinplot(
            [variant_data[variant_data["initialization"] == init]["abs_error"].dropna()
             for init in ["random_triangular", "standard", "identity"]],
            positions=[0, 1, 2],
            showmeans=True,
        )
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_xticklabels(["Random Tri.", "Standard", "Identity"], rotation=15)
        axes[idx].set_ylabel("Absolute Error")
        axes[idx].set_title(f"{variant.capitalize()} ICA")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "initialization_comparison_violin.pdf"))
    plt.close()

    print(f"\nAll plots saved to {output_dir}/")


def run_extended_ablation(
    n_seeds=20,
    output_dir="figures/ica/initialization_ablation_extended",
):
    """
    Run extended ablation across multiple parameter settings.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds per configuration
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Test across different sparsity levels
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    all_extended_results = []

    for sparse_prob in sparsity_levels:
        print(f"\n{'='*60}")
        print(f"Testing sparsity level: {sparse_prob}")
        print(f"{'='*60}")

        results_df = run_initialization_ablation(
            n_seeds=n_seeds,
            sparse_prob=sparse_prob,
            output_dir=os.path.join(output_dir, f"sparse_{sparse_prob:.1f}"),
        )

        results_df["sparse_prob"] = sparse_prob
        all_extended_results.append(results_df)

    # Combine all results
    combined_df = pd.concat(all_extended_results, ignore_index=True)

    # Save combined results
    combined_file = os.path.join(output_dir, "extended_ablation_results.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"\nCombined results saved to {combined_file}")

    # Create sparsity-specific plots
    plot_sparsity_ablation(combined_df, output_dir)

    return combined_df


def plot_sparsity_ablation(results_df, output_dir):
    """
    Plot results across different sparsity levels.

    Parameters
    ----------
    results_df : pd.DataFrame
        Combined results across sparsity levels
    output_dir : str
        Directory to save plots
    """
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    results_converged = results_df[results_df["converged"] == True].copy()

    # Plot absolute error vs sparsity for each variant/initialization combo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, variant in enumerate(["triangular", "constrained", "regularized"]):
        variant_data = results_converged[results_converged["variant"] == variant]

        for init in ["random_triangular", "standard", "identity"]:
            init_data = variant_data[variant_data["initialization"] == init]
            grouped = init_data.groupby("sparse_prob")["abs_error"].agg(["mean", "std"])

            axes[idx].errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                label=init,
                marker="o",
                capsize=5,
            )

        axes[idx].set_xlabel("Sparsity Probability")
        axes[idx].set_ylabel("Absolute Error (Mean ± Std)")
        axes[idx].set_title(f"{variant.capitalize()} ICA")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sparsity_vs_error.pdf"))
    plt.close()

    print(f"Sparsity ablation plots saved to {output_dir}/")


def main():
    """Run the main ablation study."""
    print("="*60)
    print("ICA Initialization Ablation Study")
    print("="*60)

    # Basic ablation
    print("\n1. Running basic initialization ablation...")
    results_df = run_initialization_ablation(n_seeds=20)

    print("\n2. Creating visualizations...")
    plot_ablation_results(results_df)

    # Extended ablation (uncomment to run)
    # print("\n3. Running extended ablation across sparsity levels...")
    # extended_results_df = run_extended_ablation(n_seeds=10)

    print("\n" + "="*60)
    print("Ablation study complete!")
    print("="*60)


if __name__ == "__main__":
    main()
