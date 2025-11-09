"""
Comparison script for original vs fixed ICA data generation methods.

This script demonstrates the diagonal mixing matrix problem in the original
implementation and shows how the fixed version resolves it.
"""

import numpy as np
import torch
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import seaborn as sns

from ica import generate_ica_data
from ica_fixed import generate_ica_data_with_mixing, generate_ica_data_simple_mixing
from mcc import calc_disent_metrics


def analyze_mixing_matrix(mixing_matrix, title="Mixing Matrix"):
    """Analyze and print statistics about a mixing matrix."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"Shape: {mixing_matrix.shape}")
    print(f"\nMatrix:\n{mixing_matrix}")

    # Compute diagonality metrics
    diag_elements = np.abs(np.diag(mixing_matrix))
    off_diag_mask = ~np.eye(mixing_matrix.shape[0], dtype=bool)
    off_diag_elements = np.abs(mixing_matrix[off_diag_mask])

    mean_diag = np.mean(diag_elements)
    mean_off_diag = np.mean(off_diag_elements)

    print(f"\nDiagonal Statistics:")
    print(f"  Mean |diagonal|: {mean_diag:.4f}")
    print(f"  Mean |off-diagonal|: {mean_off_diag:.4f}")
    print(f"  Ratio (diag/off-diag): {mean_diag / (mean_off_diag + 1e-10):.4f}")

    diag_norm = np.linalg.norm(diag_elements)
    off_diag_norm = np.linalg.norm(off_diag_elements)
    diagonality = diag_norm / (diag_norm + off_diag_norm + 1e-10)

    print(f"\nDiagonality Measure: {diagonality:.4f}")
    print(f"  (1.0 = perfectly diagonal, 0.5 = balanced)")

    return {
        "mean_diag": mean_diag,
        "mean_off_diag": mean_off_diag,
        "ratio": mean_diag / (mean_off_diag + 1e-10),
        "diagonality": diagonality,
    }


def compare_methods():
    """Compare original and fixed ICA data generation methods."""

    print("=" * 80)
    print("COMPARING ORIGINAL VS FIXED ICA DATA GENERATION")
    print("=" * 80)

    # Configuration
    n_covariates = 5
    n_treatments = 1
    batch_size = 2000
    random_state = 42

    # ==========================================================================
    # Method 1: Original (problematic)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: ORIGINAL (Problematic)")
    print("=" * 80)

    S_orig, X_orig, theta_orig = generate_ica_data(
        n_covariates=n_covariates,
        n_treatments=n_treatments,
        batch_size=batch_size,
        slope=1.0,
        sparse_prob=0.3,
        beta=1.0,
    )

    print(f"\nTrue treatment effect (theta): {theta_orig.numpy()}")

    # Check if covariates are identical to sources
    covariate_match = torch.allclose(
        S_orig[:, :n_covariates], X_orig[:, :n_covariates], atol=1e-6
    )
    print(f"\nCovariates identical to sources: {covariate_match}")
    if covariate_match:
        print("  ⚠️  PROBLEM: Covariates are direct copies of sources!")

    # Fit ICA
    ica_orig = FastICA(
        n_components=X_orig.shape[1], random_state=random_state, max_iter=1000
    )
    S_hat_orig = ica_orig.fit_transform(X_orig.numpy())

    results_orig = calc_disent_metrics(S_orig.numpy(), S_hat_orig)
    permuted_mixing_orig = ica_orig.mixing_[
        :, results_orig["munkres_sort_idx"].astype(int)
    ]
    scaled_mixing_orig = permuted_mixing_orig / permuted_mixing_orig.diagonal()

    stats_orig = analyze_mixing_matrix(scaled_mixing_orig, "Original Method - Mixing Matrix (scaled)")

    treatment_effect_est_orig = scaled_mixing_orig[-1, n_covariates:-1]
    print(f"\nEstimated treatment effect: {treatment_effect_est_orig}")
    print(f"True treatment effect: {theta_orig.numpy()}")
    print(f"Absolute error: {np.abs(treatment_effect_est_orig - theta_orig.numpy())}")
    print(f"MCC score: {results_orig['permutation_disentanglement_score']:.4f}")

    # ==========================================================================
    # Method 2: Fixed with random mixing
    # ==========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: FIXED (Random Mixing)")
    print("=" * 80)

    S_fixed, X_fixed, theta_fixed, mixing_info = generate_ica_data_with_mixing(
        n_covariates=n_covariates,
        n_treatments=n_treatments,
        batch_size=batch_size,
        slope=1.0,
        sparse_prob=0.3,
        beta=1.0,
        mixing_type="random",
        mixing_strength=1.0,
        n_extra_latent=0,
    )

    print(f"\nTrue treatment effect (theta): {theta_fixed.numpy()}")
    print(f"\nCovariate mixing matrix shape: {mixing_info['A_cov'].shape}")
    print(f"Covariate mixing matrix:\n{mixing_info['A_cov']}")

    # Check if covariates are identical to sources
    covariate_match_fixed = torch.allclose(
        S_fixed[:, :n_covariates], X_fixed[:, :n_covariates], atol=1e-6
    )
    print(f"\nCovariates identical to sources: {covariate_match_fixed}")
    if not covariate_match_fixed:
        print("  ✓ GOOD: Covariates are proper mixtures of sources!")

    # Fit ICA
    ica_fixed = FastICA(
        n_components=X_fixed.shape[1], random_state=random_state, max_iter=1000
    )
    S_hat_fixed = ica_fixed.fit_transform(X_fixed.numpy())

    results_fixed = calc_disent_metrics(S_fixed.numpy(), S_hat_fixed)
    permuted_mixing_fixed = ica_fixed.mixing_[
        :, results_fixed["munkres_sort_idx"].astype(int)
    ]
    scaled_mixing_fixed = permuted_mixing_fixed / permuted_mixing_fixed.diagonal()

    stats_fixed = analyze_mixing_matrix(scaled_mixing_fixed, "Fixed Method - Mixing Matrix (scaled)")

    treatment_effect_est_fixed = scaled_mixing_fixed[-1, n_covariates:-1]
    print(f"\nEstimated treatment effect: {treatment_effect_est_fixed}")
    print(f"True treatment effect: {theta_fixed.numpy()}")
    print(f"Absolute error: {np.abs(treatment_effect_est_fixed - theta_fixed.numpy())}")
    print(f"MCC score: {results_fixed['permutation_disentanglement_score']:.4f}")

    # ==========================================================================
    # Method 3: Fixed with orthogonal mixing
    # ==========================================================================
    print("\n" + "=" * 80)
    print("METHOD 3: FIXED (Random Orthogonal Mixing)")
    print("=" * 80)

    S_ortho, X_ortho, theta_ortho, mixing_info_ortho = generate_ica_data_with_mixing(
        n_covariates=n_covariates,
        n_treatments=n_treatments,
        batch_size=batch_size,
        slope=1.0,
        sparse_prob=0.3,
        beta=1.0,
        mixing_type="random_orthogonal",
        mixing_strength=1.0,
        n_extra_latent=0,
    )

    print(f"\nTrue treatment effect (theta): {theta_ortho.numpy()}")
    print(f"\nCovariate mixing matrix (orthogonal):\n{mixing_info_ortho['A_cov']}")

    # Fit ICA
    ica_ortho = FastICA(
        n_components=X_ortho.shape[1], random_state=random_state, max_iter=1000
    )
    S_hat_ortho = ica_ortho.fit_transform(X_ortho.numpy())

    results_ortho = calc_disent_metrics(S_ortho.numpy(), S_hat_ortho)
    permuted_mixing_ortho = ica_ortho.mixing_[
        :, results_ortho["munkres_sort_idx"].astype(int)
    ]
    scaled_mixing_ortho = permuted_mixing_ortho / permuted_mixing_ortho.diagonal()

    stats_ortho = analyze_mixing_matrix(
        scaled_mixing_ortho, "Orthogonal Method - Mixing Matrix (scaled)"
    )

    treatment_effect_est_ortho = scaled_mixing_ortho[-1, n_covariates:-1]
    print(f"\nEstimated treatment effect: {treatment_effect_est_ortho}")
    print(f"True treatment effect: {theta_ortho.numpy()}")
    print(f"Absolute error: {np.abs(treatment_effect_est_ortho - theta_ortho.numpy())}")
    print(f"MCC score: {results_ortho['permutation_disentanglement_score']:.4f}")

    # ==========================================================================
    # Summary comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print("\nDiagonality Measures (lower is better for ICA):")
    print(f"  Original:         {stats_orig['diagonality']:.4f}")
    print(f"  Fixed (random):   {stats_fixed['diagonality']:.4f}")
    print(f"  Fixed (orthog):   {stats_ortho['diagonality']:.4f}")

    print("\nDiagonal/Off-diagonal Ratio (lower is better):")
    print(f"  Original:         {stats_orig['ratio']:.4f}")
    print(f"  Fixed (random):   {stats_fixed['ratio']:.4f}")
    print(f"  Fixed (orthog):   {stats_ortho['ratio']:.4f}")

    print("\nMCC Scores (higher is better):")
    print(f"  Original:         {results_orig['permutation_disentanglement_score']:.4f}")
    print(f"  Fixed (random):   {results_fixed['permutation_disentanglement_score']:.4f}")
    print(f"  Fixed (orthog):   {results_ortho['permutation_disentanglement_score']:.4f}")

    # Compute correlation matrices for visualization
    return {
        "original": {
            "mixing": scaled_mixing_orig,
            "stats": stats_orig,
            "mcc": results_orig["permutation_disentanglement_score"],
            "theta_true": theta_orig.numpy(),
            "theta_est": treatment_effect_est_orig,
        },
        "fixed_random": {
            "mixing": scaled_mixing_fixed,
            "stats": stats_fixed,
            "mcc": results_fixed["permutation_disentanglement_score"],
            "theta_true": theta_fixed.numpy(),
            "theta_est": treatment_effect_est_fixed,
        },
        "fixed_orthogonal": {
            "mixing": scaled_mixing_ortho,
            "stats": stats_ortho,
            "mcc": results_ortho["permutation_disentanglement_score"],
            "theta_true": theta_ortho.numpy(),
            "theta_est": treatment_effect_est_ortho,
        },
    }


def visualize_comparison(results):
    """Create visualization comparing the methods."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = ["original", "fixed_random", "fixed_orthogonal"]
    titles = ["Original (Diagonal)", "Fixed (Random)", "Fixed (Orthogonal)"]

    for ax, method, title in zip(axes, methods, titles):
        mixing = results[method]["mixing"]
        diagonality = results[method]["stats"]["diagonality"]
        mcc = results[method]["mcc"]

        sns.heatmap(
            mixing,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            cbar_kws={"label": "Mixing coefficient"},
        )
        ax.set_title(f"{title}\nDiagonality: {diagonality:.3f}, MCC: {mcc:.3f}")
        ax.set_xlabel("Source Index")
        ax.set_ylabel("Observation Index")

    plt.tight_layout()
    plt.savefig("figures/ica/mixing_matrix_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: figures/ica/mixing_matrix_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Ensure output directory exists
    import os
    os.makedirs("figures/ica", exist_ok=True)

    # Run comparison
    results = compare_methods()

    # Create visualization
    try:
        visualize_comparison(results)
    except Exception as e:
        print(f"\nNote: Visualization skipped due to: {e}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The original implementation creates a diagonal (or near-diagonal) mixing matrix
because covariates are direct copies of independent sources. This defeats the
purpose of ICA.

The fixed implementations create proper non-diagonal mixing matrices by:
1. Generating latent sources
2. Creating observed covariates as mixtures of latent sources
3. Maintaining causal structure for treatments and outcomes

The orthogonal mixing version is particularly well-conditioned and may provide
the best balance of ICA recoverability and numerical stability.
    """)
