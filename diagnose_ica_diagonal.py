"""Diagnostic script to investigate diagonal mixing matrix issue in ICA."""

import numpy as np
import torch
from sklearn.decomposition import FastICA

from ica import generate_ica_data
from mcc import calc_disent_metrics


def diagnose_mixing_matrix():
    """Generate data and examine the mixing matrix structure."""

    # Generate ICA data with simple configuration
    n_covariates = 5
    n_treatments = 1
    batch_size = 1000

    S, X, theta = generate_ica_data(
        n_covariates=n_covariates,
        n_treatments=n_treatments,
        batch_size=batch_size,
        slope=1.0,
        sparse_prob=0.3,
        beta=1.0,
    )

    print("=" * 70)
    print("DIAGNOSTIC: ICA Mixing Matrix Analysis")
    print("=" * 70)
    print(f"\nData shape: {X.shape}")
    print(f"n_covariates: {n_covariates}, n_treatments: {n_treatments}")
    print(f"True treatment effect (theta): {theta.numpy()}")

    # Fit ICA
    ica = FastICA(n_components=X.shape[1], random_state=42, max_iter=1000)
    S_hat = ica.fit_transform(X)

    print("\n" + "=" * 70)
    print("MIXING MATRIX (before permutation/scaling):")
    print("=" * 70)
    print(ica.mixing_)

    # Check if mixing matrix is approximately diagonal
    mixing = ica.mixing_
    diag_elements = np.abs(np.diag(mixing))
    off_diag_elements = np.abs(mixing - np.diag(np.diag(mixing)))

    print(f"\nDiagonal elements (abs): {diag_elements}")
    print(f"Mean diagonal element: {np.mean(diag_elements):.4f}")
    print(f"Mean off-diagonal element: {np.mean(off_diag_elements):.4f}")
    print(f"Ratio (diag/off-diag): {np.mean(diag_elements) / (np.mean(off_diag_elements) + 1e-10):.4f}")

    # Calculate diagonality measure
    diag_norm = np.linalg.norm(diag_elements)
    off_diag_norm = np.linalg.norm(off_diag_elements)
    diagonality = diag_norm / (diag_norm + off_diag_norm)
    print(f"\nDiagonality measure: {diagonality:.4f} (1.0 = perfectly diagonal)")

    # Apply permutation and scaling
    results = calc_disent_metrics(S, S_hat)
    permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
    permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

    print("\n" + "=" * 70)
    print("MIXING MATRIX (after permutation and scaling):")
    print("=" * 70)
    print(permuted_scaled_mixing)

    # Extract treatment effect
    treatment_effect_estimate = permuted_scaled_mixing[-1, n_covariates:-1]
    print(f"\nExtracted treatment effect: {treatment_effect_estimate}")
    print(f"True treatment effect: {theta.numpy()}")
    print(f"Error: {np.abs(treatment_effect_estimate - theta.numpy())}")

    # Analyze the data generation
    print("\n" + "=" * 70)
    print("DATA GENERATION ANALYSIS:")
    print("=" * 70)

    # Check correlation between S and X
    S_np = S.numpy()
    X_np = X.numpy()

    print("\nCorrelation between sources S and observations X:")
    for i in range(S.shape[1]):
        for j in range(X.shape[1]):
            corr = np.corrcoef(S_np[:, i], X_np[:, j])[0, 1]
            if np.abs(corr) > 0.5:
                print(f"  S[{i}] <-> X[{j}]: {corr:.4f}")

    # Check if covariates are identical to sources
    covariate_match = torch.allclose(S[:, :n_covariates], X[:, :n_covariates], atol=1e-6)
    print(f"\nCovariates X[:, :n_covariates] == S[:, :n_covariates]: {covariate_match}")

    if covariate_match:
        print("  ⚠️  ISSUE IDENTIFIED: Covariates are direct copies of sources!")
        print("  This causes the mixing matrix to be diagonal for covariate components.")

    return ica, S, X, theta, permuted_scaled_mixing


if __name__ == "__main__":
    diagnose_mixing_matrix()
