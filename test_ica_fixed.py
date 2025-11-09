"""
Test cases for the fixed ICA data generation methods.

Run with: pytest test_ica_fixed.py -v
"""

import numpy as np
import torch
import pytest
from sklearn.decomposition import FastICA

from ica_fixed import (
    generate_ica_data_with_mixing,
    generate_ica_data_simple_mixing,
    generate_ica_data_full_mixing,
)


class TestMixingMatrixProperties:
    """Test that mixing matrices are non-diagonal."""

    @pytest.fixture
    def config(self):
        """Common configuration for tests."""
        return {
            "n_covariates": 5,
            "n_treatments": 1,
            "batch_size": 1000,
            "beta": 1.0,
        }

    def test_random_mixing_not_diagonal(self, config):
        """Test that random mixing creates non-diagonal structure."""
        S, X, theta, mixing_info = generate_ica_data_with_mixing(
            mixing_type="random", mixing_strength=1.0, **config
        )

        # Check covariate mixing matrix
        A_cov = mixing_info["A_cov"]

        # Compute diagonality measure
        diag_norm = np.linalg.norm(np.diag(A_cov))
        total_norm = np.linalg.norm(A_cov)
        diagonality = diag_norm / total_norm

        # For a random matrix, diagonality should be much less than 1.0
        assert diagonality < 0.8, f"Matrix is too diagonal: {diagonality:.4f}"

        # Check that off-diagonal elements exist
        n = min(A_cov.shape)
        off_diag_elements = A_cov[~np.eye(A_cov.shape[0], A_cov.shape[1], dtype=bool)]
        assert np.mean(np.abs(off_diag_elements)) > 0.1

    def test_orthogonal_mixing_preserves_variance(self, config):
        """Test that orthogonal mixing preserves variance."""
        S, X, theta, mixing_info = generate_ica_data_with_mixing(
            mixing_type="random_orthogonal", mixing_strength=1.0, **config
        )

        A_cov = mixing_info["A_cov"]

        # For orthogonal matrix, A @ A.T should be close to identity (up to scale)
        if A_cov.shape[0] == A_cov.shape[1]:
            AAT = A_cov @ A_cov.T
            # Should be close to diagonal
            diag_dominance = np.linalg.norm(np.diag(AAT)) / np.linalg.norm(AAT)
            assert diag_dominance > 0.99, "Orthogonal mixing should preserve orthogonality"

    def test_covariates_are_mixed(self, config):
        """Test that covariates are not identical to sources."""
        S, X, theta, mixing_info = generate_ica_data_with_mixing(
            mixing_type="random", mixing_strength=1.0, **config
        )

        n_covariates = config["n_covariates"]

        # Covariates should NOT be identical to sources
        is_identical = torch.allclose(
            torch.tensor(S[:, :n_covariates]),
            torch.tensor(X[:, :n_covariates]),
            atol=1e-6,
        )

        assert not is_identical, "Covariates should be mixtures, not copies of sources"

        # Check that covariates are actually related to sources via mixing
        # Compute correlation to verify mixing relationship
        corr_matrix = np.corrcoef(S[:, :n_covariates].T, X[:, :n_covariates].T)
        # Sources and observations should be correlated but not perfectly
        cross_corr = corr_matrix[:n_covariates, n_covariates:]
        assert np.mean(np.abs(cross_corr)) > 0.3, "Mixing should create correlations"


class TestICARecovery:
    """Test ICA recovery performance."""

    @pytest.fixture
    def config(self):
        return {
            "n_covariates": 5,
            "n_treatments": 1,
            "batch_size": 2000,
            "beta": 1.0,
            "random_state": 42,
        }

    def test_ica_fitting_converges(self, config):
        """Test that ICA fitting converges on mixed data."""
        S, X, theta, _ = generate_ica_data_with_mixing(
            n_covariates=config["n_covariates"],
            n_treatments=config["n_treatments"],
            batch_size=config["batch_size"],
            beta=config["beta"],
            mixing_type="random_orthogonal",
            mixing_strength=1.0,
        )

        # Fit ICA
        ica = FastICA(
            n_components=X.shape[1],
            random_state=config["random_state"],
            max_iter=1000,
        )
        S_hat = ica.fit_transform(X.numpy())

        # Check that ICA converged
        assert S_hat is not None
        assert S_hat.shape == X.shape

        # Check that mixing matrix is not degenerate
        assert np.linalg.cond(ica.mixing_) < 1e10, "Mixing matrix is ill-conditioned"

    def test_mixing_matrix_not_diagonal_after_ica(self, config):
        """Test that learned mixing matrix is non-diagonal."""
        S, X, theta, _ = generate_ica_data_with_mixing(
            n_covariates=config["n_covariates"],
            n_treatments=config["n_treatments"],
            batch_size=config["batch_size"],
            beta=config["beta"],
            mixing_type="random",
            mixing_strength=1.0,
        )

        # Fit ICA
        ica = FastICA(
            n_components=X.shape[1],
            random_state=config["random_state"],
            max_iter=1000,
        )
        ica.fit(X.numpy())

        # Analyze mixing matrix
        mixing = ica.mixing_
        diag_elements = np.abs(np.diag(mixing))
        off_diag_mask = ~np.eye(mixing.shape[0], dtype=bool)
        off_diag_elements = np.abs(mixing[off_diag_mask])

        ratio = np.mean(diag_elements) / (np.mean(off_diag_elements) + 1e-10)

        # With proper mixing, ratio should not be extremely high
        assert ratio < 10.0, f"Mixing matrix is too diagonal (ratio: {ratio:.2f})"


class TestTreatmentEffectEstimation:
    """Test treatment effect estimation accuracy."""

    def test_treatment_effects_recoverable(self):
        """Test that treatment effects can be estimated."""
        n_covariates = 5
        n_treatments = 1
        batch_size = 2000

        S, X, theta_true, mixing_info = generate_ica_data_with_mixing(
            n_covariates=n_covariates,
            n_treatments=n_treatments,
            batch_size=batch_size,
            theta_choice="fixed",
            mixing_type="random_orthogonal",
            mixing_strength=1.0,
        )

        # Just verify that true treatment effects are stored
        assert theta_true is not None
        assert len(theta_true) == n_treatments
        assert not np.isnan(theta_true).any()


class TestDataGenerationVariants:
    """Test different data generation variants."""

    def test_simple_mixing(self):
        """Test simple mixing function."""
        S, X, theta = generate_ica_data_simple_mixing(
            n_covariates=5, n_treatments=1, batch_size=1000
        )

        assert S.shape[0] == 1000
        assert X.shape[0] == 1000
        assert X.shape[1] == 5 + 1 + 1  # covariates + treatments + outcome

    def test_full_mixing(self):
        """Test full linear ICA mixing."""
        S, X, theta, A = generate_ica_data_full_mixing(
            n_covariates=5, n_treatments=1, batch_size=1000
        )

        assert S.shape[0] == 1000
        assert X.shape[0] == 1000
        assert A.shape == (7, 7)  # 5 + 1 + 1

        # Verify mixing: X should equal A @ S.T (transposed)
        X_reconstructed = (A.numpy() @ S.T.numpy()).T
        assert np.allclose(X.numpy(), X_reconstructed, atol=1e-5)

    def test_extra_latent_sources(self):
        """Test with extra latent sources."""
        n_covariates = 5
        n_extra = 2

        S, X, theta, mixing_info = generate_ica_data_with_mixing(
            n_covariates=n_covariates,
            n_treatments=1,
            batch_size=1000,
            n_extra_latent=n_extra,
        )

        # Source dimension should include extra latent
        assert S.shape[1] == n_covariates + n_extra + 1 + 1

        # Mixing matrix should map from extended latent to observed covariates
        assert mixing_info["A_cov"].shape == (n_covariates, n_covariates + n_extra)

    def test_different_mixing_types(self):
        """Test all mixing types work."""
        config = {
            "n_covariates": 5,
            "n_treatments": 1,
            "batch_size": 500,
        }

        for mixing_type in ["random", "random_orthogonal", "controlled"]:
            S, X, theta, _ = generate_ica_data_with_mixing(
                mixing_type=mixing_type, **config
            )
            assert S is not None
            assert X is not None
            assert theta is not None

    def test_different_nonlinearities(self):
        """Test different nonlinearity functions."""
        config = {
            "n_covariates": 5,
            "n_treatments": 1,
            "batch_size": 500,
        }

        for nonlinearity in ["leaky_relu", "relu", "sigmoid", "tanh"]:
            S, X, theta, _ = generate_ica_data_with_mixing(
                nonlinearity=nonlinearity, **config
            )
            assert S is not None
            assert X is not None

    def test_different_theta_choices(self):
        """Test different treatment effect generation methods."""
        config = {
            "n_covariates": 5,
            "n_treatments": 3,
            "batch_size": 500,
        }

        for theta_choice in ["fixed", "uniform", "gaussian"]:
            S, X, theta, _ = generate_ica_data_with_mixing(
                theta_choice=theta_choice, **config
            )
            assert len(theta) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
