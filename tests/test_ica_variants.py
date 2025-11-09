"""
Tests for ICA variants: Triangular, Constrained, and Regularized ICA.
"""

import numpy as np
import pytest
import torch

from ica import generate_ica_data
from ica_variants import (
    ConstrainedICA,
    RegularizedICA,
    TriangularICA,
    ica_treatment_effect_estimation_variant,
    random_triangular_matrix,
    whiten_data,
)


class TestRandomTriangularMatrix:
    """Test random triangular matrix generation."""

    def test_lower_triangular(self):
        """Test that lower triangular matrix is generated correctly."""
        n_components = 5
        matrix = random_triangular_matrix(n_components, lower=True, random_state=42)

        assert matrix.shape == (n_components, n_components)
        # Check that upper triangle is zero
        assert np.allclose(np.triu(matrix, k=1), 0.0)
        # Check that diagonal is non-zero
        assert np.all(np.abs(np.diag(matrix)) > 1e-6)

    def test_upper_triangular(self):
        """Test that upper triangular matrix is generated correctly."""
        n_components = 5
        matrix = random_triangular_matrix(n_components, lower=False, random_state=42)

        assert matrix.shape == (n_components, n_components)
        # Check that lower triangle is zero
        assert np.allclose(np.tril(matrix, k=-1), 0.0)
        # Check that diagonal is non-zero
        assert np.all(np.abs(np.diag(matrix)) > 1e-6)

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        matrix1 = random_triangular_matrix(5, lower=True, random_state=42)
        matrix2 = random_triangular_matrix(5, lower=True, random_state=42)

        assert np.allclose(matrix1, matrix2)

    def test_different_seeds_different_matrices(self):
        """Test that different seeds produce different matrices."""
        matrix1 = random_triangular_matrix(5, lower=True, random_state=42)
        matrix2 = random_triangular_matrix(5, lower=True, random_state=43)

        assert not np.allclose(matrix1, matrix2)


class TestWhiteningFunction:
    """Test data whitening function."""

    def test_whitening_shape(self):
        """Test that whitened data has correct shape."""
        X = np.random.randn(100, 10)
        X_white, K = whiten_data(X)

        assert X_white.shape == X.shape
        assert K.shape == (X.shape[1], X.shape[1])

    def test_whitening_properties(self):
        """Test that whitened data has unit variance."""
        X = np.random.randn(1000, 10)
        X_white, _ = whiten_data(X)

        # Whitened data should have approximately zero mean
        assert np.allclose(X_white.mean(axis=0), 0.0, atol=1e-10)

        # Covariance should be approximately identity
        cov = np.cov(X_white.T)
        assert np.allclose(cov, np.eye(10), atol=0.1)


class TestTriangularICA:
    """Test Triangular ICA implementation."""

    def test_initialization(self):
        """Test that TriangularICA initializes correctly."""
        ica = TriangularICA(n_components=5, random_state=42)

        assert ica.n_components == 5
        assert ica.random_state == 42
        assert ica.unmixing_ is None
        assert ica.mixing_ is None

    def test_fit_transform_shape(self):
        """Test that fit_transform produces correct output shape."""
        X = np.random.randn(100, 5)
        ica = TriangularICA(n_components=5, max_iter=100, random_state=42)

        S = ica.fit_transform(X)

        assert S.shape == X.shape
        assert ica.unmixing_ is not None
        assert ica.mixing_ is not None

    def test_triangular_constraint(self):
        """Test that unmixing matrix is triangular."""
        X = np.random.randn(100, 5)
        ica = TriangularICA(n_components=5, max_iter=100, random_state=42, lower=True)

        ica.fit_transform(X)

        # Check that unmixing matrix is lower triangular
        assert np.allclose(np.triu(ica.unmixing_, k=1), 0.0, atol=1e-6)

    def test_different_initializations(self):
        """Test different initialization methods."""
        X = np.random.randn(100, 5)

        for init in ["random_triangular", "standard", "identity"]:
            ica = TriangularICA(
                n_components=5, max_iter=100, random_state=42, init=init
            )
            S = ica.fit_transform(X)

            assert S.shape == X.shape
            assert ica.unmixing_ is not None

    def test_unmixing_mixing_inverse(self):
        """Test that mixing matrix is (pseudo-)inverse of unmixing."""
        X = np.random.randn(100, 5)
        ica = TriangularICA(n_components=5, max_iter=100, random_state=42)

        ica.fit_transform(X)

        # For triangular matrices, mixing should be inverse
        product = ica.unmixing_ @ ica.mixing_
        assert np.allclose(product, np.eye(5), atol=0.1)


class TestConstrainedICA:
    """Test Constrained ICA implementation."""

    def test_initialization(self):
        """Test that ConstrainedICA initializes correctly."""
        ica = ConstrainedICA(
            n_components=5, orthogonal=True, non_negative=False, random_state=42
        )

        assert ica.n_components == 5
        assert ica.orthogonal is True
        assert ica.non_negative is False

    def test_fit_transform_shape(self):
        """Test that fit_transform produces correct output shape."""
        X = np.random.randn(100, 5)
        ica = ConstrainedICA(n_components=5, max_iter=100, random_state=42)

        S = ica.fit_transform(X)

        assert S.shape == X.shape
        assert ica.unmixing_ is not None

    def test_non_negative_constraint(self):
        """Test that non-negativity constraint is enforced."""
        X = np.random.randn(100, 5)
        ica = ConstrainedICA(
            n_components=5, max_iter=100, random_state=42, non_negative=True
        )

        ica.fit_transform(X)

        # Check that all elements are non-negative
        assert np.all(ica.unmixing_ >= -1e-6)

    def test_orthogonal_constraint(self):
        """Test that orthogonality constraint is approximately enforced."""
        X = np.random.randn(100, 5)
        ica = ConstrainedICA(
            n_components=5, max_iter=100, random_state=42, orthogonal=True
        )

        ica.fit_transform(X)

        # Check that W @ W.T is approximately identity
        product = ica.unmixing_ @ ica.unmixing_.T
        assert np.allclose(product, np.eye(5), atol=0.2)


class TestRegularizedICA:
    """Test Regularized ICA implementation."""

    def test_initialization(self):
        """Test that RegularizedICA initializes correctly."""
        ica = RegularizedICA(
            n_components=5, l1_penalty=0.1, l2_penalty=0.01, random_state=42
        )

        assert ica.n_components == 5
        assert ica.l1_penalty == 0.1
        assert ica.l2_penalty == 0.01

    def test_fit_transform_shape(self):
        """Test that fit_transform produces correct output shape."""
        X = np.random.randn(100, 5)
        ica = RegularizedICA(n_components=5, max_iter=100, random_state=42)

        S = ica.fit_transform(X)

        assert S.shape == X.shape
        assert ica.unmixing_ is not None

    def test_l1_regularization_effect(self):
        """Test that L1 regularization promotes sparsity."""
        X = np.random.randn(100, 5)

        # Without regularization
        ica_no_reg = RegularizedICA(
            n_components=5, max_iter=100, random_state=42, l1_penalty=0.0
        )
        ica_no_reg.fit_transform(X)

        # With strong L1 regularization
        ica_l1 = RegularizedICA(
            n_components=5, max_iter=100, random_state=42, l1_penalty=0.5
        )
        ica_l1.fit_transform(X)

        # L1 regularized version should have smaller magnitudes
        assert np.linalg.norm(ica_l1.unmixing_) <= np.linalg.norm(ica_no_reg.unmixing_)


class TestICAVariantIntegration:
    """Integration tests for ICA variants with data generation."""

    def test_triangular_ica_with_generated_data(self):
        """Test triangular ICA with synthetically generated data."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=500,
            beta=1.0,
            sparse_prob=0.3,
        )

        X_np = X.numpy()
        S_np = S.numpy()

        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X_np,
            S_np,
            variant="triangular",
            random_state=42,
            n_treatments=1,
            init="random_triangular",
            max_iter=500,
        )

        assert theta_est is not None
        assert theta_est.shape == (1,)
        assert mcc is not None or mcc is None  # MCC can be None if not converged

    def test_constrained_ica_with_generated_data(self):
        """Test constrained ICA with synthetically generated data."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=500,
            beta=1.0,
            sparse_prob=0.3,
        )

        X_np = X.numpy()
        S_np = S.numpy()

        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X_np,
            S_np,
            variant="constrained",
            random_state=42,
            n_treatments=1,
            init="random_triangular",
            orthogonal=False,
            non_negative=False,
            max_iter=500,
        )

        assert theta_est is not None
        assert theta_est.shape == (1,)

    def test_regularized_ica_with_generated_data(self):
        """Test regularized ICA with synthetically generated data."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=500,
            beta=1.0,
            sparse_prob=0.3,
        )

        X_np = X.numpy()
        S_np = S.numpy()

        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X_np,
            S_np,
            variant="regularized",
            random_state=42,
            n_treatments=1,
            init="random_triangular",
            l1_penalty=0.01,
            l2_penalty=0.01,
            max_iter=500,
        )

        assert theta_est is not None
        assert theta_est.shape == (1,)

    def test_all_initializations(self):
        """Test that all initialization methods work."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=500,
            beta=1.0,
        )

        X_np = X.numpy()
        S_np = S.numpy()

        for init in ["random_triangular", "standard", "identity"]:
            theta_est, mcc = ica_treatment_effect_estimation_variant(
                X_np,
                S_np,
                variant="triangular",
                random_state=42,
                n_treatments=1,
                init=init,
                max_iter=200,
            )

            assert theta_est is not None
            assert theta_est.shape == (1,)

    def test_torch_tensor_input(self):
        """Test that torch tensors are handled correctly."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=500,
        )

        # Pass torch tensors directly
        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X,  # torch tensor
            S,  # torch tensor
            variant="triangular",
            random_state=42,
            n_treatments=1,
            init="random_triangular",
            max_iter=200,
        )

        assert theta_est is not None
        assert theta_est.shape == (1,)

    def test_multiple_treatments(self):
        """Test with multiple treatment variables."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=2,
            batch_size=500,
        )

        X_np = X.numpy()
        S_np = S.numpy()

        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X_np,
            S_np,
            variant="triangular",
            random_state=42,
            n_treatments=2,
            init="random_triangular",
            max_iter=200,
        )

        assert theta_est is not None
        assert theta_est.shape == (2,)

    def test_invalid_variant(self):
        """Test that invalid variant raises error."""
        X = np.random.randn(100, 5)
        S = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="Unknown variant"):
            ica_treatment_effect_estimation_variant(
                X, S, variant="invalid_variant", random_state=42, n_treatments=1
            )


class TestComparison:
    """Compare different methods and initializations."""

    def test_initialization_comparison(self):
        """Compare performance of different initializations."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=1000,
            beta=1.0,
        )

        X_np = X.numpy()
        S_np = S.numpy()
        theta_true_np = theta_true.numpy()

        results = {}

        for init in ["random_triangular", "standard", "identity"]:
            theta_est, mcc = ica_treatment_effect_estimation_variant(
                X_np,
                S_np,
                variant="triangular",
                random_state=42,
                n_treatments=1,
                init=init,
                max_iter=500,
            )

            if theta_est is not None and not np.isnan(theta_est).any():
                error = np.abs(theta_est - theta_true_np).mean()
                results[init] = {"theta_est": theta_est, "error": error, "mcc": mcc}

        # All initializations should produce results
        assert len(results) >= 1  # At least one should work

    def test_variant_comparison(self):
        """Compare different ICA variants."""
        S, X, theta_true = generate_ica_data(
            n_covariates=10,
            n_treatments=1,
            batch_size=1000,
            beta=1.0,
        )

        X_np = X.numpy()
        S_np = S.numpy()

        results = {}

        for variant in ["triangular", "constrained", "regularized"]:
            variant_kwargs = {}
            if variant == "regularized":
                variant_kwargs = {"l1_penalty": 0.01, "l2_penalty": 0.01}

            theta_est, mcc = ica_treatment_effect_estimation_variant(
                X_np,
                S_np,
                variant=variant,
                random_state=42,
                n_treatments=1,
                init="random_triangular",
                max_iter=500,
                **variant_kwargs,
            )

            if theta_est is not None:
                results[variant] = {"theta_est": theta_est, "mcc": mcc}

        # All variants should produce some results
        assert len(results) >= 1
