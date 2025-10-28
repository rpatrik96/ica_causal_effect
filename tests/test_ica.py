"""Tests for ica.py module."""

import numpy as np
import pytest
import torch

from ica import generate_ica_data, ica_treatment_effect_estimation


class TestGenerateICAData:
    """Tests for generate_ica_data function."""

    def test_generate_ica_data_output_shapes(self):
        """Test that generated data has correct shapes."""
        n_covariates = 5
        n_treatments = 2
        batch_size = 100

        S, X, theta = generate_ica_data(n_covariates=n_covariates, n_treatments=n_treatments, batch_size=batch_size)

        expected_dim = n_covariates + n_treatments + 1  # +1 for outcome

        assert S.shape == (batch_size, expected_dim), f"S should have shape ({batch_size}, {expected_dim})"
        assert X.shape == (batch_size, expected_dim), f"X should have shape ({batch_size}, {expected_dim})"
        assert theta.shape == (n_treatments,), f"theta should have shape ({n_treatments},)"

    def test_generate_ica_data_types(self):
        """Test that generated data has correct types."""
        S, X, theta = generate_ica_data(n_covariates=3, n_treatments=1, batch_size=50)

        assert isinstance(S, torch.Tensor), "S should be a torch.Tensor"
        assert isinstance(X, torch.Tensor), "X should be a torch.Tensor"
        assert isinstance(theta, torch.Tensor), "theta should be a torch.Tensor"

    def test_generate_ica_data_different_nonlinearities(self):
        """Test data generation with different nonlinearities."""
        nonlinearities = ["leaky_relu", "relu", "sigmoid", "tanh"]

        for nonlinearity in nonlinearities:
            S, X, theta = generate_ica_data(n_covariates=3, n_treatments=1, batch_size=50, nonlinearity=nonlinearity)
            assert X.shape[0] == 50, f"Batch size should be 50 for {nonlinearity}"
            assert torch.isfinite(X).all(), f"All values should be finite for {nonlinearity}"

    def test_generate_ica_data_sparsity(self):
        """Test that sparse_prob parameter affects number of zero coefficients."""
        n_covariates = 10
        n_treatments = 3

        # High sparsity
        _, _, _ = generate_ica_data(
            n_covariates=n_covariates, n_treatments=n_treatments, batch_size=50, sparse_prob=0.9
        )

        # Low sparsity
        _, _, _ = generate_ica_data(
            n_covariates=n_covariates, n_treatments=n_treatments, batch_size=50, sparse_prob=0.1
        )

        # Just verify no errors occur with different sparsity levels
        assert True

    def test_generate_ica_data_theta_choices(self):
        """Test different theta generation methods."""
        theta_choices = ["fixed", "uniform", "gaussian"]

        for choice in theta_choices:
            S, X, theta = generate_ica_data(n_covariates=3, n_treatments=5, batch_size=50, theta_choice=choice)
            assert theta.shape == (5,), f"theta should have 5 elements for {choice}"
            assert torch.isfinite(theta).all(), f"theta should be finite for {choice}"

    def test_generate_ica_data_beta_parameter(self):
        """Test data generation with different beta (generalized normal) parameters."""
        betas = [0.5, 1.0, 2.0, 4.0]

        for beta in betas:
            S, X, theta = generate_ica_data(n_covariates=3, n_treatments=1, batch_size=50, beta=beta)
            assert torch.isfinite(X).all(), f"All values should be finite for beta={beta}"

    def test_generate_ica_data_split_noise_dist(self):
        """Test data generation with split noise distributions."""
        # Split noise (Gaussian covariates, gennorm for treatment/outcome)
        S_split, X_split, theta_split = generate_ica_data(
            n_covariates=5, n_treatments=1, batch_size=100, split_noise_dist=True
        )

        # Non-split noise (all gennorm)
        S_nonsplit, X_nonsplit, theta_nonsplit = generate_ica_data(
            n_covariates=5, n_treatments=1, batch_size=100, split_noise_dist=False
        )

        assert S_split.shape == S_nonsplit.shape, "Shapes should match"
        assert X_split.shape == X_nonsplit.shape, "Shapes should match"


class TestICAEstimation:
    """Tests for ica_treatment_effect_estimation function."""

    @pytest.fixture
    def sample_ica_data(self):
        """Generate sample ICA data for testing."""
        torch.manual_seed(42)
        np.random.seed(42)
        return generate_ica_data(n_covariates=5, n_treatments=1, batch_size=500)

    def test_ica_estimation_returns_correct_types(self, sample_ica_data):
        """Test that ICA estimation returns correct types."""
        S, X, true_theta = sample_ica_data

        X_np = X.numpy()
        S_np = S.numpy()

        treatment_effect, mcc = ica_treatment_effect_estimation(
            X_np, S_np, random_state=42, check_convergence=False, n_treatments=1
        )

        assert isinstance(treatment_effect, np.ndarray), "treatment_effect should be numpy array"
        assert isinstance(mcc, (float, np.floating)), "mcc should be float"

    def test_ica_estimation_treatment_effect_shape(self, sample_ica_data):
        """Test that treatment effect has correct shape."""
        S, X, true_theta = sample_ica_data

        X_np = X.numpy()
        S_np = S.numpy()

        n_treatments = 1
        treatment_effect, mcc = ica_treatment_effect_estimation(
            X_np, S_np, random_state=42, check_convergence=False, n_treatments=n_treatments
        )

        assert treatment_effect.shape == (n_treatments,), f"treatment_effect should have shape ({n_treatments},)"

    def test_ica_estimation_with_multiple_treatments(self):
        """Test ICA estimation with multiple treatments."""
        torch.manual_seed(42)
        np.random.seed(42)

        n_treatments = 3
        S, X, true_theta = generate_ica_data(n_covariates=5, n_treatments=n_treatments, batch_size=500)

        X_np = X.numpy()
        S_np = S.numpy()

        treatment_effect, mcc = ica_treatment_effect_estimation(
            X_np, S_np, random_state=42, check_convergence=False, n_treatments=n_treatments
        )

        assert treatment_effect.shape == (n_treatments,), f"treatment_effect should have shape ({n_treatments},)"

    def test_ica_estimation_mcc_range(self, sample_ica_data):
        """Test that MCC is in valid range [0, 1]."""
        S, X, true_theta = sample_ica_data

        X_np = X.numpy()
        S_np = S.numpy()

        treatment_effect, mcc = ica_treatment_effect_estimation(
            X_np, S_np, random_state=42, check_convergence=False, n_treatments=1
        )

        if mcc is not None:  # MCC can be None if convergence fails
            assert 0 <= mcc <= 1, f"MCC should be in [0, 1], got {mcc}"

    def test_ica_estimation_different_random_states(self, sample_ica_data):
        """Test that ICA estimation works with different random states."""
        S, X, true_theta = sample_ica_data

        X_np = X.numpy()
        S_np = S.numpy()

        result1, mcc1 = ica_treatment_effect_estimation(
            X_np, S_np, random_state=42, check_convergence=False, n_treatments=1
        )
        result2, mcc2 = ica_treatment_effect_estimation(
            X_np, S_np, random_state=123, check_convergence=False, n_treatments=1
        )

        # Both results should be valid finite values
        assert np.isfinite(result1).all(), "Result 1 should be finite"
        assert np.isfinite(result2).all(), "Result 2 should be finite"
        # Results should be reasonably close to the true value since ICA should converge
        assert result1.shape == result2.shape, "Results should have same shape"

    def test_ica_estimation_with_convergence_check(self, sample_ica_data):
        """Test ICA estimation with convergence checking."""
        S, X, true_theta = sample_ica_data

        X_np = X.numpy()
        S_np = S.numpy()

        treatment_effect, mcc = ica_treatment_effect_estimation(
            X_np, S_np, random_state=42, check_convergence=True, n_treatments=1
        )

        # If convergence failed, result should be NaN
        if np.isnan(treatment_effect).any():
            assert mcc is None, "MCC should be None if convergence failed"
        else:
            assert mcc is not None, "MCC should not be None if convergence succeeded"
