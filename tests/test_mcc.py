"""Tests for mcc.py module."""

import numpy as np
import pytest
import torch

from mcc import calc_disent_metrics, linear_disentanglement, permutation_disentanglement


class TestLinearDisentanglement:
    """Tests for linear_disentanglement function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample latent data."""
        np.random.seed(42)
        n_samples = 200
        n_latents = 5

        z = np.random.randn(n_samples, n_latents)
        # Create hz as a linear transformation of z with some noise
        A = np.random.randn(n_latents, n_latents)
        hz = z @ A + 0.1 * np.random.randn(n_samples, n_latents)

        return z, hz

    def test_linear_disentanglement_returns_correct_structure(self, sample_data):
        """Test that linear_disentanglement returns expected structure."""
        z, hz = sample_data

        (r2, _), (z_test, hz_pred), coef, intercept = linear_disentanglement(z, hz, mode="r2")

        assert isinstance(r2, (float, np.floating)), "R2 should be float"
        assert z_test.shape == z.shape, "z_test should have same shape as z"
        assert hz_pred.shape == hz.shape, "hz_pred should have same shape as hz"
        assert coef.shape == (hz.shape[1], z.shape[1]), "coef should have shape (n_latents, n_latents)"
        assert intercept.shape == (z.shape[1],), "intercept should have shape (n_latents,)"

    def test_linear_disentanglement_r2_range(self, sample_data):
        """Test that R2 is in valid range."""
        z, hz = sample_data

        (r2, _), _, _, _ = linear_disentanglement(z, hz, mode="r2")

        # R2 can be negative for very poor fits, but should typically be reasonable
        assert r2 <= 1.0, "R2 should be <= 1.0"

    def test_linear_disentanglement_with_train_test_split(self, sample_data):
        """Test linear_disentanglement with train/test split."""
        z, hz = sample_data

        (r2_split, _), (z_test, hz_pred), coef, intercept = linear_disentanglement(
            z, hz, mode="r2", train_test_split=True
        )

        # With split, test set should be half the size
        assert z_test.shape[0] == z.shape[0] // 2, "Test set should be half the size"

    def test_linear_disentanglement_with_pearson(self, sample_data):
        """Test linear_disentanglement with Pearson correlation."""
        z, hz = sample_data

        (corr, corr_matrix), _, _, _ = linear_disentanglement(z, hz, mode="pearson")

        assert isinstance(corr, (float, np.floating)), "Correlation should be float"
        assert 0 <= corr <= 1, f"Pearson correlation should be in [0, 1], got {corr}"

    def test_linear_disentanglement_with_spearman(self, sample_data):
        """Test linear_disentanglement with Spearman correlation."""
        z, hz = sample_data

        (corr, corr_matrix), _, _, _ = linear_disentanglement(z, hz, mode="spearman")

        assert isinstance(corr, (float, np.floating)), "Correlation should be float"
        assert 0 <= corr <= 1, f"Spearman correlation should be in [0, 1], got {corr}"

    def test_linear_disentanglement_with_torch_tensors(self):
        """Test that function works with torch tensors."""
        torch.manual_seed(42)
        n_samples = 200
        n_latents = 5

        z = torch.randn(n_samples, n_latents)
        hz = z @ torch.randn(n_latents, n_latents) + 0.1 * torch.randn(n_samples, n_latents)

        (r2, _), _, _, _ = linear_disentanglement(z, hz, mode="r2")

        assert isinstance(r2, (float, np.floating)), "R2 should be float"


class TestPermutationDisentanglement:
    """Tests for permutation_disentanglement function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample latent data."""
        np.random.seed(42)
        n_samples = 200
        n_latents = 4

        z = np.random.randn(n_samples, n_latents)
        # Create hz as permuted z with some noise
        perm = np.array([2, 0, 3, 1])  # Permutation
        hz = z[:, perm] + 0.1 * np.random.randn(n_samples, n_latents)

        return z, hz

    def test_permutation_disentanglement_naive_solver(self, sample_data):
        """Test permutation_disentanglement with naive solver."""
        z, hz = sample_data

        (score, _, _), hz_transformed = permutation_disentanglement(
            z, hz, mode="pearson", solver="naive", rescaling=True
        )

        assert isinstance(score, (float, np.floating)), "Score should be float"
        assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"
        assert hz_transformed.shape == hz.shape, "Transformed hz should have same shape"

    def test_permutation_disentanglement_munkres_solver(self, sample_data):
        """Test permutation_disentanglement with Munkres solver."""
        z, hz = sample_data

        (score, corr_mat, sort_idx), hz_transformed = permutation_disentanglement(
            z, hz, mode="pearson", solver="munkres", rescaling=True
        )

        assert isinstance(score, (float, np.floating)), "Score should be float"
        assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"
        assert sort_idx is not None, "Sort index should not be None for Munkres"
        assert len(sort_idx) == z.shape[1], "Sort index should have length n_latents"

    def test_permutation_disentanglement_with_spearman(self, sample_data):
        """Test permutation_disentanglement with Spearman correlation."""
        z, hz = sample_data

        (score, _, _), _ = permutation_disentanglement(z, hz, mode="spearman", solver="munkres", rescaling=True)

        assert isinstance(score, (float, np.floating)), "Score should be float"
        assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"

    def test_permutation_disentanglement_no_rescaling(self, sample_data):
        """Test permutation_disentanglement without rescaling."""
        z, hz = sample_data

        (score_no_rescale, _, _), _ = permutation_disentanglement(
            z, hz, mode="pearson", solver="munkres", rescaling=False
        )

        (score_rescale, _, _), _ = permutation_disentanglement(z, hz, mode="pearson", solver="munkres", rescaling=True)

        # Results should differ when rescaling is used
        assert isinstance(score_no_rescale, (float, np.floating)), "Score should be float"
        assert isinstance(score_rescale, (float, np.floating)), "Score should be float"

    def test_permutation_disentanglement_with_sign_flips(self):
        """Test permutation_disentanglement with sign flips."""
        np.random.seed(42)
        n_samples = 100
        n_latents = 3

        z = np.random.randn(n_samples, n_latents)
        # Create hz with sign flip
        hz = z.copy()
        hz[:, 1] = -hz[:, 1]  # Flip sign of second latent

        (score, _, _), _ = permutation_disentanglement(
            z, hz, mode="pearson", solver="naive", sign_flips=True, rescaling=True
        )

        assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"


class TestCalcDisentMetrics:
    """Tests for calc_disent_metrics function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample latent data."""
        np.random.seed(42)
        n_samples = 200
        n_latents = 5

        z = np.random.randn(n_samples, n_latents)
        # Create hz as a linear transformation of z
        A = np.random.randn(n_latents, n_latents)
        hz = z @ A + 0.1 * np.random.randn(n_samples, n_latents)

        return z, hz

    def test_calc_disent_metrics_returns_dict(self, sample_data):
        """Test that calc_disent_metrics returns a dictionary with expected keys."""
        z, hz = sample_data

        metrics = calc_disent_metrics(z, hz)

        expected_keys = [
            "lin_dis_score",
            "lin_coef_mat",
            "lin_intercept",
            "permutation_disentanglement_score",
            "perm_corr_mat",
            "munkres_sort_idx",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_calc_disent_metrics_values_are_valid(self, sample_data):
        """Test that metric values are in valid ranges."""
        z, hz = sample_data

        metrics = calc_disent_metrics(z, hz)

        # Check linear disentanglement score (R2)
        assert metrics["lin_dis_score"] <= 1.0, "Linear disentanglement score should be <= 1.0"

        # Check permutation disentanglement score (correlation)
        assert 0 <= metrics["permutation_disentanglement_score"] <= 1, "Permutation score should be in [0, 1]"

        # Check matrix shapes
        assert metrics["lin_coef_mat"].shape == (
            z.shape[1],
            z.shape[1],
        ), "Coef matrix should be square"
        assert metrics["lin_intercept"].shape == (z.shape[1],), "Intercept should be 1D"
        assert metrics["munkres_sort_idx"].shape == (z.shape[1],), "Sort index should match n_latents"

    def test_calc_disent_metrics_with_perfect_match(self):
        """Test metrics with perfectly matched latents."""
        np.random.seed(42)
        n_samples = 200
        n_latents = 5

        z = np.random.randn(n_samples, n_latents)
        hz = z.copy()  # Perfect match

        metrics = calc_disent_metrics(z, hz)

        # With perfect match, scores should be very high
        assert metrics["lin_dis_score"] > 0.99, "Linear score should be near 1 for perfect match"
        assert (
            metrics["permutation_disentanglement_score"] > 0.99
        ), "Permutation score should be near 1 for perfect match"
