"""Tests for main_estimation.py module."""

import numpy as np
import pytest
from sklearn.linear_model import LassoCV

from main_estimation import all_together, all_together_cross_fitting


class TestAllTogether:
    """Tests for all_together function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        treatment_effect = 2.0

        x = np.random.randn(n_samples, n_features)
        eta = np.random.randn(n_samples)
        epsilon = np.random.randn(n_samples)

        treatment_support = np.array([0, 1, 2])
        outcome_support = np.array([0, 1, 2])
        treatment_coef = np.array([1.0, -0.5, 0.8])
        outcome_coef = np.array([0.5, 1.2, -0.3])

        eta_second_moment = np.var(eta)
        eta_third_moment = np.mean((eta - np.mean(eta)) ** 3)

        return {
            "x": x,
            "treatment_support": treatment_support,
            "treatment_coef": treatment_coef,
            "outcome_support": outcome_support,
            "outcome_coef": outcome_coef,
            "eta": eta,
            "epsilon": epsilon,
            "treatment_effect": treatment_effect,
            "eta_second_moment": eta_second_moment,
            "eta_third_moment": eta_third_moment,
        }

    def test_all_together_returns_four_estimates(self, sample_data):
        """Test that all_together returns four estimates."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        results = all_together(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["eta_second_moment"],
            sample_data["eta_third_moment"],
        )

        assert len(results) == 4, "Should return 4 estimates"

    def test_all_together_estimates_are_finite(self, sample_data):
        """Test that all estimates are finite."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        results = all_together(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["eta_second_moment"],
            sample_data["eta_third_moment"],
        )

        for i, result in enumerate(results):
            assert np.isfinite(result), f"Estimate {i} should be finite"

    def test_all_together_with_custom_models(self, sample_data):
        """Test all_together with custom model parameters."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        model_treatment = LassoCV(alphas=[0.1, 1.0, 10.0])
        model_outcome = LassoCV(alphas=[0.1, 1.0, 10.0])

        results = all_together(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["eta_second_moment"],
            sample_data["eta_third_moment"],
            model_treatment=model_treatment,
            model_outcome=model_outcome,
        )

        assert len(results) == 4, "Should return 4 estimates"


class TestAllTogetherCrossFitting:
    """Tests for all_together_cross_fitting function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        treatment_effect = 2.0

        x = np.random.randn(n_samples, n_features)
        eta = np.random.randn(n_samples)
        epsilon = np.random.randn(n_samples)

        treatment_support = np.array([0, 1, 2])
        outcome_support = np.array([0, 1, 2])
        treatment_coef = np.array([1.0, -0.5, 0.8])
        outcome_coef = np.array([0.5, 1.2, -0.3])

        treatment_second_moment = np.var(eta)
        treatment_third_moment = np.mean((eta - np.mean(eta)) ** 3)

        return {
            "x": x,
            "treatment_support": treatment_support,
            "treatment_coef": treatment_coef,
            "outcome_support": outcome_support,
            "outcome_coef": outcome_coef,
            "eta": eta,
            "epsilon": epsilon,
            "treatment_effect": treatment_effect,
            "treatment_second_moment": treatment_second_moment,
            "treatment_third_moment": treatment_third_moment,
        }

    def test_cross_fitting_returns_six_values(self, sample_data):
        """Test that cross_fitting returns six values."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        results = all_together_cross_fitting(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["treatment_second_moment"],
            sample_data["treatment_third_moment"],
        )

        assert len(results) == 6, "Should return 6 values (4 estimates + 2 coef arrays)"

    def test_cross_fitting_estimates_are_finite(self, sample_data):
        """Test that all cross-fitting estimates are finite."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        results = all_together_cross_fitting(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["treatment_second_moment"],
            sample_data["treatment_third_moment"],
        )

        # Test first 4 estimates
        for i in range(4):
            assert np.isfinite(results[i]), f"Estimate {i} should be finite"

    def test_cross_fitting_coefs_shape(self, sample_data):
        """Test that coefficient arrays have correct shape."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        results = all_together_cross_fitting(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["treatment_second_moment"],
            sample_data["treatment_third_moment"],
        )

        treatment_coef = results[4]
        outcome_coef = results[5]

        assert treatment_coef.shape[0] == sample_data["x"].shape[1], "Treatment coef should match number of features"
        assert outcome_coef.shape[0] == sample_data["x"].shape[1], "Outcome coef should match number of features"

    def test_cross_fitting_reproducibility(self, sample_data):
        """Test that cross_fitting produces consistent results with same data."""
        treatment = (
            np.dot(sample_data["x"][:, sample_data["treatment_support"]], sample_data["treatment_coef"])
            + sample_data["eta"]
        )
        outcome = (
            sample_data["treatment_effect"] * treatment
            + np.dot(sample_data["x"][:, sample_data["outcome_support"]], sample_data["outcome_coef"])
            + sample_data["epsilon"]
        )

        results1 = all_together_cross_fitting(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["treatment_second_moment"],
            sample_data["treatment_third_moment"],
        )

        results2 = all_together_cross_fitting(
            sample_data["x"],
            treatment,
            outcome,
            sample_data["treatment_second_moment"],
            sample_data["treatment_third_moment"],
        )

        for i in range(4):
            np.testing.assert_almost_equal(results1[i], results2[i], decimal=10)
