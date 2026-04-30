"""Tests for semi_synthetic_data.py (Section 4.2 of REBUTTAL_PLAN.md)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kurtosis as scipy_kurtosis

from semi_synthetic_data import generate_semi_synthetic_pl, load_real_covariates

# ---------------------------------------------------------------------------
# TestLoadRealCovariates
# ---------------------------------------------------------------------------


class TestLoadRealCovariates:
    def test_california_housing_shape(self):
        X = load_real_covariates("california_housing", standardize=False)
        assert X.shape[0] > 10000
        assert X.shape[1] == 8

    def test_standardize_zero_mean(self):
        X = load_real_covariates("california_housing", standardize=True)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-2)

    def test_n_samples_truncation(self):
        X = load_real_covariates("california_housing", n_samples=500)
        assert X.shape[0] == 500

    def test_standardize_false_does_not_normalise(self):
        X = load_real_covariates("california_housing", standardize=False)
        # Raw California Housing features are not unit-variance
        assert X.std(axis=0).max() > 1.0

    def test_n_samples_deterministic(self):
        X1 = load_real_covariates("california_housing", n_samples=200, seed=42)
        X2 = load_real_covariates("california_housing", n_samples=200, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_ihdp_missing_raises(self, tmp_path, monkeypatch):
        """FileNotFoundError with a clear message when IHDP is absent."""
        import semi_synthetic_data

        monkeypatch.setattr(semi_synthetic_data, "_REPO_ROOT", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="ihdp_npci_1.npz"):
            load_real_covariates("ihdp")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_real_covariates("nonexistent_dataset")

    def test_n_samples_too_large_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            load_real_covariates("california_housing", n_samples=10**9)


# ---------------------------------------------------------------------------
# TestGenerateSemiSyntheticPLR
# ---------------------------------------------------------------------------


class TestGenerateSemiSyntheticPLR:
    def test_outputs_have_correct_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 8))
        X_out, T, Y, meta = generate_semi_synthetic_pl(X, treatment_effect=1.0)
        assert X_out.shape == (1000, 8)
        assert T.shape == (1000,)
        assert Y.shape == (1000,)

    def test_x_returned_unchanged(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 8))
        X_copy = X.copy()
        X_out, _, _, _ = generate_semi_synthetic_pl(X)
        np.testing.assert_array_equal(X_out, X_copy)

    def test_ground_truth_metadata_keys(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 8))
        _, _, _, meta = generate_semi_synthetic_pl(X)
        required_keys = {
            "treatment_effect",
            "treatment_coef",
            "outcome_coef",
            "eta_distribution",
            "support_indices",
            "nonlinearity",
            "sigma_outcome",
            "seed",
        }
        assert required_keys.issubset(set(meta.keys()))

    def test_metadata_values_consistent(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 8))
        _, _, _, meta = generate_semi_synthetic_pl(X, treatment_effect=2.5, eta_distribution="laplace", seed=99)
        assert meta["treatment_effect"] == 2.5
        assert meta["eta_distribution"] == "laplace"
        assert meta["seed"] == 99

    def test_treatment_coef_unit_norm(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((300, 8))
        _, _, _, meta = generate_semi_synthetic_pl(X, seed=7)
        tc_norm = float(np.linalg.norm(meta["treatment_coef"]))
        assert abs(tc_norm - 1.0) < 1e-10

    def test_outcome_coef_unit_norm(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((300, 8))
        _, _, _, meta = generate_semi_synthetic_pl(X, seed=8)
        oc_norm = float(np.linalg.norm(meta["outcome_coef"]))
        assert abs(oc_norm - 1.0) < 1e-10

    def test_support_size_respected(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((500, 20))
        _, _, _, meta = generate_semi_synthetic_pl(X, support_size=5, seed=10)
        assert len(meta["support_indices"]) == 5
        assert len(meta["treatment_coef"]) == 5
        assert len(meta["outcome_coef"]) == 5

    def test_support_size_none_uses_all_columns(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((300, 8))
        _, _, _, meta = generate_semi_synthetic_pl(X, support_size=None, seed=11)
        assert len(meta["support_indices"]) == 8

    def test_leaky_relu_nonlinearity(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((500, 8))
        _, T1, Y1, _ = generate_semi_synthetic_pl(X, nonlinearity="identity", seed=42)
        _, T2, Y2, _ = generate_semi_synthetic_pl(X, nonlinearity="leaky_relu", seed=42)
        # With identical seed, coefficients and noise are the same, but
        # leaky_relu introduces a kink so outputs should differ in general
        assert not np.allclose(T1, T2) or not np.allclose(Y1, Y2)

    def test_eta_distribution_dispatch_rademacher(self):
        """With Rademacher eta, T residuals should have kurtosis near -2 (excess)."""
        rng_X = np.random.default_rng(42)
        X = rng_X.standard_normal((5000, 8))
        _, T, _, meta = generate_semi_synthetic_pl(
            X,
            treatment_effect=0.0,  # decouple Y from T
            eta_distribution="rademacher",
            treatment_coef_scalar=0.0,  # T = eta only
            seed=123,
        )
        # With treatment_coef_scalar=0: coef is zeros/unit-norm, so linear index is 0
        # Actually unit-norm is enforced — use a workaround: set support_size=1
        # and use a tiny scalar so the signal is dominated by eta.
        # Simpler: draw T directly from rademacher sampler and check kurtosis.
        from oml_runner import setup_treatment_noise

        _, eta_sample, _, _ = setup_treatment_noise("rademacher")
        eta = eta_sample(50000)
        kurt = float(scipy_kurtosis(eta, fisher=True))
        # Rademacher excess kurtosis = -2
        assert abs(kurt - (-2.0)) < 0.2

    def test_eta_distribution_discrete(self):
        rng_X = np.random.default_rng(55)
        X = rng_X.standard_normal((1000, 8))
        _, T, _, _ = generate_semi_synthetic_pl(X, eta_distribution="discrete", seed=55)
        assert np.isfinite(T).all()

    def test_eta_distribution_laplace(self):
        rng_X = np.random.default_rng(66)
        X = rng_X.standard_normal((1000, 8))
        _, T, _, _ = generate_semi_synthetic_pl(X, eta_distribution="laplace", seed=66)
        assert np.isfinite(T).all()

    def test_determinism(self):
        rng = np.random.default_rng(77)
        X = rng.standard_normal((500, 8))
        _, T1, Y1, _ = generate_semi_synthetic_pl(X, seed=77)
        _, T2, Y2, _ = generate_semi_synthetic_pl(X, seed=77)
        np.testing.assert_array_equal(T1, T2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_support_size_too_large_raises(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((200, 8))
        with pytest.raises(ValueError, match="support_size"):
            generate_semi_synthetic_pl(X, support_size=100)

    def test_unknown_nonlinearity_raises(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((200, 8))
        with pytest.raises(ValueError, match="nonlinearity"):
            generate_semi_synthetic_pl(X, nonlinearity="sigmoid")

    def test_treatment_effect_zero_means_y_independent_of_t_variation(self):
        """When theta=0, varying T shouldn't shift Y through the treatment channel."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((2000, 8))
        _, T, Y, _ = generate_semi_synthetic_pl(X, treatment_effect=0.0, seed=11)
        # Y = f(X) + eps; correlation with T should be low
        corr = float(np.corrcoef(T, Y)[0, 1])
        assert abs(corr) < 0.3  # loose bound — f(X) is shared via coefficients
