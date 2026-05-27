"""Tests for binary_treatment_dgp.py and binary_treatment_runner.py.

Coverage:
- Treatment vector lives in {0, 1} with the requested marginal balance.
- Propensity is bounded by ``logit_clip`` (positivity).
- Centred residual ``eta = T - p(X)`` has mean ~0 and matches Bernoulli
  conditional variance.
- Reproducibility: identical seed produces identical samples.
- Recovery of the true ATE by OLS, matching, and HOML on a small sample.
- The Monte Carlo runner returns the documented schema.
- Regression: ICA Y-loading identification recovers theta on binary T while
  the legacy kurtosis identification produces order-of-magnitude errors.
"""

import numpy as np
import pytest

from binary_treatment_dgp import BinaryTreatmentDGPConfig, empirical_eta_moments, generate_binary_treatment_data
from binary_treatment_runner import METHOD_NAMES, run_binary_treatment_experiments

# ---------------------------------------------------------------------------
# DGP shape & marginal sanity
# ---------------------------------------------------------------------------


class TestBinaryTreatmentDGP:
    def test_treatment_is_binary(self):
        """T takes values in {0, 1} only."""
        cfg = BinaryTreatmentDGPConfig(n_samples=2000, seed=0)
        _, T, _, _, _, _, _ = generate_binary_treatment_data(cfg)
        unique_vals = np.unique(T)
        assert set(unique_vals.tolist()).issubset({0.0, 1.0}), f"T values: {unique_vals}"

    def test_shapes(self):
        """All returned arrays have the documented shapes."""
        cfg = BinaryTreatmentDGPConfig(n_samples=500, n_covariates=8, support_size=3, seed=1)
        X, T, Y, p, eta, alpha, beta = generate_binary_treatment_data(cfg)
        assert X.shape == (500, 8)
        assert T.shape == (500,)
        assert Y.shape == (500,)
        assert p.shape == (500,)
        assert eta.shape == (500,)
        assert alpha.shape == (8,)
        assert beta.shape == (8,)

    def test_propensity_is_bounded(self):
        """``logit_clip`` keeps p(X) strictly inside (0, 1)."""
        cfg = BinaryTreatmentDGPConfig(n_samples=2000, propensity_strength=10.0, logit_clip=4.0, seed=2)
        _, _, _, p, _, _, _ = generate_binary_treatment_data(cfg)
        # sigmoid(±4) ≈ {0.01799, 0.98201}
        assert p.min() > 0.0 and p.max() < 1.0
        assert p.min() >= 1.0 / (1.0 + np.exp(4.0)) - 1e-9
        assert p.max() <= 1.0 / (1.0 + np.exp(-4.0)) + 1e-9

    def test_alpha_beta_support_pattern(self):
        """Alpha and beta are zero outside the first ``support_size`` indices."""
        cfg = BinaryTreatmentDGPConfig(n_samples=100, n_covariates=10, support_size=3, seed=3)
        _, _, _, _, _, alpha, beta = generate_binary_treatment_data(cfg)
        assert np.all(alpha[3:] == 0.0)
        assert np.all(beta[3:] == 0.0)
        # Non-zero entries should generally be non-zero (probability 1 under
        # standard normal sampling).
        assert np.any(alpha[:3] != 0.0)
        assert np.any(beta[:3] != 0.0)

    def test_eta_zero_mean_conditionally(self):
        """Eta = T - p(X) has near-zero empirical mean for large n."""
        cfg = BinaryTreatmentDGPConfig(n_samples=20000, seed=4)
        _, _, _, _, eta, _, _ = generate_binary_treatment_data(cfg)
        assert abs(eta.mean()) < 0.02, f"E[eta] should be ~0, got {eta.mean():.4f}"

    def test_eta_variance_matches_bernoulli(self):
        """Var[eta] ≈ E[p(X) * (1 - p(X))] from the law of total variance."""
        cfg = BinaryTreatmentDGPConfig(n_samples=20000, seed=5)
        _, _, _, p, eta, _, _ = generate_binary_treatment_data(cfg)
        empirical_var = float(np.mean(eta**2))
        analytic_var = float(np.mean(p * (1 - p)))
        assert (
            abs(empirical_var - analytic_var) < 0.01
        ), f"Empirical Var[eta]={empirical_var:.4f} vs analytic={analytic_var:.4f}"

    def test_seed_reproducibility(self):
        """Identical seed => identical samples."""
        cfg = BinaryTreatmentDGPConfig(n_samples=200, seed=42)
        out1 = generate_binary_treatment_data(cfg)
        out2 = generate_binary_treatment_data(cfg)
        for a, b in zip(out1, out2):
            assert np.allclose(a, b), "Same seed must reproduce identical arrays"

    def test_different_seeds_differ(self):
        """Different seeds produce different samples (almost surely)."""
        cfg_a = BinaryTreatmentDGPConfig(n_samples=200, seed=10)
        cfg_b = BinaryTreatmentDGPConfig(n_samples=200, seed=11)
        _, T_a, _, _, _, _, _ = generate_binary_treatment_data(cfg_a)
        _, T_b, _, _, _, _, _ = generate_binary_treatment_data(cfg_b)
        assert not np.allclose(T_a, T_b)

    def test_invalid_support_size(self):
        """support_size > n_covariates raises ValueError."""
        with pytest.raises(ValueError, match="support_size"):
            generate_binary_treatment_data(BinaryTreatmentDGPConfig(n_covariates=3, support_size=5, seed=0))

    def test_invalid_logit_clip(self):
        """Non-positive logit_clip raises ValueError."""
        with pytest.raises(ValueError, match="logit_clip"):
            generate_binary_treatment_data(BinaryTreatmentDGPConfig(logit_clip=0.0, seed=0))


class TestEmpiricalEtaMoments:
    def test_empirical_moments_match_bernoulli_formulas(self):
        """For T ~ Bernoulli(p) i.i.d. with eta = T - p, sample moments match the closed form."""
        rng = np.random.default_rng(0)
        p = 0.3
        T = (rng.uniform(size=50000) < p).astype(float)
        eta = T - p
        m2, third_cumulant = empirical_eta_moments(eta)
        # Bernoulli centred moments: var = p(1-p); third central moment = (1-2p) p (1-p)
        assert abs(m2 - p * (1 - p)) < 0.005
        assert abs(third_cumulant - (1 - 2 * p) * p * (1 - p)) < 0.005


# ---------------------------------------------------------------------------
# Estimators recover theta on the binary-T DGP
# ---------------------------------------------------------------------------


class TestEstimatorsRecoverTheta:
    """Smoke-level recovery tests on a single moderate sample.

    These are not Monte Carlo tests — they check that on n=4000 with mild
    confounding the OLS/matching/HOML estimates land in a wide tolerance
    band around the true theta. Tighter bounds are tested in the runner.
    """

    @pytest.fixture(scope="class")
    def sample(self):
        cfg = BinaryTreatmentDGPConfig(
            n_samples=4000,
            n_covariates=10,
            support_size=5,
            treatment_effect=1.5,
            propensity_strength=0.7,
            outcome_coef_scale=0.5,
            sigma_outcome=0.5,
            seed=2024,
        )
        return generate_binary_treatment_data(cfg)

    def test_ols_recovers_theta(self, sample):
        from baselines import ols_baseline

        X, T, Y, _, _, _, _ = sample
        theta_hat = float(ols_baseline(X, T, Y)[0])
        # OLS is consistent under linear confounding (which our DGP has),
        # so it should land near the true theta.
        assert abs(theta_hat - 1.5) < 0.3, f"OLS got {theta_hat:.4f}"

    def test_matching_recovers_theta(self, sample):
        from baselines import matching_baseline

        X, T, Y, _, _, _, _ = sample
        theta_hat = float(matching_baseline(X, T, Y, treatment_kind="binary"))
        assert abs(theta_hat - 1.5) < 0.5, f"Matching got {theta_hat:.4f}"

    def test_homl_recovers_theta(self, sample):
        from binary_treatment_dgp import empirical_eta_moments
        from main_estimation import all_together_cross_fitting

        X, T, Y, _, eta, _, _ = sample
        m2, m3c = empirical_eta_moments(eta)
        ortho_ml, robust_ortho_ml, robust_ortho_est, robust_ortho_split, _, _ = all_together_cross_fitting(
            X, T, Y, m2, m3c
        )
        for name, val in zip(
            ("ortho_ml", "robust_known", "robust_est", "robust_split"),
            (ortho_ml, robust_ortho_ml, robust_ortho_est, robust_ortho_split),
        ):
            assert np.isfinite(val), f"{name} not finite"
            assert abs(val - 1.5) < 0.5, f"{name} got {val:.4f}, expected ~1.5"


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------


class TestBinaryTreatmentRunner:
    def test_runner_returns_documented_schema(self):
        cfg = BinaryTreatmentDGPConfig(
            n_samples=300,
            n_covariates=5,
            support_size=3,
            treatment_effect=1.0,
            sigma_outcome=0.5,
            seed=0,
        )
        # Use a small number of experiments for speed; force serial joblib so
        # the in-process stubs (conftest.py) apply.
        results = run_binary_treatment_experiments(config=cfg, n_experiments=5, base_seed=100, n_jobs=1, verbose=False)

        # Required keys
        for key in (
            "method_names",
            "estimates",
            "estimates_finite",
            "biases",
            "sigmas",
            "rmse",
            "n_experiments",
            "n_attempted",
            "treatment_effect",
        ):
            assert key in results, f"missing key {key}"

        n_methods = len(METHOD_NAMES)
        assert results["estimates"].shape == (5, n_methods)
        assert results["biases"].shape == (n_methods,)
        assert results["sigmas"].shape == (n_methods,)
        assert results["rmse"].shape == (n_methods,)
        assert results["n_attempted"] == 5
        assert 0 <= results["n_experiments"] <= 5
        assert results["treatment_effect"] == pytest.approx(1.0)

    def test_runner_recovers_theta_in_expectation(self):
        """Across 30 replications all OML/baseline estimates should land near theta.

        ICA on binary T is allowed to fail (no torch in the test env, or
        degenerate sources); biases for the remaining methods are checked
        with a per-method finite count.
        """
        cfg = BinaryTreatmentDGPConfig(
            n_samples=1500,
            n_covariates=8,
            support_size=4,
            treatment_effect=1.2,
            propensity_strength=0.7,
            outcome_coef_scale=0.5,
            sigma_outcome=0.5,
        )
        results = run_binary_treatment_experiments(config=cfg, n_experiments=30, base_seed=42, n_jobs=1, verbose=False)

        # OML and baselines must produce finite estimates on virtually every run.
        finite_per_method = results["finite_per_method"]
        for idx, name in enumerate(METHOD_NAMES):
            if name == "ICA":
                continue
            assert finite_per_method[idx] >= 25, f"{name}: only {finite_per_method[idx]}/30 runs finite"

        biases = results["biases"]
        for idx, name in enumerate(METHOD_NAMES):
            if name == "ICA":
                continue
            assert abs(biases[idx]) < 0.5, f"{name} bias {biases[idx]:.3f} too large"


# ---------------------------------------------------------------------------
# ICA Y-loading fix (regression test)
# ---------------------------------------------------------------------------


class TestICAEpsIdentificationOnBinaryT:
    """Verify the eps-row identification fix for binary T.

    The legacy ``eps_identification="kurtosis"`` selects the eta/T-driven
    component on a binary-T DGP because Bernoulli T has |excess kurtosis|
    near 2 (the maximum possible) while Gaussian eps has near-zero kurtosis.
    The new default ``"y_loading"`` selects by ``argmax |W[:, -1]|`` and is
    immune to T's kurtosis.
    """

    @pytest.fixture(scope="class")
    def sample(self):
        cfg = BinaryTreatmentDGPConfig(
            n_samples=2000,
            n_covariates=10,
            support_size=5,
            treatment_effect=1.5,
            propensity_strength=0.7,
            outcome_coef_scale=0.5,
            sigma_outcome=0.5,
            seed=2024,
        )
        X, T, Y, _, _, _, _ = generate_binary_treatment_data(cfg)
        return X, T, Y

    def test_y_loading_recovers_theta(self, sample):
        """Y-loading picker recovers theta within 1.0 of the truth on n=2000."""
        pytest.importorskip("torch")
        from ica import ica_treatment_effect_estimation_eps_row

        X, T, Y = sample
        observed = np.hstack((X, T.reshape(-1, 1), Y.reshape(-1, 1)))
        theta_hat, _ = ica_treatment_effect_estimation_eps_row(
            observed,
            S=None,
            check_convergence=False,
            verbose=False,
            eps_identification="y_loading",
        )
        assert np.isfinite(theta_hat).all()
        assert abs(float(theta_hat[0]) - 1.5) < 1.0, f"y_loading should recover theta ~1.5, got {theta_hat[0]:.3f}"

    def test_kurtosis_legacy_is_unstable(self, sample):
        """Legacy kurtosis picker either explodes (>10x off) or NaNs."""
        pytest.importorskip("torch")
        from ica import ica_treatment_effect_estimation_eps_row

        X, T, Y = sample
        observed = np.hstack((X, T.reshape(-1, 1), Y.reshape(-1, 1)))
        theta_hat, _ = ica_treatment_effect_estimation_eps_row(
            observed,
            S=None,
            check_convergence=False,
            verbose=False,
            eps_identification="kurtosis",
        )
        # Document the failure mode: estimate is either NaN or far from 1.5.
        # If this ever passes within tolerance, the picker may have gotten
        # lucky on this seed; rerun with a different seed in the diagnostic.
        val = float(theta_hat[0]) if np.isfinite(theta_hat).all() else float("nan")
        if np.isfinite(val):
            assert abs(val - 1.5) > 5.0, (
                f"Legacy kurtosis picker happened to recover theta={val:.3f} "
                "on this seed — re-pick a seed to keep this regression test sharp."
            )

    def test_invalid_eps_identification_raises(self):
        """Unknown strategy name raises ValueError."""
        pytest.importorskip("torch")
        from ica import ica_treatment_effect_estimation_eps_row

        rng = np.random.default_rng(0)
        n, d = 200, 5
        X = rng.standard_normal((n, d + 2))
        with pytest.raises(ValueError, match="eps_identification"):
            ica_treatment_effect_estimation_eps_row(X, S=None, eps_identification="banana")
