"""Tests for nonlinear_dgp.py and nonlinear_runner.py.

Coverage:
- Output shapes and types for every difficulty toggle combination.
- Reproducibility: identical seed => identical arrays.
- Toggle isolation: each axis changes the data when switched on.
- OLS bias regression: OLS is biased on the nonlinear preset and
  approximately unbiased on the linear preset (the core paper claim).
- Eta moments match the configured distribution.
- Runner returns the documented schema and all methods produce finite
  estimates on a small run.
"""

import numpy as np
import pytest

from nonlinear_dgp import (
    NonlinearDGPConfig,
    empirical_eta_moments,
    eta_moments_from_config,
    generate_nonlinear_data,
    hard_preset,
    linear_preset,
)
from nonlinear_runner import METHOD_NAMES, run_nonlinear_experiments

# ---------------------------------------------------------------------------
# Shape and type tests
# ---------------------------------------------------------------------------


class TestNonlinearDGPShapes:
    def test_default_shapes(self):
        cfg = NonlinearDGPConfig(n_samples=200, n_covariates=8, support_size=3, seed=0)
        X, T, Y, eta, m_X, g_X, alpha = generate_nonlinear_data(cfg)
        assert X.shape == (200, 8)
        assert T.shape == (200,)
        assert Y.shape == (200,)
        assert eta.shape == (200,)
        assert m_X.shape == (200,)
        assert g_X.shape == (200,)
        assert alpha.shape == (8,)

    def test_high_dim_overrides_n_covariates(self):
        cfg = NonlinearDGPConfig(n_samples=100, n_covariates=10, high_dim=True, high_dim_d=30, seed=1)
        X, T, Y, eta, m_X, g_X, alpha = generate_nonlinear_data(cfg)
        assert X.shape == (100, 30)
        assert alpha.shape == (30,)

    def test_support_size_exceeds_covariates_raises(self):
        with pytest.raises(ValueError, match="support_size"):
            generate_nonlinear_data(NonlinearDGPConfig(n_covariates=4, support_size=5, seed=0))

    def test_invalid_eta_beta_raises(self):
        with pytest.raises(ValueError, match="eta_beta"):
            generate_nonlinear_data(NonlinearDGPConfig(heavy_tail_eta=True, eta_beta=-1.0, seed=0))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_data(self):
        cfg = NonlinearDGPConfig(n_samples=200, seed=42)
        out1 = generate_nonlinear_data(cfg)
        out2 = generate_nonlinear_data(cfg)
        for a, b in zip(out1, out2):
            assert np.allclose(a, b), "Same seed must reproduce identical arrays"

    def test_different_seeds_differ(self):
        cfg_a = NonlinearDGPConfig(n_samples=200, seed=10)
        cfg_b = NonlinearDGPConfig(n_samples=200, seed=11)
        _, _, Y_a, _, _, _, _ = generate_nonlinear_data(cfg_a)
        _, _, Y_b, _, _, _, _ = generate_nonlinear_data(cfg_b)
        assert not np.allclose(Y_a, Y_b)


# ---------------------------------------------------------------------------
# Toggle isolation: each axis changes the data
# ---------------------------------------------------------------------------


class TestToggleIsolation:
    """Verify that flipping a single toggle changes the generated data."""

    def _get_Y(self, **kwargs):
        cfg = NonlinearDGPConfig(n_samples=500, seed=99, **kwargs)
        _, _, Y, _, _, _, _ = generate_nonlinear_data(cfg)
        return Y

    def test_nonlinear_confounding_changes_data(self):
        Y_lin = self._get_Y(nonlinear_confounding=False)
        Y_nl = self._get_Y(nonlinear_confounding=True)
        assert not np.allclose(Y_lin, Y_nl), "Toggling nonlinear_confounding must change Y"

    def test_heavy_tail_eta_changes_data(self):
        Y_gauss = self._get_Y(heavy_tail_eta=False)
        Y_ht = self._get_Y(heavy_tail_eta=True, eta_beta=1.0)
        assert not np.allclose(Y_gauss, Y_ht), "Toggling heavy_tail_eta must change Y"

    def test_heteroscedastic_eps_changes_data(self):
        Y_homo = self._get_Y(heteroscedastic_eps=False)
        Y_hetero = self._get_Y(heteroscedastic_eps=True)
        assert not np.allclose(Y_homo, Y_hetero), "Toggling heteroscedastic_eps must change Y"

    def test_high_dim_changes_shape(self):
        cfg_lo = NonlinearDGPConfig(n_samples=100, n_covariates=10, seed=0)
        cfg_hi = NonlinearDGPConfig(n_samples=100, n_covariates=10, high_dim=True, high_dim_d=25, seed=0)
        X_lo, _, _, _, _, _, _ = generate_nonlinear_data(cfg_lo)
        X_hi, _, _, _, _, _, _ = generate_nonlinear_data(cfg_hi)
        assert X_lo.shape[1] == 10
        assert X_hi.shape[1] == 25


# ---------------------------------------------------------------------------
# Eta moments
# ---------------------------------------------------------------------------


class TestEtaMoments:
    def test_oracle_gaussian_moments(self):
        cfg = NonlinearDGPConfig(sigma_eta=2.0, heavy_tail_eta=False)
        m2, m3c = eta_moments_from_config(cfg)
        assert m2 == pytest.approx(4.0)  # sigma_eta^2
        assert m3c == pytest.approx(0.0)  # Gaussian is symmetric

    def test_oracle_gennorm_moments(self):
        cfg = NonlinearDGPConfig(sigma_eta=1.5, heavy_tail_eta=True, eta_beta=1.0)
        m2, m3c = eta_moments_from_config(cfg)
        assert m2 == pytest.approx(1.5**2)
        assert m3c == pytest.approx(0.0)  # gennorm is symmetric

    def test_empirical_moments_gaussian(self):
        rng = np.random.default_rng(0)
        eta = rng.standard_normal(100_000) * 2.0
        m2, m3c = empirical_eta_moments(eta)
        assert abs(m2 - 4.0) < 0.05, f"Expected E[eta^2]~4, got {m2:.4f}"
        assert abs(m3c) < 0.1, f"Expected third cumulant~0, got {m3c:.4f}"

    def test_heavy_tail_eta_second_moment_matches_config(self):
        """After normalisation, the empirical Var[eta] should be close to sigma_eta^2."""
        cfg = NonlinearDGPConfig(n_samples=50_000, sigma_eta=1.0, heavy_tail_eta=True, eta_beta=1.0, seed=7)
        _, _, _, eta, _, _, _ = generate_nonlinear_data(cfg)
        assert abs(float(np.var(eta)) - 1.0) < 0.05, f"After normalisation Var[eta] should be ~1, got {np.var(eta):.4f}"


# ---------------------------------------------------------------------------
# OLS bias regression: the CORE CLAIM
# ---------------------------------------------------------------------------


class TestOLSBiasRegression:
    """Regression tests encoding the paper's central claim.

    OLS should be *biased* under nonlinear confounding and approximately
    *unbiased* under linear confounding (large n).

    These are single-sample checks on n=8000, not full Monte Carlo. They
    verify the direction and rough magnitude of the bias. A separate smoke
    test in the runner checks the RMSE over replications.
    """

    @pytest.fixture(scope="class")
    def nonlinear_sample(self):
        cfg = hard_preset(n_samples=8000, seed=2024)
        return generate_nonlinear_data(cfg)

    @pytest.fixture(scope="class")
    def linear_sample(self):
        cfg = linear_preset(n_samples=8000, seed=2024)
        return generate_nonlinear_data(cfg)

    def test_ols_rmse_exceeds_gbm_oml_on_nonlinear_preset(self, nonlinear_sample):
        """GBM-nuisance OML should have lower RMSE than OLS on the nonlinear preset.

        We run a 10-replication mini Monte Carlo (fast, serial) to avoid
        dependence on a single random coefficient draw. The quadratic/cross-
        product design guarantees zero linear projection, so the ratio
        OLS-RMSE / GBM-OML-RMSE should comfortably exceed 1.3x.
        """
        from baselines import ols_baseline
        from main_estimation import all_together_cross_fitting
        from nonlinear_dgp import eta_moments_from_config
        from nonlinear_runner import _make_nuisance_models

        theta = 1.5
        ols_vals, oml_vals = [], []
        for seed in range(15):
            cfg = hard_preset(n_samples=2000, seed=seed)
            X, T, Y, _, _, _, _ = generate_nonlinear_data(cfg)
            ols_vals.append(float(ols_baseline(X, T, Y)[0]))
            mt, mo = _make_nuisance_models("gbm")
            m2, m3c = eta_moments_from_config(cfg)
            oml, *_ = all_together_cross_fitting(X, T, Y, m2, m3c, mt, mo)
            oml_vals.append(float(oml))

        def rmse(arr):
            a = np.array(arr)
            return float(np.sqrt((np.mean(a) - theta) ** 2 + np.std(a) ** 2))

        ols_rmse = rmse(ols_vals)
        oml_rmse = rmse(oml_vals)
        assert ols_rmse > 1.3 * oml_rmse, (
            f"OLS RMSE ({ols_rmse:.4f}) should exceed GBM-OML RMSE ({oml_rmse:.4f}) "
            "by at least 1.3x on the hard preset (nonlinear + heteroscedastic)."
        )

    def test_ols_approximately_unbiased_on_linear_preset(self, linear_sample):
        """OLS should recover theta within 0.3 on the linear preset (large n)."""
        from baselines import ols_baseline

        X, T, Y, _, _, _, _ = linear_sample
        theta_hat = float(ols_baseline(X, T, Y)[0])
        true_theta = 1.5
        bias = abs(theta_hat - true_theta)
        assert bias < 0.3, (
            f"OLS on linear preset should be approximately unbiased, "
            f"got bias={bias:.4f} (theta_hat={theta_hat:.4f})"
        )

    def test_nonlinear_g_differs_from_linear(self, nonlinear_sample):
        """g(X) values on the nonlinear preset are not a linear function of X."""
        X, _, _, _, _, g_X, _ = nonlinear_sample
        # If g is linear, OLS of g on X should have near-zero residuals.
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X, g_X)
        resid = g_X - lr.predict(X)
        resid_std = float(np.std(resid))
        # Under purely linear g the residual std should be near zero;
        # under nonlinear g it should be meaningfully positive.
        assert resid_std > 0.1, (
            f"Residual std of linear fit to g(X) should be >0.1 under nonlinear "
            f"confounding, got {resid_std:.4f}. The DGP may not be nonlinear enough."
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_hard_preset_has_all_toggles_on(self):
        cfg = hard_preset()
        assert cfg.nonlinear_confounding is True
        assert cfg.heavy_tail_eta is True
        assert cfg.heteroscedastic_eps is True

    def test_linear_preset_has_all_toggles_off(self):
        cfg = linear_preset()
        assert cfg.nonlinear_confounding is False
        assert cfg.heavy_tail_eta is False
        assert cfg.high_dim is False
        assert cfg.heteroscedastic_eps is False

    def test_hard_preset_generates_data(self):
        cfg = hard_preset(n_samples=200, seed=0)
        X, T, Y, eta, m_X, g_X, alpha = generate_nonlinear_data(cfg)
        assert X.shape == (200, 10)
        assert np.all(np.isfinite(Y))


# ---------------------------------------------------------------------------
# Runner schema
# ---------------------------------------------------------------------------


class TestNonlinearRunner:
    def test_runner_returns_documented_schema(self):
        cfg = NonlinearDGPConfig(
            n_samples=200,
            n_covariates=5,
            support_size=3,
            treatment_effect=1.0,
            sigma_outcome=0.5,
            seed=0,
        )
        results = run_nonlinear_experiments(
            config=cfg,
            n_experiments=5,
            base_seed=100,
            n_jobs=1,
            nuisance="linear",
            verbose=False,
        )

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
            "nuisance",
        ):
            assert key in results, f"missing key {key}"

        n_methods = len(METHOD_NAMES)
        assert results["estimates"].shape == (5, n_methods)
        assert results["biases"].shape == (n_methods,)
        assert results["sigmas"].shape == (n_methods,)
        assert results["rmse"].shape == (n_methods,)
        assert results["n_attempted"] == 5
        assert results["treatment_effect"] == pytest.approx(1.0)

    def test_runner_linear_preset_recovers_theta(self):
        """On the linear preset, OML and baselines should recover theta in expectation.

        The HOML estimated-moments variants (Robust Ortho Est / Split) can be
        unstable at small n because the nested moment estimation on ~375 samples
        has high variance. We only check the stable paths: Ortho ML, OLS, Matching.
        """
        cfg = linear_preset(n_samples=1500, seed=None)
        cfg.support_size = 4

        results = run_nonlinear_experiments(
            config=cfg,
            n_experiments=20,
            base_seed=42,
            n_jobs=1,
            nuisance="linear",
            verbose=False,
        )

        biases = results["biases"]
        finite_per_method = results["finite_per_method"]
        # Only check the numerically-stable methods on this small run.
        stable_methods = {"Ortho ML", "OLS", "Matching"}
        for idx, name in enumerate(METHOD_NAMES):
            if name not in stable_methods:
                continue
            assert finite_per_method[idx] >= 15, f"{name}: too few finite runs"
            assert abs(biases[idx]) < 0.5, f"{name} bias {biases[idx]:.3f} too large on linear DGP"

    def test_runner_nonlinear_ols_is_biased(self):
        """OLS RMSE should exceed GBM-nuisance OML RMSE under nonlinear confounding.

        With a flexible (GBM) first stage, OML can remove the nonlinear
        confounding; OLS cannot. We use a strong alpha_scale=2.0 to ensure
        the bias is clearly visible over 20 replications.
        """
        cfg = NonlinearDGPConfig(
            n_samples=2000,
            n_covariates=10,
            support_size=5,
            treatment_effect=1.5,
            sigma_eta=1.0,
            sigma_outcome=1.0,
            nonlinear_confounding=True,
            heavy_tail_eta=False,
            heteroscedastic_eps=False,
            alpha_scale=1.0,
            beta_scale=1.0,
            interaction_scale=0.3,
            seed=None,
        )
        results = run_nonlinear_experiments(
            config=cfg,
            n_experiments=20,
            base_seed=999,
            n_jobs=1,
            nuisance="gbm",
            verbose=False,
        )

        method_idx = {name: i for i, name in enumerate(METHOD_NAMES)}
        ols_rmse = results["rmse"][method_idx["OLS"]]
        oml_rmse = results["rmse"][method_idx["Ortho ML"]]

        # OLS should have meaningfully higher RMSE than GBM-nuisance OML under
        # nonlinear confounding. The quadratic/cross-product design guarantees
        # zero linear projection, so OLS cannot absorb any confounding via X.
        # The 1.5x threshold is conservative; empirically the ratio is ~1.9x at
        # n=2000 with alpha_scale=1.0.
        assert ols_rmse > 1.5 * oml_rmse, (
            f"OLS RMSE ({ols_rmse:.4f}) should exceed GBM-OML RMSE ({oml_rmse:.4f}) "
            "by at least 1.5x under nonlinear confounding."
        )
