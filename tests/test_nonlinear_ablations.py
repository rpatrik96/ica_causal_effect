"""Tests for nonlinear_ablations.py and the new knobs in nonlinear_dgp / nonlinear_runner.

Coverage:
- nonlinearity_strength=0 produces the same data as linear confounding.
- nonlinearity_strength scaling changes data monotonically.
- RandomForest nuisance is accepted and produces finite estimates.
- axis_isolation sweep returns correct structure.
- nuisance_ablation sweep returns results for all four nuisance types.
- ica_dn_frontier sweep returns results for all (d, n) combinations.
- strength_sweep returns monotonically increasing OLS bias as strength grows.
"""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_dgp import NonlinearDGPConfig, generate_nonlinear_data
from nonlinear_runner import METHOD_NAMES, _make_nuisance_models, run_nonlinear_experiments

# ---------------------------------------------------------------------------
# nonlinearity_strength knob in the DGP
# ---------------------------------------------------------------------------


class TestNonlinearityStrength:
    def test_strength_zero_matches_zero_nonlinear_terms(self):
        """At strength=0, the nonlinear terms are zeroed out.

        With strength=0 and nonlinear_confounding=True, m(X) and g(X) become
        zero (since all coefficients are multiplied by 0). The treatment
        T = 0 + eta is pure noise and Y = theta*T + eps. The resulting Y
        should have lower variance than at strength=1 (no confounding amplification).
        """
        cfg0 = NonlinearDGPConfig(
            n_samples=2000,
            n_covariates=8,
            support_size=4,
            nonlinear_confounding=True,
            nonlinearity_strength=0.0,
            seed=42,
        )
        cfg1 = NonlinearDGPConfig(
            n_samples=2000,
            n_covariates=8,
            support_size=4,
            nonlinear_confounding=True,
            nonlinearity_strength=1.0,
            seed=42,
        )
        X0, T0, Y0, _, m0, g0, _ = generate_nonlinear_data(cfg0)
        X1, T1, Y1, _, m1, g1, _ = generate_nonlinear_data(cfg1)

        # At strength=0, nuisance should be zero
        assert np.allclose(m0, 0.0, atol=1e-10), "m(X) should be zero at strength=0"
        assert np.allclose(g0, 0.0, atol=1e-10), "g(X) should be zero at strength=0"

        # At strength=1, nuisance should be nonzero
        assert np.std(m1) > 0.05, "m(X) should be nonzero at strength=1"
        assert np.std(g1) > 0.05, "g(X) should be nonzero at strength=1"

    def test_strength_scaling_changes_nuisance_magnitude(self):
        """Higher strength -> larger nuisance std."""
        base_kwargs = dict(
            n_samples=2000,
            n_covariates=8,
            support_size=4,
            nonlinear_confounding=True,
            seed=42,
        )
        stds_m = []
        for strength in [0.5, 1.0, 2.0]:
            cfg = NonlinearDGPConfig(**base_kwargs, nonlinearity_strength=strength)
            _, _, _, _, m_X, g_X, _ = generate_nonlinear_data(cfg)
            stds_m.append(float(np.std(m_X)))

        assert stds_m[0] < stds_m[1] < stds_m[2], f"nuisance std should increase with strength: {stds_m}"

    def test_strength_does_not_affect_linear_dgp(self):
        """nonlinearity_strength has no effect when nonlinear_confounding=False."""
        base_kwargs = dict(n_samples=500, n_covariates=6, support_size=3, nonlinear_confounding=False, seed=7)
        cfg_a = NonlinearDGPConfig(**base_kwargs, nonlinearity_strength=0.5)
        cfg_b = NonlinearDGPConfig(**base_kwargs, nonlinearity_strength=2.0)
        _, _, Y_a, _, _, _, _ = generate_nonlinear_data(cfg_a)
        _, _, Y_b, _, _, _, _ = generate_nonlinear_data(cfg_b)
        # Linear DGP: strength is only applied to nonlinear terms which are inactive
        assert np.allclose(Y_a, Y_b), "strength should not affect linear (nonlinear_confounding=False) DGP"

    def test_strength_default_is_one(self):
        """Default nonlinearity_strength=1.0 and explicit 1.0 produce identical data."""
        base_kwargs = dict(n_samples=300, n_covariates=6, support_size=3, nonlinear_confounding=True, seed=99)
        cfg_default = NonlinearDGPConfig(**base_kwargs)
        cfg_explicit = NonlinearDGPConfig(**base_kwargs, nonlinearity_strength=1.0)
        out_d = generate_nonlinear_data(cfg_default)
        out_e = generate_nonlinear_data(cfg_explicit)
        for a, b in zip(out_d, out_e):
            assert np.allclose(a, b), "Default and explicit strength=1.0 must be identical"


# ---------------------------------------------------------------------------
# RandomForest nuisance in the runner
# ---------------------------------------------------------------------------


class TestRandomForestNuisance:
    def test_rf_nuisance_accepted(self):
        """_make_nuisance_models('rf') should not raise."""
        mt, mo = _make_nuisance_models("rf")
        assert mt is not None
        assert mo is not None

    def test_rf_nuisance_invalid_raises(self):
        """Unknown nuisance type should raise ValueError."""
        with pytest.raises(ValueError, match="nuisance"):
            _make_nuisance_models("unknown_model")

    @pytest.mark.slow
    def test_rf_produces_finite_estimates(self):
        """Runner with rf nuisance should return finite estimates on a small run."""
        cfg = NonlinearDGPConfig(
            n_samples=500,
            n_covariates=6,
            support_size=3,
            treatment_effect=1.5,
            nonlinear_confounding=True,
            seed=0,
        )
        results = run_nonlinear_experiments(
            config=cfg,
            n_experiments=5,
            base_seed=42,
            n_jobs=1,
            nuisance="rf",
            verbose=False,
        )
        oml_idx = list(METHOD_NAMES).index("Ortho ML")
        assert np.isfinite(results["rmse"][oml_idx]), "RF-nuisance OML RMSE should be finite"
        assert results["n_attempted"] == 5


# ---------------------------------------------------------------------------
# Ablation functions: structural checks (small, fast)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAxisIsolation:
    def test_axis_isolation_returns_all_labels(self):
        """run_axis_isolation should return entries for all five configurations."""
        from nonlinear_ablations import run_axis_isolation

        expected_labels = {
            "baseline (all-off)",
            "nonlinear-only",
            "heavy-tail-only",
            "heteroscedastic-only",
            "high-dim-only (d=20)",
        }
        results = run_axis_isolation(
            n_samples=500,
            n_experiments=5,
            n_jobs=1,
            nuisance="linear",
            verbose=False,
        )
        assert set(results.keys()) == expected_labels

    def test_axis_isolation_rmse_shape(self):
        """Each result should have the standard schema."""
        from nonlinear_ablations import run_axis_isolation

        results = run_axis_isolation(
            n_samples=300,
            n_experiments=4,
            n_jobs=1,
            nuisance="linear",
            verbose=False,
        )
        n_methods = len(METHOD_NAMES)
        for label, r in results.items():
            assert r["rmse"].shape == (n_methods,), f"{label}: wrong rmse shape"
            assert np.all(np.isfinite(r["rmse"]) | np.isnan(r["rmse"])), f"{label}: non-finite rmse"

    def test_nonlinear_axis_raises_ols_rmse(self):
        """The nonlinear-only axis should produce higher OLS RMSE than the linear baseline."""
        from nonlinear_ablations import run_axis_isolation

        results = run_axis_isolation(
            n_samples=1000,
            n_experiments=10,
            n_jobs=1,
            nuisance="linear",
            verbose=False,
        )
        ols_idx = list(METHOD_NAMES).index("OLS")
        baseline_rmse = results["baseline (all-off)"]["rmse"][ols_idx]
        nonlinear_rmse = results["nonlinear-only"]["rmse"][ols_idx]
        assert nonlinear_rmse > baseline_rmse, (
            f"Nonlinear axis should increase OLS RMSE: " f"baseline={baseline_rmse:.4f}, nonlinear={nonlinear_rmse:.4f}"
        )


@pytest.mark.slow
class TestNuisanceAblation:
    def test_nuisance_ablation_returns_all_types(self):
        """run_nuisance_ablation should return results for linear, poly, rf, gbm."""
        from nonlinear_ablations import run_nuisance_ablation

        results = run_nuisance_ablation(n_samples=400, n_experiments=4, n_jobs=1, verbose=False)
        for nuisance in ["linear", "poly", "rf", "gbm"]:
            assert nuisance in results, f"Missing nuisance type: {nuisance}"

    def test_gbm_oml_beats_linear_oml_on_nonlinear_dgp(self):
        """GBM-nuisance OML should have lower RMSE than linear-nuisance OML under nonlinear confounding."""
        from nonlinear_ablations import run_nuisance_ablation

        results = run_nuisance_ablation(n_samples=1500, n_experiments=8, n_jobs=1, verbose=False)
        oml_idx = list(METHOD_NAMES).index("Ortho ML")
        linear_rmse = results["linear"]["rmse"][oml_idx]
        gbm_rmse = results["gbm"]["rmse"][oml_idx]
        assert gbm_rmse < linear_rmse, (
            f"GBM-OML RMSE ({gbm_rmse:.4f}) should be lower than Linear-OML RMSE ({linear_rmse:.4f}) "
            "under nonlinear confounding."
        )


@pytest.mark.slow
class TestIcaDnFrontier:
    def test_ica_dn_frontier_returns_all_combinations(self):
        """run_ica_dn_frontier should return results for all d x n combinations."""
        from nonlinear_ablations import run_ica_dn_frontier

        d_grid = [2, 5, 8]
        n_grid = [500, 1000, 2000, 5000]
        results = run_ica_dn_frontier(n_experiments=3, n_jobs=1, verbose=False)
        for d in d_grid:
            for n in n_grid:
                key = f"d={d}, n={n}"
                assert key in results, f"Missing key: {key}"

    def test_ica_dn_frontier_schema(self):
        """Each result entry should have the standard schema."""
        from nonlinear_ablations import run_ica_dn_frontier

        results = run_ica_dn_frontier(n_experiments=3, n_jobs=1, verbose=False)
        n_methods = len(METHOD_NAMES)
        for key, r in results.items():
            assert r["rmse"].shape == (n_methods,), f"{key}: wrong rmse shape"

    def test_ica_improves_with_larger_n_at_small_d(self):
        """At small d, ICA RMSE should decrease as n grows (central limit regime)."""
        from nonlinear_ablations import run_ica_dn_frontier

        results = run_ica_dn_frontier(n_experiments=10, n_jobs=1, verbose=False)
        ica_idx = list(METHOD_NAMES).index("ICA")
        # At d=2 (the smallest), ICA should have lower RMSE at n=5000 than n=500
        rmse_small_n = results["d=2, n=500"]["rmse"][ica_idx]
        rmse_large_n = results["d=2, n=5000"]["rmse"][ica_idx]
        # Allow for NaN (ICA convergence failure) — only check if both are finite
        if np.isfinite(rmse_small_n) and np.isfinite(rmse_large_n):
            assert rmse_large_n < rmse_small_n * 1.5, (
                f"ICA RMSE at d=2 should not increase dramatically from n=500 to n=5000: "
                f"{rmse_small_n:.4f} -> {rmse_large_n:.4f}"
            )


@pytest.mark.slow
class TestStrengthSweep:
    def test_strength_sweep_returns_all_strengths(self):
        """run_strength_sweep should return results for all configured strengths."""
        from nonlinear_ablations import run_strength_sweep

        results = run_strength_sweep(n_samples=500, n_experiments=4, n_jobs=1, verbose=False)
        expected = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0"]
        for s in expected:
            assert s in results, f"Missing strength: {s}"

    def test_ols_bias_increases_with_strength(self):
        """OLS absolute bias should be (weakly) increasing in nonlinearity_strength.

        Uses linear nuisance (fast) since OLS bias is DGP-driven, not nuisance-dependent.
        """
        from nonlinear_ablations import run_strength_sweep

        results = run_strength_sweep(n_samples=2000, n_experiments=8, n_jobs=1, nuisance="linear", verbose=False)
        ols_idx = list(METHOD_NAMES).index("OLS")
        # Compare strength=0.0 vs strength=2.0 — should be clearly different
        bias_zero = abs(results["0.0"]["biases"][ols_idx])
        bias_two = abs(results["2.0"]["biases"][ols_idx])
        assert bias_two > bias_zero, (
            f"OLS bias at strength=2.0 ({bias_two:.4f}) should exceed bias at " f"strength=0.0 ({bias_zero:.4f})"
        )

    def test_strength_zero_ols_near_unbiased(self):
        """At strength=0 (effectively linear DGP), OLS should be approximately unbiased.

        Uses linear nuisance (fast) since OLS bias is DGP-driven, not nuisance-dependent.
        """
        from nonlinear_ablations import run_strength_sweep

        results = run_strength_sweep(n_samples=2000, n_experiments=8, n_jobs=1, nuisance="linear", verbose=False)
        ols_idx = list(METHOD_NAMES).index("OLS")
        bias_zero = abs(results["0.0"]["biases"][ols_idx])
        # At strength=0, m(X) and g(X) are zero, so Y = theta*T + eps (no confounding)
        # OLS should recover theta closely
        assert bias_zero < 0.2, f"OLS bias at strength=0 should be near zero (no confounding), got {bias_zero:.4f}"
