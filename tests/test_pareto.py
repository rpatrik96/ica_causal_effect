"""
Tests for the Pareto-frontier / selector logic in explorations/pareto/pareto_analysis.py.

Coverage
--------
* kurtosis_selector: deterministic, correct threshold behaviour.
* apply_selector: returns per-row RMSE consistent with selector rule.
* oracle_envelope: is a per-row lower bound on every fixed-method strategy.
* oracle_envelope_is_lower_bound: utility returns True on valid data.
* estimate_excess_kurtosis: finite, monotonic-ish, edge cases.
* homl_blowup_detected: fires when HOML/OML ratio >= threshold near ek~0.
* compute_oracle_selector_stats: structure and invariants.
* gennorm_excess_kurtosis: known analytical values.
* HOML near-Gaussian blow-up regression test against cached grid data.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from explorations.pareto.pareto_analysis import (  # noqa: E402
    _SELECTOR_EK_HIGH,
    apply_selector,
    compute_oracle_selector_stats,
    estimate_excess_kurtosis,
    gennorm_excess_kurtosis,
    homl_blowup_detected,
    kurtosis_selector,
    oracle_envelope,
    oracle_envelope_is_lower_bound,
)

# -------------------------------------------------------------------------
# Fixtures / helpers
# -------------------------------------------------------------------------

CACHE_PATH = os.path.join(REPO, "explorations", "pareto", "grid_results_v2.npy")
BLOWUP_CACHE_PATH = os.path.join(REPO, "explorations", "pareto", "blowup_results.npy")


def _make_row(
    ek: float = 1.0,
    rmse_oml: float = 0.1,
    rmse_ica: float = 0.2,
    rmse_homl_known: float = 0.15,
    rmse_homl_est: float = 0.16,
    rmse_homl_split: float = 0.17,
) -> Dict:
    """Construct a minimal row dict for unit tests."""
    return dict(
        eta_excess_kurtosis=ek,
        rmse_oml=rmse_oml,
        rmse_ica=rmse_ica,
        rmse_homl_known=rmse_homl_known,
        rmse_homl_est=rmse_homl_est,
        rmse_homl_split=rmse_homl_split,
    )


def _load_npy(path: str) -> List[Dict]:
    """Load an npy file of dicts; skip test if absent."""
    if not os.path.exists(path):
        pytest.skip(f"Cache not found: {path}")
    return list(np.load(path, allow_pickle=True))


def _load_grid() -> List[Dict]:
    return _load_npy(CACHE_PATH)


def _load_blowup() -> List[Dict]:
    return _load_npy(BLOWUP_CACHE_PATH)


# -------------------------------------------------------------------------
# 1. kurtosis_selector: deterministic & correct
# -------------------------------------------------------------------------


class TestKurtosisSelector:
    def test_high_kurtosis_selects_ica(self):
        assert kurtosis_selector(5.0) == "ICA"

    def test_low_kurtosis_selects_oml(self):
        assert kurtosis_selector(-1.0) == "OML"

    def test_near_gaussian_selects_oml(self):
        # ek == 0 (Gaussian) -> OML (HOML would blow up)
        assert kurtosis_selector(0.0) == "OML"

    def test_moderate_kurtosis_selects_oml(self):
        assert kurtosis_selector(1.5) == "OML"

    def test_exactly_at_threshold_oml(self):
        # ek == ek_high (not strictly above) -> OML
        assert kurtosis_selector(_SELECTOR_EK_HIGH) == "OML"

    def test_just_above_threshold_ica(self):
        assert kurtosis_selector(_SELECTOR_EK_HIGH + 0.01) == "ICA"

    def test_nan_kurtosis_selects_oml(self):
        assert kurtosis_selector(np.nan) == "OML"

    def test_inf_kurtosis_selects_oml(self):
        assert kurtosis_selector(np.inf) == "OML"

    def test_custom_thresholds(self):
        assert kurtosis_selector(2.5, ek_high=2.0) == "ICA"
        assert kurtosis_selector(1.9, ek_high=2.0) == "OML"

    def test_deterministic(self):
        for ek in [-2.0, 0.0, 0.76, 3.01, 22.2]:
            assert kurtosis_selector(ek) == kurtosis_selector(ek)


# -------------------------------------------------------------------------
# 2. apply_selector: returns correct RMSE per row
# -------------------------------------------------------------------------


class TestApplySelector:
    def _rows(self):
        return [
            _make_row(ek=5.0, rmse_oml=0.10, rmse_ica=0.05),  # high ek -> ICA
            _make_row(ek=0.0, rmse_oml=0.10, rmse_ica=0.50),  # Gaussian -> OML
            _make_row(ek=1.5, rmse_oml=0.08, rmse_ica=0.09),  # moderate -> OML
        ]

    def test_ica_row_uses_rmse_ica(self):
        sel = apply_selector(self._rows())
        assert sel[0] == pytest.approx(0.05)

    def test_oml_row_uses_rmse_oml(self):
        sel = apply_selector(self._rows())
        assert sel[1] == pytest.approx(0.10)
        assert sel[2] == pytest.approx(0.08)

    def test_length_matches_input(self):
        rows = self._rows()
        assert len(apply_selector(rows)) == len(rows)

    def test_selector_picks_ica_when_clearly_better(self):
        rows = [_make_row(ek=10.0, rmse_oml=0.20, rmse_ica=0.05)]
        assert apply_selector(rows)[0] == pytest.approx(0.05)

    def test_nan_kurtosis_falls_back_to_oml(self):
        rows = [_make_row(ek=np.nan, rmse_oml=0.10, rmse_ica=0.05)]
        assert apply_selector(rows)[0] == pytest.approx(0.10)


# -------------------------------------------------------------------------
# 3. oracle_envelope: lower bound on every fixed method
# -------------------------------------------------------------------------


class TestOracleEnvelope:
    def _rows(self):
        return [
            _make_row(rmse_oml=0.10, rmse_ica=0.05, rmse_homl_known=0.08),
            _make_row(rmse_oml=0.03, rmse_ica=0.07, rmse_homl_known=0.05),
            _make_row(rmse_oml=0.06, rmse_ica=0.06, rmse_homl_known=0.04),
        ]

    def test_oracle_le_oml(self):
        env = oracle_envelope(self._rows(), include_homl=True)
        oml = np.array([r["rmse_oml"] for r in self._rows()])
        assert np.all(env <= oml + 1e-12)

    def test_oracle_le_ica(self):
        env = oracle_envelope(self._rows(), include_homl=True)
        ica = np.array([r["rmse_ica"] for r in self._rows()])
        assert np.all(env <= ica + 1e-12)

    def test_oracle_le_homl_known(self):
        rows = self._rows()
        env = oracle_envelope(rows, include_homl=True)
        hk = np.array([r["rmse_homl_known"] for r in rows])
        finite = np.isfinite(hk)
        assert np.all(env[finite] <= hk[finite] + 1e-12)

    def test_oracle_with_homl_le_without(self):
        rows = self._rows()
        assert np.all(oracle_envelope(rows, include_homl=True) <= oracle_envelope(rows, include_homl=False) + 1e-12)

    def test_oracle_ignores_inf_homl(self):
        rows = [_make_row(rmse_oml=0.10, rmse_ica=0.05, rmse_homl_known=np.inf)]
        assert oracle_envelope(rows, include_homl=True)[0] == pytest.approx(0.05)

    def test_oracle_envelope_is_lower_bound_utility(self):
        assert oracle_envelope_is_lower_bound(self._rows()) is True

    def test_oracle_length(self):
        rows = self._rows()
        assert len(oracle_envelope(rows)) == len(rows)


# -------------------------------------------------------------------------
# 4. estimate_excess_kurtosis: finite, edge cases
# -------------------------------------------------------------------------


class TestEstimateExcessKurtosis:
    def test_gaussian_near_zero(self):
        rng = np.random.default_rng(0)
        ek = estimate_excess_kurtosis(rng.normal(0, 1, 50_000))
        assert abs(ek) < 0.15

    def test_laplace_near_three(self):
        rng = np.random.default_rng(1)
        ek = estimate_excess_kurtosis(rng.laplace(0, 1, 50_000))
        assert abs(ek - 3.0) < 0.3

    def test_uniform_near_minus1p2(self):
        rng = np.random.default_rng(2)
        ek = estimate_excess_kurtosis(rng.uniform(-1, 1, 50_000))
        assert abs(ek - (-1.2)) < 0.1

    def test_heavier_tail_higher_kurtosis(self):
        rng = np.random.default_rng(3)
        laplace_ek = estimate_excess_kurtosis(rng.laplace(0, 1, 10_000))
        gauss_ek = estimate_excess_kurtosis(rng.normal(0, 1, 10_000))
        assert laplace_ek > gauss_ek

    def test_constant_array_returns_zero(self):
        assert estimate_excess_kurtosis(np.ones(100)) == pytest.approx(0.0)

    def test_returns_nan_for_short_array(self):
        assert np.isnan(estimate_excess_kurtosis(np.array([1.0, 2.0])))

    def test_handles_empty_array(self):
        assert np.isnan(estimate_excess_kurtosis(np.array([])))

    def test_finite_for_typical_residuals(self):
        rng = np.random.default_rng(4)
        assert np.isfinite(estimate_excess_kurtosis(rng.standard_t(df=5, size=1000)))


# -------------------------------------------------------------------------
# 5. homl_blowup_detected: encodes the near-Gaussian instability claim
# -------------------------------------------------------------------------


class TestHomlBlowupDetected:
    def test_fires_when_homl_dominates(self):
        rows = [_make_row(ek=0.0, rmse_oml=0.02, rmse_homl_known=1.0)]
        assert homl_blowup_detected(rows, threshold=10.0) is True

    def test_does_not_fire_when_homl_ok(self):
        rows = [_make_row(ek=0.0, rmse_oml=0.02, rmse_homl_known=0.04)]
        assert homl_blowup_detected(rows, threshold=10.0) is False

    def test_does_not_fire_for_high_kurtosis(self):
        rows = [_make_row(ek=10.0, rmse_oml=0.02, rmse_homl_known=1.0)]
        assert homl_blowup_detected(rows, threshold=10.0) is False

    def test_custom_threshold(self):
        rows = [_make_row(ek=0.1, rmse_oml=0.02, rmse_homl_known=0.12)]
        assert homl_blowup_detected(rows, threshold=5.0) is True
        assert homl_blowup_detected(rows, threshold=10.0) is False

    def test_fires_only_on_near_gauss_rows(self):
        rows = [
            _make_row(ek=0.0, rmse_oml=0.02, rmse_homl_known=1.0),
            _make_row(ek=5.0, rmse_oml=0.02, rmse_homl_known=0.01),
        ]
        assert homl_blowup_detected(rows, threshold=10.0) is True

    def test_handles_nan_homl(self):
        rows = [_make_row(ek=0.0, rmse_oml=0.02, rmse_homl_known=np.nan)]
        assert homl_blowup_detected(rows, threshold=10.0) is False


# -------------------------------------------------------------------------
# 6. compute_oracle_selector_stats: structure and invariants
# -------------------------------------------------------------------------


class TestComputeOracleSelectorStats:
    def _rows(self):
        return [
            _make_row(ek=5.0, rmse_oml=0.10, rmse_ica=0.05, rmse_homl_known=0.12),
            _make_row(ek=0.5, rmse_oml=0.06, rmse_ica=0.09, rmse_homl_known=0.08),
            _make_row(ek=-0.5, rmse_oml=0.04, rmse_ica=0.07, rmse_homl_known=0.06),
        ]

    def test_returns_dict_with_required_keys(self):
        stats = compute_oracle_selector_stats(self._rows())
        required = {
            "n_rows",
            "mean_oracle",
            "mean_oml",
            "mean_ica",
            "mean_homl_known",
            "mean_selector",
            "best_fixed_method",
            "best_fixed_rmse",
            "oracle_gap_pct",
            "selector_gap_recovery_pct",
        }
        assert required.issubset(stats.keys())

    def test_oracle_le_best_fixed(self):
        stats = compute_oracle_selector_stats(self._rows())
        assert stats["mean_oracle"] <= stats["best_fixed_rmse"] + 1e-12

    def test_n_rows_correct(self):
        stats = compute_oracle_selector_stats(self._rows())
        assert stats["n_rows"] == len(self._rows())

    def test_empty_rows_returns_empty_dict(self):
        assert compute_oracle_selector_stats([]) == {}

    def test_all_nan_ica_excluded(self):
        rows = [_make_row(rmse_ica=np.nan)] * 3
        assert compute_oracle_selector_stats(rows) == {}

    def test_best_fixed_is_minimum_of_fixed_methods(self):
        stats = compute_oracle_selector_stats(self._rows())
        assert stats["best_fixed_rmse"] <= stats["mean_oml"] + 1e-12
        assert stats["best_fixed_rmse"] <= stats["mean_ica"] + 1e-12

    def test_oracle_gap_pct_nonneg(self):
        stats = compute_oracle_selector_stats(self._rows())
        assert stats["oracle_gap_pct"] >= -1e-9


# -------------------------------------------------------------------------
# 7. gennorm_excess_kurtosis: known analytical values
# -------------------------------------------------------------------------


class TestGennormExcessKurtosis:
    def test_gaussian_is_zero(self):
        assert abs(gennorm_excess_kurtosis(2.0)) < 1e-10

    def test_laplace_is_three(self):
        assert abs(gennorm_excess_kurtosis(1.0) - 3.0) < 1e-6

    def test_heavy_tail_positive(self):
        assert gennorm_excess_kurtosis(0.5) > 0
        assert gennorm_excess_kurtosis(1.5) > 0

    def test_light_tail_negative(self):
        assert gennorm_excess_kurtosis(3.0) < 0
        assert gennorm_excess_kurtosis(4.0) < 0

    def test_monotone_decreasing_in_beta(self):
        betas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        ks = [gennorm_excess_kurtosis(b) for b in betas]
        for i in range(len(ks) - 1):
            assert ks[i] > ks[i + 1], f"Not monotone at beta={betas[i]}"

    def test_returns_float(self):
        assert isinstance(gennorm_excess_kurtosis(1.5), float)


# -------------------------------------------------------------------------
# 8. Regression tests against cached grid data
# -------------------------------------------------------------------------


class TestGridRegressions:
    """Encode the key empirical claims as regression tests."""

    def test_homl_blowup_at_gaussian(self):
        """HOML-known RMSE >> OML RMSE when beta=2 (Gaussian, ek=0)."""
        rows = _load_grid()
        gauss_rows = [r for r in rows if abs(r.get("eta_excess_kurtosis", 99)) < 0.01]
        assert len(gauss_rows) >= 1, "No near-Gaussian rows in grid"
        for r in gauss_rows:
            ratio = r["rmse_homl_known"] / r["rmse_oml"]
            assert ratio >= 10.0, (
                f"Expected HOML blow-up (ratio>=10) at ek~0, " f"got ratio={ratio:.2f} (n={r['n_samples']})"
            )

    def test_oml_beats_all_homl_variants_on_average(self):
        """OML mean RMSE < all HOML variant means across the grid."""
        rows = _load_grid()
        stats = compute_oracle_selector_stats(rows)
        assert stats["mean_oml"] < stats["mean_homl_known"]
        assert stats["mean_oml"] < stats["mean_homl_est"]
        assert stats["mean_oml"] < stats["mean_homl_split"]

    def test_homl_never_improves_oracle(self):
        """Adding HOML to oracle candidates does NOT improve mean oracle RMSE."""
        rows = _load_grid()
        oracle_with = oracle_envelope(rows, include_homl=True)
        oracle_without = oracle_envelope(rows, include_homl=False)
        improvement = oracle_without.mean() - oracle_with.mean()
        # Improvement should be < 0.1% relative — HOML never wins any cell
        assert (
            improvement < 0.001 * oracle_without.mean()
        ), f"Unexpected HOML oracle improvement: {100*improvement/oracle_without.mean():.3f}%"

    def test_ica_wins_heavy_tail_regime(self):
        """ICA RMSE < OML RMSE for rows with ek > 20 and n >= 2000.

        At small n (500, 1000) finite-sample noise prevents ICA from reliably
        beating OML even in the very heavy-tail regime (beta=0.5, ek~22).
        The advantage consolidates at n >= 2000.
        """
        rows = _load_grid()
        heavy_rows = [r for r in rows if r.get("eta_excess_kurtosis", 0) > 20 and r["n_samples"] >= 2000]
        assert len(heavy_rows) >= 2, "Insufficient heavy-tail rows with n>=2000"
        for r in heavy_rows:
            assert r["rmse_ica"] < r["rmse_oml"], (
                f"ICA should beat OML at ek>20, n>=2000: "
                f"ICA={r['rmse_ica']:.5f} OML={r['rmse_oml']:.5f} n={r['n_samples']}"
            )

    def test_oml_wins_near_gaussian_regime(self):
        """OML RMSE < ICA RMSE when eta is Gaussian (ek=0)."""
        rows = _load_grid()
        gauss_rows = [r for r in rows if abs(r.get("eta_excess_kurtosis", 99)) < 0.01]
        assert len(gauss_rows) >= 1
        for r in gauss_rows:
            assert r["rmse_oml"] < r["rmse_ica"], (
                f"OML should beat ICA at ek~0: " f"OML={r['rmse_oml']:.5f} ICA={r['rmse_ica']:.5f}"
            )

    def test_oracle_is_lower_bound(self):
        """oracle_envelope <= every fixed-method RMSE for all grid rows."""
        assert oracle_envelope_is_lower_bound(_load_grid())

    def test_oracle_gap_below_5_pct(self):
        """The oracle gap (best-fixed vs oracle) should be below 5%."""
        rows = _load_grid()
        stats = compute_oracle_selector_stats(rows)
        assert stats["oracle_gap_pct"] < 5.0, f"Oracle gap {stats['oracle_gap_pct']:.2f}% exceeds 5%"


class TestBlowupSweepRegressions:
    """Regression tests against the near-Gaussian blow-up sweep."""

    def test_blowup_detected_in_sweep(self):
        rows = _load_blowup()
        assert homl_blowup_detected(rows, threshold=10.0) is True

    def test_blowup_ratio_large_near_gaussian(self):
        """Mean HOML-known/OML ratio >= 10x for rows with |ek| < 0.2."""
        rows = _load_blowup()
        near_g = [r for r in rows if abs(r.get("eta_excess_kurtosis", 99)) < 0.2]
        assert len(near_g) >= 2, "Insufficient near-Gaussian rows"
        ratios = [
            r["rmse_homl_known"] / r["rmse_oml"]
            for r in near_g
            if np.isfinite(r.get("rmse_homl_known", np.nan)) and r["rmse_oml"] > 1e-10
        ]
        assert ratios
        assert np.mean(ratios) >= 10.0, f"Mean HOML/OML ratio near Gaussian = {np.mean(ratios):.1f}x, expected >= 10x"

    def test_oml_stable_across_blowup_sweep(self):
        """OML RMSE < 0.10 across all sweep rows (no blow-up for OML)."""
        for r in _load_blowup():
            assert r["rmse_oml"] < 0.10, f"OML unexpectedly large: {r['rmse_oml']:.5f} at beta={r.get('beta', '?')}"

    def test_homl_all_variants_blow_up_near_gauss(self):
        """All three HOML variants exceed OML RMSE for rows with |ek| < 0.15."""
        rows = _load_blowup()
        near_g = [r for r in rows if abs(r.get("eta_excess_kurtosis", 99)) < 0.15]
        assert len(near_g) >= 2
        for r in near_g:
            for key in ("rmse_homl_known", "rmse_homl_est", "rmse_homl_split"):
                val = r.get(key, np.nan)
                if np.isfinite(val):
                    assert val > r["rmse_oml"], (
                        f"{key}={val:.4f} should exceed OML={r['rmse_oml']:.4f} "
                        f"near ek~0 (beta={r.get('beta', '?')})"
                    )
