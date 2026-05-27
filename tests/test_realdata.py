"""Tests for realdata_loaders and realdata_runner.

All tests use synthetic fixtures so they run without network access.
The synthetic fixture is triggered by passing use_fixture_on_failure=True
(the default) when no real data is cached.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the repo root is on the path when tests are run from the tests/ dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realdata_loaders import (
    _ihdp_synthetic_fixture,
    _jobs_synthetic_fixture,
    _parse_ihdp_array,
    load_ihdp,
    load_jobs,
    load_jobs_observational,
)

# ---------------------------------------------------------------------------
# IHDP loader tests
# ---------------------------------------------------------------------------


class TestIHDPFixture:
    """Tests using the synthetic IHDP fixture (no network required)."""

    def test_fixture_shape(self):
        arr = _ihdp_synthetic_fixture()
        assert arr.shape == (747, 30), f"Expected (747,30), got {arr.shape}"

    def test_fixture_treatment_binary(self):
        arr = _ihdp_synthetic_fixture()
        t = arr[:, 0]
        assert set(np.unique(t)).issubset({0.0, 1.0}), "Treatment must be binary"

    def test_fixture_reasonable_ate(self):
        arr = _ihdp_synthetic_fixture()
        _, _, _, ate, att = _parse_ihdp_array(arr)
        # Fixture is constructed with ATE ~0.4; just check it is reasonable
        assert -1.0 < ate < 2.0, f"ATE out of reasonable range: {ate}"
        assert -1.0 < att < 2.0, f"ATT out of reasonable range: {att}"

    def test_parse_ihdp_array_shapes(self):
        arr = _ihdp_synthetic_fixture()
        x, t, y, ate, att = _parse_ihdp_array(arr)
        assert x.shape == (747, 25), f"X shape: {x.shape}"
        assert t.shape == (747,), f"T shape: {t.shape}"
        assert y.shape == (747,), f"Y shape: {y.shape}"
        assert isinstance(ate, float)
        assert isinstance(att, float)


class TestLoadIHDP:
    """Tests for load_ihdp() using fixture fallback."""

    def test_single_replication_shapes(self, tmp_path):
        # Use a temp dir with no cached files to force fixture
        X, T, Y, att = load_ihdp(replication=99, data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert X.shape[1] == 25, "Expect 25 covariates"
        assert T.shape[0] == X.shape[0]
        assert Y.shape[0] == X.shape[0]
        assert isinstance(att, float)

    def test_treatment_binary(self, tmp_path):
        X, T, Y, att = load_ihdp(replication=99, data_dir=str(tmp_path), use_fixture_on_failure=True)
        unique_T = np.unique(T)
        assert set(unique_T).issubset({0.0, 1.0}), f"T not binary: {unique_T}"

    def test_no_nans_in_X_T_Y(self, tmp_path):
        X, T, Y, att = load_ihdp(replication=99, data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert not np.any(np.isnan(X)), "NaN in X"
        assert not np.any(np.isnan(T)), "NaN in T"
        assert not np.any(np.isnan(Y)), "NaN in Y"

    def test_stacked_replications(self, tmp_path):
        X, T, Y, atts = load_ihdp(
            replication=None,
            n_replications=3,
            data_dir=str(tmp_path),
            use_fixture_on_failure=True,
        )
        assert X.shape[0] == 747 * 3, f"Expected 3*747 rows, got {X.shape[0]}"
        assert atts.shape == (3,), f"Expected 3 ATT values, got {atts.shape}"

    def test_failure_raises_without_fixture(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ihdp(replication=999, data_dir=str(tmp_path), use_fixture_on_failure=False)

    def test_real_data_if_cached(self):
        """If the real IHDP file is cached, verify it loads correctly."""
        from realdata_loaders import _ihdp_csv_path

        path = _ihdp_csv_path(1)
        if not os.path.exists(path):
            pytest.skip("Real IHDP replication 1 not cached; skipping real-data test.")
        X, T, Y, att = load_ihdp(replication=1, use_fixture_on_failure=False)
        assert X.shape == (747, 25)
        assert T.shape == (747,)
        assert set(np.unique(T)).issubset({0.0, 1.0})
        # ATT for real IHDP rep 1 should be in a sane range
        assert 0.0 < att < 1.0, f"Unexpected ATT={att} for real IHDP rep 1"


# ---------------------------------------------------------------------------
# Jobs loader tests
# ---------------------------------------------------------------------------


class TestJobsFixture:
    """Tests using the synthetic Jobs fixture (no network required)."""

    def test_fixture_shapes(self):
        X, T, Y = _jobs_synthetic_fixture()
        assert X.shape[1] == 8, "Expect 8 covariates (age, educ, black, hisp, married, nodegree, re74, re75)"
        assert T.shape[0] == X.shape[0]
        assert Y.shape[0] == X.shape[0]

    def test_fixture_treatment_binary(self):
        X, T, Y = _jobs_synthetic_fixture()
        assert set(np.unique(T)).issubset({0.0, 1.0}), "T must be binary"

    def test_fixture_correct_att(self):
        X, T, Y = _jobs_synthetic_fixture()
        att = Y[T == 1].mean() - Y[T == 0].mean()
        # Fixture is constructed so ATT == 1794 exactly
        assert abs(att - 1794.0) < 1.0, f"ATT={att:.1f}, expected ~1794"

    def test_fixture_group_sizes(self):
        X, T, Y = _jobs_synthetic_fixture()
        assert int((T == 1).sum()) == 185
        assert int((T == 0).sum()) == 260


class TestLoadJobs:
    """Tests for load_jobs() using fixture fallback."""

    def test_shapes(self, tmp_path):
        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert X.ndim == 2
        assert T.ndim == 1
        assert Y.ndim == 1
        assert X.shape[0] == T.shape[0] == Y.shape[0]

    def test_treatment_binary(self, tmp_path):
        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert set(np.unique(T)).issubset({0.0, 1.0})

    def test_meta_keys(self, tmp_path):
        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        for key in ("att_benchmark", "att_experimental", "is_real", "n_treated", "n_control", "source"):
            assert key in meta, f"Missing key: {key}"

    def test_meta_is_real_flag_when_fixture(self, tmp_path, monkeypatch):
        # Simulate no network / no cached data so the fixture is used.
        import realdata_loaders

        monkeypatch.setattr(realdata_loaders, "_download_nber_dta", lambda *a, **kw: False)
        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert meta["is_real"] is False

    def test_no_nans(self, tmp_path):
        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(T))
        assert not np.any(np.isnan(Y))

    def test_failure_raises_without_fixture(self, tmp_path, monkeypatch):
        import realdata_loaders

        monkeypatch.setattr(realdata_loaders, "_download_nber_dta", lambda *a, **kw: False)
        with pytest.raises(FileNotFoundError):
            load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=False)

    def test_att_experimental_key_present_and_finite(self, tmp_path):
        """att_experimental must be a finite float regardless of fixture/real."""
        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert np.isfinite(meta["att_experimental"]), "att_experimental should be finite"

    def test_real_data_if_cached(self):
        """If the real NSW data is cached, check n=445 and ATT near benchmark."""
        from realdata_loaders import _JOBS_NBER_CSV_NAME, DATA_DIR

        csv_path = os.path.join(DATA_DIR, _JOBS_NBER_CSV_NAME)
        if not os.path.exists(csv_path):
            pytest.skip("Real Jobs data not cached; skipping real-data test.")
        X, T, Y, meta = load_jobs(use_fixture_on_failure=False)
        assert meta["is_real"] is True
        assert meta["n_treated"] == 185, f"Expected 185 treated, got {meta['n_treated']}"
        assert meta["n_control"] == 260, f"Expected 260 control, got {meta['n_control']}"
        # Experimental ATT from data should be close to the D&W benchmark
        assert (
            abs(meta["att_experimental"] - 1794.0) < 50.0
        ), f"Experimental ATT={meta['att_experimental']:.1f}, expected ~1794"


class TestLoadJobsObservational:
    """Tests for load_jobs_observational() using fixture fallback."""

    def test_shapes_cps_fixture(self, tmp_path):
        X, T, Y, meta = load_jobs_observational(comparison="cps", data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert X.ndim == 2
        assert T.ndim == 1
        assert Y.ndim == 1
        assert X.shape[0] == T.shape[0] == Y.shape[0]

    def test_meta_comparison_key(self, tmp_path):
        X, T, Y, meta = load_jobs_observational(comparison="psid", data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert meta["comparison"] == "PSID"

    def test_invalid_comparison_raises(self, tmp_path):
        with pytest.raises(ValueError, match="comparison must be"):
            load_jobs_observational(comparison="invalid", data_dir=str(tmp_path))

    def test_treated_group_is_nsw_treated(self, tmp_path):
        """Treated rows in observational set are the NSW treated units."""
        X, T, Y, meta = load_jobs_observational(comparison="cps", data_dir=str(tmp_path), use_fixture_on_failure=True)
        assert meta["n_treated"] > 0
        assert meta["n_control"] > 0
        # n_treated should equal the NSW treated count from fixture (185)
        assert meta["n_treated"] == 185

    def test_failure_raises_without_fixture(self, tmp_path, monkeypatch):
        import realdata_loaders

        monkeypatch.setattr(realdata_loaders, "_download_nber_dta", lambda *a, **kw: False)
        with pytest.raises(FileNotFoundError):
            load_jobs_observational(comparison="cps", data_dir=str(tmp_path), use_fixture_on_failure=False)


# ---------------------------------------------------------------------------
# Runner smoke tests
# ---------------------------------------------------------------------------


class TestRunnerSmoke:
    """Smoke tests: load fixture data and run all estimators end-to-end."""

    def test_ihdp_single_replication_runs(self, tmp_path):
        """Single IHDP replication through the runner returns 7 estimates."""
        from realdata_runner import _run_single_replication

        X, T, Y, att = load_ihdp(replication=1, data_dir=str(tmp_path), use_fixture_on_failure=True)
        est = _run_single_replication(X, T, Y, nuisance="linear")
        assert len(est) == 7, f"Expected 7 estimates, got {len(est)}"
        # Check the non-ICA estimates are finite (ICA may be nan on small data)
        for i, (name, val) in enumerate(zip(("OML", "HOML-k", "HOML-e", "HOML-s"), est[:4])):
            assert np.isfinite(val), f"{name} returned non-finite: {val}"

    def test_ihdp_single_replication_gbm_nuisance(self, tmp_path):
        """GBM nuisance path runs without error and returns 7 finite estimates."""
        from realdata_runner import _run_single_replication

        X, T, Y, att = load_ihdp(replication=1, data_dir=str(tmp_path), use_fixture_on_failure=True)
        est = _run_single_replication(X, T, Y, nuisance="gbm")
        assert len(est) == 7
        for name, val in zip(("OML", "HOML-k", "HOML-e", "HOML-s"), est[:4]):
            assert np.isfinite(val), f"GBM nuisance: {name} returned non-finite: {val}"

    def test_jobs_runner_runs(self, tmp_path):
        """Jobs runner executes without error and returns 7 estimates."""
        from realdata_runner import _run_single_replication

        X, T, Y, meta = load_jobs(data_dir=str(tmp_path), use_fixture_on_failure=True)
        est = _run_single_replication(X, T, Y, nuisance="linear")
        assert len(est) == 7

    def test_run_ihdp_function(self, tmp_path, monkeypatch):
        """run_ihdp() with 2 fixture replications completes and returns dict."""
        import realdata_runner
        from realdata_runner import run_ihdp

        orig_load = realdata_runner.load_ihdp

        def patched_load(replication, **kwargs):
            return orig_load(replication, data_dir=str(tmp_path), use_fixture_on_failure=True)

        monkeypatch.setattr(realdata_runner, "load_ihdp", patched_load)

        results = run_ihdp(n_replications=2, nuisance="linear", verbose=False)
        assert results["dataset"] == "IHDP"
        assert results["estimates"].shape == (2, 7)
        assert results["biases"].shape == (7,)
        assert results["rmse"].shape == (7,)
        # New CI keys must be present
        assert "rmse_se" in results
        assert "rmse_ci_lo" in results
        assert "rmse_ci_hi" in results
        assert results["rmse_ci_lo"].shape == (7,)
        assert results["rmse_ci_hi"].shape == (7,)
        assert results["nuisance"] == "linear"

    def test_run_ihdp_gbm_nuisance(self, tmp_path, monkeypatch):
        """run_ihdp() with GBM nuisance returns dict with nuisance='gbm'."""
        import realdata_runner
        from realdata_runner import run_ihdp

        orig_load = realdata_runner.load_ihdp

        def patched_load(replication, **kwargs):
            return orig_load(replication, data_dir=str(tmp_path), use_fixture_on_failure=True)

        monkeypatch.setattr(realdata_runner, "load_ihdp", patched_load)

        results = run_ihdp(n_replications=2, nuisance="gbm", verbose=False)
        assert results["nuisance"] == "gbm"
        assert results["estimates"].shape == (2, 7)

    def test_run_jobs_function(self, tmp_path, monkeypatch):
        """run_jobs() with fixture data completes and returns dict."""
        import realdata_runner
        from realdata_runner import run_jobs

        orig_load = realdata_runner.load_jobs

        def patched_load(**kwargs):
            return orig_load(data_dir=str(tmp_path), use_fixture_on_failure=True)

        monkeypatch.setattr(realdata_runner, "load_jobs", patched_load)

        results = run_jobs(verbose=False)
        assert results["dataset"] == "Jobs"
        assert results["estimates"].shape == (7,)
        assert "att_benchmark" in results
        assert results["att_benchmark"] == pytest.approx(1794.0)
        # New keys
        assert "att_experimental" in results
        assert "nuisance" in results

    def test_run_jobs_observational(self, tmp_path, monkeypatch):
        """run_jobs_observational() with fixture data completes and returns dict."""
        import realdata_runner
        from realdata_runner import run_jobs_observational

        orig_load = realdata_runner.load_jobs_observational

        def patched_load(**kwargs):
            return orig_load(comparison="cps", data_dir=str(tmp_path), use_fixture_on_failure=True)

        monkeypatch.setattr(realdata_runner, "load_jobs_observational", patched_load)

        results = run_jobs_observational(comparison="cps", verbose=False)
        assert "Jobs" in results["dataset"]
        assert results["estimates"].shape == (7,)
        assert results["att_benchmark"] == pytest.approx(1794.0)
        assert "comparison" in results

    def test_nuisance_model_factory_all_types(self):
        """_make_nuisance_models returns valid sklearn estimator pairs."""
        from realdata_runner import _make_nuisance_models

        for ntype in ("linear", "gbm", "poly"):
            mt, mo = _make_nuisance_models(ntype)
            assert hasattr(mt, "fit"), f"{ntype}: mt has no .fit"
            assert hasattr(mo, "fit"), f"{ntype}: mo has no .fit"

    def test_nuisance_model_factory_invalid(self):
        from realdata_runner import _make_nuisance_models

        with pytest.raises(ValueError, match="nuisance must be"):
            _make_nuisance_models("random_forest_extreme")
