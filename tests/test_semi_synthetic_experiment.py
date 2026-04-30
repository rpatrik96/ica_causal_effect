"""Tests for semi_synthetic_experiment.py.

Three test classes:
- ``TestMainSmoke``: verifies that ``run(...)`` creates the expected output files.
- ``TestSummaryMdShape``: checks the produced markdown table has the expected columns.
- ``TestRunnerHandlesPartialMethodSubset``: single-method run contains only that method.

ICA and HOML are skipped in the test suite (FastICA stochasticity / convergence
issues on tiny n; HOML requires oml_runner / torch-dependent imports).  Only
OLS and Matching are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_tiny(
    tmp_path,
    methods: str = "ols,matching",
    n_experiments: int = 2,
    n_samples: int = 500,
) -> dict:
    """Invoke ``semi_synthetic_experiment.run`` with a small configuration."""
    from semi_synthetic_experiment import run

    return run(
        dataset="california_housing",
        n_samples=n_samples,
        n_experiments=n_experiments,
        treatment_effect=1.0,
        eta_distribution="discrete",
        nonlinearity="identity",
        support_size=None,
        seed=42,
        output_dir=str(tmp_path),
        methods=methods,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Skip guard: skip if network is unavailable (fetch_california_housing)
# ---------------------------------------------------------------------------

_CALIFORNIA_AVAILABLE = True
try:
    from sklearn.datasets import fetch_california_housing

    fetch_california_housing()
except Exception:
    _CALIFORNIA_AVAILABLE = False

needs_california = pytest.mark.skipif(
    not _CALIFORNIA_AVAILABLE,
    reason="California Housing dataset not available (network/cache missing).",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMainSmoke:
    """Smoke test: run produces expected output files."""

    @needs_california
    def test_output_files_created(self, tmp_path):
        _run_tiny(tmp_path)
        npy = tmp_path / "semi_synthetic_california_housing_results.npy"
        md = tmp_path / "semi_synthetic_california_housing_summary.md"
        assert npy.exists(), f".npy file not found at {npy}"
        assert md.exists(), f".md file not found at {md}"

    @needs_california
    def test_npy_contains_correct_rep_count(self, tmp_path):
        out = _run_tiny(tmp_path, n_experiments=2)
        results = out["results"]
        assert len(results) == 2

    @needs_california
    def test_npy_loadable(self, tmp_path):
        _run_tiny(tmp_path, n_experiments=2)
        npy = tmp_path / "semi_synthetic_california_housing_results.npy"
        loaded = np.load(str(npy), allow_pickle=True)
        assert len(loaded) == 2
        assert isinstance(loaded[0], dict)


class TestSummaryMdShape:
    """Verify the markdown table has the expected columns."""

    @needs_california
    def test_md_has_expected_columns(self, tmp_path):
        _run_tiny(tmp_path)
        md_path = tmp_path / "semi_synthetic_california_housing_summary.md"
        content = md_path.read_text(encoding="utf-8")
        header_line = content.splitlines()[0]
        for col in ("method", "bias", "std", "rmse", "n_reps"):
            assert col in header_line, f"Column '{col}' missing from header: {header_line}"

    @needs_california
    def test_md_has_method_rows(self, tmp_path):
        _run_tiny(tmp_path, methods="ols,matching")
        md_path = tmp_path / "semi_synthetic_california_housing_summary.md"
        content = md_path.read_text(encoding="utf-8")
        assert "ols" in content
        assert "matching" in content

    @needs_california
    def test_md_rows_are_parseable(self, tmp_path):
        _run_tiny(tmp_path, methods="ols")
        md_path = tmp_path / "semi_synthetic_california_housing_summary.md"
        lines = [ln for ln in md_path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
        # header + separator + at least one data row
        assert len(lines) >= 3


class TestRunnerHandlesPartialMethodSubset:
    """Single-method run should only store results for that method."""

    @needs_california
    def test_ols_only_results(self, tmp_path):
        out = _run_tiny(tmp_path, methods="ols", n_experiments=2)
        results = out["results"]
        for rep in results:
            assert "ols" in rep, "Expected 'ols' key in every rep."
            assert "matching" not in rep, "Unexpected 'matching' key with methods='ols'."
            assert "homl" not in rep, "Unexpected 'homl' key with methods='ols'."
            assert "ica" not in rep, "Unexpected 'ica' key with methods='ols'."

    @needs_california
    def test_ols_only_npy_schema(self, tmp_path):
        _run_tiny(tmp_path, methods="ols", n_experiments=2)
        npy = tmp_path / "semi_synthetic_california_housing_results.npy"
        loaded = np.load(str(npy), allow_pickle=True)
        for rep in loaded:
            assert set(rep.keys()) <= {"ols", "ground_truth", "rep_seed"}

    @needs_california
    def test_matching_only(self, tmp_path):
        out = _run_tiny(tmp_path, methods="matching", n_experiments=2)
        for rep in out["results"]:
            assert "matching" in rep
            assert "ols" not in rep

    @needs_california
    def test_invalid_method_raises(self, tmp_path):
        from semi_synthetic_experiment import run

        with pytest.raises(ValueError, match="Unknown method"):
            run(
                dataset="california_housing",
                n_samples=200,
                n_experiments=1,
                output_dir=str(tmp_path),
                methods="bogus_method",
                n_jobs=1,
            )
