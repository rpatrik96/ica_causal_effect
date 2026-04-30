"""Integration tests for the central experiment dispatcher.

Verifies that the new OLS and matching baselines are appended to the
per-experiment result list at indices 5 and 6 in ``ablation_utils``, and
that the ``disable_baselines`` opt-out emits NaN placeholders at the same
positions.

We call ``run_single_experiment`` directly rather than going through
``run_parallel_experiments`` because the latter uses joblib/loky which
forks worker processes; those workers do not inherit the conftest stubs
for ``torch``/``seaborn`` and break on import.  Calling
``run_single_experiment`` in-process exercises the same code path with
the same return contract.
"""

from __future__ import annotations

import numpy as np

from ablation_utils import (
    ICA_IDX,
    MATCHING_IDX,
    OLS_IDX,
    extract_treatment_estimates,
    run_single_experiment,
)


def _make_synthetic_inputs(*, n_samples: int, support_size: int = 5, seed: int = 0):
    """Build a tiny linear PLR (Gaussian X, Gaussian eta, Uniform eps).

    Returns the ``run_single_experiment`` kwargs needed to drive one rep.
    """
    rng = np.random.default_rng(seed)
    treatment_support = np.arange(support_size)
    outcome_support = np.arange(support_size)
    treatment_coef = rng.standard_normal(support_size)
    outcome_coef = rng.standard_normal(support_size)
    x = rng.standard_normal((n_samples, support_size))
    eta = rng.standard_normal(n_samples)
    epsilon = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n_samples)
    return dict(
        x=x,
        eta=eta,
        epsilon=epsilon,
        treatment_effect=1.0,
        treatment_support=treatment_support,
        treatment_coef=treatment_coef,
        outcome_support=outcome_support,
        outcome_coef=outcome_coef,
        eta_second_moment=1.0,
        eta_third_moment=0.0,
        lambda_reg=0.05,
        check_convergence=False,
        verbose=False,
        oracle_support=True,
    )


def _run_n_reps(n_reps: int, *, disable_baselines: bool, n_samples: int = 300):
    """Run ``n_reps`` independent reps via the in-process single-experiment API."""
    return [
        run_single_experiment(
            **_make_synthetic_inputs(n_samples=n_samples, seed=i),
            disable_baselines=disable_baselines,
        )
        for i in range(n_reps)
    ]


class TestDispatcherIncludesBaselines:
    """``run_single_experiment`` must surface OLS / matching at indices 5/6."""

    def test_per_experiment_list_has_seven_entries(self):
        results = _run_n_reps(2, disable_baselines=False)
        ortho_rec_tau = extract_treatment_estimates(results)

        assert len(ortho_rec_tau) == 2
        for row in ortho_rec_tau:
            assert len(row) == 7, f"Expected 7 entries (5+1+1), got {len(row)}"

    def test_baselines_finite_when_enabled(self):
        results = _run_n_reps(2, disable_baselines=False)
        ortho_rec_tau = extract_treatment_estimates(results)

        for row in ortho_rec_tau:
            ols_value = row[OLS_IDX]
            matching_value = row[MATCHING_IDX]
            ols_arr = np.atleast_1d(np.asarray(ols_value, dtype=float))
            matching_arr = np.atleast_1d(np.asarray(matching_value, dtype=float))
            assert np.all(np.isfinite(ols_arr)), f"OLS at index {OLS_IDX} must be finite, got {ols_value}"
            assert np.all(
                np.isfinite(matching_arr)
            ), f"Matching at index {MATCHING_IDX} must be finite, got {matching_value}"

    def test_baselines_nan_when_disabled(self):
        results = _run_n_reps(2, disable_baselines=True)
        ortho_rec_tau = extract_treatment_estimates(results)

        for row in ortho_rec_tau:
            assert len(row) == 7
            ols_arr = np.atleast_1d(np.asarray(row[OLS_IDX], dtype=float))
            matching_arr = np.atleast_1d(np.asarray(row[MATCHING_IDX], dtype=float))
            assert np.all(np.isnan(ols_arr)), f"Disabled OLS must be NaN, got {row[OLS_IDX]}"
            assert np.all(np.isnan(matching_arr)), f"Disabled matching must be NaN, got {row[MATCHING_IDX]}"

    def test_ica_index_unchanged(self):
        """ICA still lives at index 4 — the append-only invariant."""
        results = _run_n_reps(2, disable_baselines=False)
        ortho_rec_tau = extract_treatment_estimates(results)

        for row in ortho_rec_tau:
            ica_value = row[ICA_IDX]
            assert ica_value is not None


class TestRunSingleExperimentReturnShape:
    """Smoke test for the raw 10-tuple returned by ``run_single_experiment``."""

    def test_returns_ten_element_tuple(self):
        result = run_single_experiment(
            **_make_synthetic_inputs(n_samples=200, seed=123),
            disable_baselines=False,
        )
        assert len(result) == 10, f"run_single_experiment must return 10-tuple, got {len(result)}"
        ols_estimate = result[-2]
        matching_estimate = result[-1]
        assert ols_estimate is not None
        assert matching_estimate is not None
        ols_arr = np.atleast_1d(np.asarray(ols_estimate, dtype=float))
        matching_arr = np.atleast_1d(np.asarray(matching_estimate, dtype=float))
        assert np.all(np.isfinite(ols_arr))
        assert np.all(np.isfinite(matching_arr))

    def test_disable_baselines_yields_nan(self):
        result = run_single_experiment(
            **_make_synthetic_inputs(n_samples=200, seed=456),
            disable_baselines=True,
        )
        assert len(result) == 10
        ols_arr = np.atleast_1d(np.asarray(result[-2], dtype=float))
        matching_arr = np.atleast_1d(np.asarray(result[-1], dtype=float))
        assert np.all(np.isnan(ols_arr)), "Disabled OLS must be NaN-filled"
        assert np.all(np.isnan(matching_arr)), "Disabled matching must be NaN-filled"
