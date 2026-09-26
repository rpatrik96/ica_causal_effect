"""Smoke tests for the E2 theory-validation experiment."""

import numpy as np

from theory_validation import ESTIMATORS, build_cells, get_law, predicted_variances, run_experiment


def test_e2_laws_are_standardized_and_paper_epsilon_is_uniform():
    discrete = get_law("discrete")
    paper_eps = get_law("paper_default")
    requested_uniform = get_law("uniform")

    np.testing.assert_allclose(np.dot(discrete.support, discrete.probabilities), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.dot(discrete.support**2, discrete.probabilities), 1.0)
    np.testing.assert_allclose(paper_eps.m4, requested_uniform.m4)
    np.testing.assert_allclose(paper_eps.m6, requested_uniform.m6)


def test_e2_quick_grid_smoke():
    cells = build_cells(quick=True)
    predictions, comparison = predicted_variances(cells[0])
    rows = run_experiment(quick=True, n_jobs=1, bootstrap_draws=40)

    assert len(cells) == 1
    assert set(predictions) == set(ESTIMATORS)
    assert np.isfinite(comparison)
    assert len(rows) == len(ESTIMATORS)
    assert {row["estimator"] for row in rows} == set(ESTIMATORS)
    assert all(row["n_success"] > 0 for row in rows)
    assert all(np.isfinite(row["observed_n_mse"]) for row in rows)
