"""Regression tests for common-finite Monte Carlo summaries."""

import numpy as np
import pytest

from nonlinear_runner import _summarize_estimates
from selector_runner import _rmse


def test_statistics_use_common_finite_rows():
    estimates = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, np.nan, 10.0],
            [2.0, 2.0, 2.0],
        ]
    )

    finite_mask, finite, finite_per_method, biases, sigmas, rmse = _summarize_estimates(estimates, 0.0)

    np.testing.assert_array_equal(finite_mask, [True, False, True])
    np.testing.assert_array_equal(finite_per_method, [3, 2, 3])
    np.testing.assert_array_equal(finite, estimates[[0, 2]])
    np.testing.assert_allclose(biases, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(sigmas, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(rmse, np.sqrt(2.0))
    assert _rmse(estimates[:, 0], 0.0, finite_mask) == pytest.approx(np.sqrt(2.0))
