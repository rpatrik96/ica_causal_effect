"""Regression tests for analytical treatment-noise moments."""

import numpy as np
import pytest

from ablation_utils import calculate_homl_moments
from oml_runner import DEFAULT_TREATMENT_DISCOUNTS, DEFAULT_TREATMENT_PROBS, setup_treatment_noise
from oml_utils import AsymptoticVarianceCalculator, compute_distribution_moments


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("variance", [0.25, 1.0, 4.0])
def test_gennorm_moments_match_monte_carlo(beta, variance):
    """The known moments use the same scale as the treatment-noise sampler."""
    scale = np.sqrt(variance)
    params, eta_sample, mean, probs = setup_treatment_noise("gennorm", scale=scale, gennorm_beta=beta)
    moments = calculate_homl_moments("gennorm", params, mean, probs, scale=scale)

    np.random.seed(2)
    samples = eta_sample(1_000_000)

    assert moments["eta_second_moment"] == pytest.approx(np.mean(samples**2), rel=0.02)
    assert moments["eta_fourth_moment"] == pytest.approx(np.mean(samples**4), rel=0.02)


def test_default_discrete_moments_and_homl_variance():
    moments = compute_distribution_moments("discrete", DEFAULT_TREATMENT_DISCOUNTS, DEFAULT_TREATMENT_PROBS)
    homl = AsymptoticVarianceCalculator.calc_homl_asymptotic_var(
        DEFAULT_TREATMENT_DISCOUNTS,
        np.dot(DEFAULT_TREATMENT_DISCOUNTS, DEFAULT_TREATMENT_PROBS),
        DEFAULT_TREATMENT_PROBS,
    )

    assert moments["excess_kurtosis"] == pytest.approx(5.050, abs=0.01)
    assert moments["cubed_variance"] == pytest.approx(87.30, abs=0.1)
    assert homl[5] == pytest.approx(1.882, abs=0.005)


@pytest.mark.parametrize(
    ("distribution", "expected_kappa", "expected_cubed_variance", "expected_homl"),
    [
        ("laplace", 3.0, 90.0, 7.0),
        ("uniform", -1.2, None, 1.4286),
        ("rademacher", -2.0, None, 1.0),
    ],
)
def test_symmetric_unit_variance_homl_formulas(distribution, expected_kappa, expected_cubed_variance, expected_homl):
    moments = compute_distribution_moments(distribution)
    homl = AsymptoticVarianceCalculator.calc_homl_asymptotic_var_from_distribution(distribution)

    assert moments["excess_kurtosis"] == pytest.approx(expected_kappa)
    if expected_cubed_variance is not None:
        assert moments["cubed_variance"] == pytest.approx(expected_cubed_variance)
    assert homl[5] == pytest.approx(expected_homl, abs=5e-5)
