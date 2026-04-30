"""Tests for asymmetric Bernoulli treatment-noise distribution.

Covers:
- Zero-mean sampling from setup_treatment_noise("bernoulli")
- Variance accuracy for p=0.3
- Excess kurtosis matches theory
- parse_distribution_spec parses "bernoulli:0.4" correctly
- Rademacher regression: p=0.5 symmetric case unchanged
"""

import numpy as np
import pytest
from scipy.stats import kurtosis as scipy_kurtosis

from eta_ablation_experiments import parse_distribution_spec
from oml_runner import setup_treatment_noise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_P = 0.3
N_SMALL = 100_000
N_LARGE = 200_000
RNG_SEED = 42


def _make_bernoulli_sampler(p: float = DEFAULT_P):
    """Return (support, eta_sample, mean, probs) for bernoulli(p)."""
    # gennorm_beta slot carries the Bernoulli p value
    return setup_treatment_noise(distribution="bernoulli", gennorm_beta=p)


# ---------------------------------------------------------------------------
# 1. Zero mean
# ---------------------------------------------------------------------------


def test_setup_treatment_noise_bernoulli_zero_mean():
    """100k samples from default Bernoulli(0.3) should have |mean| < 0.01."""
    np.random.seed(RNG_SEED)
    _, eta_sample, mean_discount, _ = _make_bernoulli_sampler()

    # Analytical mean must be exactly 0
    assert mean_discount == 0.0, f"Analytical mean should be 0.0, got {mean_discount}"

    samples = eta_sample(N_SMALL)
    assert abs(samples.mean()) < 0.01, f"|empirical mean| = {abs(samples.mean()):.5f} >= 0.01"


# ---------------------------------------------------------------------------
# 2. Variance
# ---------------------------------------------------------------------------


def test_setup_treatment_noise_bernoulli_variance():
    """For p=0.3, sample variance should be ≈ p*(1-p) = 0.21 ± 0.01."""
    np.random.seed(RNG_SEED)
    p = DEFAULT_P
    theoretical_var = p * (1.0 - p)  # 0.21

    _, eta_sample, _, _ = _make_bernoulli_sampler(p)
    samples = eta_sample(N_SMALL)
    empirical_var = samples.var(ddof=1)

    assert (
        abs(empirical_var - theoretical_var) < 0.01
    ), f"Empirical variance {empirical_var:.4f} not within 0.01 of theoretical {theoretical_var:.4f}"


# ---------------------------------------------------------------------------
# 3. Excess kurtosis
# ---------------------------------------------------------------------------


def test_setup_treatment_noise_bernoulli_kurtosis_matches_theory():
    """Excess kurtosis for Bernoulli(0.3) on centered support ≈ (1-6p(1-p))/(p(1-p)).

    For p=0.3: (1 - 6*0.21) / 0.21 = (1 - 1.26) / 0.21 ≈ -1.238095...
    Note: excess kurtosis of a Bernoulli(p) random variable on {0,1} is
        (1 - 6p(1-p)) / (p(1-p))
    Centering does not change the kurtosis.
    """
    np.random.seed(RNG_SEED)
    p = DEFAULT_P
    theoretical_excess_kurtosis = (1.0 - 6.0 * p * (1.0 - p)) / (p * (1.0 - p))

    _, eta_sample, _, _ = _make_bernoulli_sampler(p)
    samples = eta_sample(N_LARGE)
    empirical_excess_kurtosis = scipy_kurtosis(samples, fisher=True)

    # Allow 5% relative tolerance on a 200k sample
    rtol = 0.05
    assert abs(empirical_excess_kurtosis - theoretical_excess_kurtosis) <= rtol * abs(theoretical_excess_kurtosis), (
        f"Empirical excess kurtosis {empirical_excess_kurtosis:.4f} not within "
        f"{rtol*100:.0f}% of theoretical {theoretical_excess_kurtosis:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. parse_distribution_spec
# ---------------------------------------------------------------------------


def test_parse_distribution_spec_bernoulli():
    """'bernoulli:0.4' parses to name='bernoulli', p=0.4."""
    name, p = parse_distribution_spec("bernoulli:0.4")
    assert name == "bernoulli", f"Expected 'bernoulli', got '{name}'"
    assert p == pytest.approx(0.4), f"Expected p=0.4, got {p}"


def test_parse_distribution_spec_bernoulli_default():
    """'bernoulli' without param parses to name='bernoulli', p=0.3."""
    name, p = parse_distribution_spec("bernoulli")
    assert name == "bernoulli"
    assert p == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 5. Rademacher regression
# ---------------------------------------------------------------------------


def test_setup_treatment_noise_rademacher_unchanged():
    """Rademacher path still returns a ±1 sampler (regression guard)."""
    np.random.seed(RNG_SEED)
    support, eta_sample, mean_discount, probs = setup_treatment_noise(distribution="rademacher")

    assert mean_discount == 0.0
    assert set(support.tolist()) == {1.0, -1.0}, f"Rademacher support should be {{-1, 1}}, got {set(support.tolist())}"
    assert np.allclose(probs, [0.5, 0.5])

    samples = eta_sample(N_SMALL)
    unique_vals = set(np.unique(samples).tolist())
    assert unique_vals == {-1.0, 1.0}, f"Rademacher samples should only contain ±1, got {unique_vals}"
    assert abs(samples.mean()) < 0.01
