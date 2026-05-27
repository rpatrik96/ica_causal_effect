"""Binary-treatment data-generating process for the partially linear model.

This module provides a DGP where the treatment T is genuinely binary
(T in {0, 1}) — not merely a continuous T = m(X) + eta with binary eta.

The reviewer concern this addresses: the canonical setting in causal
inference is binary treatment (RCT, observational study with binary
intervention). The other DGPs in this repo all produce continuous T,
even when the eta noise is Bernoulli, because T = m(X) + eta with X
continuous. Here T is sampled directly from Bernoulli(p(X)).

Model
-----
::

    X        ~ N(0, I_d)                          # covariates
    p(X)     = sigmoid(propensity_strength * X @ alpha)  # propensity
    T | X    ~ Bernoulli(p(X))                    # binary treatment
    eta      = T - p(X)                           # centred propensity residual
    eps      ~ N(0, sigma_outcome^2)              # outcome noise
    Y        = theta * T + X @ beta + eps         # partially linear outcome

The propensity is bounded away from 0 and 1 by clipping the logit at a
configurable cap so propensity-score matching never sees extreme
weights.

The Bernoulli residual ``eta = T - p(X)`` is *heteroscedastic*
(Var[eta | X] = p(X) * (1 - p(X))) — this matches the canonical OML
treatment-residual formulation, so the cross-fitted Lasso nuisance
machinery in :mod:`main_estimation` applies without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class BinaryTreatmentDGPConfig:
    """Configuration for the binary-treatment DGP.

    Attributes
    ----------
    n_samples : int
        Number of i.i.d. samples to draw.
    n_covariates : int
        Covariate dimensionality d.
    support_size : int
        Number of non-zero entries in the propensity coefficient ``alpha``
        and the outcome coefficient ``beta``. The first ``support_size``
        coordinates of X drive both the treatment assignment and the
        outcome confounding.
    treatment_effect : float
        True ATE theta.
    propensity_strength : float
        Multiplier on the linear logit ``alpha @ X``. Larger values yield
        more deterministic treatment assignment (more confounding).
    outcome_coef_scale : float
        Scale of the outcome confounding coefficients ``beta``.
    sigma_outcome : float
        Standard deviation of the additive Gaussian outcome noise.
    logit_clip : float
        Clip ``alpha @ X`` to ``[-logit_clip, +logit_clip]`` so the
        propensity stays in a positivity-safe interval. Default 6 keeps
        ``p(X) in [~0.0025, ~0.9975]``.
    seed : int, optional
        Random seed for reproducibility. ``None`` uses fresh entropy.
    """

    n_samples: int = 2000
    n_covariates: int = 10
    support_size: int = 5
    treatment_effect: float = 1.5
    propensity_strength: float = 1.0
    outcome_coef_scale: float = 1.0
    sigma_outcome: float = 1.0
    logit_clip: float = 6.0
    seed: Optional[int] = None


def generate_binary_treatment_data(
    config: BinaryTreatmentDGPConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (X, T, Y) plus diagnostics from the binary-treatment DGP.

    Returns
    -------
    X : np.ndarray
        Covariate matrix of shape ``(n_samples, n_covariates)``.
    T : np.ndarray
        Binary treatment vector of shape ``(n_samples,)`` with values in
        ``{0, 1}``.
    Y : np.ndarray
        Outcome vector of shape ``(n_samples,)``.
    propensity : np.ndarray
        True propensity ``p(X)`` of shape ``(n_samples,)``. Useful for
        diagnostics (e.g. checking positivity, oracle propensity-score
        weighting).
    eta : np.ndarray
        Centred Bernoulli residual ``T - p(X)`` of shape ``(n_samples,)``.
        Mean zero by construction; Var[eta | X] = p(X) * (1 - p(X)).
    alpha : np.ndarray
        Propensity coefficient vector of shape ``(n_covariates,)``. The
        first ``support_size`` entries are non-zero.
    beta : np.ndarray
        Outcome confounding coefficient vector of shape
        ``(n_covariates,)``. Same support pattern as ``alpha``.
    """
    if config.support_size > config.n_covariates:
        raise ValueError(f"support_size ({config.support_size}) must be <= n_covariates ({config.n_covariates})")
    if config.logit_clip <= 0:
        raise ValueError(f"logit_clip must be positive, got {config.logit_clip}")

    rng = np.random.default_rng(config.seed)

    n, d, s = config.n_samples, config.n_covariates, config.support_size

    alpha = np.zeros(d)
    alpha[:s] = rng.standard_normal(s)
    beta = np.zeros(d)
    beta[:s] = rng.standard_normal(s) * config.outcome_coef_scale

    X = rng.standard_normal((n, d))

    logit = config.propensity_strength * (X @ alpha)
    logit = np.clip(logit, -config.logit_clip, config.logit_clip)
    propensity = _sigmoid(logit)

    T = (rng.uniform(size=n) < propensity).astype(float)
    eta = T - propensity

    eps = rng.standard_normal(n) * config.sigma_outcome
    Y = config.treatment_effect * T + X @ beta + eps

    return X, T, Y, propensity, eta, alpha, beta


def empirical_eta_moments(eta: np.ndarray) -> Tuple[float, float]:
    """Return ``(E[eta^2], E[eta^3] - 3 * E[eta] * E[eta^2])`` from samples.

    Used to feed the HOML estimator with the *empirical* second-moment
    and third-cumulant of the Bernoulli propensity residual when running
    the binary-treatment DGP. Even though E[eta | X] = 0, the marginal
    moments of eta are the relevant quantities for the cross-fitted
    HOML score (cf. ``main_estimation.all_together_cross_fitting``).
    """
    eta = np.asarray(eta, dtype=float).ravel()
    m1 = float(np.mean(eta))
    m2 = float(np.mean(eta**2))
    m3 = float(np.mean(eta**3))
    third_cumulant = m3 - 3.0 * m1 * m2
    return m2, third_cumulant
