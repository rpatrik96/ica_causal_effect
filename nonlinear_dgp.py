"""Nonlinear confounding DGP for the partially linear model.

This module provides a DGP where the nuisance functions g(X) and m(X) can be
nonlinear, exposing the bias of OLS and the advantage of flexible first-stage
estimation in OML/HOML/ICA.

The reviewer concern this addresses: prior experiments used *linear* confounding
(g(X) = X @ beta, m(X) = X @ alpha), so OLS was consistent and showed no
disadvantage relative to the orthogonal methods. Here we support four
independently-toggleable difficulty axes:

1. **Nonlinear confounding** (``nonlinear_confounding=True``): g and m are
   nonlinear functions of X (sum of sin terms + interactions/quadratic). OLS
   regresses Y on [T, X] linearly and is biased; OML/HOML/ICA with a flexible
   first-stage nuisance survive.

2. **High-dimensional covariates** (``high_dim=True``): d is set to
   ``high_dim_d`` (default 50) and the signal is spread over all d covariates
   with decaying weights, stressing Lasso-based nuisance estimation.

3. **Heavy-tailed / non-Gaussian eta** (``heavy_tail_eta=True``): the treatment
   noise eta is drawn from a generalised-normal distribution with shape parameter
   ``eta_beta`` < 2 (beta=1 is Laplace). This is the regime where ICA's
   non-Gaussianity assumption is most useful; the HOML variance coefficient
   depends on eta's excess kurtosis.

4. **Heteroscedastic outcome noise** (``heteroscedastic_eps=True``): the outcome
   noise variance depends on X (Var[eps|X] = sigma_outcome^2 * exp(X @ gamma_eps)),
   violating the homoscedastic PLR assumption used to derive the HOML asymptotic
   variance.

Model
-----
::

    X        ~ N(0, I_d)                                  # covariates
    eta      ~ gennorm(beta=eta_beta)  (or N(0, sigma_eta^2) if not heavy_tail)
    T        = m(X) + eta                                  # treatment
    eps      ~ N(0, sigma_eps(X)^2)                        # outcome noise
    Y        = theta * T + g(X) + eps                      # partially linear

where::

    m(X) = sum_j alpha_j * sin(pi * X_j)                  [if nonlinear_confounding]
         + sum_{j<k, j,k in supp} gamma_{jk} * X_j * X_k  [interaction terms]
    or
    m(X) = X @ alpha                                       [if linear]

    g(X) = sum_j beta_j * sin(pi * X_j)                   [if nonlinear_confounding]
         + 0.5 * sum_j beta2_j * X_j^2                    [quadratic terms]
    or
    g(X) = X @ beta                                        [if linear]

    sigma_eps(X) = sigma_outcome * exp(0.5 * X @ gamma_eps)  [if heteroscedastic]
    or
    sigma_eps(X) = sigma_outcome                              [if homoscedastic]

Notes on OLS bias under nonlinear confounding
---------------------------------------------
OLS regresses Y = theta*T + g(X) + eps on [T, X] with a *linear* model for
g(X). When g is nonlinear, the residual [g(X) - X @ beta_ols] is not zero and
is correlated with T (because T = m(X) + eta and m is also nonlinear in X).
Hence the OLS coefficient on T absorbs part of the confounding and is biased.

OML/HOML use a first-stage fit for m and g. With a *linear* nuisance (LassoCV)
they suffer the same bias. With a *flexible* nuisance (GradientBoosting,
polynomial pipeline) they can approximate m and g well enough to eliminate
the bias — this is the central story we want to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import gennorm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NonlinearDGPConfig:
    """Configuration for the nonlinear-confounding DGP.

    Attributes
    ----------
    n_samples : int
        Number of i.i.d. samples to draw.
    n_covariates : int
        Covariate dimensionality d (before any high-dim override).
    support_size : int
        Number of active covariates (non-zero entries in alpha/beta).
        Must be <= n_covariates.
    treatment_effect : float
        True treatment effect theta.
    sigma_eta : float
        Standard deviation of the treatment noise eta when using Gaussian
        noise (``heavy_tail_eta=False``). When ``heavy_tail_eta=True`` this
        is used as the gennorm scale parameter so that Var[eta] ≈ sigma_eta^2.
    sigma_outcome : float
        Baseline standard deviation of the outcome noise eps. Under
        heteroscedastic noise this is a scale multiplier.
    nonlinear_confounding : bool
        If True, use sin + interaction terms in m and g. OLS will be biased.
        If False, m and g are linear — OLS is consistent (sanity-check mode).
    heavy_tail_eta : bool
        If True, draw eta from gennorm(beta=eta_beta) rather than N(0, sigma_eta^2).
        This is the regime where ICA has a theoretical advantage.
    eta_beta : float
        Shape parameter for the gennorm treatment-noise distribution when
        ``heavy_tail_eta=True``. beta=1 is Laplace, beta=2 is Gaussian,
        beta<2 is heavy-tailed. Default 1.0 (Laplace).
    high_dim : bool
        If True, overrides n_covariates with ``high_dim_d`` and spreads
        signal over all d covariates with harmonically decaying weights.
    high_dim_d : int
        Covariate dimension to use when ``high_dim=True``. Default 50.
    heteroscedastic_eps : bool
        If True, the outcome noise variance depends on X via
        ``Var[eps|X] = sigma_outcome^2 * exp(X @ gamma_eps)``.
    alpha_scale : float
        Scale of the propensity coefficient vector alpha (treatment nuisance).
    beta_scale : float
        Scale of the outcome coefficient vector beta (outcome nuisance).
    interaction_scale : float
        Scale of the pairwise interaction terms added to m and g when
        ``nonlinear_confounding=True``.
    heteroscedastic_scale : float
        Scale of the heteroscedastic variance exponent gamma_eps when
        ``heteroscedastic_eps=True``. Larger values create stronger
        variance heterogeneity.
    nonlinearity_strength : float
        Scalar multiplier applied to the nonlinear terms in m(X) and g(X)
        when ``nonlinear_confounding=True``. At strength=0 the nonlinear
        terms vanish and the DGP reduces to linear (OLS is unbiased); at
        strength=1 the standard design from the paper is recovered; larger
        values amplify the confounding. Default 1.0.
    seed : int, optional
        Random seed for reproducibility. None uses fresh entropy.
    """

    n_samples: int = 2000
    n_covariates: int = 10
    support_size: int = 5
    treatment_effect: float = 1.5
    sigma_eta: float = 1.0
    sigma_outcome: float = 1.0
    # Difficulty toggles
    nonlinear_confounding: bool = False
    heavy_tail_eta: bool = False
    eta_beta: float = 1.0  # Laplace when heavy_tail_eta=True
    high_dim: bool = False
    high_dim_d: int = 50
    heteroscedastic_eps: bool = False
    # Coefficient scales
    alpha_scale: float = 1.0
    beta_scale: float = 1.0
    interaction_scale: float = 0.3
    heteroscedastic_scale: float = 0.5
    nonlinearity_strength: float = 1.0
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Hard preset (all four axes ON)
# ---------------------------------------------------------------------------


def hard_preset(n_samples: int = 2000, seed: Optional[int] = None) -> NonlinearDGPConfig:
    """Return a config with all four difficulty axes active.

    This is the setting where OLS should show clearly elevated RMSE relative
    to OML/HOML with a flexible first stage, and where ICA benefits from
    non-Gaussian eta.

    Parameters
    ----------
    n_samples : int
        Sample size. Default 2000.
    seed : int, optional
        Random seed.
    """
    return NonlinearDGPConfig(
        n_samples=n_samples,
        n_covariates=10,
        support_size=5,
        treatment_effect=1.5,
        sigma_eta=1.0,
        sigma_outcome=1.0,
        nonlinear_confounding=True,
        heavy_tail_eta=True,
        eta_beta=1.0,
        high_dim=False,
        heteroscedastic_eps=True,
        alpha_scale=1.0,
        beta_scale=1.0,
        interaction_scale=0.3,
        heteroscedastic_scale=0.5,
        seed=seed,
    )


def linear_preset(n_samples: int = 2000, seed: Optional[int] = None) -> NonlinearDGPConfig:
    """Return a fully linear config (OLS sanity-check baseline).

    All four difficulty toggles are off. OLS should be consistent here, so
    a test that checks OLS is approximately unbiased on this preset encodes
    the 'linear DGP => OLS is fine' invariant.
    """
    return NonlinearDGPConfig(
        n_samples=n_samples,
        n_covariates=10,
        support_size=5,
        treatment_effect=1.5,
        sigma_eta=1.0,
        sigma_outcome=1.0,
        nonlinear_confounding=False,
        heavy_tail_eta=False,
        high_dim=False,
        heteroscedastic_eps=False,
        alpha_scale=0.8,
        beta_scale=0.8,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# DGP helper functions
# ---------------------------------------------------------------------------


def _nonlinear_m(X: np.ndarray, alpha: np.ndarray, gamma: np.ndarray, support_size: int) -> np.ndarray:
    """Nonlinear treatment nuisance with zero linear projection under N(0, I).

    m(X) = sum_{j<k, j,k in supp} gamma_{jk} * X_j * X_k
           + sum_j alpha_j * (X_j^2 - 1)

    Both terms have E[m(X)] = 0 and zero population OLS coefficient on X
    (cross-products and centred quadratics are orthogonal to X under N(0,I)).
    This guarantees that OLS cannot remove the confounding by including X
    linearly — the bias is structural, not a finite-sample fluke.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    alpha : np.ndarray, shape (d,) — centred-quadratic coefficients
    gamma : np.ndarray, shape (support_size, support_size) — cross-product coefficients
    support_size : int — number of active covariates
    """
    s = support_size
    Xs = X[:, :s]  # (n, s)
    # Centred quadratic terms: E[X_j^2 - 1] = 0, uncorrelated with X
    m = (Xs**2 - 1.0) @ alpha[:s]
    # Pairwise cross-products: E[X_j * X_k] = 0 for j != k, uncorrelated with X
    for j in range(s):
        for k in range(j + 1, s):
            m = m + gamma[j, k] * Xs[:, j] * Xs[:, k]
    return m


def _linear_m(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Linear treatment nuisance m(X) = X @ alpha."""
    return X @ alpha


def _nonlinear_g(X: np.ndarray, beta: np.ndarray, beta2: np.ndarray, support_size: int) -> np.ndarray:
    """Nonlinear outcome nuisance with zero linear projection under N(0, I).

    g(X) = sum_j beta_j * (X_j^2 - 1)
           + sum_j beta2_j * tanh(X_j)

    The centred-quadratic term is orthogonal to X. The tanh term has
    small but nonzero linear projection (tanh is odd), but the dominant
    nonlinearity is the quadratic, ensuring a clear OLS bias.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    beta : np.ndarray, shape (d,) — centred-quadratic coefficients
    beta2 : np.ndarray, shape (d,) — tanh coefficients
    support_size : int
    """
    Xs = X[:, :support_size]
    # Centred quadratic terms
    g = (Xs**2 - 1.0) @ beta[:support_size]
    # Tanh nonlinearity — mildly nonlinear, adds curvature without linear cancellation
    g = g + np.tanh(Xs) @ beta2[:support_size]
    return g


def _linear_g(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Linear outcome nuisance g(X) = X @ beta."""
    return X @ beta


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_nonlinear_data(
    config: NonlinearDGPConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (X, T, Y) plus diagnostics from the nonlinear-confounding DGP.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_covariates)
        Covariate matrix. Dimensionality is ``config.high_dim_d`` when
        ``config.high_dim=True``, otherwise ``config.n_covariates``.
    T : np.ndarray, shape (n_samples,)
        Continuous treatment T = m(X) + eta.
    Y : np.ndarray, shape (n_samples,)
        Outcome Y = theta * T + g(X) + eps.
    eta : np.ndarray, shape (n_samples,)
        Treatment noise drawn from N(0, sigma_eta^2) or gennorm(eta_beta).
    m_X : np.ndarray, shape (n_samples,)
        True treatment nuisance m(X). Useful for diagnostics / first-stage MSE.
    g_X : np.ndarray, shape (n_samples,)
        True outcome nuisance g(X). Useful for diagnostics.
    alpha : np.ndarray, shape (n_covariates,)
        Coefficient vector used in m (linear case) or sin argument (nonlinear).
    """
    # Resolve effective dimension
    d = config.high_dim_d if config.high_dim else config.n_covariates
    s = config.support_size
    if s > d:
        raise ValueError(f"support_size ({s}) must be <= effective n_covariates ({d})")
    if config.eta_beta <= 0:
        raise ValueError(f"eta_beta must be positive, got {config.eta_beta}")

    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    # ------------------------------------------------------------------ #
    # Coefficients
    # ------------------------------------------------------------------ #
    if config.high_dim:
        # Spread signal over all d covariates with harmonically decaying weights.
        # This makes Lasso-with-dense-support hard.
        alpha = rng.standard_normal(d) * config.alpha_scale / np.arange(1, d + 1)
        beta = rng.standard_normal(d) * config.beta_scale / np.arange(1, d + 1)
        beta2 = rng.standard_normal(d) * config.beta_scale * 0.5 / np.arange(1, d + 1)
    else:
        alpha = np.zeros(d)
        alpha[:s] = rng.standard_normal(s) * config.alpha_scale
        beta = np.zeros(d)
        beta[:s] = rng.standard_normal(s) * config.beta_scale
        beta2 = np.zeros(d)
        beta2[:s] = rng.standard_normal(s) * config.beta_scale * 0.5

    # Pairwise interaction coefficients (used only when nonlinear_confounding=True)
    gamma = rng.standard_normal((s, s)) * config.interaction_scale

    # Heteroscedastic noise direction (used only when heteroscedastic_eps=True)
    gamma_eps = np.zeros(d)
    gamma_eps[:s] = rng.standard_normal(s) * config.heteroscedastic_scale

    # ------------------------------------------------------------------ #
    # Covariates
    # ------------------------------------------------------------------ #
    X = rng.standard_normal((n, d))

    # ------------------------------------------------------------------ #
    # Treatment nuisance m(X)
    # ------------------------------------------------------------------ #
    if config.nonlinear_confounding:
        m_X = _nonlinear_m(X, alpha * config.nonlinearity_strength, gamma * config.nonlinearity_strength, s)
    else:
        m_X = _linear_m(X, alpha)

    # ------------------------------------------------------------------ #
    # Treatment noise eta
    # ------------------------------------------------------------------ #
    if config.heavy_tail_eta:
        # gennorm(beta) has variance = Gamma(3/beta) / Gamma(1/beta)
        # We normalise to match sigma_eta^2 so that the moments are comparable
        # across the heavy-tail vs Gaussian settings.
        raw = gennorm.rvs(beta=config.eta_beta, size=n, random_state=rng)
        # Empirical std of gennorm(beta) with loc=0, scale=1
        _std = float(gennorm.std(config.eta_beta))
        eta = (raw / _std) * config.sigma_eta
    else:
        eta = rng.standard_normal(n) * config.sigma_eta

    T = m_X + eta

    # ------------------------------------------------------------------ #
    # Outcome nuisance g(X) and noise eps
    # ------------------------------------------------------------------ #
    if config.nonlinear_confounding:
        g_X = _nonlinear_g(X, beta * config.nonlinearity_strength, beta2 * config.nonlinearity_strength, s)
    else:
        g_X = _linear_g(X, beta)

    if config.heteroscedastic_eps:
        log_std = np.clip(X @ gamma_eps, -3.0, 3.0)  # Clip to avoid extreme variances
        eps_std = config.sigma_outcome * np.exp(0.5 * log_std)
        eps = rng.standard_normal(n) * eps_std
    else:
        eps = rng.standard_normal(n) * config.sigma_outcome

    Y = config.treatment_effect * T + g_X + eps

    return X, T, Y, eta, m_X, g_X, alpha


# ---------------------------------------------------------------------------
# Moment helpers (for HOML)
# ---------------------------------------------------------------------------


def eta_moments_from_config(config: NonlinearDGPConfig) -> Tuple[float, float]:
    """Return the theoretical (second moment, third cumulant) of the eta distribution.

    These are the *oracle* population moments, derived analytically from the
    chosen distribution. They are fed to the HOML 'known moments' estimator.

    For Gaussian eta:
        E[eta^2] = sigma_eta^2,  kappa_3 = 0.

    For gennorm(beta) (normalised to variance sigma_eta^2):
        E[eta^2] = sigma_eta^2 (by construction of the normalisation).
        kappa_3 = 0  (gennorm is symmetric => odd cumulants vanish).

    Returns
    -------
    second_moment : float
        E[eta^2].
    third_cumulant : float
        kappa_3(eta) = E[eta^3] - 3 * E[eta] * E[eta^2].  Zero for any
        zero-mean symmetric distribution.
    """
    second_moment = config.sigma_eta**2
    third_cumulant = 0.0  # gennorm is symmetric; all odd cumulants are zero
    return second_moment, third_cumulant


def empirical_eta_moments(eta: np.ndarray) -> Tuple[float, float]:
    """Compute empirical (second moment, third cumulant) from eta samples.

    Mirrors the analogous function in ``binary_treatment_dgp`` for API
    consistency. Use the oracle version ``eta_moments_from_config`` when
    running the 'known moments' HOML path.

    Parameters
    ----------
    eta : np.ndarray, shape (n,)
        Observed treatment residuals eta = T - m(X).

    Returns
    -------
    second_moment : float
    third_cumulant : float
    """
    eta = np.asarray(eta, dtype=float).ravel()
    m1 = float(np.mean(eta))
    m2 = float(np.mean(eta**2))
    m3 = float(np.mean(eta**3))
    third_cumulant = m3 - 3.0 * m1 * m2
    return m2, third_cumulant
