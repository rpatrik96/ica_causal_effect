"""
Experiment execution framework for OML experiments.

This module provides a unified interface for running Monte Carlo experiments
across different parameter configurations for Orthogonal Machine Learning.
"""

from typing import Callable, Optional, Tuple

import numpy as np

from oml_utils import OMLExperimentConfig

# Default discrete treatment noise distribution parameters
DEFAULT_TREATMENT_DISCOUNTS = np.array([0, -0.5, -2.0, -4.0])
DEFAULT_TREATMENT_PROBS = np.array([0.65, 0.2, 0.1, 0.05])


def setup_treatment_noise(  # pylint: disable=no-else-return
    distribution: str = "discrete",
    rademacher: bool = False,
    scale: float = 1.0,
    gennorm_beta: float = None,
) -> Tuple[np.ndarray, Callable, float, np.ndarray]:
    """Setup treatment noise distribution and return sampling infrastructure.

    All continuous distributions are parameterized so that Var[eta] = scale².

    Parameters
    ----------
    distribution : str
        Type of noise distribution:
        - ``"discrete"``: default 4-point discrete distribution with fixed discounts
        - ``"laplace"``: Laplace(0, scale/sqrt(2)), variance = scale²
        - ``"uniform"``: Uniform(-sqrt(3)*scale, sqrt(3)*scale), variance = scale²
        - ``"rademacher"``: {-scale, +scale} with probability 0.5 each
        - ``"gennorm_heavy"``: generalized normal with beta=1 (Laplace-tailed)
        - ``"gennorm_light"``: generalized normal with beta=4 (lighter than Gaussian)
        - ``"gennorm"``: generalized normal with custom beta (requires ``gennorm_beta``)
    rademacher : bool
        Legacy flag — if True and ``distribution="discrete"``, uses Rademacher
        instead of the default discrete distribution.
    scale : float
        Scale parameter; default 1.0 gives unit variance for all distributions.
    gennorm_beta : float, optional
        Shape parameter beta for ``distribution="gennorm"``.

    Returns
    -------
    discounts_or_params : np.ndarray
        Discount values (discrete/rademacher) or distribution parameter array
        [loc, scale] / [beta, loc, scale] (continuous).
    eta_sample : Callable[[int], np.ndarray]
        Function that draws n treatment noise samples.
    mean_discount : float
        Mean of the distribution (0.0 for zero-mean continuous distributions).
    probs_or_None : np.ndarray or None
        Probability vector for discrete/rademacher; None for continuous.

    Raises
    ------
    ValueError
        If an unsupported distribution name is given, or ``gennorm_beta`` is
        missing when ``distribution="gennorm"``.
    """
    if distribution == "discrete":
        # Original discrete distribution
        if not rademacher:
            discounts = DEFAULT_TREATMENT_DISCOUNTS.copy()
            probs = DEFAULT_TREATMENT_PROBS.copy()
        else:
            discounts = np.array([1, -1])
            probs = np.array([0.5, 0.5])

        mean_discount = np.dot(discounts, probs)

        def eta_sample(x):
            return np.array(
                [discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)]
            )

        return discounts, eta_sample, mean_discount, probs

    elif distribution == "laplace":
        # Heavy-tailed Laplace distribution
        # Laplace(0, b) has variance 2*b^2, so b=1/sqrt(2) gives var=1
        laplace_scale = scale / np.sqrt(2)
        params = np.array([0.0, laplace_scale])  # [loc, scale]

        def eta_sample(x):
            return np.random.laplace(loc=0.0, scale=laplace_scale, size=x)

        return params, eta_sample, 0.0, None

    elif distribution == "uniform":
        # Bounded uniform distribution
        # Uniform(-a, a) has variance a^2/3, so a=sqrt(3)*scale gives var=scale^2
        half_width = np.sqrt(3) * scale
        params = np.array([-half_width, half_width])  # [low, high]

        def eta_sample(x):
            return np.random.uniform(low=-half_width, high=half_width, size=x)

        return params, eta_sample, 0.0, None

    elif distribution == "rademacher":
        # Bounded Rademacher distribution: {-scale, +scale} with equal probability
        # Variance = scale^2
        discounts = np.array([scale, -scale])
        probs = np.array([0.5, 0.5])

        def eta_sample(x):
            return np.random.choice(discounts, size=x, p=probs)

        return discounts, eta_sample, 0.0, probs

    elif distribution == "gennorm_heavy":
        # Generalized normal with beta=1 (equivalent to Laplace, heavy tails)
        from scipy.stats import gennorm

        beta = 1.0
        # gennorm variance = Gamma(3/beta) / Gamma(1/beta), for beta=1 this is 2
        # To get unit variance, scale by sqrt(variance)
        gn_scale = scale / np.sqrt(gennorm.var(beta))
        params = np.array([beta, 0.0, gn_scale])  # [beta, loc, scale]

        def eta_sample(x):
            return gennorm.rvs(beta, loc=0.0, scale=gn_scale, size=x)

        return params, eta_sample, 0.0, None

    elif distribution == "gennorm_light":
        # Generalized normal with beta=4 (lighter tails than Gaussian)
        from scipy.stats import gennorm

        beta = 4.0
        gn_scale = scale / np.sqrt(gennorm.var(beta))
        params = np.array([beta, 0.0, gn_scale])  # [beta, loc, scale]

        def eta_sample(x):
            return gennorm.rvs(beta, loc=0.0, scale=gn_scale, size=x)

        return params, eta_sample, 0.0, None

    elif distribution == "gennorm":
        # Generalized normal with custom beta
        from scipy.stats import gennorm

        if gennorm_beta is None:
            raise ValueError("gennorm_beta parameter required for distribution='gennorm'")
        beta = gennorm_beta
        gn_scale = scale / np.sqrt(gennorm.var(beta))
        params = np.array([beta, 0.0, gn_scale])  # [beta, loc, scale]

        def eta_sample(x):
            return gennorm.rvs(beta, loc=0.0, scale=gn_scale, size=x)

        return params, eta_sample, 0.0, None

    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            "Valid options: discrete, laplace, uniform, rademacher, gennorm_heavy, gennorm_light, gennorm"
        )


def setup_covariate_pdf(config: OMLExperimentConfig, beta: float) -> Callable[[int, int], np.ndarray]:
    """Setup covariate sampling function based on PDF choice.

    Parameters
    ----------
    config : OMLExperimentConfig
        Experiment configuration; uses ``config.covariate_pdf`` to select the
        distribution ("gauss", "uniform", or "gennorm").
    beta : float
        Shape parameter for the generalized normal distribution (ignored for
        "gauss" and "uniform").

    Returns
    -------
    Callable[[int, int], np.ndarray]
        Function ``f(n_samples, n_dimensions) -> np.ndarray`` that draws
        covariates from the chosen distribution.

    Raises
    ------
    ValueError
        If ``config.covariate_pdf`` is not one of the supported options.
    """
    if config.covariate_pdf == "gauss":
        return lambda n, d: np.random.normal(size=(n, d))
    if config.covariate_pdf == "uniform":
        return lambda n, d: np.random.uniform(-1, 1, size=(n, d))
    if config.covariate_pdf == "gennorm":
        from scipy.stats import gennorm

        return lambda n, d: gennorm.rvs(beta, size=(n, d))

    raise ValueError(f"Unknown covariate PDF: {config.covariate_pdf}")


def setup_treatment_outcome_coefs(
    cov_dim_max: int,
    config: OMLExperimentConfig,
    outcome_coef_array: Optional[np.ndarray],
    outcome_coef_list: np.ndarray,
    outcome_coefficient: Optional[float],
    support_size: int,
    treatment_coef_array: Optional[np.ndarray],
    treatment_coef_list: np.ndarray,
    treatment_coefficient: Optional[float],
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Setup treatment and outcome coefficients and their active supports.

    In scalar mode (``config.scalar_coeffs=True`` or ``config.asymptotic_var=True``),
    the first ``support_size`` covariates form the support and a single scalar
    coefficient is broadcast across them. In full mode, support indices are drawn
    randomly from [0, cov_dim_max).

    Parameters
    ----------
    cov_dim_max : int
        Total covariate dimension (upper bound for random support sampling).
    config : OMLExperimentConfig
        Experiment configuration controlling scalar/full mode and coefficient
        matching.
    outcome_coef_array : np.ndarray or None
        Pre-generated outcome coefficient array used in full (non-scalar) mode.
    outcome_coef_list : np.ndarray
        Mutable array updated in-place with scalar outcome coefficients.
    outcome_coefficient : float or None
        Scalar outcome coefficient value (used when ``config.scalar_coeffs=True``).
    support_size : int
        Number of active covariates.
    treatment_coef_array : np.ndarray or None
        Pre-generated treatment coefficient array used in full (non-scalar) mode.
    treatment_coef_list : np.ndarray
        Mutable array updated in-place with scalar treatment coefficients.
    treatment_coefficient : float or None
        Scalar treatment coefficient value (used when ``config.scalar_coeffs=True``).

    Returns
    -------
    outcome_coef : np.ndarray
        Outcome coefficients of shape (support_size,).
    outcome_coefficient : float or None
        (Possibly overridden) scalar outcome coefficient.
    outcome_support : np.ndarray
        Indices of active covariates for the outcome, shape (support_size,).
    treatment_coef : np.ndarray
        Treatment coefficients of shape (support_size,).
    treatment_support : np.ndarray
        Indices of active covariates for treatment, shape (support_size,).
    """
    # Adjust coefficient matching based on configuration
    if config.asymptotic_var:
        outcome_coefficient = treatment_coefficient

    if config.matched_coefficients:
        outcome_coefficient = -treatment_coefficient

    if config.asymptotic_var or config.scalar_coeffs:
        # Scalar coefficient mode
        treatment_coef_list[0] = treatment_coefficient
        outcome_coef_list[0] = outcome_coefficient

        outcome_support = treatment_support = np.array(range(support_size))

        treatment_coef = treatment_coef_list[:support_size]
        outcome_coef = outcome_coef_list[:support_size]
    else:
        # Full coefficient array mode
        outcome_support = treatment_support = np.random.choice(range(cov_dim_max), size=support_size, replace=False)
        treatment_coef = treatment_coef_array[treatment_support]
        outcome_coef = outcome_coef_array[outcome_support]

    return outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support
