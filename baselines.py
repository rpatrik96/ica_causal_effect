"""Baseline estimators for the partially linear model.

Implements two baseline estimators for comparison in the paper:

- **OLS baseline**: Ordinary least squares of outcome on [treatment, covariates].
  Consistent under the PLR only when confounding is linear; serves as the
  naive benchmark.
- **Matching baseline**: k-nearest-neighbour matching estimator.  For binary
  treatment uses propensity-score matching via logistic regression; for
  continuous treatment uses GPS-residual matching (Imbens & Rubin 2015,
  Ch. 18) with a LassoCV nuisance model matching ``main_estimation.py``.

Dependencies are restricted to ``numpy`` and ``scikit-learn`` so the shared
cluster virtual environment at
``/is/cluster/fast/preizinger/nl-causal-representations/care`` is not
invalidated by heavy optional packages such as ``causalml`` (numba) or
``econml`` (tensorflow).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Regularisation grid reused from main_estimation.py for apples-to-apples
# nuisance estimation in the continuous-treatment matching path.
DEFAULT_LASSO_ALPHAS = [0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]


def ols_baseline(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Estimate the treatment effect via ordinary least squares.

    Fits ``sklearn.linear_model.LinearRegression`` on the design matrix
    ``[treatment, covariates]`` (treatment columns placed first) and returns
    the first ``m`` coefficients as the treatment-effect estimate.

    Handles both univariate (``m=1``) and multivariate (``m>=2``) treatment
    uniformly — treatment is reshaped to ``(n, m)`` on entry so the caller
    never needs to branch.

    Implements the partially linear model::

        treatment = g(covariates) + eta
        outcome   = theta * treatment + f(covariates) + epsilon

    Under a linear DGP with no unmeasured confounding this is the BLUE
    (Gauss–Markov).  In the general PLR it is biased whenever ``f`` or ``g``
    are nonlinear.

    Args:
        covariates: Covariate matrix of shape ``(n_samples, n_features)``.
        treatment: Treatment array of shape ``(n_samples,)`` or
            ``(n_samples, m_treatments)``.
        outcome: Outcome vector of shape ``(n_samples,)``.
        fit_intercept: Whether to fit an intercept term.  Defaults to
            ``True``.

    Returns:
        Treatment-effect estimate of shape ``(m_treatments,)``.  Even for a
        univariate treatment (``m=1``) this is a one-element array so that
        downstream callers can index it uniformly.
    """
    treatment = np.atleast_2d(treatment.T).T  # (n, 1) if 1-D, else (n, m)
    _, m = treatment.shape

    X_design = np.concatenate([treatment, covariates], axis=1)  # (n, m + d)

    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X_design, outcome)

    return reg.coef_[:m].copy()  # shape (m,)


def matching_baseline(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_neighbors: int = 5,
    treatment_kind: str = "auto",
) -> float:
    """Estimate the treatment effect via k-nearest-neighbour matching.

    Two estimation paths are available, selected by ``treatment_kind``:

    **Binary path** (``treatment_kind="binary"``):
        Fits a logistic regression of treatment on covariates to obtain
        propensity scores ``p̂(X)``.  For each treated unit the ``n_neighbors``
        nearest untreated units (by propensity distance) are found via
        ``sklearn.neighbors.NearestNeighbors``.  The ATE contribution from the
        treated side is ``mean(Y_treated - mean(Y_matched_untreated))``.  The
        procedure is repeated symmetrically for untreated → treated and the
        two halves are averaged.

    **Continuous path** (``treatment_kind="continuous"``):
        Residualises ``T`` on ``X`` via ``LassoCV`` (same regularisation grid
        as ``main_estimation.py``) to obtain ``T_resid``.  For each unit the
        ``n_neighbors`` nearest units in covariate space are found.  The
        treatment effect is estimated as the OLS slope of
        ``(Y_i - Ȳ_match_i)`` on ``(T_i - T̄_match_i)`` (Imbens & Rubin 2015,
        Ch. 18 GPS-residual estimator).

    **Auto detection** (``treatment_kind="auto"``):
        Uses the binary path when ``np.unique(treatment).size <= 2`` and all
        unique values lie in ``{0, 1}`` or ``{-1, 1}``; otherwise uses the
        continuous path.

    Args:
        covariates: Covariate matrix of shape ``(n_samples, n_features)``.
        treatment: Treatment vector of shape ``(n_samples,)``.  Must be 1-D.
        outcome: Outcome vector of shape ``(n_samples,)``.
        n_neighbors: Number of nearest neighbours used for matching.
            Defaults to ``5``.
        treatment_kind: One of ``"binary"``, ``"continuous"``, or ``"auto"``.
            Defaults to ``"auto"``.

    Returns:
        Scalar treatment-effect estimate.  Returns ``np.nan`` if the
        estimation denominator is numerically zero (degenerate data).

    Raises:
        ValueError: If ``treatment_kind`` is not one of the three allowed
            values.
    """
    treatment = np.asarray(treatment, dtype=float).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    covariates = np.asarray(covariates, dtype=float)

    if treatment_kind not in ("binary", "continuous", "auto"):
        raise ValueError(f"treatment_kind must be 'binary', 'continuous', or 'auto'; got '{treatment_kind}'")

    if treatment_kind == "auto":
        unique_vals = np.unique(treatment)
        is_binary = (unique_vals.size <= 2) and np.all(np.isin(unique_vals, [0.0, 1.0, -1.0]))
        treatment_kind = "binary" if is_binary else "continuous"

    if treatment_kind == "binary":
        return _matching_binary(covariates, treatment, outcome, n_neighbors)
    return _matching_continuous(covariates, treatment, outcome, n_neighbors)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _matching_binary(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_neighbors: int,
) -> float:
    """Propensity-score matching for binary treatment.

    Estimates the ATE by averaging over both the treated→control and
    control→treated matching directions.

    Args:
        covariates: ``(n, d)`` covariate matrix.
        treatment: ``(n,)`` binary treatment vector (values in {0,1} or
            {-1,1}).
        outcome: ``(n,)`` outcome vector.
        n_neighbors: Number of nearest neighbours per unit.

    Returns:
        Scalar ATE estimate, or ``np.nan`` if either treatment arm is empty.
    """
    # Normalise to {0, 1} so indexing is unambiguous
    t_min = treatment.min()
    t01 = (treatment - t_min).astype(float)
    t01 = (t01 / t01.max()).astype(float)  # maps {0,1} or {-1,1} → {0,1}

    treated_mask = t01 == 1.0
    control_mask = t01 == 0.0

    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        return np.nan

    # Fit propensity model
    prop_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    prop_model.fit(covariates, t01)
    prop_scores = prop_model.predict_proba(covariates)[:, 1].reshape(-1, 1)  # (n, 1)

    # --- Treated → Control direction ---
    k_ctrl = min(n_neighbors, control_mask.sum())
    nn_ctrl = NearestNeighbors(n_neighbors=k_ctrl, algorithm="auto", metric="euclidean")
    nn_ctrl.fit(prop_scores[control_mask])
    _, idx_ctrl = nn_ctrl.kneighbors(prop_scores[treated_mask])
    # idx_ctrl references rows within the control subset
    control_outcomes = outcome[control_mask]
    matched_ctrl_mean = control_outcomes[idx_ctrl].mean(axis=1)  # (n_treated,)
    ate_treated = float(np.mean(outcome[treated_mask] - matched_ctrl_mean))

    # --- Control → Treated direction ---
    k_trt = min(n_neighbors, treated_mask.sum())
    nn_trt = NearestNeighbors(n_neighbors=k_trt, algorithm="auto", metric="euclidean")
    nn_trt.fit(prop_scores[treated_mask])
    _, idx_trt = nn_trt.kneighbors(prop_scores[control_mask])
    treated_outcomes = outcome[treated_mask]
    matched_trt_mean = treated_outcomes[idx_trt].mean(axis=1)  # (n_control,)
    ate_control = float(np.mean(matched_trt_mean - outcome[control_mask]))

    return (ate_treated + ate_control) / 2.0


def _matching_continuous(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_neighbors: int,
) -> float:
    """GPS-residual k-NN matching for continuous treatment.

    Residualises T on X via LassoCV, then for each unit finds neighbours in
    covariate space and estimates theta as the OLS slope of
    ``(Y_i - Ȳ_match)`` on ``(T_i - T̄_match)``.

    Args:
        covariates: ``(n, d)`` covariate matrix.
        treatment: ``(n,)`` continuous treatment vector.
        outcome: ``(n,)`` outcome vector.
        n_neighbors: Number of nearest neighbours per unit.

    Returns:
        Scalar treatment-effect estimate, or ``np.nan`` if the denominator is
        numerically zero.
    """
    # Residualise T on X
    lasso = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
    lasso.fit(covariates, treatment)
    t_resid = treatment - lasso.predict(covariates)  # (n,)

    # Find k-NN in covariate space (include self so offset by 1)
    k = min(n_neighbors + 1, covariates.shape[0])
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(covariates)
    _, indices = nn.kneighbors(covariates)  # (n, k); first column is self
    # Exclude self (index 0 is always the query point itself)
    match_indices = indices[:, 1:]  # (n, n_neighbors)

    # Per-unit outcome and treatment residual of matched neighbours
    y_match_mean = outcome[match_indices].mean(axis=1)  # (n,)
    t_match_mean = t_resid[match_indices].mean(axis=1)  # (n,)

    delta_y = outcome - y_match_mean  # (n,)
    delta_t = t_resid - t_match_mean  # (n,)

    denom = float(np.dot(delta_t, delta_t))
    if abs(denom) < 1e-12:
        return np.nan

    return float(np.dot(delta_t, delta_y) / denom)
