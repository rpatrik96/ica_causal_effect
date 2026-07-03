#!/usr/bin/env python3
"""WS4 multi-feature estimator selector across the whole campaign regime map.

Where `selector.py` chose ICA-vs-OML on one feature (ε-kurtosis), this selector
decides among FOUR candidates — OLS, OML(linear nuisance), OML(gbm nuisance), ICA
— using a small feature vector that captures each regime the campaign charted:

  feature              regime it detects                       campaign evidence
  -------------------  -------------------------------------   -----------------
  nonlinearity_score   nonlinear g(X): GBM outcome-fit R^2      r04, r05, r08
                       gain over a linear fit  -> OML(gbm)
  d_over_n             high dimension d/n -> avoid ICA          r06 (ICA blows up;
                       (catastrophic), use OML(gbm)                 OLS double-descent)
  eps_excess_kurtosis  heavy-tailed outcome noise, linear,      r09, r10, r11, r12
                       n>>d -> ICA efficiency edge
  (else)               linear, well-specified, light noise      r02, r07
                       -> OLS (efficient / simplest)

Rule tree (checked in order; first match wins) — deliberately interpretable, its
thresholds grounded in the round findings, not fit to a training set:

  if nonlinearity_score > tau_nl:   -> "OML_gbm"    # misspecified conditional mean
  elif d_over_n         > tau_dn:   -> "OML_gbm"    # high-dim; ICA unusable
  elif eps_excess_kurt  > tau_eps:  -> "ICA"        # heavy outcome noise
  else:                             -> "OLS"        # well-specified linear

`nonlinearity_score` and the kurtoses are computed on a capped subsample so the
selector's own cost stays bounded even at large n.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Default thresholds (calibrated against the round findings; see r13). `n_gate` is
# the minimum sample size for the flexible-nuisance (gbm) branch: below it GBM
# overfits and the simpler estimator wins (the r08 lesson, confirmed in r13).
DEFAULT_THRESHOLDS = {"nl": 0.05, "dn": 0.5, "eps": 3.0, "n_gate": 1000}
CANDIDATES = ("OLS", "OML_lin", "OML_gbm", "ICA")


def _r2_gain(A, y, seed=0):
    """Held-out R^2 improvement of GBM over LassoCV predicting y from A — a proxy
    for how nonlinear the conditional mean is."""
    Atr, Ate, ytr, yte = train_test_split(A, y, test_size=0.4, random_state=seed)
    lin = LassoCV(max_iter=5000).fit(Atr, ytr)
    gbm = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=seed).fit(Atr, ytr)
    return float(r2_score(yte, gbm.predict(Ate)) - r2_score(yte, lin.predict(Ate)))


def multi_features(X, T, Y, max_feat_n=2000, seed=0) -> dict:
    """Feature vector for the multi-selector. Kurtoses from partialled-out
    first-stage residuals; nonlinearity from a GBM-vs-linear R^2 gap; d/n from
    shapes. Heavy parts run on a capped subsample."""
    n_full, d = X.shape
    rng = np.random.default_rng(seed)
    if n_full > max_feat_n:
        idx = rng.choice(n_full, size=max_feat_n, replace=False)
        Xs, Ts, Ys = X[idx], T[idx], Y[idx]
    else:
        Xs, Ts, Ys = X, T, Y

    mt = LassoCV(max_iter=5000).fit(Xs, Ts)
    mo = LassoCV(max_iter=5000).fit(Xs, Ys)
    eta_hat = Ts - mt.predict(Xs)
    r_Y = Ys - mo.predict(Xs)
    denom = float(eta_hat @ eta_hat)
    theta_tilde = float(eta_hat @ r_Y / denom) if denom > 0 else 0.0
    eps_hat = r_Y - theta_tilde * eta_hat

    # nonlinearity: max R^2 gain of GBM over linear on the outcome and treatment.
    nl = max(_r2_gain(Xs, Ys, seed), _r2_gain(Xs, Ts, seed + 1))
    return {
        "eps_excess_kurtosis": float(kurtosis(eps_hat, fisher=True, bias=False)),
        "eta_excess_kurtosis": float(kurtosis(eta_hat, fisher=True, bias=False)),
        "nonlinearity_score": float(nl),
        "d_over_n": float(d) / float(n_full),
        "n": int(n_full),
        "d": int(d),
    }


def select_estimator_multi(features: dict, thresholds: dict | None = None) -> str:
    """Interpretable rule tree -> one of CANDIDATES.

    Order matters. The n-gate on the flexible-nuisance branch is the key
    correction from r13: a flexible GBM nuisance only helps when there are enough
    samples to fit it (r08); at small n the simpler estimator wins, so nonlinear/
    high-dim *small-n* cells fall back to OLS (low-dim) or OML_lin (high-dim, where
    OLS min-norm is poor but regularization helps)."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    nl_ok = features["nonlinearity_score"] > th["nl"]
    high_d = features["d_over_n"] > th["dn"]
    enough_n = features["n"] >= th["n_gate"]

    if high_d:
        # d >= n: OLS min-norm is poor and GBM overfits -> regularized OML_lin.
        return "OML_gbm" if (nl_ok and enough_n) else "OML_lin"
    if nl_ok:
        # nonlinear g(X): rescue with a flexible nuisance only if n supports it.
        return "OML_gbm" if enough_n else "OLS"
    if features["eps_excess_kurtosis"] > th["eps"]:
        return "ICA"                           # heavy outcome noise, linear, low-dim (r09-r12)
    return "OLS"                               # well-specified linear (r02/r07)
