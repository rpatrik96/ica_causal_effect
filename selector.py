#!/usr/bin/env python3
"""WS4 data-driven estimator selector: choose ICA vs OML per dataset.

Motivation (from rounds r09/r10): in the linear PLR at n >> d, ICA is the most
efficient treatment-effect estimator **iff the outcome noise ε is sufficiently
non-Gaussian (heavy-tailed)**; otherwise OML/OLS win. A practitioner cannot see ε
directly, but can *estimate* its non-Gaussianity from first-stage residuals and
switch estimators on that signal.

Selector = feature extraction + a threshold rule:

  1. Fit nuisances t̂(X), ŷ(X) (LassoCV); form partialled-out residuals
       η̂ = T − t̂(X)                    (≈ treatment noise)
       r_Y = Y − ŷ(X) ≈ θ·η̂ + ε
       θ̃  = ⟨η̂, r_Y⟩ / ⟨η̂, η̂⟩          (partialled-out OLS slope)
       ε̂ = r_Y − θ̃·η̂                   (≈ outcome noise)
  2. Feature = excess kurtosis of ε̂ (Fisher; 0 for Gaussian, >0 heavy-tailed).
  3. Rule: choose ICA if kurt(ε̂) > τ, else OML.

The r10 frontier is the ground truth used to calibrate τ and to measure regret
(selector RMSE − best-fixed-estimator RMSE).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis
from sklearn.linear_model import LassoCV

DEFAULT_TAU = 1.0  # excess-kurtosis threshold on ε̂; calibrated in r12.


def selection_features(X, T, Y, alphas=None) -> dict:
    """Estimated non-Gaussianity of the treatment/outcome noises from first-stage
    residuals, plus problem size — the inputs a selector may key on."""
    mt = LassoCV(alphas=alphas, max_iter=5000).fit(X, T)
    mo = LassoCV(alphas=alphas, max_iter=5000).fit(X, Y)
    eta_hat = T - mt.predict(X)
    r_Y = Y - mo.predict(X)
    denom = float(eta_hat @ eta_hat)
    theta_tilde = float(eta_hat @ r_Y / denom) if denom > 0 else 0.0
    eps_hat = r_Y - theta_tilde * eta_hat
    return {
        "eps_excess_kurtosis": float(kurtosis(eps_hat, fisher=True, bias=False)),
        "eta_excess_kurtosis": float(kurtosis(eta_hat, fisher=True, bias=False)),
        "theta_partialled": theta_tilde,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
    }


def select_estimator(features: dict, tau: float = DEFAULT_TAU) -> str:
    """Threshold rule: ICA when the estimated outcome-noise ε is heavy-tailed."""
    return "ICA" if features["eps_excess_kurtosis"] > tau else "OML"


def choose_estimate(estimates: dict, features: dict, tau: float = DEFAULT_TAU):
    """Return (chosen_estimator_name, chosen_theta) given a dict of candidate
    estimates (keys 'ICA', 'OML', ...) and the selection features."""
    name = select_estimator(features, tau)
    return name, estimates[name]
