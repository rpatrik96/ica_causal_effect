"""
Throwaway experiment script: test ICA mitigation strategies for binary T.

For each variant, run 10 Monte Carlo reps on a fixed BinaryTreatmentDGPConfig
and report (bias, sigma, RMSE) of the theta estimate.

Variants:
  1. Baseline eps-row (max |kurt|), fun in {logcosh, exp, cube}
  2. Y-loading picker: eps-row = argmax |W[:, last]|
     and Min-|kurt| picker: eps-row = argmin |kurt|
  3. Skip-T: among components with large |W[:, last]|, pick least non-Gaussian
  4. Residualize T (LassoCV on X), then run ICA on [X, T_resid, Y]
  5. Oracle eta source: ICA on observed [X, T, Y] but Munkres against [X, eta, eps]
  6. Drop T: ICA on [X, Y] (sanity check — shouldn't recover theta on its own)
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Callable, Dict, List

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoCV

# project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from binary_treatment_dgp import BinaryTreatmentDGPConfig, generate_binary_treatment_data  # noqa: E402
from mcc import calc_disent_metrics  # noqa: E402

warnings.filterwarnings("ignore")

ICA_MAX_ITER = 1000
ICA_TOL = 1e-4
N_REPS = 10
SEEDS = list(range(2024, 2024 + N_REPS))
THETA_TRUE = 1.5


def _fit_ica(data: np.ndarray, random_state: int, fun: str = "logcosh"):
    ica = FastICA(
        n_components=data.shape[1],
        random_state=random_state,
        max_iter=ICA_MAX_ITER,
        whiten="unit-variance",
        tol=ICA_TOL,
        fun=fun,
    )
    S_hat = ica.fit_transform(data)
    return ica, S_hat


def _theta_from_eps_row(W: np.ndarray, eps_idx: int, n_cov: int, n_treat: int) -> float:
    """Given unmixing W and identified eps row, normalize last column to 1 and
    extract -theta from the treatment columns. Returns scalar (n_treat=1)."""
    w_eps = W[eps_idx, :]
    if abs(w_eps[-1]) < 1e-12:
        return np.nan
    w = w_eps / w_eps[-1]
    return float(-w[n_cov : n_cov + n_treat].mean())


# ---------------------------------------------------------------------------
# Variant implementations
# ---------------------------------------------------------------------------


def variant_baseline_kurt(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    data = np.column_stack([X, T, Y])
    n_cov = X.shape[1]
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    abs_k = np.array([abs(kurtosis(S_hat[:, j])) for j in range(S_hat.shape[1])])
    eps_idx = int(np.argmax(abs_k))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_yloading(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    data = np.column_stack([X, T, Y])
    n_cov = X.shape[1]
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    # pick row with largest |W[:, last_col]|  (largest loading on Y)
    yload = np.abs(ica.components_[:, -1])
    eps_idx = int(np.argmax(yload))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_min_kurt(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    """Pick the row with the smallest |kurt| among the components — eps is Gaussian by construction."""
    data = np.column_stack([X, T, Y])
    n_cov = X.shape[1]
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    abs_k = np.array([abs(kurtosis(S_hat[:, j])) for j in range(S_hat.shape[1])])
    eps_idx = int(np.argmin(abs_k))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_skip_t(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    """Among components with large |W[:, last_col]|, pick least non-Gaussian (eps is Gaussian)."""
    data = np.column_stack([X, T, Y])
    n_cov = X.shape[1]
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    yload = np.abs(ica.components_[:, -1])
    abs_k = np.array([abs(kurtosis(S_hat[:, j])) for j in range(S_hat.shape[1])])
    # restrict to top-k (k=3) by Y-loading, then min |kurt|
    k = min(3, len(yload))
    top_y = np.argsort(yload)[-k:]
    sub_k = abs_k[top_y]
    eps_idx = int(top_y[np.argmin(sub_k)])
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_residualize_t_kurt(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    """Lasso-residualize T on X, then run ICA on [X, T_resid, Y], pick max |kurt|."""
    n_cov = X.shape[1]
    try:
        lasso = LassoCV(cv=3, random_state=seed, max_iter=2000).fit(X, T)
        T_resid = T - lasso.predict(X)
    except Exception:
        return np.nan
    data = np.column_stack([X, T_resid, Y])
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    abs_k = np.array([abs(kurtosis(S_hat[:, j])) for j in range(S_hat.shape[1])])
    eps_idx = int(np.argmax(abs_k))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_residualize_t_yload(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    n_cov = X.shape[1]
    try:
        lasso = LassoCV(cv=3, random_state=seed, max_iter=2000).fit(X, T)
        T_resid = T - lasso.predict(X)
    except Exception:
        return np.nan
    data = np.column_stack([X, T_resid, Y])
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    yload = np.abs(ica.components_[:, -1])
    eps_idx = int(np.argmax(yload))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_residualize_t_minkurt(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    n_cov = X.shape[1]
    try:
        lasso = LassoCV(cv=3, random_state=seed, max_iter=2000).fit(X, T)
        T_resid = T - lasso.predict(X)
    except Exception:
        return np.nan
    data = np.column_stack([X, T_resid, Y])
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    abs_k = np.array([abs(kurtosis(S_hat[:, j])) for j in range(S_hat.shape[1])])
    eps_idx = int(np.argmin(abs_k))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_oracle_eta(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    """ICA on [X, T, Y]; resolve permutations against ground-truth [X, eta, eps_proxy]."""
    data = np.column_stack([X, T, Y])
    n_cov = X.shape[1]
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    # Ground-truth sources for matching: covariates (X), eta, eps proxy (Y - theta*T - X@beta is unknown,
    # but eps is independent gaussian). We use [X, eta, residual-of-Y-on-everything] as a fallback.
    # For oracle, we cheat: we already know eta. For eps, regress Y on X,T (OLS) and use residuals.
    from numpy.linalg import lstsq

    A = np.column_stack([X, T, np.ones(len(T))])
    coef, *_ = lstsq(A, Y, rcond=None)
    eps_proxy = Y - A @ coef
    S_truth = np.column_stack([X, eta, eps_proxy])
    res = calc_disent_metrics(S_truth, S_hat)
    permuted_mixing = ica.mixing_[:, res["munkres_sort_idx"].astype(int)]
    if abs(permuted_mixing.diagonal().min()) < 1e-12:
        return np.nan
    permuted_scaled = permuted_mixing / permuted_mixing.diagonal()
    return float(permuted_scaled[-1, n_cov : n_cov + 1].mean())


def variant_oracle_eta_residualized(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    """Run ICA on [X, eta, Y] (eta is the *true* Bernoulli residual, oracle).

    This is the *upper bound* — gives ICA an i.i.d. continuous source instead of binary T.
    """
    data = np.column_stack([X, eta, Y])
    n_cov = X.shape[1]
    try:
        ica, S_hat = _fit_ica(data, seed, fun=fun)
    except Exception:
        return np.nan
    abs_k = np.array([abs(kurtosis(S_hat[:, j])) for j in range(S_hat.shape[1])])
    eps_idx = int(np.argmax(abs_k))
    return _theta_from_eps_row(ica.components_, eps_idx, n_cov, 1)


def variant_drop_t(X, T, Y, eta, seed, fun: str = "logcosh") -> float:
    """ICA on [X, Y] only — try to extract theta? In a partially linear model, theta cannot be
    identified from [X,Y] without T. Sanity check; we just attempt to read the eps-row's
    Y-coefficient and there is no theta column to read from. Return NaN as 'not applicable'."""
    return np.nan  # by construction, theta is not identifiable from [X,Y]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CONFIG = BinaryTreatmentDGPConfig(
    n_samples=2000,
    n_covariates=10,
    support_size=5,
    treatment_effect=THETA_TRUE,
    propensity_strength=0.7,
    outcome_coef_scale=0.5,
    sigma_outcome=0.5,
)


def run_variant(name: str, fn: Callable, fun: str) -> Dict[str, float]:
    estimates = []
    for seed in SEEDS:
        cfg = BinaryTreatmentDGPConfig(**{**CONFIG.__dict__, "seed": seed})
        X, T, Y, prop, eta, alpha, beta = generate_binary_treatment_data(cfg)
        try:
            theta_hat = fn(X, T, Y, eta, seed, fun=fun)
        except Exception:
            theta_hat = np.nan
        estimates.append(theta_hat)
    estimates = np.array(estimates, dtype=float)
    valid = estimates[~np.isnan(estimates)]
    if len(valid) == 0:
        return dict(name=name, fun=fun, n_valid=0, bias=np.nan, sigma=np.nan, rmse=np.nan, median=np.nan)
    bias = float(np.mean(valid) - THETA_TRUE)
    sigma = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((valid - THETA_TRUE) ** 2)))
    median = float(np.median(valid))
    return dict(name=name, fun=fun, n_valid=int(len(valid)), bias=bias, sigma=sigma, rmse=rmse, median=median)


def main():
    variants: List = []
    for fun in ["logcosh", "exp", "cube"]:
        variants.append(("baseline-kurt", variant_baseline_kurt, fun))
    for fun in ["logcosh", "exp", "cube"]:
        variants.append(("yloading", variant_yloading, fun))
    for fun in ["logcosh", "exp", "cube"]:
        variants.append(("min-kurt", variant_min_kurt, fun))
    variants.append(("skip-t", variant_skip_t, "logcosh"))
    variants.append(("residualize-t-kurt", variant_residualize_t_kurt, "logcosh"))
    variants.append(("residualize-t-yload", variant_residualize_t_yload, "logcosh"))
    variants.append(("residualize-t-minkurt", variant_residualize_t_minkurt, "logcosh"))
    variants.append(("oracle-eta-source", variant_oracle_eta, "logcosh"))
    variants.append(("oracle-eta-input", variant_oracle_eta_residualized, "logcosh"))
    variants.append(("drop-t", variant_drop_t, "logcosh"))

    results = []
    for name, fn, fun in variants:
        r = run_variant(name, fn, fun)
        results.append(r)
        print(
            f"[{name:25s} fun={fun:7s}] n_valid={r['n_valid']:2d} "
            f"bias={r['bias']:+.4f} sigma={r['sigma']:.4f} "
            f"rmse={r['rmse']:.4f} median={r['median']:+.4f}"
        )

    # Sort by RMSE
    valid_results = [r for r in results if not np.isnan(r["rmse"])]
    valid_results.sort(key=lambda r: r["rmse"])
    print("\n=== Sorted by RMSE ===")
    print(f"{'variant':30s} {'fun':8s} {'n_valid':>8s} {'bias':>10s} {'sigma':>10s} {'rmse':>10s} {'median':>10s}")
    for r in valid_results:
        print(
            f"{r['name']:30s} {r['fun']:8s} {r['n_valid']:8d} "
            f"{r['bias']:+10.4f} {r['sigma']:10.4f} {r['rmse']:10.4f} {r['median']:+10.4f}"
        )

    print("\n=== Variants achieving |bias| < 0.5 ===")
    good = [r for r in valid_results if not np.isnan(r["bias"]) and abs(r["bias"]) < 0.5]
    for r in good:
        print(f"  {r['name']:30s} fun={r['fun']:8s} bias={r['bias']:+.4f} rmse={r['rmse']:.4f}")
    if not good:
        print("  (none)")


if __name__ == "__main__":
    main()
