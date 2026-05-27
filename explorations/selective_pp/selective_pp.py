"""
Selective Projection Pursuit + HOML-framework comparison for the PLR.

Round 1 (Reviewer Lester, TMLR rebuttal):
  Full FastICA estimates a (d+2)×(d+2) unmixing matrix W to recover ALL
  latent sources, but theta is read from a single row — the eps row.
  The hypothesis was that Projection Pursuit (PP) could find that single
  direction directly, with lower variance and faster compute.
  Result: PP-2D has 2–4× HIGHER RMSE than full ICA; full ICA "borrows
  strength" from the d covariate rows.

Round 2 (this file):
  The user's reframing: the variance penalty was an artifact of staying
  inside the ICA framework. The HOML/orthogonal-score framework is
  ALREADY selective — it operates purely on (v, r), the 2D sufficient
  statistic for theta. The question is whether HOML achieves ICA-competitive
  RMSE at a fraction of the compute, i.e. "HOML is the correct selective
  estimator; ICA is the wasteful global one."

  This file adds OML, HOML-Known, and HOML-Est to the benchmark and
  measures the speed–variance frontier explicitly.

Model:
    T = g(X) + eta          (treatment, eta non-Gaussian)
    Y = theta*T + f(X) + eps (outcome)

HOML score (second-order orthogonal):
    theta_hat = E_n[r * phi(v)] / E_n[v * phi(v)]
    phi(v) = v^3 - 3*sigma^2*v - kappa_3

where sigma^2 = E[eta^2] and kappa_3 = E[eta^3] (zero-mean).

Inputs: only (v, r) = (T_res, Y_res) plus two scalars (sigma^2, kappa_3).
This is a 2D score — no (d+2)-dimensional object ever computed.
HOML is already "selective by construction."

Estimator variants in this file:
  OML         — first-order orthogonal (DML); no moment knowledge
  HOML-Known  — second-order, oracle moments
  HOML-Est    — second-order, moments estimated from training-fold residuals
  FullICA     — FastICA on all (d+2) dimensions (ica_treatment_effect_estimation_eps_row)
  PP-2D       — FastICA on 2D residual space [v, r]
  PP-fixedpoint — targeted fixed-point iteration in 2D whitened space
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

# Add repo root to path so we can import main_estimation.py etc.
# NOTE: ica.py is imported lazily (inside functions) because it imports torch
# at module level; torch 2.2.2 in this venv was compiled against NumPy 1.x
# which conflicts with the installed NumPy 2.x. The PP estimators themselves
# do not use torch.
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from main_estimation import DEFAULT_LASSO_ALPHAS  # noqa: E402

# ---------------------------------------------------------------------------
# Projection-pursuit contrast function and gradient
# ---------------------------------------------------------------------------


def _logcosh(u: np.ndarray) -> tuple:
    """Return (mean(log cosh(u)), mean(tanh(u))) — negentropy approx + gradient."""
    g_val = np.mean(np.log(np.cosh(u)))
    g_prime = np.mean(np.tanh(u))
    return g_val, g_prime


def _negentropy_1d(w: np.ndarray, Z_white: np.ndarray) -> float:
    """Negentropy approximation (logcosh) of projection w'z for maximization."""
    proj = Z_white @ w  # shape (n,)
    val, _ = _logcosh(proj)
    return -val  # negate for minimization


def _negentropy_1d_grad(w: np.ndarray, Z_white: np.ndarray) -> np.ndarray:
    """Gradient of negentropy w.r.t. w (unit-sphere constraint handled externally)."""
    proj = Z_white @ w  # (n,)
    _, g_prime = _logcosh(proj)
    grad = -(Z_white.T @ np.tanh(proj)) / len(proj)  # (2,)
    return grad


# ---------------------------------------------------------------------------
# Residualisation helper
# ---------------------------------------------------------------------------


def _residualise_cross_fit(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 2,
) -> tuple:
    """Cross-fitted residuals via LassoCV (mirrors all_together_cross_fitting).

    Returns
    -------
    v : np.ndarray, shape (n,)
        Treatment residuals T - g_hat(X).
    r : np.ndarray, shape (n,)
        Outcome residuals Y - f_hat(X).
    """
    n = X.shape[0]
    v = np.zeros(n)
    r = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(X):
        m_t = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
        m_y = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
        m_t.fit(X[train_idx], T[train_idx])
        m_y.fit(X[train_idx], Y[train_idx])
        v[test_idx] = T[test_idx] - m_t.predict(X[test_idx])
        r[test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
    return v, r


# ---------------------------------------------------------------------------
# Whitening in 2D
# ---------------------------------------------------------------------------


def _whiten_2d(Z: np.ndarray) -> tuple:
    """Whiten a 2D matrix Z of shape (n, 2).

    Returns
    -------
    Z_white : np.ndarray, shape (n, 2)
    W_white : np.ndarray, shape (2, 2)   inverse whitening matrix
    """
    cov = np.cov(Z.T)  # (2, 2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W_white = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))  # (2, 2) whitening matrix
    Z_white = Z @ W_white  # (n, 2)
    W_white_inv = np.diag(np.sqrt(eigvals)) @ eigvecs.T  # dewhitening
    return Z_white, W_white, W_white_inv


# ---------------------------------------------------------------------------
# Estimator 1: PP-2D (2D FastICA, single component)
# ---------------------------------------------------------------------------


def selective_pp_2d(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    random_state: int = 0,
    n_splits: int = 2,
    fun: str = "logcosh",
    check_convergence: bool = True,
) -> tuple:
    """Selective PP via 2D FastICA on partialled-out [v, r] residuals.

    Runs FastICA with n_components=2 in the 2D residual space [T-g(X), Y-f(X)].
    Identifies the eps component via y_loading heuristic (same as
    ica_treatment_effect_estimation_eps_row). Reads theta as -w_eps[0]/w_eps[1]
    (the treatment column normalized so Y-column = 1).

    Returns
    -------
    theta_hat : float
    converged : bool
    """
    from warnings import catch_warnings  # noqa: PLC0415

    v, r = _residualise_cross_fit(X, T, Y, n_splits=n_splits)
    Z = np.column_stack([v, r])  # (n, 2)

    with catch_warnings(record=True) as w:
        ica2 = FastICA(
            n_components=2,
            random_state=random_state,
            max_iter=2000,
            whiten="unit-variance",
            tol=1e-4,
            fun=fun,
        )
        ica2.fit(Z)
        converged = len(w) == 0

    if not converged and check_convergence:
        return np.nan, False

    # W_2d has shape (2, 2): rows are unmixing directions in Z-space
    W_2d = ica2.components_  # (2, 2)

    # Identify eps row: largest |loading on Y| = largest |W[:, 1]| (Y is column 1)
    y_loadings = np.abs(W_2d[:, 1])
    eps_idx = int(np.argmax(y_loadings))

    w_eps = W_2d[eps_idx, :]  # (2,): [w_T, w_Y]
    # Normalize so Y-entry = 1
    w_eps_norm = w_eps / w_eps[1]
    # theta: Y = theta*T + eps  =>  eps direction in unmixing space satisfies
    # w_eps' * [v, r] = w_T*v + w_Y*r = const*eps
    # normalized: (w_T/w_Y)*v + r = eps'  =>  r = eps' - (w_T/w_Y)*v
    # comparing to r = theta*v + eps: the T coefficient of the eps row is -theta
    theta_hat = float(-w_eps_norm[0])

    return theta_hat, converged


# ---------------------------------------------------------------------------
# Estimator 2: PP-fixedpoint (targeted single-component FastICA fixed-point)
# ---------------------------------------------------------------------------


def selective_pp_fixedpoint(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
    fun: str = "logcosh",
    random_state: int = 0,
    n_restarts: int = 4,
) -> tuple:
    """Targeted single-component FastICA fixed-point iteration for the eps direction.

    This is the "selective" estimator: instead of the full (d+2)-dimensional ICA,
    run a single FastICA fixed-point update in the 2D whitened residual space
    [T-g(X), Y-f(X)], seeded from the Y-loading direction (the eps heuristic).

    The fixed-point update for a single component w in whitened space is:
        w <- E[z g(w'z)] - E[g'(w'z)] w
        w <- w / ||w||
    where g = tanh (derivative of logcosh), g' = 1 - tanh^2.
    This converges to a local maximum of negentropy under the unit-norm constraint.

    After convergence, apply the same y_loading criterion as
    ica_treatment_effect_estimation_eps_row: if the found direction has
    the larger |r-loading|, it is eps; otherwise use the orthogonal direction.
    Read theta from the normalized eps row of the 2D unmixing matrix.

    Key property: this is equivalent to running deflation-mode FastICA(n_components=1)
    in 2D, which requires only O(n) operations per iteration (no (d+2)^2 matrix
    algebra). The savings over full ICA scale as (d+2) / 2 in the matrix dimension.

    Returns
    -------
    theta_hat : float
    n_iters : int  (fixed-point iterations until convergence or max_iter)
    converged : bool
    """
    v, r = _residualise_cross_fit(X, T, Y, n_splits=n_splits)
    Z = np.column_stack([v, r])  # (n, 2)
    Z_white, W_white, W_white_inv = _whiten_2d(Z)

    rng = np.random.default_rng(random_state)
    n = Z_white.shape[0]

    best_negent = -np.inf
    best_w = None
    total_iters = 0
    any_converged = False

    for restart in range(n_restarts):
        if restart == 0:
            # Seed near Y-axis (eps heuristic): Y residual (r) is output,
            # eps should have large loading on r; map r-direction to whitened space.
            # r-direction in original space = [0, 1]; whiten: w = W_white.T @ [0,1] / norm
            w = W_white.T[:, 1].copy()
            w /= np.linalg.norm(w) + 1e-12
        else:
            phi0 = rng.uniform(0, 2 * np.pi)
            w = np.array([np.cos(phi0), np.sin(phi0)])

        converged_this = False
        for i in range(max_iter):
            proj = Z_white @ w  # (n,)
            if fun == "logcosh":
                g = np.tanh(proj)
                g_prime = 1.0 - np.tanh(proj) ** 2
            elif fun == "exp":
                u = np.exp(-0.5 * proj**2)
                g = proj * u
                g_prime = (1.0 - proj**2) * u
            else:  # cube
                g = proj**3
                g_prime = 3.0 * proj**2

            w_new = (Z_white.T @ g) / n - np.mean(g_prime) * w
            w_new /= np.linalg.norm(w_new) + 1e-12

            delta = np.abs(np.abs(np.dot(w_new, w)) - 1.0)
            w = w_new
            if delta < tol:
                converged_this = True
                total_iters += i + 1
                break
        else:
            total_iters += max_iter

        if converged_this:
            any_converged = True

        # Evaluate negentropy of found direction
        negent = -_negentropy_1d(w, Z_white)
        if negent > best_negent:
            best_negent = negent
            best_w = w.copy()

    # Map best_w back to original 2D space: row of unmixing matrix
    # projection = Z_white @ w = Z @ W_white @ w = Z @ w_orig
    # => w_orig = W_white @ w  (unmixing row in original [v,r] space)
    w_orig = W_white @ best_w  # (2,): [coeff on v, coeff on r]

    # Also consider the orthogonal direction (deflationary complement)
    phi_w = np.arctan2(best_w[1], best_w[0])
    w_orth_white = np.array([-np.sin(phi_w), np.cos(phi_w)])
    w_orig_orth = W_white @ w_orth_white

    # Pick the direction with larger |r-loading| as eps (y_loading criterion)
    if abs(w_orig_orth[1]) > abs(w_orig[1]):
        w_eps = w_orig_orth
    else:
        w_eps = w_orig

    # Normalize so r (Y) coefficient = 1; theta = -(v coefficient)
    w_eps_norm = w_eps / w_eps[1]
    theta_hat = float(-w_eps_norm[0])

    return theta_hat, total_iters, any_converged


# ---------------------------------------------------------------------------
# Thin wrapper: full ICA (eps-row strategy) from ica.py
# ---------------------------------------------------------------------------


def full_ica(
    X_full: np.ndarray,
    S_full: Optional[np.ndarray] = None,
    n_treatments: int = 1,
    random_state: int = 0,
    check_convergence: bool = True,
    fun: str = "logcosh",
) -> tuple:
    """Wrapper around ica_treatment_effect_estimation_eps_row."""
    from ica import ica_treatment_effect_estimation_eps_row  # noqa: PLC0415

    est, mcc = ica_treatment_effect_estimation_eps_row(
        X_full,
        S=S_full,
        random_state=random_state,
        check_convergence=check_convergence,
        n_treatments=n_treatments,
        verbose=False,
        fun=fun,
        eps_identification="y_loading",
    )
    return est, mcc


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _make_plr_data(
    n_samples: int,
    n_covariates: int,
    theta_true: float,
    beta_eta: float,
    beta_eps: float,
    rng: np.random.Generator,
) -> tuple:
    """Pure-numpy PLR DGP matching ica.py's generate_ica_data structure.

    T = X @ A + eta   (A sparse random; eta ~ gennorm(beta_eta))
    Y = theta*T + X @ B + eps   (B random; eps ~ gennorm(beta_eps))
    Covariates X ~ N(0,1); all sources independent.

    Returns X (n, d), T (n,), Y (n,), X_full (n, d+2), S_full (n, d+2).
    X_full and S_full match ica.py column order: [covariates, treatment, outcome].
    """
    from scipy.stats import gennorm as _gennorm  # noqa: PLC0415

    X = rng.standard_normal((n_samples, n_covariates))
    # Sparse random mixing (treatment <- covariates)
    sparse_mask = rng.random((n_covariates,)) < 0.3
    A = rng.standard_normal((n_covariates,)) * sparse_mask
    B = rng.standard_normal((n_covariates,))

    eta = _gennorm(beta=beta_eta).rvs(n_samples, random_state=int(rng.integers(0, 2**31)))
    eps = _gennorm(beta=beta_eps).rvs(n_samples, random_state=int(rng.integers(0, 2**31)))

    T = X @ A + eta
    Y = theta_true * T + X @ B + eps

    # Build X_full / S_full in [covariates, treatment, outcome] order (matches ica.py)
    X_full = np.column_stack([X, T, Y])
    S_full = np.column_stack([X, eta, eps])  # sources: X are their own sources

    return X, T, Y, X_full, S_full


# ---------------------------------------------------------------------------
# HOML-framework estimators (operate on 2D residuals only)
# ---------------------------------------------------------------------------


def _oml(v: np.ndarray, r: np.ndarray) -> float:
    """First-order orthogonal (DML) estimate: OLS on partialled-out residuals.

    theta_hat = E_n[v*r] / E_n[v^2]

    Inputs: only the 2D pair (v, r). No moment knowledge required.
    """
    return float(np.mean(v * r) / np.mean(v * v))


def _homl_known(v: np.ndarray, r: np.ndarray, eta_second: float, eta_third: float) -> float:
    """Second-order orthogonal estimate with oracle moment knowledge.

    Score multiplier: phi(v) = v^3 - 3*sigma^2*v - kappa_3
    theta_hat = E_n[r * phi(v)] / E_n[v * phi(v)]

    Inputs: (v, r) plus two oracle scalars (sigma^2, kappa_3).
    """
    phi = v**3 - 3.0 * eta_second * v - eta_third
    return float(np.mean(r * phi) / np.mean(v * phi))


def _homl_est(v_train: np.ndarray, v_test: np.ndarray, r_test: np.ndarray) -> float:
    """Second-order orthogonal estimate with moments estimated from training residuals.

    Estimates sigma^2 = E[v^2] and kappa_3 = E[v^3] from v_train, then
    applies the HOML score on (v_test, r_test).

    This is the "directional-moment" variant: estimates *only* the scalar
    cumulants along the treatment residual direction, nothing more.
    No full-dimensional objects. No global system.
    """
    sigma2 = float(np.mean(v_train**2))
    kappa3 = float(np.mean(v_train**3) - 3.0 * np.mean(v_train) * sigma2)
    phi = v_test**3 - 3.0 * sigma2 * v_test - kappa3
    return float(np.mean(r_test * phi) / np.mean(v_test * phi))


def homl_variants_cross_fit(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    eta_second: float,
    eta_third: float,
    n_splits: int = 2,
) -> dict:
    """Cross-fitted OML, HOML-Known, and HOML-Est from scratch on raw (X, T, Y).

    Runs 2-fold cross-fitting: fits LassoCV on each training fold, accumulates
    residuals, then computes all three score variants.

    Returns dict with keys: "OML", "HOML-Known", "HOML-Est", plus timing.
    """
    n = X.shape[0]
    v = np.zeros(n)
    r = np.zeros(n)
    # For HOML-Est we need train-fold residuals paired with each test fold
    # so we store them fold-by-fold.
    fold_records: list = []

    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(X):
        m_t = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
        m_y = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
        m_t.fit(X[train_idx], T[train_idx])
        m_y.fit(X[train_idx], Y[train_idx])
        v[test_idx] = T[test_idx] - m_t.predict(X[test_idx])
        r[test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
        # Training residuals for moment estimation
        v_tr = T[train_idx] - m_t.predict(X[train_idx])
        fold_records.append((v_tr, test_idx))

    # OML
    oml = _oml(v, r)

    # HOML-Known: uses full cross-fitted residuals + oracle moments
    homl_known = _homl_known(v, r, eta_second, eta_third)

    # HOML-Est: per-fold moment estimation from the complementary training residuals
    phi_est = np.zeros(n)
    for v_tr, test_idx in fold_records:
        sigma2 = float(np.mean(v_tr**2))
        kappa3 = float(np.mean(v_tr**3) - 3.0 * np.mean(v_tr) * sigma2)
        phi_est[test_idx] = v[test_idx] ** 3 - 3.0 * sigma2 * v[test_idx] - kappa3
    homl_est = float(np.mean(r * phi_est) / np.mean(v * phi_est))

    return {"OML": oml, "HOML-Known": homl_known, "HOML-Est": homl_est}


# ---------------------------------------------------------------------------
# Extended experiment runner (Round 2: HOML-framework comparison)
# ---------------------------------------------------------------------------


def _run_single_experiment(args: tuple) -> Optional[dict]:
    """Module-level worker function (picklable for multiprocessing).

    args = (seed, n_samples, n_covariates, true_theta, beta_eta, beta_eps,
            n_treatments, check_convergence)
    """
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")

    seed, n_samples, n_covariates, true_theta, beta_eta, beta_eps, n_treatments, check_conv = args
    from scipy.stats import gennorm as _gennorm  # noqa: PLC0415

    rng_local = np.random.default_rng(seed)
    X, T, Y, X_full, S_full = _make_plr_data(n_samples, n_covariates, true_theta, beta_eta, beta_eps, rng_local)

    eta_dist = _gennorm(beta=beta_eta)
    eta_second = float(eta_dist.moment(2))
    eta_third = float(eta_dist.moment(3))

    results = {"seed": seed}

    # ---- HOML-framework variants (OML, HOML-Known, HOML-Est) ----
    t0 = time.perf_counter()
    homl_variants = homl_variants_cross_fit(X, T, Y, eta_second, eta_third)
    t_homl_block = time.perf_counter() - t0
    # All three share the same nuisance estimation pass; time is for the full block.
    # For individual timing, the nuisance cost dominates — all variants are equally fast.
    results["OML"] = homl_variants["OML"]
    results["HOML-Known"] = homl_variants["HOML-Known"]
    results["HOML-Est"] = homl_variants["HOML-Est"]
    results["t_oml"] = t_homl_block  # nuisance cost shared; label as OML cost
    results["t_homl_known"] = t_homl_block
    results["t_homl_est"] = t_homl_block

    # ---- Full ICA ----
    t0 = time.perf_counter()
    ica_est, _ = full_ica(
        X_full,
        S_full,
        n_treatments=n_treatments,
        random_state=seed,
        check_convergence=check_conv,
    )
    results["FullICA"] = float(ica_est[0]) if not np.isnan(ica_est[0]) else np.nan
    results["ica_converged"] = not np.isnan(ica_est[0])
    results["t_fullica"] = time.perf_counter() - t0

    # ---- PP-2D ----
    t0 = time.perf_counter()
    pp2d_est, _ = selective_pp_2d(X, T, Y, random_state=seed, check_convergence=check_conv)
    results["PP-2D"] = float(pp2d_est)
    results["t_pp2d"] = time.perf_counter() - t0

    # ---- PP-fixedpoint ----
    t0 = time.perf_counter()
    ppfp_est, _, _ = selective_pp_fixedpoint(X, T, Y, random_state=seed)
    results["PP-fixedpoint"] = float(ppfp_est)
    results["t_ppfp"] = time.perf_counter() - t0

    return results


METHODS_ALL = ["OML", "HOML-Known", "HOML-Est", "FullICA", "PP-2D", "PP-fixedpoint"]
_TIME_KEYS = {
    "OML": "t_oml",
    "HOML-Known": "t_homl_known",
    "HOML-Est": "t_homl_est",
    "FullICA": "t_fullica",
    "PP-2D": "t_pp2d",
    "PP-fixedpoint": "t_ppfp",
}


def run_experiment(
    n_samples: int = 1000,
    n_experiments: int = 200,
    n_covariates: int = 3,
    n_treatments: int = 1,
    beta: float = 1.0,  # Laplace eps noise
    split_eta_eps: bool = True,
    beta_eta: float = 1.0,  # Laplace eta noise
    n_jobs: int = 4,
    random_seed: int = 0,
    check_convergence: bool = True,
) -> dict:
    """Monte Carlo comparison: OML, HOML-Known, HOML-Est, FullICA, PP-2D, PP-fixedpoint.

    Uses a pure-numpy PLR DGP (no torch dependency). Heavy-tailed noise (Laplace,
    beta=1) is where ICA is supposed to beat first-order methods.

    Returns dict with keys: per-method bias/sigma/rmse/timing, plus meta.
    """
    import multiprocessing  # noqa: PLC0415

    true_theta = 1.55  # matches ica.py DEFAULT_TREATMENT_EFFECTS[0]

    all_estimates = {m: [] for m in METHODS_ALL}
    timing = {m: 0.0 for m in METHODS_ALL}
    n_ica_converged = 0
    n_total_attempted = 0

    rng = np.random.default_rng(random_seed)
    seeds = rng.integers(0, 100000, size=n_experiments).tolist()

    job_args = [
        (s, n_samples, n_covariates, true_theta, beta_eta, beta, n_treatments, check_convergence) for s in seeds
    ]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_jobs) as pool:
        raw_results = pool.map(_run_single_experiment, job_args)

    for r in raw_results:
        if r is None:
            continue
        for m in METHODS_ALL:
            val = r.get(m, np.nan)
            all_estimates[m].append(val)
        for m, tk in _TIME_KEYS.items():
            timing[m] += r.get(tk, 0.0)
        n_total_attempted += 1
        if r.get("ica_converged", False):
            n_ica_converged += 1

    summary = {}
    for m in METHODS_ALL:
        ests = np.array(all_estimates[m], dtype=float)
        valid = ests[~np.isnan(ests)]
        n_valid = len(valid)
        bias = float(np.mean(valid) - true_theta) if n_valid > 0 else np.nan
        sigma = float(np.std(valid)) if n_valid > 0 else np.nan
        rmse = float(np.sqrt(bias**2 + sigma**2)) if n_valid > 0 else np.nan
        summary[m] = {
            "bias": bias,
            "sigma": sigma,
            "rmse": rmse,
            "n_valid": n_valid,
            "mean_time_ms": 1000 * timing[m] / max(n_total_attempted, 1),
        }

    summary["meta"] = {
        "true_theta": true_theta,
        "n_experiments": n_experiments,
        "n_total_attempted": n_total_attempted,
        "ica_convergence_rate": n_ica_converged / max(n_total_attempted, 1),
        "n_samples": n_samples,
        "n_covariates": n_covariates,
        "beta_eta": beta_eta,
        "beta_eps": beta,
    }

    return summary


def print_speed_variance_table(results: dict, methods_order: list = None) -> None:
    """Print the speed–variance frontier table.

    For each method prints Bias, Sigma, RMSE, Time(ms), and relative speedup
    vs FullICA (the reference).
    """
    if methods_order is None:
        methods_order = METHODS_ALL

    ref_time = results["FullICA"]["mean_time_ms"]

    header = f"{'Method':<18} {'Bias':>8} {'Sigma':>8} {'RMSE':>8} " f"{'Time(ms)':>10} {'vs ICA':>8} {'N_valid':>8}"
    print(header)
    print("-" * len(header))
    for m in methods_order:
        if m not in results:
            continue
        s = results[m]
        speedup = ref_time / s["mean_time_ms"] if s["mean_time_ms"] > 0 else float("inf")
        print(
            f"{m:<18} {s['bias']:>8.4f} {s['sigma']:>8.4f} {s['rmse']:>8.4f} "
            f"{s['mean_time_ms']:>10.1f} {speedup:>7.1f}x {s['n_valid']:>8}"
        )
    print()


# ---------------------------------------------------------------------------
# Main: run experiment suite and print tables
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Selective PP + HOML-framework comparison (Round 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_experiments", type=int, default=200)
    parser.add_argument("--n_covariates", type=int, default=3)
    parser.add_argument("--beta_eta", type=float, default=1.0, help="Shape for treatment noise (1=Laplace)")
    parser.add_argument("--beta_eps", type=float, default=1.0, help="Shape for outcome noise (1=Laplace)")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suite",
        action="store_true",
        default=False,
        help="Run full 5-configuration suite (overrides n_samples/n_covariates/beta_eta/beta_eps)",
    )
    args = parser.parse_args()

    SUITE_CONFIGS = [
        dict(n_samples=1000, n_covariates=3, beta_eta=1.0, beta_eps=1.0, label="n=1000, d=3, Laplace"),
        dict(n_samples=2000, n_covariates=3, beta_eta=1.0, beta_eps=1.0, label="n=2000, d=3, Laplace"),
        dict(n_samples=1000, n_covariates=10, beta_eta=1.0, beta_eps=1.0, label="n=1000, d=10, Laplace"),
        dict(n_samples=1000, n_covariates=3, beta_eta=1.8, beta_eps=1.8, label="n=1000, d=3, near-Gaussian"),
        dict(n_samples=5000, n_covariates=20, beta_eta=1.0, beta_eps=1.0, label="n=5000, d=20, Laplace"),
    ]

    if args.suite:
        configs = SUITE_CONFIGS
    else:
        configs = [
            dict(
                n_samples=args.n_samples,
                n_covariates=args.n_covariates,
                beta_eta=args.beta_eta,
                beta_eps=args.beta_eps,
                label=f"n={args.n_samples}, d={args.n_covariates}, "
                f"beta_eta={args.beta_eta}, beta_eps={args.beta_eps}",
            )
        ]

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        print(f"{'='*70}")
        print(f"n_experiments={args.n_experiments}, n_jobs={args.n_jobs}\n")

        res = run_experiment(
            n_samples=cfg["n_samples"],
            n_experiments=args.n_experiments,
            n_covariates=cfg["n_covariates"],
            beta=cfg["beta_eps"],
            beta_eta=cfg["beta_eta"],
            split_eta_eps=True,
            n_jobs=args.n_jobs,
            random_seed=args.seed,
        )

        meta = res.pop("meta")
        print(f"True theta = {meta['true_theta']:.4f}")
        print(
            f"ICA convergence rate: {meta['ica_convergence_rate']:.2%}  "
            f"(n_valid: {res['FullICA']['n_valid']}/{meta['n_experiments']})\n"
        )
        print_speed_variance_table(res)

        # Restore meta for any downstream use
        res["meta"] = meta
