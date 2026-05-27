"""Runner for real-data causal-inference benchmarks: IHDP and Jobs (LaLonde).

Runs all seven estimators on IHDP (semi-synthetic, known ground-truth ATE/ATT)
and on the LaLonde/NSW Jobs dataset (binary treatment, ATT vs. experimental
benchmark ~$1,794).

Estimators
----------
- OLS               : ordinary least squares of Y ~ T + X
- Matching          : propensity-score k-NN matching (binary path)
- Ortho ML          : Double ML / first-order orthogonal
- Robust Ortho ML   : second-order orthogonal with known eta moments
- Robust Ortho Est  : second-order orthogonal with estimated eta moments
- Robust Ortho Split: second-order orthogonal with nested-split moments
- ICA               : ICA-based eps-row estimator

Nuisance models (``--nuisance``)
---------------------------------
``linear``   LassoCV with a coarse alpha grid (default; consistent with main
             paper experiments).
``gbm``      GradientBoostingRegressor (200 trees, depth 4).  Reduces bias
             on nonlinear DGPs such as IHDP "setting B".
``poly``     Polynomial features (degree 2) + Ridge.

Example
-------
::

    python realdata_runner.py --dataset ihdp --n_replications 100 --nuisance gbm \\
        --output_dir figures/realdata
    python realdata_runner.py --dataset jobs --output_dir figures/realdata
    python realdata_runner.py --dataset jobs_obs --comparison cps \\
        --output_dir figures/realdata
    python realdata_runner.py --dataset both --n_replications 10 \\
        --nuisance linear --output_dir figures/realdata
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from baselines import matching_baseline, ols_baseline
from binary_treatment_dgp import empirical_eta_moments
from main_estimation import DEFAULT_LASSO_ALPHAS, all_together_cross_fitting
from realdata_loaders import load_ihdp, load_jobs, load_jobs_observational

METHOD_NAMES: Tuple[str, ...] = (
    "Ortho ML",
    "Robust Ortho ML",
    "Robust Ortho Est",
    "Robust Ortho Split",
    "ICA",
    "OLS",
    "Matching",
)

_LASSO_ALPHAS = DEFAULT_LASSO_ALPHAS


def _make_nuisance_models(nuisance: str):
    """Return (model_treatment, model_outcome) for the chosen nuisance type.

    Parameters
    ----------
    nuisance : str
        One of ``"linear"``, ``"gbm"``, or ``"poly"``.

    Returns
    -------
    model_treatment, model_outcome : sklearn estimator instances (unfitted clones)
    """
    if nuisance == "linear":
        mt = LassoCV(alphas=_LASSO_ALPHAS)
        mo = LassoCV(alphas=_LASSO_ALPHAS)
    elif nuisance == "gbm":
        mt = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0)
        mo = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=1)
    elif nuisance == "poly":
        mt = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        mo = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
    else:
        raise ValueError(f"nuisance must be 'linear', 'gbm', or 'poly'; got '{nuisance}'")
    return mt, mo


def _estimate_eta_moments_from_resid(T: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """Fit a logistic propensity model and compute eta = T - p_hat moments."""
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X, T.astype(int))
    p_hat = clf.predict_proba(X)[:, 1]
    eta = T - p_hat
    return empirical_eta_moments(eta)


def _run_single_replication(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisance: str = "linear",
) -> Tuple[float, ...]:
    """Run all seven estimators on a single (X, T, Y) dataset.

    Returns estimates in the order of METHOD_NAMES.
    """
    eta_second, eta_third = _estimate_eta_moments_from_resid(T, X)

    mt, mo = _make_nuisance_models(nuisance)

    (
        ortho_ml,
        robust_ortho_ml,
        robust_ortho_est_ml,
        robust_ortho_est_split_ml,
        _tcoef,
        _ycoef,
    ) = all_together_cross_fitting(
        X,
        T,
        Y,
        treatment_second_moment=eta_second,
        treatment_third_moment=eta_third,
        model_treatment=mt,
        model_outcome=mo,
    )

    try:
        from ica import ica_treatment_effect_estimation_eps_row  # pylint: disable=import-outside-toplevel

        ica_est, _ = ica_treatment_effect_estimation_eps_row(
            np.hstack((X, T.reshape(-1, 1), Y.reshape(-1, 1))),
            S=None,
            check_convergence=False,
            verbose=False,
        )
        ica_value = float(ica_est[0]) if np.isfinite(ica_est).all() else float("nan")
    except Exception:  # pylint: disable=broad-exception-caught
        ica_value = float("nan")

    ols_value = float(ols_baseline(X, T, Y)[0])
    matching_value = float(matching_baseline(X, T, Y, treatment_kind="binary"))

    return (
        float(ortho_ml),
        float(robust_ortho_ml),
        float(robust_ortho_est_ml),
        float(robust_ortho_est_split_ml),
        ica_value,
        ols_value,
        matching_value,
    )


def run_ihdp(
    n_replications: int = 100,
    nuisance: str = "linear",
    verbose: bool = True,
) -> Dict:
    """Run all estimators on the first n_replications IHDP NPCI replications.

    Each replication is treated as an independent dataset. Bias/sigma/RMSE and
    95% bootstrap CIs are computed relative to the per-replication true ATT.

    Parameters
    ----------
    n_replications : int
        Number of IHDP replications to evaluate (max 1000 available).
    nuisance : str
        Nuisance model type — ``"linear"``, ``"gbm"``, or ``"poly"``.
    verbose : bool
        Print per-replication progress.

    Returns
    -------
    dict with keys: dataset, method_names, estimates, true_att, biases, sigmas,
    rmse, rmse_se, rmse_ci_lo, rmse_ci_hi, n_replications, is_real, nuisance.
    """
    estimates_list: List[Tuple[float, ...]] = []
    true_atts: List[float] = []
    is_real_flags: List[bool] = []

    for rep in range(1, n_replications + 1):
        X, T, Y, true_att = load_ihdp(replication=rep, use_fixture_on_failure=True)

        from realdata_loaders import _ihdp_csv_path  # pylint: disable=import-outside-toplevel

        is_real_flags.append(os.path.exists(_ihdp_csv_path(rep)))

        est = _run_single_replication(X, T, Y, nuisance=nuisance)
        estimates_list.append(est)
        true_atts.append(float(true_att))
        if verbose:
            print(
                f"  IHDP rep {rep:3d}/{n_replications}: " f"true_ATT={true_att:.4f}, OML={est[0]:.4f}, ICA={est[4]:.4f}"
            )

    estimates = np.array(estimates_list, dtype=float)  # (n_rep, 7)
    true_att_arr = np.array(true_atts)[:, None]  # (n_rep, 1)

    biases = np.nanmean(estimates - true_att_arr, axis=0)
    sigmas = np.nanstd(estimates, axis=0)
    rmse = np.sqrt(biases**2 + sigmas**2)

    # Per-replication squared errors for SE of RMSE
    sq_err = (estimates - true_att_arr) ** 2  # (n_rep, 7)
    rmse_se = np.nanstd(sq_err, axis=0) / (2.0 * rmse * np.sqrt(np.sum(~np.isnan(sq_err), axis=0)))
    rmse_ci_lo = rmse - 1.96 * rmse_se
    rmse_ci_hi = rmse + 1.96 * rmse_se

    is_real = all(is_real_flags)

    if verbose:
        print(
            f"\nIHDP results ({n_replications} replications, "
            f"{'real data' if is_real else 'SYNTHETIC FIXTURE'}, "
            f"nuisance={nuisance}):"
        )
        print(f"{'Method':<22}{'bias':>10}{'sigma':>10}{'RMSE':>10}{'SE(RMSE)':>12}{'95% CI':>22}")
        for name, b, s, r, se, lo, hi in zip(METHOD_NAMES, biases, sigmas, rmse, rmse_se, rmse_ci_lo, rmse_ci_hi):
            print(f"{name:<22}{b:>10.4f}{s:>10.4f}{r:>10.4f}{se:>12.4f}  [{lo:.4f}, {hi:.4f}]")

    return {
        "dataset": "IHDP",
        "method_names": np.array(METHOD_NAMES),
        "estimates": estimates,
        "true_att": np.array(true_atts),
        "biases": biases,
        "sigmas": sigmas,
        "rmse": rmse,
        "rmse_se": rmse_se,
        "rmse_ci_lo": rmse_ci_lo,
        "rmse_ci_hi": rmse_ci_hi,
        "n_replications": n_replications,
        "is_real": is_real,
        "nuisance": nuisance,
    }


def run_jobs(
    nuisance: str = "linear",
    verbose: bool = True,
) -> Dict:
    """Run all estimators on the LaLonde/NSW Jobs dataset (experimental).

    Evaluation is against the experimental ATT benchmark (~$1,794).
    The naive difference-in-means from the data is also reported.

    Parameters
    ----------
    nuisance : str
        Nuisance model type — ``"linear"``, ``"gbm"``, or ``"poly"``.
    verbose : bool
        Print results table.
    """
    X, T, Y, meta = load_jobs(use_fixture_on_failure=True)
    if verbose:
        print(
            f"\nJobs dataset: n_treated={meta['n_treated']}, n_control={meta['n_control']}, "
            f"{'REAL DATA' if meta['is_real'] else 'SYNTHETIC FIXTURE'}"
        )
        print(f"  Source: {meta['source']}")
        exp_att = meta.get("att_experimental", meta["att_benchmark"])
        print(f"  Naive experimental ATT (raw diff-in-means): ${exp_att:,.1f}")

    est = _run_single_replication(X, T, Y, nuisance=nuisance)
    estimates = np.array(est)

    att_benchmark = meta["att_benchmark"]
    att_experimental = meta.get("att_experimental", att_benchmark)
    deviations_benchmark = estimates - att_benchmark
    deviations_experimental = estimates - att_experimental

    if verbose:
        print(f"\nJobs results (nuisance={nuisance}):")
        print(f"  Benchmark ATT (D&W 1999): ${att_benchmark:,.0f}")
        print(f"  Experimental ATT (data diff-in-means): ${att_experimental:,.1f}")
        print(f"{'Method':<22}{'estimate':>12}{'dev vs D&W':>14}{'dev vs DIM':>14}")
        for name, e, db, de in zip(METHOD_NAMES, estimates, deviations_benchmark, deviations_experimental):
            print(f"{name:<22}{e:>12.1f}{db:>14.1f}{de:>14.1f}")

    return {
        "dataset": "Jobs",
        "method_names": np.array(METHOD_NAMES),
        "estimates": estimates,
        "att_benchmark": att_benchmark,
        "att_experimental": att_experimental,
        "deviations": deviations_benchmark,
        "deviations_experimental": deviations_experimental,
        "n_treated": meta["n_treated"],
        "n_control": meta["n_control"],
        "is_real": meta["is_real"],
        "source": meta["source"],
        "nuisance": nuisance,
    }


def run_jobs_observational(
    comparison: str = "cps",
    nuisance: str = "linear",
    verbose: bool = True,
) -> Dict:
    """Run all estimators on NSW treated + CPS/PSID observational controls.

    This is the classical observational bias experiment: how well does each
    estimator recover the experimental ATT (~$1,794) when the control group
    is non-experimental?

    Parameters
    ----------
    comparison : str
        ``"cps"`` or ``"psid"``.
    nuisance : str
        Nuisance model type.
    verbose : bool
        Print results table.
    """
    X, T, Y, meta = load_jobs_observational(comparison=comparison, use_fixture_on_failure=True)
    if verbose:
        print(
            f"\nJobs ({comparison.upper()}) observational: "
            f"n_treated={meta['n_treated']}, n_control={meta['n_control']}, "
            f"{'REAL DATA' if meta['is_real'] else 'SYNTHETIC/FIXTURE'}"
        )

    est = _run_single_replication(X, T, Y, nuisance=nuisance)
    estimates = np.array(est)

    att_benchmark = meta["att_benchmark"]
    att_experimental = meta.get("att_experimental", att_benchmark)
    deviations = estimates - att_benchmark

    if verbose:
        print(f"\nJobs ({comparison.upper()}) observational results (nuisance={nuisance}):")
        print(f"  Experimental ATT benchmark: ${att_benchmark:,.0f}")
        print(f"  Naive observational ATT (raw DIM): ${float(Y[T == 1].mean() - Y[T == 0].mean()):,.1f}")
        print(f"{'Method':<22}{'estimate':>12}{'dev vs benchmark':>18}")
        for name, e, d in zip(METHOD_NAMES, estimates, deviations):
            print(f"{name:<22}{e:>12.1f}{d:>18.1f}")

    return {
        "dataset": f"Jobs_{comparison.upper()}",
        "method_names": np.array(METHOD_NAMES),
        "estimates": estimates,
        "att_benchmark": att_benchmark,
        "att_experimental": att_experimental,
        "deviations": deviations,
        "n_treated": meta["n_treated"],
        "n_control": meta["n_control"],
        "is_real": meta["is_real"],
        "comparison": comparison.upper(),
        "nuisance": nuisance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["ihdp", "jobs", "jobs_obs", "both"],
        default="both",
        help="Which dataset to run (default: both).",
    )
    parser.add_argument(
        "--n_replications",
        type=int,
        default=100,
        help="Number of IHDP replications to use (default: 100; max 1000).",
    )
    parser.add_argument(
        "--nuisance",
        choices=["linear", "gbm", "poly"],
        default="linear",
        help="Nuisance model for OML/HOML family (default: linear).",
    )
    parser.add_argument(
        "--comparison",
        choices=["cps", "psid"],
        default="cps",
        help="Observational comparison group for jobs_obs (default: cps).",
    )
    parser.add_argument("--output_dir", default="figures/realdata")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)
    nuisance_tag = opts.nuisance

    if opts.dataset in ("ihdp", "both"):
        print(f"\n=== Running IHDP ({opts.n_replications} replications, nuisance={nuisance_tag}) ===")
        ihdp_results = run_ihdp(
            n_replications=opts.n_replications,
            nuisance=opts.nuisance,
            verbose=opts.verbose,
        )
        ihdp_path = os.path.join(opts.output_dir, f"ihdp_results_n{opts.n_replications}_{nuisance_tag}.npy")
        np.save(ihdp_path, ihdp_results)
        print(f"Saved IHDP results to: {ihdp_path}")

    if opts.dataset in ("jobs", "both"):
        print(f"\n=== Running Jobs (LaLonde/NSW, nuisance={nuisance_tag}) ===")
        jobs_results = run_jobs(nuisance=opts.nuisance, verbose=opts.verbose)
        jobs_path = os.path.join(opts.output_dir, f"jobs_results_{nuisance_tag}.npy")
        np.save(jobs_path, jobs_results)
        print(f"Saved Jobs results to: {jobs_path}")

    if opts.dataset == "jobs_obs":
        print(f"\n=== Running Jobs observational ({opts.comparison.upper()}, nuisance={nuisance_tag}) ===")
        obs_results = run_jobs_observational(
            comparison=opts.comparison,
            nuisance=opts.nuisance,
            verbose=opts.verbose,
        )
        obs_path = os.path.join(opts.output_dir, f"jobs_{opts.comparison}_obs_results_{nuisance_tag}.npy")
        np.save(obs_path, obs_results)
        print(f"Saved Jobs observational results to: {obs_path}")


if __name__ == "__main__":
    main()
