#!/usr/bin/env python3
"""Monte-Carlo runner for the WS2 semi-synthetic benchmarks.

Loads real covariates (``semisynth_loaders.py``), pre-disentangles them, imposes
a partially linear model with a controllable ``η``, and runs the five headline
estimators (OLS, OML, HOML, ICA, matching — plus the two extra robust variants),
reporting per-method RMSE against the *exact* imposed ``θ``. Real covariate
structure, exact ground truth, full η control.

The output payload matches ``nonlinear_runner.py``'s schema (precomputed per-method
``rmse``/``biases``/``sigmas`` arrays in ``METHOD_NAMES`` order + metadata), so
``autoresearch/analyze_nonlinear_round.py`` extracts it unchanged.

Each Monte-Carlo experiment: subsample ``n_samples`` rows of the (once-)
pre-disentangled covariates, redraw ``η, ε`` (fixed PLR coefficients), estimate.
Row subsampling supplies sampling variability; the coefficients and θ are fixed.

Example::

    python semisynth_runner.py --dataset housing --n_components 10 \
        --eta_beta 1.0 --nuisance linear --n_experiments 25 --output_dir out/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from joblib import Parallel, delayed

from baselines import matching_baseline, ols_baseline
from main_estimation import all_together_cross_fitting
from nonlinear_runner import METHOD_NAMES, _make_nuisance_models
import semisynth_loaders as ssl

_DEFAULT_METHOD = "svd"  # for sparse text; overridden per-dataset below
DENSE_DATASETS = {"housing", "synthetic", "synthetic_hd"}


def _one_experiment(seed, Zfull, n_samples, theta, coefs, sigma_eps,
                    eta_beta, nuisance, nonlinear, eps_beta, bootstrap, eta_eps_corr):
    """Return the 7 per-method estimates for one MC replication."""
    m_coef, g_coef, m_quad, g_quad = coefs
    rng = np.random.default_rng(seed)
    # Without bootstrap, n is capped at the dataset size (subsample w/o replacement).
    # With bootstrap, n may exceed it (resample with replacement) — needed for the
    # n>>d probe on datasets smaller than the target n.
    replace = bool(bootstrap) or n_samples > Zfull.shape[0]
    n = n_samples if replace else min(n_samples, Zfull.shape[0])
    idx = rng.choice(Zfull.shape[0], size=n, replace=replace)
    Z = Zfull[idx]
    T, Y, _theta, m2, m3 = ssl.impose_plr(
        Z, theta, m_coef, g_coef, sigma_eps=sigma_eps, eta_beta=eta_beta, seed=seed,
        nonlinear=nonlinear, treatment_quad=m_quad, outcome_quad=g_quad, eps_beta=eps_beta,
        eta_eps_corr=eta_eps_corr,
    )
    mt, mo = _make_nuisance_models(nuisance)
    ortho_ml, robust_ml, robust_est, robust_split, _tc, _oc = all_together_cross_fitting(
        Z, T, Y, treatment_second_moment=m2, treatment_third_moment=m3,
        model_treatment=mt, model_outcome=mo,
    )
    try:
        from ica import ica_treatment_effect_estimation_eps_row

        ica_est, _ = ica_treatment_effect_estimation_eps_row(
            np.hstack((Z, T.reshape(-1, 1), Y.reshape(-1, 1))),
            S=None, check_convergence=False, verbose=False,
        )
        ica_value = float(ica_est[0]) if np.isfinite(ica_est).all() else float("nan")
    except Exception:  # noqa: BLE001  pylint: disable=broad-exception-caught
        ica_value = float("nan")
    ols_value = float(ols_baseline(Z, T, Y)[0])
    matching_value = float(matching_baseline(Z, T, Y, treatment_kind="continuous"))
    return (float(ortho_ml), float(robust_ml), float(robust_est), float(robust_split),
            ica_value, ols_value, matching_value)


def run(dataset, n_components, eta_beta, theta, sigma_eps, nuisance,
        n_samples, n_experiments, coef_scale, base_seed, n_jobs,
        nonlinear=False, eps_beta=None, bootstrap=False, eta_eps_corr=0.0):
    method = "pca" if dataset in DENSE_DATASETS else "svd"
    Xreal = ssl.load_covariates(dataset)
    Zfull = ssl.predisentangle(Xreal, n_components=n_components, method=method)
    d = Zfull.shape[1]
    coefs = ssl.make_coefficients(d, seed=base_seed, scale=coef_scale)
    # actual per-experiment sample size (bootstrap allows n > dataset rows)
    actual_n = n_samples if (bootstrap or n_samples > Zfull.shape[0]) else min(n_samples, Zfull.shape[0])

    seeds = [base_seed + 1 + i for i in range(n_experiments)]
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one_experiment)(s, Zfull, n_samples, theta, coefs, sigma_eps,
                                 eta_beta, nuisance, nonlinear, eps_beta, bootstrap, eta_eps_corr)
        for s in seeds
    )
    estimates = np.asarray(results, dtype=float)  # (n_exp, 7)
    finite_mask = np.isfinite(estimates).all(axis=1)
    finite = estimates[finite_mask]
    err = estimates - theta
    with np.errstate(invalid="ignore"):
        rmse = np.sqrt(np.nanmean(err**2, axis=0))
        biases = np.nanmean(err, axis=0)
        sigmas = np.nanstd(estimates, axis=0)
    finite_per_col = np.isfinite(estimates).sum(axis=0)
    print(f"dataset={dataset} d'={d} n={actual_n} bootstrap={bootstrap} nonlinear={nonlinear} "
          f"eta_beta={eta_beta} nuisance={nuisance} theta={theta}")
    print(f"{'method':<22}{'bias':>10}{'sigma':>10}{'rmse':>10}")
    for name, b, s, r in zip(METHOD_NAMES, biases, sigmas, rmse):
        print(f"{name:<22}{b:>10.4f}{s:>10.4f}{r:>10.4f}")
    return {
        "method_names": np.array(METHOD_NAMES),
        "estimates": estimates,
        "estimates_finite": finite,
        "biases": biases,
        "sigmas": sigmas,
        "rmse": rmse,
        "n_experiments": int(finite_mask.sum()),
        "finite_per_method": finite_per_col,
        "n_attempted": len(seeds),
        "treatment_effect": float(theta),
        "n_samples": int(actual_n),
        "n_covariates": int(d),
        "support_size": int(d),
        "nuisance": nuisance,
        # reuse the nonlinear schema fields the analyzer reads:
        "nonlinear_confounding": bool(nonlinear),
        "heavy_tail_eta": True,
        "eta_beta": float(eta_beta),
        "high_dim": False,
        "heteroscedastic_eps": False,
        "sigma_eta": 1.0,
        "sigma_outcome": float(sigma_eps),
        # WS2-specific provenance:
        "dataset": dataset,
        "predisentangle_method": method,
        "n_components": int(n_components),
        "eps_beta": (float(eps_beta) if eps_beta is not None else None),
        "bootstrap": bool(bootstrap),
        "eta_eps_corr": float(eta_eps_corr),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(ssl.LOADERS))
    p.add_argument("--n_components", type=int, default=10)
    p.add_argument("--eta_beta", type=float, default=1.0)
    p.add_argument("--treatment_effect", type=float, default=1.0)
    p.add_argument("--sigma_outcome", type=float, default=1.0)
    p.add_argument("--nuisance", default="linear", choices=["linear", "gbm", "poly", "rf"])
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_experiments", type=int, default=25)
    p.add_argument("--coef_scale", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=13337)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--nonlinear", action="store_true", help="nonlinear m(X)/g(X) recast")
    p.add_argument("--eps_beta", type=float, default=None, help="heavy-tailed eps gennorm beta")
    p.add_argument("--bootstrap", action="store_true", help="resample rows w/ replacement (n>dataset)")
    p.add_argument("--eta_eps_corr", type=float, default=0.0, help="WS3: Corr(eta,eps) violation rho")
    p.add_argument("--output_dir", default="figures/semisynth")
    p.add_argument("--results_file", default=None)
    opts = p.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)
    results = run(
        dataset=opts.dataset, n_components=opts.n_components, eta_beta=opts.eta_beta,
        theta=opts.treatment_effect, sigma_eps=opts.sigma_outcome, nuisance=opts.nuisance,
        n_samples=opts.n_samples, n_experiments=opts.n_experiments,
        coef_scale=opts.coef_scale, base_seed=opts.base_seed, n_jobs=opts.n_jobs,
        nonlinear=opts.nonlinear, eps_beta=opts.eps_beta, bootstrap=opts.bootstrap,
        eta_eps_corr=opts.eta_eps_corr,
    )
    nl = "nl" if opts.nonlinear else "lin"
    fname = opts.results_file or (
        f"semisynth_{opts.dataset}_{nl}_d{opts.n_components}_ht{opts.eta_beta}_"
        f"{opts.nuisance}_n{opts.n_samples}_rho{opts.eta_eps_corr}.npy"
    )
    out = os.path.join(opts.output_dir, fname)
    np.save(out, results)
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
