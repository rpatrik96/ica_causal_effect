#!/usr/bin/env python3
"""Evaluate the WS4 estimator selector (selector.py) on the semi-synthetic PLR.

For each config (dataset, eta_beta, eps_beta, n), over `n_experiments` draws it
computes the candidate estimates (ICA, OML, OLS), the selection features
(estimated ε/η excess kurtosis), and the selector's per-experiment pick, then
reports:
  * RMSE of always-ICA, always-OML, always-OLS
  * selector RMSE (per-experiment choice via the ε-kurtosis rule)
  * oracle_fixed = min(ICA, OML) RMSE  (best single estimator for this config)
  * regret = selector_rmse - oracle_fixed   (WS4 target: ~0)
  * ICA pick-rate and mean estimated ε/η kurtosis (for τ calibration)

Payload is analyzer-compatible (per-method `rmse` array + metadata) with extra
selector.* fields. Example:
    python selector_runner.py --dataset synthetic --eta_beta 1.0 --eps_beta 0.5 \
        --n_samples 20000 --tau 1.0 --output_dir out/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from joblib import Parallel, delayed

from baselines import ols_baseline
from ica import ica_treatment_effect_estimation_eps_row
from main_estimation import all_together_cross_fitting
from nonlinear_runner import METHOD_NAMES, _make_nuisance_models
import selector as sel
import semisynth_loaders as ssl

DENSE = {"housing", "synthetic"}


def _one(seed, Zfull, n_samples, theta, coefs, sigma_eps, eta_beta, nuisance,
         eps_beta, bootstrap, tau):
    m_coef, g_coef, m_quad, g_quad = coefs
    rng = np.random.default_rng(seed)
    replace = bool(bootstrap) or n_samples > Zfull.shape[0]
    n = n_samples if replace else min(n_samples, Zfull.shape[0])
    Z = Zfull[rng.choice(Zfull.shape[0], size=n, replace=replace)]
    T, Y, _th, m2, m3 = ssl.impose_plr(Z, theta, m_coef, g_coef, sigma_eps=sigma_eps,
                                       eta_beta=eta_beta, seed=seed, eps_beta=eps_beta)
    mt, mo = _make_nuisance_models(nuisance)
    oml, rob, est, split, _tc, _oc = all_together_cross_fitting(
        Z, T, Y, treatment_second_moment=m2, treatment_third_moment=m3,
        model_treatment=mt, model_outcome=mo)
    try:
        ica_est, _ = ica_treatment_effect_estimation_eps_row(
            np.hstack((Z, T.reshape(-1, 1), Y.reshape(-1, 1))), S=None,
            check_convergence=False, verbose=False)
        ica = float(ica_est[0]) if np.isfinite(ica_est).all() else np.nan
    except Exception:  # noqa: BLE001  pylint: disable=broad-exception-caught
        ica = np.nan
    ols = float(ols_baseline(Z, T, Y)[0])
    feats = sel.selection_features(Z, T, Y)
    name = sel.select_estimator(feats, tau)
    ests = {"ICA": ica, "OML": float(oml), "OLS": ols}
    return {"ica": ica, "oml": float(oml), "ols": ols, "rob": float(rob),
            "est": float(est), "split": float(split),
            "pick": name, "chosen": ests[name],
            "eps_kurt": feats["eps_excess_kurtosis"], "eta_kurt": feats["eta_excess_kurtosis"]}


def _rmse(v, theta):
    v = np.asarray(v, float)
    return float(np.sqrt(np.nanmean((v - theta) ** 2)))


def run(dataset, n_components, eta_beta, eps_beta, theta, sigma_eps, nuisance,
        n_samples, n_experiments, coef_scale, base_seed, n_jobs, tau, bootstrap):
    method = "pca" if dataset in DENSE else "svd"
    Zfull = ssl.predisentangle(ssl.load_covariates(dataset), n_components=n_components, method=method)
    d = Zfull.shape[1]
    coefs = ssl.make_coefficients(d, seed=base_seed, scale=coef_scale)
    actual_n = n_samples if (bootstrap or n_samples > Zfull.shape[0]) else min(n_samples, Zfull.shape[0])
    seeds = [base_seed + 1 + i for i in range(n_experiments)]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one)(s, Zfull, n_samples, theta, coefs, sigma_eps, eta_beta,
                      nuisance, eps_beta, bootstrap, tau) for s in seeds)

    ica = [r["ica"] for r in rows]; oml = [r["oml"] for r in rows]; ols = [r["ols"] for r in rows]
    chosen = [r["chosen"] for r in rows]
    ica_rmse, oml_rmse, ols_rmse = _rmse(ica, theta), _rmse(oml, theta), _rmse(ols, theta)
    sel_rmse = _rmse(chosen, theta)
    oracle_fixed = min(ica_rmse, oml_rmse)
    pick_ica = float(np.mean([r["pick"] == "ICA" for r in rows]))
    # per-method rmse array in METHOD_NAMES order (OML,HOML,est,split,ICA,OLS,Matching)
    rmse_arr = np.array([oml_rmse, _rmse([r["rob"] for r in rows], theta),
                         _rmse([r["est"] for r in rows], theta),
                         _rmse([r["split"] for r in rows], theta),
                         ica_rmse, ols_rmse, np.nan])
    print(f"{dataset} eta={eta_beta} eps={eps_beta} n={actual_n} tau={tau}: "
          f"ICA={ica_rmse:.4f} OML={oml_rmse:.4f} SELECTOR={sel_rmse:.4f} "
          f"oracle={oracle_fixed:.4f} regret={sel_rmse-oracle_fixed:+.4f} "
          f"pick_ICA={pick_ica:.2f} eps_kurt={np.nanmean([r['eps_kurt'] for r in rows]):.2f}")
    return {
        "method_names": np.array(METHOD_NAMES), "rmse": rmse_arr,
        "biases": np.full(7, np.nan), "sigmas": np.full(7, np.nan),
        "n_experiments": int(np.sum(np.isfinite(ica))), "n_attempted": len(seeds),
        "finite_per_method": np.full(7, len(seeds)),
        "treatment_effect": float(theta), "n_samples": int(actual_n),
        "n_covariates": int(d), "support_size": int(d), "nuisance": nuisance,
        "nonlinear_confounding": False, "heavy_tail_eta": True,
        "eta_beta": float(eta_beta), "eps_beta": float(eps_beta),
        "high_dim": False, "heteroscedastic_eps": False, "sigma_eta": 1.0,
        "sigma_outcome": float(sigma_eps), "dataset": dataset,
        "predisentangle_method": method, "n_components": int(n_components),
        "bootstrap": bool(bootstrap),
        # selector-specific
        "selector_tau": float(tau), "selector_rmse": sel_rmse,
        "oracle_fixed_rmse": oracle_fixed, "selector_regret": sel_rmse - oracle_fixed,
        "selector_pick_ica_rate": pick_ica,
        "mean_eps_kurt": float(np.nanmean([r["eps_kurt"] for r in rows])),
        "mean_eta_kurt": float(np.nanmean([r["eta_kurt"] for r in rows])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(ssl.LOADERS))
    p.add_argument("--n_components", type=int, default=5)
    p.add_argument("--eta_beta", type=float, default=1.0)
    p.add_argument("--eps_beta", type=float, default=1.0)
    p.add_argument("--treatment_effect", type=float, default=1.0)
    p.add_argument("--sigma_outcome", type=float, default=1.0)
    p.add_argument("--nuisance", default="linear", choices=["linear", "gbm", "poly", "rf"])
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--n_experiments", type=int, default=25)
    p.add_argument("--coef_scale", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=13337)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--tau", type=float, default=sel.DEFAULT_TAU)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--output_dir", default="figures/selector")
    p.add_argument("--results_file", default=None)
    o = p.parse_args()
    os.makedirs(o.output_dir, exist_ok=True)
    res = run(o.dataset, o.n_components, o.eta_beta, o.eps_beta, o.treatment_effect,
              o.sigma_outcome, o.nuisance, o.n_samples, o.n_experiments, o.coef_scale,
              o.base_seed, o.n_jobs, o.tau, o.bootstrap)
    fname = o.results_file or (f"selector_{o.dataset}_eta{o.eta_beta}_eps{o.eps_beta}_"
                               f"n{o.n_samples}_tau{o.tau}.npy")
    np.save(os.path.join(o.output_dir, fname), res)
    print(f"Saved -> {os.path.join(o.output_dir, fname)}")


if __name__ == "__main__":
    main()
