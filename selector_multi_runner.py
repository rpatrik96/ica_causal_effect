#!/usr/bin/env python3
"""Evaluate the multi-feature selector (selector_multi.py) across the regime map.

Per config, over `n_experiments` draws it computes FOUR candidate estimates —
OLS, OML(linear nuisance), OML(gbm nuisance), ICA — plus the selector features and
pick, then reports each candidate's RMSE, the selector RMSE, the oracle-of-four
RMSE (best single candidate for the config), regret, and the pick distribution.

Covers the whole campaign map by sweeping: nonlinear {F,T} × ε {heavy,Gaussian} ×
d {low, high} × n {small, large}. Uses synthetic_hd covariates (30000×400) so the
high-d cells reach d ≳ n.

Payload is analyzer-compatible (per-method `rmse` in METHOD_NAMES order with the
four candidates slotted in) plus selector.* fields.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

from baselines import ols_baseline
from ica import ica_treatment_effect_estimation_eps_row
from main_estimation import all_together_cross_fitting
from nonlinear_runner import METHOD_NAMES, _make_nuisance_models
import selector_multi as selm
import semisynth_loaders as ssl

DENSE = {"housing", "synthetic", "synthetic_hd"}


def _oml(Z, T, Y, m2, m3, nuisance):
    mt, mo = _make_nuisance_models(nuisance)
    oml, *_ = all_together_cross_fitting(
        Z, T, Y, treatment_second_moment=m2, treatment_third_moment=m3,
        model_treatment=mt, model_outcome=mo)
    return float(oml)


def _one(seed, Zfull, n_samples, theta, coefs, sigma_eps, eta_beta, eps_beta,
         nonlinear, bootstrap, thresholds):
    m_coef, g_coef, m_quad, g_quad = coefs
    rng = np.random.default_rng(seed)
    replace = bool(bootstrap) or n_samples > Zfull.shape[0]
    n = n_samples if replace else min(n_samples, Zfull.shape[0])
    Z = Zfull[rng.choice(Zfull.shape[0], size=n, replace=replace)]
    T, Y, _th, m2, m3 = ssl.impose_plr(
        Z, theta, m_coef, g_coef, sigma_eps=sigma_eps, eta_beta=eta_beta, seed=seed,
        nonlinear=nonlinear, treatment_quad=m_quad, outcome_quad=g_quad, eps_beta=eps_beta)

    cand = {"OLS": float(ols_baseline(Z, T, Y)[0]),
            "OML_lin": _oml(Z, T, Y, m2, m3, "linear"),
            "OML_gbm": _oml(Z, T, Y, m2, m3, "gbm")}
    try:
        ica_est, _ = ica_treatment_effect_estimation_eps_row(
            np.hstack((Z, T.reshape(-1, 1), Y.reshape(-1, 1))), S=None,
            check_convergence=False, verbose=False)
        cand["ICA"] = float(ica_est[0]) if np.isfinite(ica_est).all() else np.nan
    except Exception:  # noqa: BLE001  pylint: disable=broad-exception-caught
        cand["ICA"] = np.nan

    feats = selm.multi_features(Z, T, Y, seed=seed)
    pick = selm.select_estimator_multi(feats, thresholds)
    return {"cand": cand, "pick": pick, "chosen": cand[pick], "feats": feats}


def _rmse(v, theta):
    return float(np.sqrt(np.nanmean((np.asarray(v, float) - theta) ** 2)))


def run(dataset, n_components, eta_beta, eps_beta, theta, sigma_eps, n_samples,
        n_experiments, base_seed, n_jobs, nonlinear, bootstrap, thresholds):
    method = "pca" if dataset in DENSE else "svd"
    Zfull = ssl.predisentangle(ssl.load_covariates(dataset), n_components=n_components, method=method)
    d = Zfull.shape[1]
    coefs = ssl.make_coefficients(d, seed=base_seed)
    actual_n = n_samples if (bootstrap or n_samples > Zfull.shape[0]) else min(n_samples, Zfull.shape[0])
    seeds = [base_seed + 1 + i for i in range(n_experiments)]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one)(s, Zfull, n_samples, theta, coefs, sigma_eps, eta_beta,
                      eps_beta, nonlinear, bootstrap, thresholds) for s in seeds)

    cand_rmse = {c: _rmse([r["cand"][c] for r in rows], theta) for c in selm.CANDIDATES}
    sel_rmse = _rmse([r["chosen"] for r in rows], theta)
    oracle = min(cand_rmse.values())
    oracle_name = min(cand_rmse, key=cand_rmse.get)
    picks = Counter(r["pick"] for r in rows)
    mode_pick = picks.most_common(1)[0][0]
    mf = {k: float(np.nanmean([r["feats"][k] for r in rows]))
          for k in ("eps_excess_kurtosis", "nonlinearity_score", "d_over_n")}
    # per-method rmse array in METHOD_NAMES order: OML(=OML_lin),HOML,est,split,ICA,OLS,Matching
    rmse_arr = np.array([cand_rmse["OML_lin"], np.nan, np.nan, np.nan,
                         cand_rmse["ICA"], cand_rmse["OLS"], np.nan])
    regime = f"{'NL' if nonlinear else 'lin'}/eps{eps_beta}/d{d}/n{actual_n}"
    print(f"[{regime}] cand={{ {' '.join(f'{c}:{v:.4f}' for c,v in cand_rmse.items())} }} "
          f"| SELECTOR={sel_rmse:.4f} pick={mode_pick} oracle={oracle:.4f}({oracle_name}) "
          f"regret={sel_rmse-oracle:+.4f} | feats nl={mf['nonlinearity_score']:.3f} "
          f"d/n={mf['d_over_n']:.3f} epsk={mf['eps_excess_kurtosis']:.1f}")
    return {
        "method_names": np.array(METHOD_NAMES), "rmse": rmse_arr,
        "biases": np.full(7, np.nan), "sigmas": np.full(7, np.nan),
        "n_experiments": n_experiments, "n_attempted": len(seeds),
        "finite_per_method": np.full(7, len(seeds)),
        "treatment_effect": float(theta), "n_samples": int(actual_n),
        "n_covariates": int(d), "support_size": int(d), "nuisance": "multi",
        "nonlinear_confounding": bool(nonlinear), "heavy_tail_eta": True,
        "eta_beta": float(eta_beta), "eps_beta": float(eps_beta),
        "high_dim": bool(d >= 100), "heteroscedastic_eps": False, "sigma_eta": 1.0,
        "sigma_outcome": float(sigma_eps), "dataset": dataset,
        "predisentangle_method": method, "n_components": int(n_components),
        "bootstrap": bool(bootstrap),
        # multi-selector fields
        "cand_rmse_OLS": cand_rmse["OLS"], "cand_rmse_OML_lin": cand_rmse["OML_lin"],
        "cand_rmse_OML_gbm": cand_rmse["OML_gbm"], "cand_rmse_ICA": cand_rmse["ICA"],
        "selector_rmse": sel_rmse, "oracle_rmse": oracle, "oracle_name": oracle_name,
        "selector_regret": sel_rmse - oracle, "selector_pick_mode": mode_pick,
        "selector_pick_counts": dict(picks),
        "mean_nonlinearity": mf["nonlinearity_score"], "mean_d_over_n": mf["d_over_n"],
        "mean_eps_kurt": mf["eps_excess_kurtosis"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="synthetic_hd", choices=sorted(ssl.LOADERS))
    p.add_argument("--n_components", type=int, default=5)
    p.add_argument("--eta_beta", type=float, default=1.0)
    p.add_argument("--eps_beta", type=float, default=1.0)
    p.add_argument("--treatment_effect", type=float, default=1.0)
    p.add_argument("--sigma_outcome", type=float, default=1.0)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--n_experiments", type=int, default=15)
    p.add_argument("--base_seed", type=int, default=13337)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--nonlinear", action="store_true")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--tau_nl", type=float, default=selm.DEFAULT_THRESHOLDS["nl"])
    p.add_argument("--tau_dn", type=float, default=selm.DEFAULT_THRESHOLDS["dn"])
    p.add_argument("--tau_eps", type=float, default=selm.DEFAULT_THRESHOLDS["eps"])
    p.add_argument("--output_dir", default="figures/selector_multi")
    p.add_argument("--results_file", default=None)
    o = p.parse_args()
    os.makedirs(o.output_dir, exist_ok=True)
    thresholds = {"nl": o.tau_nl, "dn": o.tau_dn, "eps": o.tau_eps}
    res = run(o.dataset, o.n_components, o.eta_beta, o.eps_beta, o.treatment_effect,
              o.sigma_outcome, o.n_samples, o.n_experiments, o.base_seed, o.n_jobs,
              o.nonlinear, o.bootstrap, thresholds)
    nl = "NL" if o.nonlinear else "lin"
    fname = o.results_file or (f"selmulti_{o.dataset}_{nl}_d{o.n_components}_"
                               f"eps{o.eps_beta}_n{o.n_samples}.npy")
    np.save(os.path.join(o.output_dir, fname), res)
    print(f"Saved -> {os.path.join(o.output_dir, fname)}")


if __name__ == "__main__":
    main()
