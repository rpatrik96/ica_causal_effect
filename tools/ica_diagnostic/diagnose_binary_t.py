"""Diagnostic: why does FastICA blow up when the treatment T is binary?

Reproduces a single binary-T sample, builds the observed and source matrices,
runs FastICA in several configurations, and inspects per-component kurtosis
and the eps-row identification logic from
``ica.ica_treatment_effect_estimation_eps_row``.
"""

# ruff: noqa: E402, E501, F841
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from binary_treatment_dgp import BinaryTreatmentDGPConfig, generate_binary_treatment_data
from ica import ica_treatment_effect_estimation, ica_treatment_effect_estimation_eps_row


def banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def main() -> None:
    cfg = BinaryTreatmentDGPConfig(
        n_samples=2000,
        n_covariates=10,
        support_size=5,
        treatment_effect=1.5,
        propensity_strength=0.7,
        outcome_coef_scale=0.5,
        sigma_outcome=0.5,
        seed=2024,
    )
    X_cov, T, Y, propensity, eta, alpha, beta = generate_binary_treatment_data(cfg)

    n, d = X_cov.shape
    theta = cfg.treatment_effect

    # Observed matrix [X, T, Y]  (n x 12)
    obs = np.column_stack([X_cov, T, Y])
    # Ground-truth sources [X, eta, eps] -- treat continuous X as their own sources
    eps = Y - theta * T - X_cov @ beta
    sources = np.column_stack([X_cov, eta, eps])

    banner("DGP summary")
    print(f"n={n} d={d}, theta={theta}")
    print(f"T mean={T.mean():.4f}, var={T.var():.4f} (Bernoulli => var<=0.25)")
    print(f"Y mean={Y.mean():.4f}, var={Y.var():.4f}")
    print(f"propensity range: [{propensity.min():.4f}, {propensity.max():.4f}]")
    print(f"eta mean={eta.mean():.4e}, var={eta.var():.4f}, " f"kurt(fisher)={kurtosis(eta, fisher=True):.4f}")
    print(f"eps mean={eps.mean():.4e}, var={eps.var():.4f}, " f"kurt(fisher)={kurtosis(eps, fisher=True):.4f}")
    print(
        f"T  kurt(fisher)={kurtosis(T, fisher=True):.4f}  "
        "(Bernoulli p~0.5 => kurt = -2; p far from 0.5 => large positive)"
    )

    # Per-source kurtosis
    banner("Per-source kurtosis (ground-truth S)")
    for j in range(sources.shape[1]):
        name = f"X_{j}" if j < d else ("eta" if j == d else "eps")
        print(f"  src[{j:2d}] {name:>5s}  kurt={kurtosis(sources[:, j], fisher=True):+.4f}")

    # Per-observed-column kurtosis
    banner("Per-observed-column kurtosis ([X | T | Y])")
    for j in range(obs.shape[1]):
        name = f"X_{j}" if j < d else ("T" if j == d else "Y")
        print(f"  obs[{j:2d}] {name:>3s}  kurt={kurtosis(obs[:, j], fisher=True):+.4f}")

    # Run FastICA on observed [X, T, Y]
    banner("FastICA on [X, T, Y] (logcosh, seed=0)")
    ica = FastICA(
        n_components=obs.shape[1],
        random_state=0,
        max_iter=1000,
        whiten="unit-variance",
        tol=1e-4,
        fun="logcosh",
    )
    S_hat = ica.fit_transform(obs)
    A = ica.mixing_  # shape (12, 12), X_obs ~ A @ S_hat (centred)
    W = ica.components_  # shape (12, 12), S_hat ~ X_obs @ W.T

    print(f"S_hat shape: {S_hat.shape}")
    abs_kurt = np.array([np.abs(kurtosis(S_hat[:, j], fisher=True)) for j in range(S_hat.shape[1])])
    raw_kurt = np.array([kurtosis(S_hat[:, j], fisher=True) for j in range(S_hat.shape[1])])

    print("Recovered components (sorted by |kurt| desc):")
    order = np.argsort(-abs_kurt)
    for j in order:
        # Project on each ground-truth source: which is closest?
        corrs = np.array([np.corrcoef(S_hat[:, j], sources[:, k])[0, 1] for k in range(sources.shape[1])])
        best = int(np.argmax(np.abs(corrs)))
        best_name = f"X_{best}" if best < d else ("eta" if best == d else "eps")
        print(
            f"  comp[{j:2d}] kurt={raw_kurt[j]:+8.3f}  |kurt|={abs_kurt[j]:7.3f}  "
            f"max|corr|={np.abs(corrs[best]):.3f} with {best_name}"
        )

    eps_idx = int(np.argmax(abs_kurt))
    print(f"\neps_row picker chose: comp[{eps_idx}] (|kurt|={abs_kurt[eps_idx]:.3f})")

    # What does THAT component correspond to vs the ground-truth eps?
    eps_col = d + 1
    corrs_eps = np.array([np.corrcoef(S_hat[:, j], sources[:, eps_col])[0, 1] for j in range(S_hat.shape[1])])
    eta_col = d
    corrs_eta = np.array([np.corrcoef(S_hat[:, j], sources[:, eta_col])[0, 1] for j in range(S_hat.shape[1])])
    true_eps_match = int(np.argmax(np.abs(corrs_eps)))
    true_eta_match = int(np.argmax(np.abs(corrs_eta)))
    print(f"True eps best matches comp[{true_eps_match}] (|corr|={np.abs(corrs_eps[true_eps_match]):.3f})")
    print(f"True eta best matches comp[{true_eta_match}] (|corr|={np.abs(corrs_eta[true_eta_match]):.3f})")

    # theta from each candidate row (using the production formula)
    n_cov = obs.shape[1] - 1 - 1  # n_treatments=1

    def theta_from_row(idx: int) -> float:
        w = W[idx, :]
        wn = w / w[-1]
        return float(-wn[n_cov : n_cov + 1])

    print(f"\nTheta from picked row  (comp {eps_idx}): {theta_from_row(eps_idx):+.4f}  (true theta = {theta})")
    print(f"Theta from true-eps row(comp {true_eps_match}): {theta_from_row(true_eps_match):+.4f}")
    print(f"Theta from true-eta row(comp {true_eta_match}): {theta_from_row(true_eta_match):+.4f}")
    print("\nTheta from EVERY row:")
    for j in range(W.shape[0]):
        print(
            f"  comp[{j:2d}]  theta_hat={theta_from_row(j):+12.4f}  "
            f"|kurt|={abs_kurt[j]:7.3f}  W[j,-1]={W[j,-1]:+.4f}"
        )

    # Run the production eps-row routine
    banner("Production: ica_treatment_effect_estimation_eps_row")
    theta_eps_row, mcc_eps = ica_treatment_effect_estimation_eps_row(
        obs, S=sources, random_state=0, n_treatments=1, verbose=True, fun="logcosh"
    )
    print(f"theta_hat (eps_row) = {theta_eps_row} ; MCC = {mcc_eps}")

    # Run the Munkres-matched routine with ground-truth sources
    banner("Production: ica_treatment_effect_estimation (Munkres-matched)")
    theta_munkres, mcc_munkres = ica_treatment_effect_estimation(
        obs, S=sources, random_state=0, n_treatments=1, verbose=True, fun="logcosh"
    )
    print(f"theta_hat (munkres) = {theta_munkres} ; MCC = {mcc_munkres}")

    # Sweep seeds and contrast functions
    banner("Sweep: seeds 0..4 x fun in {logcosh, exp, cube}, both routines")
    print(
        f"{'seed':>4s} {'fun':>8s} | {'theta_eps':>12s} {'theta_munkres':>14s} {'MCC_munk':>9s} {'eps_idx':>7s} {'|kurt|max':>10s}"
    )
    for fun in ("logcosh", "exp", "cube"):
        for seed in range(5):
            try:
                t_eps, _ = ica_treatment_effect_estimation_eps_row(
                    obs, S=sources, random_state=seed, n_treatments=1, verbose=False, fun=fun
                )
            except Exception:  # noqa: BLE001
                t_eps = np.array([np.nan])
                _ = None
            try:
                t_mun, mcc_mun = ica_treatment_effect_estimation(
                    obs, S=sources, random_state=seed, n_treatments=1, verbose=False, fun=fun
                )
            except Exception:  # noqa: BLE001
                t_mun, mcc_mun = np.array([np.nan]), None

            # Re-fit once to grab the eps_idx + |kurt| max for diagnostics
            try:
                ica_local = FastICA(
                    n_components=obs.shape[1],
                    random_state=seed,
                    max_iter=1000,
                    whiten="unit-variance",
                    tol=1e-4,
                    fun=fun,
                )
                Sh = ica_local.fit_transform(obs)
                ak = np.array([np.abs(kurtosis(Sh[:, j], fisher=True)) for j in range(Sh.shape[1])])
                eps_idx_local = int(np.argmax(ak))
                kmax = ak[eps_idx_local]
            except Exception:  # noqa: BLE001
                eps_idx_local, kmax = -1, np.nan

            mcc_str = f"{mcc_mun:.3f}" if mcc_mun is not None and not np.isnan(mcc_mun) else "  nan"
            print(
                f"{seed:>4d} {fun:>8s} | "
                f"{float(t_eps[0]):>12.4f} {float(t_mun[0]):>14.4f} "
                f"{mcc_str:>9s} {eps_idx_local:>7d} {kmax:>10.3f}"
            )

    # Sanity: regress eta vs T to confirm binary-induced extreme kurtosis
    banner("Sanity: T binary => kurt(T) is large; FastICA whitening hits this")
    print(
        f"  E[T]={T.mean():.4f}; var(T)={T.var():.4f}; "
        f"theoretical kurt(Bernoulli p) = (1-6p(1-p))/(p(1-p)) - relative to fisher offset"
    )
    p_emp = float(T.mean())
    if 0 < p_emp < 1:
        # excess kurtosis of Bernoulli(p): (1 - 6 p (1-p)) / (p (1-p))
        bern_kurt = (1.0 - 6.0 * p_emp * (1.0 - p_emp)) / (p_emp * (1.0 - p_emp))
        print(f"  closed-form excess kurt(Bernoulli p={p_emp:.3f}) = {bern_kurt:+.4f}")
        print(f"  empirical excess kurt(T)                    = {kurtosis(T, fisher=True):+.4f}")
        print(f"  empirical excess kurt(eps)                  = {kurtosis(eps, fisher=True):+.4f}")
        print(f"  empirical excess kurt(eta=T-p(X))           = {kurtosis(eta, fisher=True):+.4f}")


if __name__ == "__main__":
    main()
