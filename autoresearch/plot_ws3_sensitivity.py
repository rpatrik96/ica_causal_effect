#!/usr/bin/env python3
"""WS3 sensitivity curves: eta-Gaussianisation and eta-eps dependence.

Two figures. (1) ws3_gauss_eta: RMSE vs eta_beta at heavy vs Gaussian eps (log y),
showing ICA graceful when eps is non-Gaussian but abrupt at the Gaussian wall, and
HOML detonating at Gaussian eta. (2) ws3_eta_eps_dep: RMSE vs rho=Corr(eta,eps),
showing every estimator biases ~linearly (RMSE = rho reference line).
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("autoresearch/results")
# Paul Tol bright palette (CVD-safe), fixed estimator order.
COL = {"OLS": "#4477AA", "OML": "#228833", "HOML": "#CCBB44", "ICA": "#EE6677", "matching": "#AA3377"}
IDX = {"OML": 0, "HOML": 1, "ICA": 4, "OLS": 5, "matching": 6}


def load(round_name, keyfn):
    G = {}
    for f in glob.glob(str(RES / round_name / "**" / "*.npy"), recursive=True):
        d = np.load(f, allow_pickle=True).item()
        r = np.asarray(d["rmse"], float)
        G[keyfn(d)] = {m: r[i] for m, i in IDX.items()}
    return G


def gauss_eta():
    G = load("r16_gauss_eta", lambda d: (round(float(d["eps_beta"]), 2), round(float(d["eta_beta"]), 2)))
    etas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (eps, title) in zip(axes, [(0.5, "heavy-tailed ε (β=0.5)"),
                                        (2.0, "Gaussian ε (β=2)")]):
        for m in ("OLS", "OML", "HOML", "ICA", "matching"):
            ax.plot(etas, [G[(eps, e)][m] for e in etas], "-o", ms=4, lw=1.8,
                    color=COL[m], label=m)
        ax.set_yscale("log"); ax.axvline(2.0, color="0.6", ls=":", lw=1)
        ax.set_xlabel("η shape  β   (2 = Gaussian →)", fontsize=9.5)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", color="0.92")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("treatment-effect RMSE (log)", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
    axes[1].annotate("ICA non-identifiability\nwall (both noises Gaussian)",
                     xy=(2.0, G[(2.0, 2.0)]["ICA"]), xytext=(1.15, 0.9), fontsize=8.2,
                     color="#a3384a", arrowprops=dict(arrowstyle="->", color="#a3384a"))
    fig.suptitle("WS3 — sensitivity to Gaussianising η: OLS/OML flat (robust); "
                 "ICA graceful iff ε non-Gaussian; HOML detonates at Gaussian η", fontsize=10.5)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RES / "r16_gauss_eta" / f"ws3_gauss_eta.{e}", dpi=160, bbox_inches="tight")
    print("saved ws3_gauss_eta")


def eta_eps_dep():
    G = load("r16_eta_eps_dep", lambda d: round(float(d["eta_eps_corr"]), 2))
    rhos = sorted(G)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(rhos, rhos, "--", color="0.6", lw=1.2, label="RMSE = ρ (pure bias)")
    for m in ("OLS", "OML", "HOML", "ICA", "matching"):
        ax.plot(rhos, [G[r][m] for r in rhos], "-o", ms=4, lw=1.8, color=COL[m], label=m)
    ax.set_xlabel("ρ = Corr(η, ε)   (independence violation)", fontsize=9.5)
    ax.set_ylabel("treatment-effect RMSE", fontsize=9.5)
    ax.grid(True, color="0.92")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.6, ncol=2, loc="upper left")
    ax.set_title("WS3 — sensitivity to η–ε dependence: every estimator biases ~linearly in ρ\n"
                 "no method is robust — violating exogeneity changes the estimand", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RES / "r16_eta_eps_dep" / f"ws3_eta_eps_dep.{e}", dpi=160, bbox_inches="tight")
    print("saved ws3_eta_eps_dep")


if __name__ == "__main__":
    gauss_eta()
    eta_eps_dep()
