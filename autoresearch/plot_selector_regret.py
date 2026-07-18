#!/usr/bin/env python3
"""Chart the WS4 selector's performance vs fixed strategies (r12).

Grouped bars of RMS treatment-effect RMSE over the (eta,eps) grid for four
strategies: always-ICA, always-OML, the ε-kurtosis Selector, and the per-cell
oracle. Log y (range spans always-ICA's Gaussian-cell blow-up to the near-tie of
the rest). The Selector bar is highlighted; each bar is direct-labeled.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("autoresearch/results/r12_selector_eval")
TAU = 8.0


def main():
    sel, orc, ica, oml = [], [], [], []
    for f in glob.glob(str(R / "**" / "*.npy"), recursive=True):
        d = np.load(f, allow_pickle=True).item()
        if float(d["selector_tau"]) != TAU:
            continue
        rmse = np.asarray(d["rmse"], float)
        sel.append(d["selector_rmse"]); orc.append(d["oracle_fixed_rmse"])
        ica.append(rmse[4]); oml.append(rmse[0])
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    vals = [rms(ica), rms(oml), rms(sel), rms(orc)]
    labels = ["always\nICA", "always\nOML", "Selector\n(ε-kurtosis)", "per-cell\noracle"]
    # muted grey for references, one accent for the Selector, light grey for oracle
    colors = ["#9aa0a6", "#9aa0a6", "#2f6fb0", "#c7ccd1"]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bars = ax.bar(range(4), vals, color=colors, width=0.62, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("RMSE vs true θ  (RMS over the η×ε grid, log scale)", fontsize=9.5)
    ax.grid(axis="y", color="0.9", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.06, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if i == 2 else "normal",
                color="#1a1a1a")
    ax.text(0, vals[0] * 1.28, "24× worse\n(ICA blows up at\nGaussian-ε cells)",
            ha="center", va="bottom", fontsize=8, color="#7a2f2f")
    ax.set_title("WS4 selector matches the oracle and avoids ICA's blow-ups\n"
                 "linear PLR, synthetic X, n=20 000, d=5 · pick ICA if est. ε-kurtosis > 8",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"selector_regret.{ext}", dpi=160, bbox_inches="tight")
    print(f"saved -> {R/'selector_regret.png'}  vals={dict(zip(['ICA','OML','SEL','oracle'],[round(v,4) for v in vals]))}")


if __name__ == "__main__":
    main()
