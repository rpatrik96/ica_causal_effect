#!/usr/bin/env python3
"""Chart the multi-feature selector vs fixed strategies and the oracle (r13).

Bars of RMS treatment-effect RMSE over the completed regime-map cells for: the
four fixed estimators, the naive rule (v1), the n-gated rule (v2), and the
per-cell oracle-of-four. Log y (always-ICA blows up by 2 orders). v2 and oracle
highlighted.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import selector_multi as selm

R = Path("autoresearch/results/r13_multiselector_eval")


def main():
    rows = [np.load(f, allow_pickle=True).item()
            for f in glob.glob(str(R / "**" / "*.npy"), recursive=True)]
    ols, omll, omlg, ica, v1, v2, orc = ([] for _ in range(7))
    for d in rows:
        cr = {"OLS": d["cand_rmse_OLS"], "OML_lin": d["cand_rmse_OML_lin"],
              "OML_gbm": d["cand_rmse_OML_gbm"], "ICA": d["cand_rmse_ICA"]}
        ols.append(cr["OLS"]); omll.append(cr["OML_lin"]); omlg.append(cr["OML_gbm"]); ica.append(cr["ICA"])
        orc.append(min(cr.values()))
        v1.append(cr[d["selector_pick_mode"]])
        feats = {"nonlinearity_score": d["mean_nonlinearity"], "d_over_n": d["mean_d_over_n"],
                 "eps_excess_kurtosis": d["mean_eps_kurt"], "n": int(d["n_samples"]), "d": int(d["n_covariates"])}
        v2.append(cr[selm.select_estimator_multi(feats)])
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    series = [("always\nOLS", rms(ols), "#9aa0a6"), ("always\nOML(lin)", rms(omll), "#9aa0a6"),
              ("always\nOML(gbm)", rms(omlg), "#9aa0a6"), ("always\nICA", rms(ica), "#9aa0a6"),
              ("Selector v1\n(naive)", rms(v1), "#b0632f"),
              ("Selector v2\n(n-gated)", rms(v2), "#2f6fb0"),
              ("per-cell\noracle", rms(orc), "#c7ccd1")]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    vals = [s[1] for s in series]
    bars = ax.bar(range(len(series)), vals, color=[s[2] for s in series], width=0.66, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(0.05, max(vals) * 1.7)
    ax.set_xticks(range(len(series))); ax.set_xticklabels([s[0] for s in series], fontsize=9)
    ax.set_ylabel("RMSE vs true θ  (RMS over the regime map, log scale)", fontsize=9.5)
    ax.grid(axis="y", color="0.9", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals):
        bold = series[i][0].startswith("Selector v2")
        ax.text(i, v * 1.05, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold" if bold else "normal")
    ax.text(3, vals[3] * 1.18, "naive ICA\ndetonates", ha="center", va="bottom",
            fontsize=8, color="#7a2f2f")
    ax.annotate("n-gate fixes the\nsmall-n gbm overuse", xy=(5, vals[5]), xytext=(5, vals[5] * 6),
                ha="center", fontsize=8.2, color="#20507f",
                arrowprops=dict(arrowstyle="->", color="#20507f", lw=1.1))
    ax.set_title("Multi-feature selector spans the whole regime map\n"
                 "one rule (nonlinearity, d/n, ε-kurtosis, n) over {OLS, OML-lin, OML-gbm, ICA} · "
                 "12 regime corners", fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"multiselector.{ext}", dpi=160, bbox_inches="tight")
    print("saved -> multiselector.png ; RMS:",
          {s[0].replace(chr(10), ' '): round(s[1], 4) for s in series})


if __name__ == "__main__":
    main()
