#!/usr/bin/env python3
"""Chart the ICA-win frontier from r10_ica_win_frontier.

Diverging heatmap of the ICA/OLS treatment-effect RMSE ratio over the
(eta_beta × eps_beta) non-Gaussianity grid. Ratio<1 (blue) = ICA beats OLS;
ratio>1 (red) = ICA loses. Colour is log2(ratio) so the map is symmetric about
the neutral midpoint (ratio=1); cells are annotated with the actual ratio and
the cells where ICA is best of ALL estimators (ICA < min(OLS, OML)) are ringed.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

R = Path("autoresearch/results/r10_ica_win_frontier")


def load_grid():
    ratio, best = {}, {}
    for f in glob.glob(str(R / "**" / "*.npy"), recursive=True):
        d = np.load(f, allow_pickle=True).item()
        rmse = np.asarray(d["rmse"], float)  # [OML,HOML,est,split,ICA,OLS,match]
        ols, oml, ica = rmse[5], rmse[0], rmse[4]
        k = (float(d["eta_beta"]), float(d["eps_beta"]))
        ratio[k] = ica / ols
        best[k] = ica < min(ols, oml)
    etas = sorted({k[0] for k in ratio})
    epss = sorted({k[1] for k in ratio})
    M = np.array([[ratio[(et, ep)] for ep in epss] for et in etas])
    B = np.array([[best[(et, ep)] for ep in epss] for et in etas])
    return etas, epss, M, B


def main():
    etas, epss, M, B = load_grid()
    L = np.log2(M)
    # Diverging scale centred at ratio=1 (log2=0); clip so the 99x outlier does
    # not wash out the rest. Blue = ICA wins, red = ICA loses.
    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.6)
    cmap = plt.get_cmap("RdBu_r").copy()

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(L, origin="lower", cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(epss)))
    ax.set_yticks(range(len(etas)))
    ax.set_xticklabels([f"{e:g}" for e in epss])
    ax.set_yticklabels([f"{e:g}" for e in etas])
    ax.set_xlabel("outcome-noise non-Gaussianity  ε ~ gennorm(β)\n"
                  "← heavier-tailed        β        lighter →", fontsize=10)
    ax.set_ylabel("treatment-noise non-Gaussianity  η ~ gennorm(β)\n"
                  "← heavier-tailed        β        lighter →", fontsize=10)

    # Gaussian reference lines (β=2): the source ICA cannot sharpen on.
    if 2.0 in epss:
        gx = epss.index(2.0)
        ax.axvline(gx, color="0.25", lw=1.1, ls=(0, (4, 3)), alpha=0.8)
    if 2.0 in etas:
        gy = etas.index(2.0)
        ax.axhline(gy, color="0.25", lw=1.1, ls=(0, (4, 3)), alpha=0.8)

    for i in range(len(etas)):
        for j in range(len(epss)):
            r = M[i, j]
            # readable text colour on each cell
            tc = "white" if abs(L[i, j]) > 0.75 else "0.12"
            txt = f"{r:.0f}" if r >= 10 else f"{r:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color=tc,
                    fontsize=10, fontweight="bold" if B[i, j] else "normal")
            if B[i, j]:  # ICA best of ALL estimators -> ring the cell
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#111", lw=2.4))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                        ticks=[np.log2(v) for v in (0.5, 0.7, 1, 1.4, 2, 3)])
    cbar.ax.set_yticklabels(["0.5×", "0.7×", "1×", "1.4×", "2×", "3×"])
    cbar.set_label("ICA / OLS  RMSE ratio   (blue = ICA wins)", fontsize=9)

    ax.set_title("Where does ICA win?  Treatment-effect RMSE, ICA vs OLS\n"
                 "linear PLR on real (California-Housing) covariates · n=50 000 · d=5 · "
                 "outlined = ICA best of all", fontsize=10.5)
    # legend note for the ring
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"ica_win_frontier.{ext}", dpi=160, bbox_inches="tight")
    print(f"saved -> {R/'ica_win_frontier.png'} (+ .pdf)")
    # concise console summary
    wins = [(etas[i], epss[j]) for i in range(len(etas)) for j in range(len(epss)) if B[i, j]]
    print("ICA best-of-all cells (eta,eps):", wins)


if __name__ == "__main__":
    main()
