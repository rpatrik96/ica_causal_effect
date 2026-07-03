#!/usr/bin/env python3
"""Chart r14: hand-set vs calibrated thresholds across train / held-out / real X.

Grouped bars of the selector's regret (RMS RMSE above the per-cell oracle) for the
mechanism-grounded hand-set thresholds vs thresholds calibrated to minimise regret
on the training split. The point: calibration wins only on train and overfits
(worse on held-out and, sharply, on real housing) — the principled thresholds
generalise.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import selector_multi as selm
from importlib import import_module

cal = import_module("autoresearch.calibrate_selector")
R = Path("autoresearch/results/r14_calib")

SPLITS = [("calibration\n(train)", "r14_calib"), ("held-out\n(synthetic)", "r14_test"),
          ("real X\n(housing)", "r14_real_housing"), ("real X\n(news20)", "r14_real_news20")]


def main():
    calib_cells = cal.load_cells("r14_calib")
    th_cal, _ = cal.calibrate(calib_cells)
    th_def = selm.DEFAULT_THRESHOLDS
    hs, cc = [], []
    for _, rd in SPLITS:
        cells = cal.load_cells(rd)
        hs.append(cal.total_regret(cells, th_def)["regret"])
        cc.append(cal.total_regret(cells, th_cal)["regret"])

    x = np.arange(len(SPLITS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b1 = ax.bar(x - w/2, hs, w, label="hand-set (mechanism-grounded)", color="#2f6fb0", zorder=3)
    b2 = ax.bar(x + w/2, cc, w, label="calibrated (min train regret)", color="#b0632f", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in SPLITS], fontsize=9.5)
    ax.set_ylabel("selector regret vs per-cell oracle  (RMS RMSE)", fontsize=9.5)
    ax.grid(axis="y", color="0.9", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0008,
                    f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.annotate("calibration overfits:\nbetter on train, worse everywhere else",
                xy=(2 + w/2, cc[2]), xytext=(1.4, max(cc) * 0.82), fontsize=8.4, color="#7a2f2f",
                arrowprops=dict(arrowstyle="->", color="#7a2f2f", lw=1.1))
    ax.set_title("Mechanism-grounded thresholds generalise; tuned ones overfit\n"
                 "multi-selector regret, thresholds calibrated on train vs hand-set", fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"calibration.{ext}", dpi=160, bbox_inches="tight")
    print("saved -> calibration.png ; hand-set:", [round(v, 4) for v in hs],
          "calibrated:", [round(v, 4) for v in cc])


if __name__ == "__main__":
    main()
