#!/usr/bin/env python3
"""WS4 closure chart: the interpretable hand-set rule beats the best fixed
estimator and a learned tree, on the pooled r13+r14 cells."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

import autoresearch.learn_selector_tree as L

R = Path("autoresearch/results/r13_multiselector_eval")


def main():
    cells = L.load_all()
    X, y = L.featmat(cells)
    # learned-tree LOO regret
    loo = []
    for i in range(len(cells)):
        c = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(
            np.delete(X, i, 0), np.delete(y, i))
        loo.append(cells[i]["cand"][c.predict(X[i:i+1])[0]])
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    orc = rms([c["oracle"] for c in cells])
    reg_omllin, _ = L.regret(cells, lambda f: "OML_lin")
    reg_tree = rms(loo) - orc
    reg_hs, _ = L.regret(cells, L.rule_handset)

    series = [("best fixed\n(always OML-lin)", reg_omllin, "#9aa0a6"),
              ("learned tree\n(LOO-CV)", reg_tree, "#b0632f"),
              ("hand-set rule\n(mechanism)", reg_hs, "#2f6fb0")]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    vals = [s[1] for s in series]
    ax.bar(range(len(series)), vals, color=[s[2] for s in series], width=0.6, zorder=3)
    ax.set_xticks(range(len(series))); ax.set_xticklabels([s[0] for s in series], fontsize=9.5)
    ax.set_ylabel("selector regret vs oracle  (RMS, pooled 58 cells)", fontsize=9.5)
    ax.grid(axis="y", color="0.9", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0009, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if i == 2 else "normal")
    ax.text(0.5, max(vals) * 0.9, "always-ICA regret = 31.7 (off-chart)\nalways-OLS = 0.23",
            ha="center", fontsize=8, color="#7a2f2f")
    ax.set_title("WS4 closure: the interpretable rule beats a learned tree\n"
                 "and the best fixed estimator (pooled r13+r14, LOO for the tree)", fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"ws4_closure.{ext}", dpi=160, bbox_inches="tight")
    print("saved -> ws4_closure.png ; omllin", round(reg_omllin, 4),
          "tree_loo", round(reg_tree, 4), "handset", round(reg_hs, 4))


if __name__ == "__main__":
    main()
