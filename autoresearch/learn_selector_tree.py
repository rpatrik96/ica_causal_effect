#!/usr/bin/env python3
"""Close WS4: test the ICA d-gate, and learn a shallow tree to check the rule's
thresholds are intrinsic.

Pools every per-cell record collected by selector_multi_runner across the WS4
rounds (r13 + r14 splits): features (nonlinearity, d/n, ε-kurtosis, n) + the four
candidate RMSEs + the per-cell oracle. Then:

  (1) compares the regret of the hand-set rule, a d-gated ICA variant (ICA only
      when also low-dim), and a per-cell oracle — to decide empirically whether the
      d-gate helps or breaks the (linear, high-d) ICA wins;
  (2) fits a shallow DecisionTreeClassifier on the four features → oracle label and
      prints its splits, so we can see whether a data-driven learner recovers the
      hand-set thresholds (evidence the boundaries are intrinsic). Reports the tree's
      leave-one-out regret for an honest generalisation number.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

import selector_multi as selm

RESULTS = Path("autoresearch/results")
CANDS = ("OLS", "OML_lin", "OML_gbm", "ICA")
ROUNDS = ["r13_multiselector_eval", "r14_calib", "r14_test",
          "r14_real_housing", "r14_real_news20"]
FNAMES = ["nonlinearity", "d_over_n", "eps_kurt", "log10_n"]


def load_all():
    cells = []
    for rd in ROUNDS:
        for f in glob.glob(str(RESULTS / rd / "**" / "*.npy"), recursive=True):
            d = np.load(f, allow_pickle=True).item()
            cr = {c: float(d[f"cand_rmse_{c}"]) for c in CANDS}
            feats = {"nonlinearity_score": float(d["mean_nonlinearity"]),
                     "d_over_n": float(d["mean_d_over_n"]),
                     "eps_excess_kurtosis": float(d["mean_eps_kurt"]),
                     "n": int(d["n_samples"]), "d": int(d["n_covariates"])}
            cells.append({"feats": feats, "cand": cr, "oracle": min(cr.values()),
                          "oracle_name": min(cr, key=cr.get)})
    return cells


def _rms(v):
    return float(np.sqrt(np.nanmean(np.square(v))))


def regret(cells, pickfn):
    sel = [c["cand"][pickfn(c["feats"])] for c in cells]
    match = sum(pickfn(c["feats"]) == c["oracle_name"] for c in cells)
    return _rms(sel) - _rms([c["oracle"] for c in cells]), match


def rule_handset(f):
    return selm.select_estimator_multi(f)


def rule_dgate(f, dmax=50):
    """v2 rule but the ICA branch also requires low dimension (d <= dmax)."""
    th = selm.DEFAULT_THRESHOLDS
    nl_ok = f["nonlinearity_score"] > th["nl"]
    enough_n = f["n"] >= th["n_gate"]
    if f["d_over_n"] > th["dn"]:
        return "OML_gbm" if (nl_ok and enough_n) else "OML_lin"
    if nl_ok:
        return "OML_gbm" if enough_n else "OLS"
    if f["eps_excess_kurtosis"] > th["eps"] and f["d"] <= dmax:
        return "ICA"
    return "OLS"


def featmat(cells):
    X = np.array([[c["feats"]["nonlinearity_score"], c["feats"]["d_over_n"],
                   c["feats"]["eps_excess_kurtosis"], np.log10(c["feats"]["n"])] for c in cells])
    y = np.array([c["oracle_name"] for c in cells])
    return X, y


def main():
    cells = load_all()
    print(f"pooled {len(cells)} cells from {len(ROUNDS)} rounds\n")

    print("== (1) does a d-gate on the ICA branch help? (regret, match/N) ==")
    for name, fn in [("hand-set (no d-gate)", rule_handset),
                     ("d-gated ICA (d<=50)", rule_dgate)]:
        r, m = regret(cells, fn)
        print(f"  {name:<22} regret={r:+.4f}  match={m}/{len(cells)}")
    # show the cells where the two rules disagree
    diff = [(c, rule_handset(c["feats"]), rule_dgate(c["feats"])) for c in cells
            if rule_handset(c["feats"]) != rule_dgate(c["feats"])]
    print(f"  rules disagree on {len(diff)} cells:")
    for c, a, b in diff:
        f = c["feats"]
        print(f"    d={f['d']:>4} n={f['n']:>5} eps_k={f['eps_excess_kurtosis']:>6.1f} "
              f"nl={f['nonlinearity_score']:+.2f} | handset={a} dgate={b} oracle={c['oracle_name']} "
              f"(ICA={c['cand']['ICA']:.3f} OLS={c['cand']['OLS']:.3f})")

    print("\n== (2) shallow tree on the four features -> oracle label ==")
    X, y = featmat(cells)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(X, y)
    print(export_text(clf, feature_names=FNAMES))
    tr_pick = lambda f: clf.predict([[f["nonlinearity_score"], f["d_over_n"],
                                      f["eps_excess_kurtosis"], np.log10(f["n"])]])[0]
    r_tree, m_tree = regret(cells, tr_pick)
    print(f"tree train regret={r_tree:+.4f} match={m_tree}/{len(cells)}")
    # leave-one-out CV regret (honest generalisation)
    loo_sel = []
    for i in range(len(cells)):
        Xtr = np.delete(X, i, 0); ytr = np.delete(y, i)
        c = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(Xtr, ytr)
        loo_sel.append(cells[i]["cand"][c.predict(X[i:i+1])[0]])
    print(f"tree LOO-CV regret={_rms(loo_sel) - _rms([c['oracle'] for c in cells]):+.4f}")
    r_hs, m_hs = regret(cells, rule_handset)
    print(f"(hand-set rule regret={r_hs:+.4f} match={m_hs}/{len(cells)} for reference)")
    print("\nhand-set thresholds:", dict(selm.DEFAULT_THRESHOLDS))


if __name__ == "__main__":
    main()
