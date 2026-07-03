#!/usr/bin/env python3
"""Calibrate multi-selector thresholds on a held-out split, confirm on real X.

Reads the per-cell records (mean features + four candidate RMSEs + oracle) that
`selector_multi_runner.py` writes for each round, grid-searches the rule
thresholds (nl, dn, eps, n_gate) to minimise total regret on the CALIBRATION
round, then reports regret of the calibrated thresholds — and of the hand-set
defaults — on the calibration, held-out, and real-X rounds.

Regret is computed per cell as cand_rmse[rule_pick(mean_features)] - oracle, so no
re-run is needed; the selection boundary is a per-cell (per-regime) decision.

Usage:
    python autoresearch/calibrate_selector.py \
        --calib r14_calib --test r14_test --real r14_real_housing r14_real_news20
"""
from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np

import selector_multi as selm

RESULTS = Path("autoresearch/results")
CANDS = ("OLS", "OML_lin", "OML_gbm", "ICA")


def load_cells(round_name):
    cells = []
    for f in glob.glob(str(RESULTS / round_name / "**" / "*.npy"), recursive=True):
        d = np.load(f, allow_pickle=True).item()
        cr = {c: d[f"cand_rmse_{c}"] for c in CANDS}
        cells.append({
            "feats": {"nonlinearity_score": d["mean_nonlinearity"],
                      "d_over_n": d["mean_d_over_n"],
                      "eps_excess_kurtosis": d["mean_eps_kurt"],
                      "n": int(d["n_samples"]), "d": int(d["n_covariates"])},
            "cand": cr, "oracle": min(cr.values()),
            "oracle_name": min(cr, key=cr.get),
        })
    return cells


def total_regret(cells, thresholds):
    reg, sq_sel, sq_orc, match = [], [], [], 0
    for c in cells:
        pick = selm.select_estimator_multi(c["feats"], thresholds)
        sel = c["cand"][pick]
        reg.append(sel - c["oracle"]); sq_sel.append(sel); sq_orc.append(c["oracle"])
        match += (pick == c["oracle_name"])
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    return {"rms_sel": rms(sq_sel), "rms_oracle": rms(sq_orc),
            "regret": rms(sq_sel) - rms(sq_orc), "match": match, "n": len(cells)}


def calibrate(cells):
    """Grid-search thresholds minimising RMS regret on the calibration cells."""
    grid = {
        "nl": [0.02, 0.05, 0.1, 0.2],
        "dn": [0.3, 0.5, 0.8],
        "eps": [1.0, 2.0, 3.0, 5.0],
        "n_gate": [500, 1000, 2000],
    }
    best, best_reg = None, np.inf
    for nl, dn, eps, ng in itertools.product(grid["nl"], grid["dn"], grid["eps"], grid["n_gate"]):
        th = {"nl": nl, "dn": dn, "eps": eps, "n_gate": ng}
        r = total_regret(cells, th)["regret"]
        if r < best_reg - 1e-9:
            best_reg, best = r, th
    return best, best_reg


def _report(name, cells, th):
    r = total_regret(cells, th)
    print(f"  {name:<18} regret={r['regret']:+.4f}  (sel={r['rms_sel']:.4f} "
          f"oracle={r['rms_oracle']:.4f})  match={r['match']}/{r['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="r14_calib")
    ap.add_argument("--test", default="r14_test")
    ap.add_argument("--real", nargs="*", default=["r14_real_housing", "r14_real_news20"])
    args = ap.parse_args()

    calib = load_cells(args.calib)
    th_cal, reg_cal = calibrate(calib)
    th_def = selm.DEFAULT_THRESHOLDS
    print(f"calibrated thresholds (on {args.calib}, {len(calib)} cells): {th_cal}")
    print(f"hand-set defaults:                        {dict(th_def)}\n")

    for label, th in [("HAND-SET defaults", th_def), ("CALIBRATED", th_cal)]:
        print(f"== {label} ==")
        _report(f"{args.calib} (train)", calib, th)
        for rd in [args.test, *args.real]:
            cells = load_cells(rd)
            if cells:
                _report(rd, cells, th)
        print()


if __name__ == "__main__":
    main()
