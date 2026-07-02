#!/usr/bin/env python3
"""Domain metric-extractor for autoresearch rounds.

Loads the per-job ``.npy`` payloads under ``autoresearch/results/<round>/`` and
writes a tidy per-configuration TSV of per-estimator error metrics (RMSE / bias /
std / NaN-count) for the five headline estimators, plus ICA MCC and per-config
convergence bookkeeping.

Why this exists: ``cluster/sweep_runner.py``'s generic ``aggregate()`` only records
array *shapes* for these files, because each payload is a 1-D object array of result
*dicts* (one per internal OMLParameterGrid config), not a scalar dict. Metric
extraction is domain knowledge, so it lives here; grading/judgment stays with the
orchestrating Claude instance (see ``autoresearch/PROGRAM.md``).

Estimator columns in each dict's ``ortho_rec_tau`` (univariate treatment, width 7):
    0 OML   1 HOML(robust)   2 OML-Est   3 OML-Split   4 ICA   5 OLS   6 matching
For m-variate treatment the width is ``4 + 3m``; only the univariate case is
summarised per-column here (a warning is emitted otherwise).

Usage:
    python autoresearch/analyze_round.py <round>            # e.g. r01_smoke
    python autoresearch/analyze_round.py <round> --results-root autoresearch/results
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

# ortho_rec_tau column -> estimator label (univariate treatment)
METHOD_NAMES = ["OML", "HOML", "OML-Est", "OML-Split", "ICA", "OLS", "matching"]
# The five headline estimators for the regime map (PROGRAM WS1 success criterion a).
HEADLINE = ["OLS", "OML", "HOML", "ICA", "matching"]


def _to_float_array(seq) -> np.ndarray:
    """Coerce a per-experiment column (which may contain None) to float with NaNs."""
    out = np.array([np.nan if v is None else float(v) for v in seq], dtype=float)
    return out


def per_config_metrics(d: dict) -> dict:
    """Compute per-estimator RMSE/bias/std/NaN for one result dict."""
    true = float(d["treatment_effect"])
    tau = np.asarray(d["ortho_rec_tau"], dtype=object)  # (n_exp, n_methods)
    n_exp = tau.shape[0]
    width = tau.shape[1] if tau.ndim == 2 else len(tau[0])
    row = {
        "n_samples": int(d["n_samples"]),
        "support_size": int(d["support_size"]),
        "beta": float(d["beta"]),
        "treatment_effect": true,
        "eta_noise_dist": d.get("eta_noise_dist", "discrete"),
        "sigma_outcome": float(np.asarray(d["sigma_outcome"])),
        "n_exp_kept": n_exp,
        "n_methods": width,
        "ica_excess_kurtosis": float(np.asarray(d.get("eta_excess_kurtosis", np.nan))),
    }
    if width != 7:
        row["WARN"] = f"non-univariate width={width}; per-column labels may be off"
    for k in range(min(width, len(METHOD_NAMES))):
        col = _to_float_array([tau[i][k] for i in range(n_exp)])
        err = col - true
        name = METHOD_NAMES[k]
        row[f"{name}.rmse"] = float(np.sqrt(np.nanmean(err**2))) if np.isfinite(err).any() else np.nan
        row[f"{name}.bias"] = float(np.nanmean(err)) if np.isfinite(err).any() else np.nan
        row[f"{name}.std"] = float(np.nanstd(col)) if np.isfinite(col).any() else np.nan
        row[f"{name}.nan"] = int(np.isnan(col).sum())
    return row


def collect(results_root: Path, round_name: str) -> list[dict]:
    round_dir = results_root / round_name
    rows: list[dict] = []
    for job_dir in sorted(round_dir.glob("job_*")):
        npys = sorted(glob.glob(str(job_dir / "**" / "*.npy"), recursive=True))
        if not npys:
            rows.append({"job": job_dir.name, "note": "NO .npy (failed/held/empty)"})
            continue
        for npy in npys:
            try:
                arr = np.load(npy, allow_pickle=True)
            except Exception as exc:  # noqa: BLE001
                rows.append({"job": job_dir.name, "note": f"load_error: {exc}"})
                continue
            payload = arr.tolist() if arr.ndim else [arr.item()]
            for d in payload:
                if not isinstance(d, dict):
                    continue
                row = {"job": job_dir.name}
                row.update(per_config_metrics(d))
                rows.append(row)
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(r.get(k, "")) for k in keys) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def print_headline(rows: list[dict]) -> None:
    data = [r for r in rows if "ICA.rmse" in r]
    if not data:
        print("(no metric rows)")
        return
    data.sort(key=lambda r: (r["eta_noise_dist"], r["beta"], r["support_size"], r["n_samples"], r["treatment_effect"]))
    hdr = ["eta", "beta", "d", "n", "te", "kept"] + [f"{m}.rmse" for m in HEADLINE]
    print("  ".join(f"{h:>10s}" for h in hdr))
    for r in data:
        cells = [
            f"{r['eta_noise_dist']:>10s}",
            f"{r['beta']:>10g}",
            f"{r['support_size']:>10d}",
            f"{r['n_samples']:>10d}",
            f"{r['treatment_effect']:>10g}",
            f"{r['n_exp_kept']:>10d}",
        ]
        for m in HEADLINE:
            cells.append(f"{r.get(m + '.rmse', float('nan')):>10.4g}")
        print("  ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("round", help="round name, e.g. r01_smoke")
    ap.add_argument("--results-root", default="autoresearch/results")
    args = ap.parse_args()
    results_root = Path(args.results_root)
    rows = collect(results_root, args.round)
    out = results_root / args.round / "metrics.tsv"
    write_tsv(rows, out)
    print_headline(rows)
    print(f"\n{len([r for r in rows if 'ICA.rmse' in r])} config row(s) -> {out}")
    notes = [r for r in rows if r.get("note")]
    if notes:
        print(f"WARN: {len(notes)} job(s) with no metrics:")
        for r in notes:
            print(f"  {r['job']}: {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
