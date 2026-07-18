#!/usr/bin/env python3
"""Domain metric-extractor for the *nonlinear* autoresearch rounds.

Companion to ``analyze_round.py``, for rounds whose ``script`` is
``nonlinear_runner.py`` rather than ``monte_carlo_single_instance.py``. Those
payloads have a different schema: a precomputed per-method ``rmse`` array (in
``METHOD_NAMES`` order) plus DGP metadata (``nuisance``,
``nonlinear_confounding``, ``heavy_tail_eta`` …), instead of the raw
``ortho_rec_tau`` matrix. Writes a tidy per-config ``metrics.tsv`` and prints a
headline table of the five estimators.

Usage:
    python autoresearch/analyze_nonlinear_round.py <round>   # e.g. r04_nonlinear_breakdown
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

# nonlinear_runner.py METHOD_NAMES order (index -> label).
METHOD_NAMES = [
    "Ortho ML",
    "Robust Ortho ML",
    "Robust Ortho Est",
    "Robust Ortho Split",
    "ICA",
    "OLS",
    "Matching",
]
# Short label -> METHOD_NAMES index, for the five headline estimators.
HEADLINE = {"OLS": 5, "OML": 0, "HOML": 1, "ICA": 4, "matching": 6}


def per_config_metrics(d: dict) -> dict:
    rmse = np.asarray(d["rmse"], dtype=float)
    biases = np.asarray(d["biases"], dtype=float)
    sigmas = np.asarray(d["sigmas"], dtype=float)
    finite = np.asarray(d.get("finite_per_method", []), dtype=float)
    row = {
        "dataset": str(d.get("dataset", "")),
        "n_samples": int(d["n_samples"]),
        "n_covariates": int(d.get("n_covariates", -1)),
        "support_size": int(d.get("support_size", -1)),
        "nuisance": str(d.get("nuisance", "?")),
        "nonlinear": bool(d.get("nonlinear_confounding", False)),
        "heavy_tail_eta": bool(d.get("heavy_tail_eta", False)),
        "eta_beta": float(d.get("eta_beta", np.nan)),
        "eps_beta": (float(d["eps_beta"]) if d.get("eps_beta") is not None else np.nan),
        "bootstrap": bool(d.get("bootstrap", False)),
        "hetero_eps": bool(d.get("heteroscedastic_eps", False)),
        "treatment_effect": float(d["treatment_effect"]),
        "n_exp_kept": int(d.get("n_experiments", -1)),
        "n_attempted": int(d.get("n_attempted", -1)),
    }
    for name, k in zip(METHOD_NAMES, range(len(rmse))):
        row[f"{name}.rmse"] = float(rmse[k])
        row[f"{name}.bias"] = float(biases[k]) if k < len(biases) else np.nan
        row[f"{name}.std"] = float(sigmas[k]) if k < len(sigmas) else np.nan
        row[f"{name}.finite"] = int(finite[k]) if k < len(finite) else -1
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
                d = np.load(npy, allow_pickle=True).item()
            except Exception as exc:  # noqa: BLE001  pylint: disable=broad-exception-caught
                rows.append({"job": job_dir.name, "note": f"load_error: {exc}"})
                continue
            if not (isinstance(d, dict) and "rmse" in d):
                rows.append({"job": job_dir.name, "note": "not a nonlinear payload"})
                continue
            row = {"job": job_dir.name}
            row.update(per_config_metrics(d))
            rows.append(row)
    return rows


def _fmt(v) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


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


def print_headline(rows: list[dict]) -> None:
    data = [r for r in rows if "ICA.rmse" in r]
    if not data:
        print("(no metric rows)")
        return
    data.sort(key=lambda r: (r["nonlinear"], r["nuisance"], r["n_samples"]))
    hdr = ["confound", "nuisance", "n", "kept/att"] + list(HEADLINE)
    print("  ".join(f"{h:>12s}" for h in hdr))
    for r in data:
        cells = [
            f"{'nonlin' if r['nonlinear'] else 'linear':>12s}",
            f"{r['nuisance']:>12s}",
            f"{r['n_samples']:>12d}",
            f"{str(r['n_exp_kept']) + '/' + str(r['n_attempted']):>12s}",
        ]
        for short, idx in HEADLINE.items():
            cells.append(f"{r.get(METHOD_NAMES[idx] + '.rmse', float('nan')):>12.4g}")
        print("  ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("round", help="round name, e.g. r04_nonlinear_breakdown")
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
