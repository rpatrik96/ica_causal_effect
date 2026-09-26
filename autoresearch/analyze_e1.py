#!/usr/bin/env python3
r"""Paired-comparison analysis for the r17 TMLR critical-eval reruns (E1; also E3/E4).

Loads every ``.npy`` result payload under the given result directories, turns
each simulated configuration into one record with its per-seed estimates
(``theta_hats``, written by ``ablation_utils.summarize_named_estimates``), and
writes:

* ``<out>/e1_rmse_per_config.csv``: RMSE / bias / std / failures per config and method;
* ``<out>/e1_paired_per_config.csv``: for every ICA readout R and every other method X,
  the RMSE difference RMSE_R - RMSE_X with a paired bootstrap 95% CI (seeds resampled
  jointly) and a paired sign test on the per-seed absolute errors |theta_hat - theta|;
* ``<out>/e1_win_rates.csv``: config-level win rates of each readout against each
  method, stratified by c_ica = 1 + ||b + a theta||^2 and by c_b = 1 + ||b||^2 Var(X)
  (bins < 1.5, [1.5, 5), >= 5), separately for each arm (oracle support x number of
  irrelevant covariate columns beyond the support, ``extra_cov`` = cov_dim_max - support_size);
* ``<out>/e1_summary.md``: the headline tables.

Per-seed estimates are paired: entry i of every method's array comes from the same
simulated dataset, so seeds are dropped pairwise only where either estimate is
non-finite. p-values are unadjusted for multiplicity.

Usage:
    python autoresearch/analyze_e1.py autoresearch/results/r17_e1_coef \\
        autoresearch/results/r17_e1_heatmaps --out autoresearch/results/r17_e1_analysis
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

BIN_EDGES = (1.5, 5.0)
BIN_LABELS = ("<1.5", "[1.5,5)", ">=5")
# Comparators shown in the markdown summary (the CSVs carry every method).
HEADLINE_COMPARATORS = ("ols", "oml", "homl_known", "homl_estimated", "huber")
BIN_ORDER = {label: i for i, label in enumerate(BIN_LABELS)}


def bin_label(value: float) -> str:
    """Return the c-bin label ({<1.5, [1.5,5), >=5}) for a variance coefficient."""
    if not np.isfinite(value):
        return "nan"
    if value < BIN_EDGES[0]:
        return BIN_LABELS[0]
    if value < BIN_EDGES[1]:
        return BIN_LABELS[1]
    return BIN_LABELS[2]


def _scalar_or_nan(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1) if value is not None else np.array([])
    return float(arr[0]) if arr.size == 1 else float("nan")


def _norm_or_nan(value) -> float:
    if value is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(value, dtype=float).reshape(-1)))


def _record(payload: dict, source: str, extra: dict | None = None, top: dict | None = None) -> dict | None:
    """Build one analysis record from a per-config result dict."""
    if not isinstance(payload, dict) or "theta_hats" not in payload:
        return None
    top = top or {}
    extra = extra or {}

    def pick(*keys, default=None):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
            if key in top and top[key] is not None:
                return top[key]
        return default

    theta = float(pick("theta", "treatment_effect"))
    a_val, b_val = pick("a"), pick("b")
    rec = {
        "source": source,
        "pipeline": pick("pipeline", default="unknown"),
        "oracle_support": bool(pick("oracle_support", default=True)),
        "theta": theta,
        "a": _scalar_or_nan(a_val),
        "b": _scalar_or_nan(b_val),
        "a_norm": _norm_or_nan(a_val),
        "b_norm": _norm_or_nan(b_val),
        "c_ica": float(pick("c_ica", "ica_var_coeff", default=np.nan)),
        "c_b": float(pick("c_b", default=np.nan)),
        "n_samples": pick("n_samples", default=np.nan),
        "support_size": pick("support_size", default=np.nan),
        "cov_dim_max": pick("cov_dim_max", default=np.nan),
        "covariate_beta": pick("covariate_beta", "beta", default=np.nan),
        "scalar_coeffs": pick("scalar_coeffs", default=""),
        "axis_mode": pick("axis_mode", default=""),
        "theta_hats": {k: np.asarray(v, dtype=float) for k, v in payload["theta_hats"].items()},
    }
    rec["extra_cov"] = (
        int(rec["cov_dim_max"]) - int(rec["support_size"])
        if np.isfinite(float(rec["cov_dim_max"])) and np.isfinite(float(rec["support_size"]))
        else "n/a"
    )
    rec.update(extra)
    key_fields = [
        "pipeline",
        "oracle_support",
        "n_samples",
        "support_size",
        "cov_dim_max",
        "covariate_beta",
        "scalar_coeffs",
        "a",
        "b",
        "theta",
    ] + sorted(extra)
    rec["config_id"] = "|".join(f"{k}={rec[k]}" for k in key_fields)
    return rec


def load_records(paths: list[str]) -> list[dict]:
    """Load analysis records from ``.npy`` files or directories searched recursively."""
    files: list[Path] = []
    for path in paths:
        p = Path(path)
        files.extend(sorted(p.rglob("*.npy")) if p.is_dir() else [p])
    records = []
    for f in files:
        # allow_pickle: result payloads written by this repo's own experiment scripts.
        data = np.load(f, allow_pickle=True)
        payload = data.item() if data.shape == () else list(data)
        if isinstance(payload, dict) and "grid_results" in payload:  # variance ablation
            for (eta_beta, eta_var), cell in payload["grid_results"].items():
                rec = _record(cell, str(f), {"eta_beta": eta_beta, "eta_variance": eta_var}, top=payload)
                if rec is not None:
                    records.append(rec)
        elif isinstance(payload, dict):
            rec = _record(payload, str(f))
            if rec is not None:
                records.append(rec)
        else:
            for item in payload:
                rec = _record(item, str(f))
                if rec is not None:
                    records.append(rec)
    return records


def rmse(errors: np.ndarray) -> float:
    """Root mean squared error of an error vector (NaN when empty)."""
    return float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan")


def paired_comparison(est_r: np.ndarray, est_x: np.ndarray, theta: float, n_boot: int, rng) -> dict:
    """Paired RMSE difference with bootstrap CI and a sign test on per-seed absolute errors."""
    mask = np.isfinite(est_r) & np.isfinite(est_x)
    err_r, err_x = est_r[mask] - theta, est_x[mask] - theta
    n = int(mask.sum())
    out = {"n_paired": n, "rmse_r": rmse(err_r), "rmse_x": rmse(err_x)}
    out["rmse_diff"] = out["rmse_r"] - out["rmse_x"]
    if n == 0:
        out.update(ci_low=np.nan, ci_high=np.nan, n_r_closer=0, n_x_closer=0, n_ties=0, sign_p=np.nan)
        return out
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.sqrt(np.mean(err_r[idx] ** 2, axis=1)) - np.sqrt(np.mean(err_x[idx] ** 2, axis=1))
    out["ci_low"], out["ci_high"] = (float(q) for q in np.percentile(boot, [2.5, 97.5]))
    abs_diff = np.abs(err_r) - np.abs(err_x)
    n_r = int((abs_diff < 0).sum())
    n_x = int((abs_diff > 0).sum())
    out.update(n_r_closer=n_r, n_x_closer=n_x, n_ties=n - n_r - n_x)
    out["sign_p"] = float(binomtest(n_r, n_r + n_x, 0.5).pvalue) if n_r + n_x > 0 else float("nan")
    return out


def analyse(records: list[dict], n_boot: int = 2000, seed: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    """Return per-config RMSE rows, paired-comparison rows and win-rate rows."""
    rng = np.random.default_rng(seed)
    base_cols = [
        "config_id",
        "pipeline",
        "oracle_support",
        "cov_dim_max",
        "support_size",
        "extra_cov",
        "theta",
        "a",
        "b",
        "a_norm",
        "b_norm",
        "c_ica",
        "c_b",
    ]
    rmse_rows, paired_rows = [], []
    for rec in records:
        base = {k: rec[k] for k in base_cols}
        base.update(c_ica_bin=bin_label(rec["c_ica"]), c_b_bin=bin_label(rec["c_b"]))
        for method, est in rec["theta_hats"].items():
            finite = np.isfinite(est)
            err = est[finite] - rec["theta"]
            rmse_rows.append(
                {
                    **base,
                    "method": method,
                    "n_seeds": int(est.size),
                    "n_failed": int((~finite).sum()),
                    "rmse": rmse(err),
                    "bias": float(np.mean(err)) if err.size else np.nan,
                    "std": float(np.std(est[finite])) if err.size else np.nan,
                }
            )
        readouts = sorted(m for m in rec["theta_hats"] if m.startswith("ica_"))
        for readout in readouts:
            for other in sorted(rec["theta_hats"]):
                if other == readout:
                    continue
                res = paired_comparison(rec["theta_hats"][readout], rec["theta_hats"][other], rec["theta"], n_boot, rng)
                paired_rows.append({**base, "readout": readout, "comparator": other, **res})

    win_rows = []
    for stratifier in ("c_ica_bin", "c_b_bin"):
        groups: dict[tuple, list[dict]] = {}
        for row in paired_rows:
            key = (row["readout"], row["comparator"], row["oracle_support"], row["extra_cov"], row[stratifier])
            groups.setdefault(key, []).append(row)
        for (readout, comparator, oracle, d_amb, stratum), rows in sorted(groups.items(), key=str):
            valid = [r for r in rows if np.isfinite(r["rmse_diff"])]
            n_cfg = len(valid)
            wins = sum(r["rmse_diff"] < 0 for r in valid)
            sig_wins = sum(r["ci_high"] < 0 for r in valid)
            sig_losses = sum(r["ci_low"] > 0 for r in valid)
            win_rows.append(
                {
                    "stratifier": stratifier,
                    "stratum": stratum,
                    "oracle_support": oracle,
                    "extra_cov": d_amb,
                    "readout": readout,
                    "comparator": comparator,
                    "n_configs": n_cfg,
                    "wins": wins,
                    "win_rate": wins / n_cfg if n_cfg else np.nan,
                    "ci_excludes_0_readout_better": sig_wins,
                    "ci_excludes_0_comparator_better": sig_losses,
                    "sign_p_below_0.05": sum(r["sign_p"] < 0.05 for r in valid),
                    "median_rmse_diff": float(np.median([r["rmse_diff"] for r in valid])) if valid else np.nan,
                }
            )
    return rmse_rows, paired_rows, win_rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to CSV with the union of keys as header."""
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if not np.isfinite(value) else f"{value:.4g}"
    return str(value)


def markdown_summary(records: list[dict], rmse_rows: list[dict], win_rows: list[dict], sources: list[str]) -> str:
    """Render the headline tables as markdown."""
    lines = ["# r17 E1 paired analysis", ""]
    lines.append(f"Inputs: {', '.join(sources)}  ")
    lines.append(f"Configurations: {len(records)}; seeds per config: " + _seed_summary(records))
    lines.append("")
    lines.append(
        "Win = the ICA readout has the lower RMSE in that configuration. `CI<0` counts configurations "
        "whose paired-bootstrap 95% CI of RMSE(ICA) - RMSE(X) lies below 0, `CI>0` those above 0. "
        "Sign-test p-values are per configuration and unadjusted."
    )
    lines.append("")

    lines.append("## Failures (non-finite estimates) by method")
    lines.append("")
    failures: dict[str, list[int]] = {}
    for row in rmse_rows:
        failures.setdefault(row["method"], [0, 0])
        failures[row["method"]][0] += row["n_failed"]
        failures[row["method"]][1] += row["n_seeds"]
    lines.append("| method | failed | total seeds |")
    lines.append("|---|---:|---:|")
    for method, (n_fail, n_tot) in sorted(failures.items()):
        lines.append(f"| {method} | {n_fail} | {n_tot} |")
    lines.append("")

    for stratifier, title in (("c_ica_bin", "c_ica = 1 + ||b + a theta||^2"), ("c_b_bin", "c_b = 1 + ||b||^2 Var(X)")):
        lines.append(f"## Win rates stratified by {title}")
        lines.append("")
        lines.append(
            "| oracle | extra cov | readout | vs | bin | configs | wins | win rate | CI<0 | CI>0 | median diff |"
        )
        lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
        rows = [
            r
            for r in win_rows
            if r["stratifier"] == stratifier and r["comparator"] in HEADLINE_COMPARATORS + ("ica_eps_row",)
        ]
        rows.sort(
            key=lambda r: (
                not r["oracle_support"],
                str(r["extra_cov"]),
                r["readout"],
                r["comparator"],
                BIN_ORDER.get(r["stratum"], 9),
            )
        )
        for r in rows:
            lines.append(
                f"| {r['oracle_support']} | {_fmt(r['extra_cov'])} | {r['readout']} | {r['comparator']} | {r['stratum']} | {r['n_configs']} "
                f"| {r['wins']} | {_fmt(r['win_rate'])} | {r['ci_excludes_0_readout_better']} "
                f"| {r['ci_excludes_0_comparator_better']} | {_fmt(r['median_rmse_diff'])} |"
            )
        lines.append("")

    lines.append("## Median RMSE per method, by oracle support, extra covariates and c_ica bin")
    lines.append("")
    groups: dict[tuple, list[float]] = {}
    for row in rmse_rows:
        key = (row["oracle_support"], row["extra_cov"], row["c_ica_bin"], row["method"])
        groups.setdefault(key, []).append(row["rmse"])
    lines.append("| oracle | extra cov | c_ica bin | method | configs | median RMSE |")
    lines.append("|---|---:|---|---|---:|---:|")
    for (oracle, d_amb, stratum, method), vals in sorted(groups.items(), key=str):
        median = _fmt(float(np.nanmedian(vals)))
        lines.append(f"| {oracle} | {_fmt(d_amb)} | {stratum} | {method} | {len(vals)} | {median} |")
    lines.append("")
    return "\n".join(lines)


def _seed_summary(records: list[dict]) -> str:
    counts = sorted({len(next(iter(r["theta_hats"].values()))) for r in records if r["theta_hats"]})
    return ", ".join(str(c) for c in counts) if counts else "none"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Result directories (searched recursively) or .npy files.")
    parser.add_argument("--out", required=True, help="Output directory for the CSVs and the markdown summary.")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples per comparison.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    args = parser.parse_args(argv)

    records = load_records(args.inputs)
    if not records:
        print("no records with per-seed estimates (theta_hats) found")
        return 1
    rmse_rows, paired_rows, win_rows = analyse(records, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rmse_rows, out / "e1_rmse_per_config.csv")
    write_csv(paired_rows, out / "e1_paired_per_config.csv")
    write_csv(win_rows, out / "e1_win_rates.csv")
    (out / "e1_summary.md").write_text(markdown_summary(records, rmse_rows, win_rows, args.inputs), encoding="utf-8")
    print(f"{len(records)} configurations -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
