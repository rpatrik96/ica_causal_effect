"""Top-level driver for the semi-synthetic treatment-effect experiment.

Loads real covariates (California Housing or IHDP), imposes a partially-linear
regression (PLR) DGP via :func:`~semi_synthetic_data.generate_semi_synthetic_pl`,
then evaluates ICA, HOML, OLS, and Matching estimators across multiple Monte
Carlo replications.

Outputs
-------
``<output_dir>/semi_synthetic_<dataset>_results.npy``
    List of per-rep result dicts.
``<output_dir>/semi_synthetic_<dataset>_summary.svg``
    Bar chart of bias / std / RMSE per method.
``<output_dir>/semi_synthetic_<dataset>_summary.md``
    Markdown table with columns method | bias | std | rmse | n_reps.

CLI example
-----------
.. code-block:: bash

    python semi_synthetic_experiment.py \\
        --dataset california_housing \\
        --n_experiments 20 \\
        --n_samples 5000 \\
        --eta_distribution discrete \\
        --nonlinearity identity \\
        --treatment_effect 1.0 \\
        --methods ica,homl,ols,matching \\
        --output_dir figures/semi_synthetic
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS: tuple[str, ...] = ("ica", "homl", "ols", "matching")


# ---------------------------------------------------------------------------
# Moment helpers
# ---------------------------------------------------------------------------


def _compute_eta_moments(
    eta_distribution: str,
    eta_sample_fn: Any,
    n_empirical: int = 100_000,
) -> tuple[float, float]:
    """Return (second_moment, third_moment) for the eta distribution.

    Tries an analytic look-up first; falls back to empirical estimation from a
    large sample so the function is robust to new distributions.

    Parameters
    ----------
    eta_distribution:
        Distribution name string (e.g. ``"discrete"``, ``"laplace"``,
        ``"rademacher"``, ``"bernoulli"``).
    eta_sample_fn:
        Callable ``eta_sample_fn(n) -> np.ndarray`` returned by
        :func:`~oml_runner.setup_treatment_noise`.
    n_empirical:
        Number of samples to draw when using the empirical fallback.

    Returns
    -------
    tuple[float, float]
        ``(second_moment, third_moment)`` — both central moments.
    """
    base = eta_distribution.split(":")[0].lower()

    # Symmetric distributions have third_moment = 0 analytically.
    analytic_third_zero = {"laplace", "uniform", "rademacher", "gennorm_heavy", "gennorm_light"}

    if base in analytic_third_zero:
        # Second moment: empirical (cheap and always correct)
        samples = eta_sample_fn(n_empirical)
        second_moment = float(np.mean(samples**2))
        third_moment = 0.0
        return second_moment, third_moment

    # For discrete / bernoulli / general: empirical from large draw
    samples = eta_sample_fn(n_empirical)
    second_moment = float(np.mean(samples**2))
    third_moment = float(np.mean(samples**3) - 3.0 * float(np.mean(samples)) * second_moment)
    return second_moment, third_moment


# ---------------------------------------------------------------------------
# Single-rep runner
# ---------------------------------------------------------------------------


def _run_one_rep(
    *,
    dataset: str,
    n_samples: int | None,
    treatment_effect: float,
    eta_distribution: str,
    nonlinearity: str,
    support_size: int | None,
    methods: list[str],
    rep_seed: int,
) -> dict[str, Any]:
    """Run a single Monte Carlo replication.

    Parameters
    ----------
    dataset:
        Dataset name passed to :func:`~semi_synthetic_data.load_real_covariates`.
    n_samples:
        Number of covariate rows to subsample, or ``None`` for all.
    treatment_effect:
        True theta.
    eta_distribution:
        Treatment noise distribution string.
    nonlinearity:
        Nonlinearity applied to the linear index.
    support_size:
        Number of covariate columns used in the PLR mechanism, or ``None``.
    methods:
        List of method names to evaluate (subset of ``ALL_METHODS``).
    rep_seed:
        RNG seed for this replication.

    Returns
    -------
    dict
        Keys: method names (float estimates), ``"ground_truth"``,
        ``"rep_seed"``.
    """
    from semi_synthetic_data import generate_semi_synthetic_pl, load_real_covariates

    # ---- Load covariates ----
    covariates = load_real_covariates(dataset, n_samples=n_samples, seed=rep_seed)

    # ---- Generate (X, T, Y, meta) ----
    _, treatment, outcome, meta = generate_semi_synthetic_pl(
        covariates,
        treatment_effect=treatment_effect,
        eta_distribution=eta_distribution,
        nonlinearity=nonlinearity,
        support_size=support_size,
        seed=rep_seed,
    )

    result: dict[str, Any] = {
        "ground_truth": treatment_effect,
        "rep_seed": rep_seed,
    }

    # ---- OLS ----
    if "ols" in methods:
        import baselines

        ols_hat = baselines.ols_baseline(covariates, treatment, outcome)
        result["ols"] = float(ols_hat[0])

    # ---- Matching ----
    if "matching" in methods:
        import baselines

        match_hat = baselines.matching_baseline(covariates, treatment, outcome, treatment_kind="continuous")
        result["matching"] = float(match_hat)

    # ---- HOML ----
    if "homl" in methods:
        from main_estimation import all_together_cross_fitting
        from oml_runner import setup_treatment_noise

        _, eta_sample_fn, _, _ = setup_treatment_noise(distribution=eta_distribution)
        second_moment, third_moment = _compute_eta_moments(eta_distribution, eta_sample_fn)

        homl_results = all_together_cross_fitting(
            covariates,
            treatment,
            outcome,
            second_moment,
            third_moment,
        )
        # homl_results = (ortho_ml, robust_ortho_ml, robust_ortho_est_ml,
        #                 robust_ortho_est_split_ml, treatment_coef, outcome_coef)
        # We report the "HOML" (index 1 = robust_ortho_ml with known moments)
        result["homl"] = float(homl_results[1])

    # ---- ICA ----
    if "ica" in methods:
        from ica import (
            ica_treatment_effect_estimation_eps_row,
        )
        from oml_runner import setup_treatment_noise

        # Build X_full = [covariates, T, Y] and S_full = [covariates, eta, eps]
        # We do not have eta/eps separately from generate_semi_synthetic_pl, so
        # we use ica_treatment_effect_estimation_eps_row which does not need
        # ground-truth sources.
        X_full = np.hstack([covariates, treatment.reshape(-1, 1), outcome.reshape(-1, 1)])

        # Attempt eps-row ICA (does not require ground-truth sources)
        try:
            ica_hat, _ = ica_treatment_effect_estimation_eps_row(
                X_full,
                S=None,
                n_treatments=1,
                verbose=False,
            )
            result["ica"] = float(ica_hat[0]) if np.isfinite(ica_hat[0]) else float("nan")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ICA failed for rep_seed=%d: %s", rep_seed, exc)
            result["ica"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _compute_summary(
    results: list[dict[str, Any]],
    methods: list[str],
    treatment_effect: float,
) -> dict[str, dict[str, float]]:
    """Compute bias, std, RMSE per method across replications.

    Parameters
    ----------
    results:
        List of per-rep result dicts.
    methods:
        Method names to summarise.
    treatment_effect:
        True theta (used to compute bias).

    Returns
    -------
    dict
        ``{method: {"bias": ..., "std": ..., "rmse": ..., "n_reps": ...}}``.
    """
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        estimates = np.array(
            [r[method] for r in results if method in r and np.isfinite(r[method])],
            dtype=float,
        )
        n_valid = int(estimates.size)
        if n_valid == 0:
            summary[method] = {"bias": float("nan"), "std": float("nan"), "rmse": float("nan"), "n_reps": 0}
            continue
        bias = float(np.mean(estimates - treatment_effect))
        std = float(np.std(estimates, ddof=1) if n_valid > 1 else 0.0)
        rmse = float(np.sqrt(np.mean((estimates - treatment_effect) ** 2)))
        summary[method] = {"bias": bias, "std": std, "rmse": rmse, "n_reps": n_valid}
    return summary


def _write_markdown_table(
    summary: dict[str, dict[str, float]],
    path: str,
) -> None:
    """Write a markdown table of the summary to *path*.

    Parameters
    ----------
    summary:
        Output of :func:`_compute_summary`.
    path:
        Destination file path.
    """
    header = "| method | bias | std | rmse | n_reps |\n"
    sep = "|--------|------|-----|------|--------|\n"
    rows = []
    for method, stats in summary.items():
        rows.append(
            f"| {method} | {stats['bias']:.4f} | {stats['std']:.4f} |" f" {stats['rmse']:.4f} | {stats['n_reps']} |"
        )
    content = header + sep + "\n".join(rows) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _write_summary_plot(
    summary: dict[str, dict[str, float]],
    path: str,
) -> None:
    """Write a bar plot of bias / std / RMSE per method to *path*.

    Parameters
    ----------
    summary:
        Output of :func:`_compute_summary`.
    path:
        Destination SVG file path.
    """
    from matplotlib import pyplot as plt

    try:
        from plot_utils import plot_typography

        plot_typography(preset="publication")
    except Exception:  # pylint: disable=broad-except
        pass

    methods = list(summary.keys())
    metrics = ["bias", "std", "rmse"]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(methods)), 5))
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [summary[m][metric] for m in methods]
        ax.bar(x + i * width, values, width, label=metric.upper(), color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Semi-synthetic experiment: bias / std / RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    dataset: str = "california_housing",
    n_samples: int | None = None,
    n_experiments: int = 20,
    treatment_effect: float = 1.0,
    eta_distribution: str = "discrete",
    nonlinearity: str = "identity",
    support_size: int | None = None,
    seed: int = 12143,
    output_dir: str = "figures/semi_synthetic",
    methods: str = "all",
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run the semi-synthetic experiment and write outputs.

    This is the programmatic entry point; the ``__main__`` block delegates
    here after parsing CLI flags.

    Parameters
    ----------
    dataset:
        Dataset to load (``"california_housing"`` or ``"ihdp"``).
    n_samples:
        Number of rows to subsample from the dataset, or ``None`` for all.
    n_experiments:
        Number of Monte Carlo replications.
    treatment_effect:
        True causal effect theta.
    eta_distribution:
        Treatment noise distribution (e.g. ``"discrete"``, ``"laplace"``).
    nonlinearity:
        Nonlinearity applied to the linear index (``"identity"`` or
        ``"leaky_relu"``).
    support_size:
        Number of covariate columns used; ``None`` uses all columns.
    seed:
        Base random seed; rep i uses ``seed + i``.
    output_dir:
        Directory where output files are written (created if absent).
    methods:
        Comma-separated subset of ``{ica,homl,ols,matching}`` or ``"all"``.
    n_jobs:
        Number of parallel jobs passed to ``joblib.Parallel``.  ``-1`` uses
        all available CPUs.

    Returns
    -------
    dict
        ``{"results": list[dict], "summary": dict}``.
    """
    # ---- Parse methods ----
    if methods == "all":
        method_list = list(ALL_METHODS)
    else:
        method_list = [m.strip().lower() for m in methods.split(",")]
        unknown = set(method_list) - set(ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown method(s): {unknown}. Choose from {ALL_METHODS}.")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Parallel MC loop ----
    seeds = [seed + i for i in range(n_experiments)]

    try:
        from joblib import Parallel, delayed  # type: ignore[import-untyped]

        results: list[dict[str, Any]] = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_rep)(
                dataset=dataset,
                n_samples=n_samples,
                treatment_effect=treatment_effect,
                eta_distribution=eta_distribution,
                nonlinearity=nonlinearity,
                support_size=support_size,
                methods=method_list,
                rep_seed=s,
            )
            for s in seeds
        )
    except ImportError:
        logger.warning("joblib not available — running sequentially.")
        results = [
            _run_one_rep(
                dataset=dataset,
                n_samples=n_samples,
                treatment_effect=treatment_effect,
                eta_distribution=eta_distribution,
                nonlinearity=nonlinearity,
                support_size=support_size,
                methods=method_list,
                rep_seed=s,
            )
            for s in seeds
        ]

    # ---- Save raw results ----
    npy_path = os.path.join(output_dir, f"semi_synthetic_{dataset}_results.npy")
    np.save(npy_path, results, allow_pickle=True)  # type: ignore[arg-type]
    logger.info("Saved results to %s", npy_path)

    # ---- Compute summary ----
    summary = _compute_summary(results, method_list, treatment_effect)

    # ---- Write markdown table ----
    md_path = os.path.join(output_dir, f"semi_synthetic_{dataset}_summary.md")
    _write_markdown_table(summary, md_path)
    logger.info("Saved markdown table to %s", md_path)

    # ---- Write plot ----
    svg_path = os.path.join(output_dir, f"semi_synthetic_{dataset}_summary.svg")
    try:
        _write_summary_plot(summary, svg_path)
        logger.info("Saved plot to %s", svg_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not write plot: %s", exc)

    return {"results": results, "summary": summary}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semi-synthetic treatment-effect experiment driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="california_housing",
        choices=["california_housing", "ihdp"],
        help="Real covariate dataset to use.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of rows to subsample (None = use all).",
    )
    parser.add_argument(
        "--n_experiments",
        type=int,
        default=20,
        help="Number of Monte Carlo replications.",
    )
    parser.add_argument(
        "--treatment_effect",
        type=float,
        default=1.0,
        help="True causal effect theta.",
    )
    parser.add_argument(
        "--eta_distribution",
        default="discrete",
        help="Treatment noise distribution (e.g. discrete, laplace, rademacher).",
    )
    parser.add_argument(
        "--nonlinearity",
        default="identity",
        choices=["identity", "leaky_relu"],
        help="Nonlinearity applied to the linear index.",
    )
    parser.add_argument(
        "--support_size",
        type=int,
        default=None,
        help="Number of covariate columns used in PLR (None = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12143,
        help="Base RNG seed; rep i uses seed+i.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/semi_synthetic",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset of {ica,homl,ols,matching}, or 'all'.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (joblib). -1 = all CPUs.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _parser = _build_parser()
    _args = _parser.parse_args()
    run(
        dataset=_args.dataset,
        n_samples=_args.n_samples,
        n_experiments=_args.n_experiments,
        treatment_effect=_args.treatment_effect,
        eta_distribution=_args.eta_distribution,
        nonlinearity=_args.nonlinearity,
        support_size=_args.support_size,
        seed=_args.seed,
        output_dir=_args.output_dir,
        methods=_args.methods,
        n_jobs=_args.n_jobs,
    )
