"""Producer for the multi-treatment Fig 4 / Fig E.15 results dictionary.

Runs a Monte Carlo grid over ``(sample_size, n_treatments, n_covariates)``
triples for the linear partially linear model (PLR) and dumps a results
dictionary in the schema consumed by
``regenerate_ica_heatmaps.regenerate_main_multi``.

The producer was missing from the repository (the consumer
``regenerate_main_multi`` already exists). This module is the rebuttal-time
replacement that *also* injects the new OLS and per-coordinate Higher-Order
OML baselines required by the NbV6-C2 commitment.

Output schema (all keys are flat, parallel lists indexed by a single
configuration counter; this matches the existing consumer)::

    {
        "sample_sizes":               (K,) int,
        "n_treatments":               (K,) int,
        "n_covariates":               (K,) int,
        "true_params":                (K,) list of torch.Tensor of shape (m,),
        "treatment_effects":          (K,) list of np.ndarray of shape
                                          (n_experiments, m_max) — ICA estimate
                                          (Munkres path),
        "treatment_effects_iv":       (K,) — alias of HOML for legacy consumer,
        "treatment_effects_ica_eps_row": (K,) — non-Munkres ICA fallback,
        "treatment_effects_ols":      (K,) — OLS baseline,
        "treatment_effects_homl":     (K,) — per-coordinate Higher-Order OML,
    }

Per-experiment estimates have shape ``(m_max, )`` where ``m_max`` is the
largest treatment count in ``n_treatments_grid``; configurations with
``m < m_max`` are right-padded with ``np.nan`` so a single ndarray suffices
across configurations.

Example
-------
::

    python multi_treatment_runner.py \\
        --n_experiments 20 \\
        --sample_sizes 500 1000 2000 5000 10000 \\
        --n_treatments 1 2 5 \\
        --n_covariates 10 20 50 \\
        --include_baselines \\
        --output_dir figures/multi_treatment

"""

from __future__ import annotations

import argparse
import os
from itertools import product
from typing import Any, Dict, List

import numpy as np
import torch
from joblib import Parallel, delayed

from baselines import ols_baseline
from ica import (
    generate_ica_data,
    ica_treatment_effect_estimation,
    ica_treatment_effect_estimation_eps_row,
)
from main_estimation import all_together_cross_fitting

# Mapping from rebuttal-level "eta_distribution" string to the gennorm beta
# expected by ica.generate_ica_data. The repo treats "discrete" as a
# light-tailed light-kurtosis source (mirroring oml_runner.gennorm_light),
# i.e. beta=4. "laplace" uses beta=1, "gauss" uses beta=2.
ETA_DISTRIBUTION_TO_GENNORM_BETA = {
    "discrete": 4.0,
    "gennorm_light": 4.0,
    "gennorm_heavy": 1.0,
    "laplace": 1.0,
    "gauss": 2.0,
    "uniform": 4.0,
}


def _eta_distribution_beta(eta_distribution: str) -> float:
    """Return the gennorm shape parameter for the eta distribution name."""
    try:
        return ETA_DISTRIBUTION_TO_GENNORM_BETA[eta_distribution]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported eta_distribution '{eta_distribution}'. "
            f"Supported: {sorted(ETA_DISTRIBUTION_TO_GENNORM_BETA)}"
        ) from exc


def _per_coordinate_homl(
    covariates: np.ndarray,
    treatment_matrix: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    """Apply scalar Higher-Order OML to each treatment coordinate.

    The repo's ``main_estimation.all_together_cross_fitting`` returns a scalar
    treatment effect. For ``m >= 2`` we run it ``m`` times, treating each
    column ``j`` of the treatment matrix as the active scalar treatment and
    the remaining ``m - 1`` columns as additional nuisance covariates. This
    matches the per-coordinate strategy described in Mackey et al. (2018) and
    is the strategy mandated by Risk R8 of the rebuttal plan: do **not**
    invent a vector OML.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix of shape ``(n, d)``.
    treatment_matrix : np.ndarray
        Treatment matrix of shape ``(n, m)``.
    outcome : np.ndarray
        Outcome vector of shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Per-coordinate Higher-Order OML estimates, shape ``(m,)``.
    """
    n, m = treatment_matrix.shape
    estimates = np.full(m, np.nan)
    for j in range(m):
        active = treatment_matrix[:, j]
        if m > 1:
            other_treatments = np.delete(treatment_matrix, j, axis=1)
            extended_covariates = np.concatenate([covariates, other_treatments], axis=1)
        else:
            extended_covariates = covariates

        # Empirical second / third moments of the active treatment
        # (consistent with the runtime convention of Fig 2 / Fig E.5 — the
        # "discrete"-noise default treats moments as known constants of the
        # marginal distribution; per-coordinate we use empirical estimates
        # because the per-coordinate marginal of T_j is data-dependent under
        # nonzero confounding).
        eta_second = float(np.var(active))
        eta_third = float(np.mean((active - active.mean()) ** 3))

        try:
            _, robust_ortho_ml, _, _, _, _ = all_together_cross_fitting(
                extended_covariates,
                active,
                outcome,
                treatment_second_moment=eta_second,
                treatment_third_moment=eta_third,
            )
            estimates[j] = float(robust_ortho_ml)
        except Exception:  # pylint: disable=broad-except
            estimates[j] = np.nan
    return estimates


def _run_single_experiment(
    n: int,
    m: int,
    d: int,
    nonlinearity: str,
    gennorm_beta: float,
    include_baselines: bool,
    seed: int,
) -> Dict[str, Any]:
    """Run a single Monte Carlo replicate for the (n, m, d) cell.

    Returns
    -------
    dict
        Keys: ``"theta_true"`` (torch.Tensor, shape (m,)),
        ``"theta_ica"`` (np.ndarray, shape (m,)),
        ``"theta_ica_eps_row"`` (np.ndarray, shape (m,)),
        ``"theta_ols"`` (np.ndarray, shape (m,) or NaN array),
        ``"theta_homl"`` (np.ndarray, shape (m,) or NaN array).
    """
    # Pin the seed inside the worker so results are reproducible regardless of
    # joblib backend / worker count.
    np.random.seed(seed)
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)

    S, X, theta_true = generate_ica_data(
        n_covariates=d,
        n_treatments=m,
        batch_size=n,
        beta=gennorm_beta,
        nonlinearity=nonlinearity,
    )

    X_np = X.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()

    covariates = X_np[:, :d]
    treatment_matrix = X_np[:, d : d + m]
    outcome = X_np[:, -1]

    # ICA — Munkres-based estimate (canonical Fig 4 path).
    try:
        theta_ica, _ = ica_treatment_effect_estimation(
            X_np,
            S_np,
            random_state=seed,
            n_treatments=m,
            verbose=False,
        )
    except Exception:  # pylint: disable=broad-except
        theta_ica = np.full(m, np.nan)

    # ICA fallback — eps-row (no Munkres). Risk R2 mitigation for m >= 2.
    try:
        theta_ica_eps, _ = ica_treatment_effect_estimation_eps_row(
            X_np,
            S=S_np,
            random_state=seed,
            n_treatments=m,
            verbose=False,
        )
    except Exception:  # pylint: disable=broad-except
        theta_ica_eps = np.full(m, np.nan)

    if include_baselines:
        try:
            theta_ols = ols_baseline(covariates, treatment_matrix, outcome)
        except Exception:  # pylint: disable=broad-except
            theta_ols = np.full(m, np.nan)

        theta_homl = _per_coordinate_homl(covariates, treatment_matrix, outcome)
    else:
        theta_ols = np.full(m, np.nan)
        theta_homl = np.full(m, np.nan)

    return {
        "theta_true": theta_true,  # torch.Tensor, shape (m,)
        "theta_ica": np.asarray(theta_ica, dtype=float).reshape(m),
        "theta_ica_eps_row": np.asarray(theta_ica_eps, dtype=float).reshape(m),
        "theta_ols": np.asarray(theta_ols, dtype=float).reshape(m),
        "theta_homl": np.asarray(theta_homl, dtype=float).reshape(m),
    }


def _pad_to(arr: np.ndarray, m_max: int) -> np.ndarray:
    """Right-pad a 1D array to length ``m_max`` with NaN."""
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size >= m_max:
        return arr[:m_max]
    out = np.full(m_max, np.nan)
    out[: arr.size] = arr
    return out


def run_multi_treatment_experiment(
    sample_sizes: List[int] = [500, 1000, 2000, 5000, 10000],
    n_treatments_grid: List[int] = [1, 2, 5],
    n_covariates_grid: List[int] = [10, 20, 50],
    n_experiments: int = 20,
    nonlinearity: str = "identity",
    eta_distribution: str = "discrete",
    output_path: str = "figures/multi_treatment/results_multi_treatment.npy",
    include_baselines: bool = True,
    seed: int = 12143,
    n_jobs: int = -1,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Run the Fig 4 / Fig E.15 multi-treatment Monte Carlo grid.

    For each ``(n, m, d)`` triple in
    ``product(sample_sizes, n_treatments_grid, n_covariates_grid)``, runs
    ``n_experiments`` Monte Carlo replicates and aggregates ICA, ICA
    eps-row fallback, OLS, and per-coordinate Higher-Order OML estimates.

    Parameters
    ----------
    sample_sizes : list[int]
        Sample sizes ``n`` to sweep over.
    n_treatments_grid : list[int]
        Treatment dimensions ``m`` to sweep over.
    n_covariates_grid : list[int]
        Covariate dimensions ``d`` to sweep over.
    n_experiments : int
        Monte Carlo replicates per configuration.
    nonlinearity : str
        Activation passed to :func:`ica.generate_ica_data`. ``"identity"``
        encodes the linear PLR setting required by the rebuttal commitment.
        ``ica.generate_ica_data`` does not currently accept ``"identity"``,
        so we map it to ``"leaky_relu"`` with ``slope=1.0`` (which equals
        the identity).
    eta_distribution : str
        Treatment-noise family. Translated to a gennorm shape ``beta`` via
        :data:`ETA_DISTRIBUTION_TO_GENNORM_BETA`. ``"discrete"`` → ``beta=4``
        following the convention in ``oml_runner.gennorm_light``.
    output_path : str
        Destination ``.npy`` file. Parent directories are created if missing.
    include_baselines : bool
        If ``True`` (default), compute OLS and per-coordinate HOML baselines
        in addition to the ICA estimates.
    seed : int
        Master seed. Per-experiment seeds are derived deterministically from
        the configuration index and replicate index.
    n_jobs : int
        ``joblib.Parallel`` worker count; ``-1`` uses all available cores.
    verbose : int
        Joblib verbosity.

    Returns
    -------
    dict
        Results dictionary; see module docstring for the schema.
    """
    # Map "identity" (linear PLR) to leaky_relu(slope=1) which is exactly the
    # identity. ica.generate_ica_data validates nonlinearity against a fixed
    # set; "identity" is not in that set so we route through leaky_relu.
    is_identity = nonlinearity == "identity"
    effective_nonlinearity = "leaky_relu" if is_identity else nonlinearity

    gennorm_beta = _eta_distribution_beta(eta_distribution)

    m_max = max(n_treatments_grid)

    # Build the configuration list (parallel flat lists).
    sample_sizes_out: List[int] = []
    n_treatments_out: List[int] = []
    n_covariates_out: List[int] = []
    true_params_out: List[torch.Tensor] = []
    treatment_effects_out: List[np.ndarray] = []
    treatment_effects_eps_row_out: List[np.ndarray] = []
    treatment_effects_ols_out: List[np.ndarray] = []
    treatment_effects_homl_out: List[np.ndarray] = []

    configurations = list(product(sample_sizes, n_treatments_grid, n_covariates_grid))

    for config_idx, (n, m, d) in enumerate(configurations):
        # Deterministic per-experiment seed: depends on master seed,
        # configuration index, and replicate index.
        rep_seeds = [seed + 1000 * config_idx + r for r in range(n_experiments)]

        if is_identity and effective_nonlinearity == "leaky_relu":
            # leaky_relu with slope=1 is identity; pass slope=1 via closure.
            # generate_ica_data exposes slope as a top-level kwarg.
            results: List[Dict[str, Any]] = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_run_single_experiment_with_slope)(
                    n, m, d, effective_nonlinearity, gennorm_beta, include_baselines, s, 1.0
                )
                for s in rep_seeds
            )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_run_single_experiment)(n, m, d, effective_nonlinearity, gennorm_beta, include_baselines, s)
                for s in rep_seeds
            )

        # Stack per-rep estimates into shape (n_experiments, m_max).
        theta_ica_stack = np.vstack([_pad_to(r["theta_ica"], m_max) for r in results])
        theta_eps_stack = np.vstack([_pad_to(r["theta_ica_eps_row"], m_max) for r in results])
        theta_ols_stack = np.vstack([_pad_to(r["theta_ols"], m_max) for r in results])
        theta_homl_stack = np.vstack([_pad_to(r["theta_homl"], m_max) for r in results])

        # True params: identical across reps for the (m, d) cell as long as
        # ica.generate_ica_data has theta_choice="fixed" (the default), so
        # store the first replicate's value. This matches the legacy schema
        # ``true_params[i].numpy()`` access in regenerate_ica_heatmaps.
        true_theta = results[0]["theta_true"]

        sample_sizes_out.append(int(n))
        n_treatments_out.append(int(m))
        n_covariates_out.append(int(d))
        true_params_out.append(true_theta)
        treatment_effects_out.append(theta_ica_stack)
        treatment_effects_eps_row_out.append(theta_eps_stack)
        treatment_effects_ols_out.append(theta_ols_stack)
        treatment_effects_homl_out.append(theta_homl_stack)

    # Legacy 'treatment_effects_iv' alias points at HOML per-coordinate
    # estimates so the existing consumer
    # ``regenerate_ica_heatmaps.regenerate_main_multi`` can read it without
    # modification (Risk R5).
    results_dict: Dict[str, Any] = {
        "sample_sizes": sample_sizes_out,
        "n_treatments": n_treatments_out,
        "n_covariates": n_covariates_out,
        "true_params": true_params_out,
        "treatment_effects": treatment_effects_out,
        "treatment_effects_iv": treatment_effects_homl_out,
        "treatment_effects_ica_eps_row": treatment_effects_eps_row_out,
        "treatment_effects_ols": treatment_effects_ols_out,
        "treatment_effects_homl": treatment_effects_homl_out,
        "metadata": {
            "n_experiments": n_experiments,
            "nonlinearity": nonlinearity,
            "eta_distribution": eta_distribution,
            "gennorm_beta": gennorm_beta,
            "include_baselines": include_baselines,
            "seed": seed,
            "m_max": m_max,
            "shape_per_config": "(n_experiments, m_max) NaN-padded",
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, results_dict, allow_pickle=True)

    return results_dict


def _run_single_experiment_with_slope(
    n: int,
    m: int,
    d: int,
    nonlinearity: str,
    gennorm_beta: float,
    include_baselines: bool,
    seed: int,
    slope: float,
) -> Dict[str, Any]:
    """Variant of :func:`_run_single_experiment` that passes ``slope`` through.

    Used when the user requested ``nonlinearity="identity"``; we route
    through ``leaky_relu`` with ``slope=1`` which is exactly the identity
    map. Splitting this out keeps the joblib closure cleanly serializable.
    """
    np.random.seed(seed)
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)

    S, X, theta_true = generate_ica_data(
        n_covariates=d,
        n_treatments=m,
        batch_size=n,
        beta=gennorm_beta,
        nonlinearity=nonlinearity,
        slope=slope,
    )

    X_np = X.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()

    covariates = X_np[:, :d]
    treatment_matrix = X_np[:, d : d + m]
    outcome = X_np[:, -1]

    try:
        theta_ica, _ = ica_treatment_effect_estimation(
            X_np,
            S_np,
            random_state=seed,
            n_treatments=m,
            verbose=False,
        )
    except Exception:  # pylint: disable=broad-except
        theta_ica = np.full(m, np.nan)

    try:
        theta_ica_eps, _ = ica_treatment_effect_estimation_eps_row(
            X_np,
            S=S_np,
            random_state=seed,
            n_treatments=m,
            verbose=False,
        )
    except Exception:  # pylint: disable=broad-except
        theta_ica_eps = np.full(m, np.nan)

    if include_baselines:
        try:
            theta_ols = ols_baseline(covariates, treatment_matrix, outcome)
        except Exception:  # pylint: disable=broad-except
            theta_ols = np.full(m, np.nan)

        theta_homl = _per_coordinate_homl(covariates, treatment_matrix, outcome)
    else:
        theta_ols = np.full(m, np.nan)
        theta_homl = np.full(m, np.nan)

    return {
        "theta_true": theta_true,
        "theta_ica": np.asarray(theta_ica, dtype=float).reshape(m),
        "theta_ica_eps_row": np.asarray(theta_ica_eps, dtype=float).reshape(m),
        "theta_ols": np.asarray(theta_ols, dtype=float).reshape(m),
        "theta_homl": np.asarray(theta_homl, dtype=float).reshape(m),
    }


def _parse_args() -> argparse.Namespace:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Producer for figures/multi_treatment/results_multi_treatment.npy")
    parser.add_argument("--n_experiments", type=int, default=20)
    parser.add_argument("--sample_sizes", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--n_treatments", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--n_covariates", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--nonlinearity", type=str, default="identity")
    parser.add_argument("--eta_distribution", type=str, default="discrete")
    parser.add_argument("--seed", type=int, default=12143)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/multi_treatment",
        help="Output directory; the file is named results_multi_treatment.npy.",
    )
    bl_group = parser.add_mutually_exclusive_group()
    bl_group.add_argument(
        "--include_baselines",
        dest="include_baselines",
        action="store_true",
        help="Compute OLS + per-coordinate HOML baselines (default).",
    )
    bl_group.add_argument(
        "--no_baselines",
        dest="include_baselines",
        action="store_false",
        help="Skip baselines; ICA-only.",
    )
    parser.set_defaults(include_baselines=True)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    output_path = os.path.join(args.output_dir, "results_multi_treatment.npy")

    print(
        f"Running multi-treatment experiment: "
        f"n_samples={args.sample_sizes}, "
        f"n_treatments={args.n_treatments}, "
        f"n_covariates={args.n_covariates}, "
        f"n_experiments={args.n_experiments}, "
        f"baselines={args.include_baselines}, "
        f"output={output_path}"
    )

    run_multi_treatment_experiment(
        sample_sizes=args.sample_sizes,
        n_treatments_grid=args.n_treatments,
        n_covariates_grid=args.n_covariates,
        n_experiments=args.n_experiments,
        nonlinearity=args.nonlinearity,
        eta_distribution=args.eta_distribution,
        output_path=output_path,
        include_baselines=args.include_baselines,
        seed=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
