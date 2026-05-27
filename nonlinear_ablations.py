"""Ablation experiments for the nonlinear-confounding rebuttal.

Four complementary sweeps that deepen the story established by the initial
nonlinear DGP experiments:

1. **Difficulty-axis isolation** (``--axis_isolation``):
   Turn on exactly one difficulty axis at a time and report per-method RMSE.
   Answers: which single axis breaks OLS most? which advantages ICA over GBM-OML?

2. **Nuisance flexibility crossover** (``--nuisance_ablation``):
   Sweep linear/poly/rf/gbm on the nonlinear-only preset. Finds the crossover
   where OML beats OLS and quantifies "OML is only as good as its first stage."

3. **ICA d/n frontier** (``--ica_dn_frontier``):
   Vary total dimension d and sample size n to map where ICA becomes competitive
   with GBM-OML under heavy-tailed eta. Reconciles "ICA worst on high-dim
   synthetic but best on IHDP."

4. **Nonlinearity strength sweep** (``--strength_sweep``):
   Vary the nonlinearity_strength scalar from 0 (linear) to large, tracing
   OLS-bias and each method's RMSE continuously.

All sweeps use small n_experiments (default 30) and n_jobs=4 to be polite to
concurrent agents.
"""

# kwargs-style config dicts (dict(...)) are intentional for readability here.
# pylint: disable=use-dict-literal

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from nonlinear_dgp import NonlinearDGPConfig
from nonlinear_runner import METHOD_NAMES, run_nonlinear_experiments

# Index shortcuts
_IDX = {name: i for i, name in enumerate(METHOD_NAMES)}
OML_IDX = _IDX["Ortho ML"]
OLS_IDX = _IDX["OLS"]
ICA_IDX = _IDX["ICA"]
MATCH_IDX = _IDX["Matching"]


# ---------------------------------------------------------------------------
# 1. Difficulty-axis isolation
# ---------------------------------------------------------------------------


def run_axis_isolation(
    n_samples: int = 2000,
    n_experiments: int = 30,
    n_jobs: int = 4,
    nuisance: str = "gbm",
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run each difficulty axis in isolation and report per-method RMSE.

    Parameters
    ----------
    n_samples : int
        Sample size per experiment.
    n_experiments : int
        Monte Carlo replications per configuration.
    n_jobs : int
        Joblib parallelism.
    nuisance : str
        Nuisance model for OML/HOML (default 'gbm' — flexible, so OML can de-confound).
    verbose : bool
        Print tables.

    Returns
    -------
    dict mapping axis name -> run_nonlinear_experiments result dict
    """
    # Base config: all axes OFF (linear, Gaussian, low-dim, homoscedastic)
    base_kwargs = dict(
        n_samples=n_samples,
        n_covariates=10,
        support_size=5,
        treatment_effect=1.5,
        sigma_eta=1.0,
        sigma_outcome=1.0,
        alpha_scale=1.0,
        beta_scale=1.0,
        interaction_scale=0.3,
        heteroscedastic_scale=0.5,
        nonlinearity_strength=1.0,
    )

    configurations = {
        "baseline (all-off)": dict(
            nonlinear_confounding=False,
            heavy_tail_eta=False,
            high_dim=False,
            high_dim_d=10,
            heteroscedastic_eps=False,
        ),
        "nonlinear-only": dict(
            nonlinear_confounding=True,
            heavy_tail_eta=False,
            high_dim=False,
            high_dim_d=10,
            heteroscedastic_eps=False,
        ),
        "heavy-tail-only": dict(
            nonlinear_confounding=False,
            heavy_tail_eta=True,
            eta_beta=1.0,
            high_dim=False,
            high_dim_d=10,
            heteroscedastic_eps=False,
        ),
        "heteroscedastic-only": dict(
            nonlinear_confounding=False,
            heavy_tail_eta=False,
            high_dim=False,
            high_dim_d=10,
            heteroscedastic_eps=True,
        ),
        "high-dim-only (d=20)": dict(
            nonlinear_confounding=False,
            heavy_tail_eta=False,
            high_dim=True,
            high_dim_d=20,
            heteroscedastic_eps=False,
            n_covariates=20,
            support_size=5,
        ),
    }

    results = {}
    for label, axis_kwargs in configurations.items():
        cfg_kwargs = {**base_kwargs, **axis_kwargs}
        cfg = NonlinearDGPConfig(**cfg_kwargs)
        r = run_nonlinear_experiments(
            config=cfg,
            n_experiments=n_experiments,
            base_seed=13337,
            n_jobs=n_jobs,
            nuisance=nuisance,
            use_oracle_moments=True,
            verbose=False,
        )
        results[label] = r

    if verbose:
        _print_axis_isolation_table(results, nuisance)

    return results


def _print_axis_isolation_table(results: Dict[str, dict], nuisance: str) -> None:
    """Print a compact RMSE table for the axis-isolation sweep."""
    col_methods = ["Ortho ML", "ICA", "OLS", "Matching"]
    col_idx = [_IDX[m] for m in col_methods]

    header = f"{'Axis':<30}" + "".join(f"{m:>14}" for m in col_methods)
    print(f"\n=== Axis Isolation (nuisance={nuisance}, n={next(iter(results.values()))['n_samples']}) ===")
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        rmse = r["rmse"]
        row = f"{label:<30}" + "".join(f"{rmse[i]:>14.4f}" for i in col_idx)
        print(row)
    print()

    # Highlight which axis breaks OLS most (vs baseline)
    baseline_ols = results["baseline (all-off)"]["rmse"][OLS_IDX]
    print("OLS RMSE lift over baseline (axis -> OLS RMSE / baseline OLS RMSE):")
    for label, r in results.items():
        if label == "baseline (all-off)":
            continue
        ratio = r["rmse"][OLS_IDX] / baseline_ols
        print(f"  {label:<30}  ratio={ratio:.3f}")

    # Highlight which axis advantages ICA over GBM-OML
    print("\nICA RMSE / OML RMSE ratio per axis:")
    for label, r in results.items():
        ica_rmse = r["rmse"][ICA_IDX]
        oml_rmse = r["rmse"][OML_IDX]
        ratio = ica_rmse / oml_rmse if oml_rmse > 1e-9 else float("nan")
        print(f"  {label:<30}  ICA/OML={ratio:.3f}")


# ---------------------------------------------------------------------------
# 2. Nuisance flexibility crossover
# ---------------------------------------------------------------------------


def run_nuisance_ablation(
    n_samples: int = 2000,
    n_experiments: int = 30,
    n_jobs: int = 4,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Sweep nuisance model flexibility on the nonlinear-only preset.

    Uses nonlinear confounding but Gaussian eta and homoscedastic eps so that
    the only challenge is the nonlinear nuisance — OML performance reflects
    purely the first-stage quality.

    Returns
    -------
    dict mapping nuisance name -> run_nonlinear_experiments result dict
    """
    cfg = NonlinearDGPConfig(
        n_samples=n_samples,
        n_covariates=10,
        support_size=5,
        treatment_effect=1.5,
        sigma_eta=1.0,
        sigma_outcome=1.0,
        nonlinear_confounding=True,
        heavy_tail_eta=False,
        high_dim=False,
        heteroscedastic_eps=False,
        alpha_scale=1.0,
        beta_scale=1.0,
        interaction_scale=0.3,
        nonlinearity_strength=1.0,
    )

    nuisance_options = ["linear", "poly", "rf", "gbm"]
    results = {}
    for nuisance in nuisance_options:
        r = run_nonlinear_experiments(
            config=cfg,
            n_experiments=n_experiments,
            base_seed=13337,
            n_jobs=n_jobs,
            nuisance=nuisance,
            use_oracle_moments=True,
            verbose=False,
        )
        results[nuisance] = r

    if verbose:
        _print_nuisance_ablation_table(results)

    return results


def _print_nuisance_ablation_table(results: Dict[str, dict]) -> None:
    """Print RMSE table for nuisance flexibility sweep."""
    col_methods = ["Ortho ML", "Robust Ortho ML", "ICA", "OLS"]
    col_idx = [_IDX[m] for m in col_methods]

    n_samp = next(iter(results.values()))["n_samples"]
    print(f"\n=== Nuisance Flexibility Crossover (nonlinear-only, n={n_samp}) ===")
    header = f"{'Nuisance':<12}" + "".join(f"{m:>20}" for m in col_methods)
    print(header)
    print("-" * len(header))

    ols_rmse_ref = None
    for nuisance, r in results.items():
        rmse = r["rmse"]
        row = f"{nuisance:<12}" + "".join(f"{rmse[i]:>20.4f}" for i in col_idx)
        print(row)
        if ols_rmse_ref is None:
            ols_rmse_ref = rmse[OLS_IDX]

    print("\nOML RMSE / OLS RMSE ratio (crossover when <1.0):")
    for nuisance, r in results.items():
        oml_rmse = r["rmse"][OML_IDX]
        ols_rmse = r["rmse"][OLS_IDX]
        ratio = oml_rmse / ols_rmse if ols_rmse > 1e-9 else float("nan")
        crossover = " *** CROSSOVER ***" if ratio < 1.0 else ""
        print(f"  {nuisance:<12}  OML/OLS={ratio:.3f}{crossover}")


# ---------------------------------------------------------------------------
# 3. ICA d/n frontier
# ---------------------------------------------------------------------------


def run_ica_dn_frontier(
    n_experiments: int = 30,
    n_jobs: int = 4,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Sweep (d_total, n) to find where ICA becomes competitive with GBM-OML.

    Uses heavy-tailed Laplace eta (ICA's ideal regime) and nonlinear confounding
    so GBM-OML has a real first-stage challenge. The key metric is
    ICA_RMSE / GBM-OML_RMSE; values <1.0 indicate ICA wins.

    d_total = n_covariates + 2 (treatment + outcome dimensions fed to ICA).
    We sweep d_covariate in {2, 5, 8} and n in {500, 1000, 2000, 5000}.

    Returns
    -------
    dict mapping (d, n) -> run_nonlinear_experiments result dict (key as string)
    """
    d_covariates_grid = [2, 5, 8]
    n_grid = [500, 1000, 2000, 5000]

    results = {}
    for d_cov in d_covariates_grid:
        s = min(d_cov, 3)  # keep support_size <= d
        for n in n_grid:
            key = f"d={d_cov}, n={n}"
            cfg = NonlinearDGPConfig(
                n_samples=n,
                n_covariates=d_cov,
                support_size=s,
                treatment_effect=1.5,
                sigma_eta=1.0,
                sigma_outcome=1.0,
                nonlinear_confounding=True,
                heavy_tail_eta=True,
                eta_beta=1.0,
                high_dim=False,
                heteroscedastic_eps=False,
                alpha_scale=1.0,
                beta_scale=1.0,
                interaction_scale=0.3,
                nonlinearity_strength=1.0,
            )
            r = run_nonlinear_experiments(
                config=cfg,
                n_experiments=n_experiments,
                base_seed=13337,
                n_jobs=n_jobs,
                nuisance="gbm",
                use_oracle_moments=True,
                verbose=False,
            )
            results[key] = r

    if verbose:
        _print_ica_dn_table(results, d_covariates_grid, n_grid)

    return results


def _print_ica_dn_table(results: Dict[str, dict], d_grid: List[int], n_grid: List[int]) -> None:
    """Print ICA RMSE and ICA/GBM-OML ratio as a d x n matrix."""
    print("\n=== ICA d/n Frontier (heavy-tail eta, nonlinear confounding, GBM nuisance) ===")

    # ICA RMSE table
    print("\nICA RMSE:")
    dn_label = "d\\n"
    header = f"{dn_label:<12}" + "".join(f"{n:>10}" for n in n_grid)
    print(header)
    for d in d_grid:
        row = f"d={d:<10}"
        for n in n_grid:
            key = f"d={d}, n={n}"
            row += f"{results[key]['rmse'][ICA_IDX]:>10.4f}"
        print(row)

    # GBM-OML RMSE table
    print("\nGBM-OML (Ortho ML) RMSE:")
    print(header)
    for d in d_grid:
        row = f"d={d:<10}"
        for n in n_grid:
            key = f"d={d}, n={n}"
            row += f"{results[key]['rmse'][OML_IDX]:>10.4f}"
        print(row)

    # ICA / OML ratio
    print("\nICA RMSE / GBM-OML RMSE  (< 1.0 = ICA wins):")
    print(header)
    for d in d_grid:
        row = f"d={d:<10}"
        for n in n_grid:
            key = f"d={d}, n={n}"
            ica_r = results[key]["rmse"][ICA_IDX]
            oml_r = results[key]["rmse"][OML_IDX]
            ratio = ica_r / oml_r if oml_r > 1e-9 else float("nan")
            mark = "*" if ratio < 1.0 else " "
            row += f"{ratio:>9.3f}{mark}"
        print(row)


# ---------------------------------------------------------------------------
# 4. Nonlinearity strength sweep
# ---------------------------------------------------------------------------


def run_strength_sweep(
    n_samples: int = 2000,
    n_experiments: int = 30,
    n_jobs: int = 4,
    nuisance: str = "gbm",
    verbose: bool = True,
) -> Dict[str, dict]:
    """Vary nonlinearity_strength from 0 (linear) to 3.0 and report per-method RMSE.

    At strength=0 the nonlinear terms vanish and OLS is unbiased; as strength
    grows OLS bias accumulates while GBM-OML stays controlled.

    Parameters
    ----------
    n_samples : int
        Sample size.
    n_experiments : int
        Monte Carlo replications per strength value.
    n_jobs : int
        Joblib parallelism.
    nuisance : str
        Nuisance model ('gbm' recommended to show OML's resilience).
    verbose : bool
        Print results table.

    Returns
    -------
    dict mapping str(strength) -> run_nonlinear_experiments result dict
    """
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    results = {}
    for strength in strengths:
        cfg = NonlinearDGPConfig(
            n_samples=n_samples,
            n_covariates=10,
            support_size=5,
            treatment_effect=1.5,
            sigma_eta=1.0,
            sigma_outcome=1.0,
            nonlinear_confounding=True,
            heavy_tail_eta=False,
            high_dim=False,
            heteroscedastic_eps=False,
            alpha_scale=1.0,
            beta_scale=1.0,
            interaction_scale=0.3,
            heteroscedastic_scale=0.5,
            nonlinearity_strength=strength,
        )
        r = run_nonlinear_experiments(
            config=cfg,
            n_experiments=n_experiments,
            base_seed=13337,
            n_jobs=n_jobs,
            nuisance=nuisance,
            use_oracle_moments=True,
            verbose=False,
        )
        results[str(strength)] = r

    if verbose:
        _print_strength_sweep_table(results, strengths, nuisance)

    return results


def _print_strength_sweep_table(results: Dict[str, dict], strengths: List[float], nuisance: str) -> None:
    """Print OLS bias and per-method RMSE vs nonlinearity strength."""
    col_methods = ["Ortho ML", "ICA", "OLS"]
    col_idx = [_IDX[m] for m in col_methods]

    n_samp = next(iter(results.values()))["n_samples"]
    print(f"\n=== Nonlinearity Strength Sweep (nuisance={nuisance}, n={n_samp}) ===")
    header = f"{'strength':<12}" + "".join(f"{m:>14}" for m in col_methods) + f"{'OLS bias':>14}"
    print(header)
    print("-" * len(header))
    for strength in strengths:
        r = results[str(strength)]
        rmse = r["rmse"]
        ols_bias = r["biases"][OLS_IDX]
        row = f"{strength:<12.3f}" + "".join(f"{rmse[i]:>14.4f}" for i in col_idx) + f"{ols_bias:>14.4f}"
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis_isolation", action="store_true", help="Run difficulty-axis isolation sweep")
    parser.add_argument("--nuisance_ablation", action="store_true", help="Run nuisance flexibility crossover sweep")
    parser.add_argument("--ica_dn_frontier", action="store_true", help="Run ICA d/n frontier sweep")
    parser.add_argument("--strength_sweep", action="store_true", help="Run nonlinearity strength sweep")
    parser.add_argument("--all", action="store_true", help="Run all four sweeps")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_experiments", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--nuisance", choices=["linear", "gbm", "poly", "rf"], default="gbm")
    parser.add_argument("--output_dir", default="figures/nonlinear_ablations")
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    any_selected = any([opts.axis_isolation, opts.nuisance_ablation, opts.ica_dn_frontier, opts.strength_sweep])
    run_all = opts.all or not any_selected

    if run_all or opts.axis_isolation:
        r = run_axis_isolation(
            n_samples=opts.n_samples,
            n_experiments=opts.n_experiments,
            n_jobs=opts.n_jobs,
            nuisance=opts.nuisance,
        )
        np.save(os.path.join(opts.output_dir, "axis_isolation.npy"), r)

    if run_all or opts.nuisance_ablation:
        r = run_nuisance_ablation(
            n_samples=opts.n_samples,
            n_experiments=opts.n_experiments,
            n_jobs=opts.n_jobs,
        )
        np.save(os.path.join(opts.output_dir, "nuisance_ablation.npy"), r)

    if run_all or opts.ica_dn_frontier:
        r = run_ica_dn_frontier(
            n_experiments=opts.n_experiments,
            n_jobs=opts.n_jobs,
        )
        np.save(os.path.join(opts.output_dir, "ica_dn_frontier.npy"), r)

    if run_all or opts.strength_sweep:
        r = run_strength_sweep(
            n_samples=opts.n_samples,
            n_experiments=opts.n_experiments,
            n_jobs=opts.n_jobs,
            nuisance=opts.nuisance,
        )
        np.save(os.path.join(opts.output_dir, "strength_sweep.npy"), r)

    print(f"\nAll ablation results saved to {opts.output_dir}/")


if __name__ == "__main__":
    main()
