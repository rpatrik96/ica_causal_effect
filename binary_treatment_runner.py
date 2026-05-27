"""Monte Carlo runner for the binary-treatment partially linear model.

Companion to :mod:`binary_treatment_dgp`. Runs all five estimators
(Ortho ML, HOML known, HOML est, HOML split, ICA-eps-row, OLS,
propensity-score matching) on data drawn from
:func:`binary_treatment_dgp.generate_binary_treatment_data` and reports
bias / sigma / RMSE per method.

The motivation is reviewer feedback asking for binary-treatment
experiments. The other DGPs in this repo all generate continuous T
(via T = m(X) + eta), even when the eta noise is Bernoulli. This
runner uses a *genuinely* binary T sampled from Bernoulli(p(X)).

Example
-------
::

    python binary_treatment_runner.py \
        --n_samples 2000 \
        --n_experiments 50 \
        --n_covariates 10 \
        --treatment_effect 1.5 \
        --output_dir figures/binary_treatment

The output dictionary is saved to
``<output_dir>/binary_treatment_results.npy`` and contains per-method
arrays of estimates plus aggregate statistics.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

from baselines import matching_baseline, ols_baseline
from binary_treatment_dgp import (
    BinaryTreatmentDGPConfig,
    empirical_eta_moments,
    generate_binary_treatment_data,
)
from main_estimation import all_together_cross_fitting

METHOD_NAMES: Tuple[str, ...] = (
    "Ortho ML",
    "Robust Ortho ML",
    "Robust Ortho Est",
    "Robust Ortho Split",
    "ICA",
    "OLS",
    "Matching",
)


def _single_binary_run(
    seed: int,
    config: BinaryTreatmentDGPConfig,
    use_oracle_moments: bool,
) -> Tuple[float, ...]:
    """Run all estimators on a single binary-treatment sample.

    Returns the seven per-method estimates in the order of
    :data:`METHOD_NAMES`.
    """
    cfg = BinaryTreatmentDGPConfig(
        n_samples=config.n_samples,
        n_covariates=config.n_covariates,
        support_size=config.support_size,
        treatment_effect=config.treatment_effect,
        propensity_strength=config.propensity_strength,
        outcome_coef_scale=config.outcome_coef_scale,
        sigma_outcome=config.sigma_outcome,
        logit_clip=config.logit_clip,
        seed=seed,
    )
    X, T, Y, propensity, eta_oracle, _, _ = generate_binary_treatment_data(cfg)

    # Pick the eta moments fed to HOML (known path):
    # - oracle: use the *true* eta = T - p(X) sample moments
    # - non-oracle: use a residualised T - hat{p}(X) inside the cross-fit (which
    #   the cross-fitting pipeline computes internally), but still pass a
    #   reasonable second/third moment for the "known" path. We default to the
    #   oracle eta moments for a clean comparison; reviewers can rerun with
    #   --no_oracle_moments to see the gap.
    if use_oracle_moments:
        eta_second_moment, eta_third_cumulant = empirical_eta_moments(eta_oracle)
    else:
        # Fall back to residualised T after a quick OLS-style propensity fit.
        # Marginal Bernoulli with p ≈ T.mean() gives reasonable defaults.
        p_hat = float(np.clip(np.mean(T), 1e-3, 1 - 1e-3))
        eta_second_moment = p_hat * (1.0 - p_hat)
        eta_third_cumulant = p_hat * (1.0 - p_hat) * (1.0 - 2.0 * p_hat)

    (
        ortho_ml,
        robust_ortho_ml,
        robust_ortho_est_ml,
        robust_ortho_est_split_ml,
        _treatment_coef,
        _outcome_coef,
    ) = all_together_cross_fitting(
        X,
        T,
        Y,
        treatment_second_moment=eta_second_moment,
        treatment_third_moment=eta_third_cumulant,
    )

    # ICA via the eps-row identification (does not require Munkres against
    # ground-truth sources, since for binary T the "source" is degenerate).
    # Imported lazily so the runner stays importable on environments without
    # torch (the ica module imports torch at module load).
    try:
        from ica import ica_treatment_effect_estimation_eps_row  # noqa: WPS433

        ica_estimate, _ = ica_treatment_effect_estimation_eps_row(
            np.hstack((X, T.reshape(-1, 1), Y.reshape(-1, 1))),
            S=None,
            check_convergence=False,
            verbose=False,
        )
        ica_value = float(ica_estimate[0]) if np.isfinite(ica_estimate).all() else float("nan")
    except Exception:  # pylint: disable=broad-exception-caught
        ica_value = float("nan")

    ols_value = float(ols_baseline(X, T, Y)[0])
    matching_value = float(matching_baseline(X, T, Y, treatment_kind="binary"))

    return (
        float(ortho_ml),
        float(robust_ortho_ml),
        float(robust_ortho_est_ml),
        float(robust_ortho_est_split_ml),
        ica_value,
        ols_value,
        matching_value,
    )


def run_binary_treatment_experiments(
    config: BinaryTreatmentDGPConfig,
    n_experiments: int = 50,
    base_seed: int = 12143,
    n_jobs: int = -1,
    use_oracle_moments: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Run ``n_experiments`` Monte Carlo replications of the binary-T DGP.

    Each replication uses ``base_seed + i`` as the DGP seed so the runs
    are independent yet fully reproducible.
    """
    seeds = [base_seed + i for i in range(n_experiments)]

    results: List[Tuple[float, ...]] = Parallel(n_jobs=n_jobs)(
        delayed(_single_binary_run)(seed, config, use_oracle_moments) for seed in seeds
    )

    estimates = np.array(results, dtype=float)  # (n_experiments, n_methods)
    # Per-method finite filtering: a NaN from one estimator (commonly ICA on
    # binary T, where the source structure is degenerate) must not invalidate
    # the other methods' bias/sigma/RMSE. We compute per-column statistics
    # using nan-aware reductions.
    finite_per_col = np.isfinite(estimates).sum(axis=0)
    if verbose:
        for name, n_finite in zip(METHOD_NAMES, finite_per_col):
            n_non = len(seeds) - int(n_finite)
            if n_non:
                print(
                    f"[binary_treatment_runner] {name}: {n_non}/{len(seeds)} "
                    "replications produced non-finite estimates and were dropped."
                )

    # estimates_finite: rows where every method finite (legacy companion array)
    finite_mask = np.all(np.isfinite(estimates), axis=1)
    estimates_finite = estimates[finite_mask]

    biases = np.nanmean(estimates - config.treatment_effect, axis=0)
    sigmas = np.nanstd(estimates, axis=0)
    rmse = np.sqrt(biases**2 + sigmas**2)

    if verbose:
        print(f"\nBinary-treatment results (theta = {config.treatment_effect}):")
        print(f"{'Method':<22}{'bias':>10}{'sigma':>10}{'rmse':>10}")
        for name, b, s, r in zip(METHOD_NAMES, biases, sigmas, rmse):
            print(f"{name:<22}{b:>10.4f}{s:>10.4f}{r:>10.4f}")

    return {
        "method_names": np.array(METHOD_NAMES),
        "estimates": estimates,
        "estimates_finite": estimates_finite,
        "biases": biases,
        "sigmas": sigmas,
        "rmse": rmse,
        "n_experiments": int(finite_mask.sum()),
        "finite_per_method": finite_per_col,
        "n_attempted": len(seeds),
        "treatment_effect": float(config.treatment_effect),
        "n_samples": config.n_samples,
        "n_covariates": config.n_covariates,
        "support_size": config.support_size,
        "propensity_strength": config.propensity_strength,
        "sigma_outcome": config.sigma_outcome,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_experiments", type=int, default=50)
    parser.add_argument("--n_covariates", type=int, default=10)
    parser.add_argument("--support_size", type=int, default=5)
    parser.add_argument("--treatment_effect", type=float, default=1.5)
    parser.add_argument("--propensity_strength", type=float, default=1.0)
    parser.add_argument("--outcome_coef_scale", type=float, default=1.0)
    parser.add_argument("--sigma_outcome", type=float, default=1.0)
    parser.add_argument("--logit_clip", type=float, default=6.0)
    parser.add_argument("--base_seed", type=int, default=12143)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--no_oracle_moments",
        action="store_true",
        help="Use marginal Bernoulli moments instead of the oracle eta sample moments.",
    )
    parser.add_argument("--output_dir", default="figures/binary_treatment")
    parser.add_argument(
        "--results_file",
        default=None,
        help=(
            "Output filename inside --output_dir. If omitted, encodes the key "
            "knobs into the filename so concurrent cluster jobs do not clobber "
            "each other (binary_treatment_results_n{n}_d{d}_p{prop}.npy)."
        ),
    )
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    config = BinaryTreatmentDGPConfig(
        n_samples=opts.n_samples,
        n_covariates=opts.n_covariates,
        support_size=opts.support_size,
        treatment_effect=opts.treatment_effect,
        propensity_strength=opts.propensity_strength,
        outcome_coef_scale=opts.outcome_coef_scale,
        sigma_outcome=opts.sigma_outcome,
        logit_clip=opts.logit_clip,
    )

    results = run_binary_treatment_experiments(
        config=config,
        n_experiments=opts.n_experiments,
        base_seed=opts.base_seed,
        n_jobs=opts.n_jobs,
        use_oracle_moments=not opts.no_oracle_moments,
        verbose=True,
    )

    if opts.results_file is None:
        results_file = (
            f"binary_treatment_results_n{opts.n_samples}_d{opts.n_covariates}" f"_p{opts.propensity_strength}.npy"
        )
    else:
        results_file = opts.results_file
    out_path = os.path.join(opts.output_dir, results_file)
    np.save(out_path, results, allow_pickle=True)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
