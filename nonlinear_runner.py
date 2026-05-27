"""Monte Carlo runner for the nonlinear-confounding partially linear model.

Companion to :mod:`nonlinear_dgp`. Runs all seven estimators
(Ortho ML, HOML known, HOML est, HOML split, ICA-eps-row, OLS, matching)
on data drawn from :func:`nonlinear_dgp.generate_nonlinear_data` and reports
bias / sigma / RMSE per method.

The motivation is reviewer feedback asking for experiments where OLS visibly
breaks down. With nonlinear confounding (``--nonlinear_confounding``) OLS is
biased because it cannot capture the nonlinear g(X) and m(X). OML/HOML with
a *flexible* nuisance (``--nuisance linear|gbm|poly``) survive.

Example
-------
::

    # Minimal smoke test (linear Lasso nuisance)
    python nonlinear_runner.py \\
        --n_samples 2000 --n_experiments 20 \\
        --nonlinear_confounding --heavy_tail_eta \\
        --output_dir figures/nonlinear

    # Full hard preset with GBM nuisance (recommended for the paper)
    python nonlinear_runner.py \\
        --n_samples 2000 --n_experiments 50 \\
        --nonlinear_confounding --heavy_tail_eta \\
        --heteroscedastic_eps \\
        --nuisance gbm \\
        --output_dir figures/nonlinear

The output dictionary is saved to
``<output_dir>/nonlinear_results_<tag>.npy`` and contains per-method
arrays of estimates plus aggregate statistics.

Nuisance options
----------------
``linear``
    The default ``LassoCV`` from ``main_estimation.py``. Biased when
    confounding is nonlinear -- deliberately weak, so the reader can see
    OML also breaks without a flexible first stage.

``gbm``
    ``sklearn.ensemble.GradientBoostingRegressor`` with sensible defaults.
    Can approximate nonlinear m and g; rescues OML/HOML under nonlinear
    confounding.

``poly``
    Polynomial feature pipeline (degree 2) + Ridge. Cheaper than GBM,
    good enough for mild nonlinearity.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from baselines import matching_baseline, ols_baseline
from main_estimation import DEFAULT_LASSO_ALPHAS, all_together_cross_fitting
from nonlinear_dgp import (
    NonlinearDGPConfig,
    empirical_eta_moments,
    eta_moments_from_config,
    generate_nonlinear_data,
)

METHOD_NAMES: Tuple[str, ...] = (
    "Ortho ML",
    "Robust Ortho ML",
    "Robust Ortho Est",
    "Robust Ortho Split",
    "ICA",
    "OLS",
    "Matching",
)


def _make_nuisance_models(nuisance: str):
    """Return (model_treatment, model_outcome) for the chosen nuisance type.

    Parameters
    ----------
    nuisance : str
        One of ``"linear"``, ``"gbm"``, ``"poly"``, or ``"rf"``.

    Returns
    -------
    model_treatment, model_outcome : sklearn estimator instances (unfitted)
    """
    if nuisance == "linear":
        mt = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
        mo = LassoCV(alphas=DEFAULT_LASSO_ALPHAS)
    elif nuisance == "gbm":
        mt = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0)
        mo = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=1)
    elif nuisance == "poly":
        mt = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        mo = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
    elif nuisance == "rf":
        mt = RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=5, random_state=0)
        mo = RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=5, random_state=1)
    else:
        raise ValueError(f"nuisance must be 'linear', 'gbm', 'poly', or 'rf'; got '{nuisance}'")
    return mt, mo


def _single_nonlinear_run(
    seed: int,
    config: NonlinearDGPConfig,
    nuisance: str,
    use_oracle_moments: bool,
) -> Tuple[float, ...]:
    """Run all estimators on one nonlinear-confounding sample.

    Returns the seven per-method estimates in the order of
    :data:`METHOD_NAMES`.
    """
    cfg = NonlinearDGPConfig(
        n_samples=config.n_samples,
        n_covariates=config.n_covariates,
        support_size=config.support_size,
        treatment_effect=config.treatment_effect,
        sigma_eta=config.sigma_eta,
        sigma_outcome=config.sigma_outcome,
        nonlinear_confounding=config.nonlinear_confounding,
        heavy_tail_eta=config.heavy_tail_eta,
        eta_beta=config.eta_beta,
        high_dim=config.high_dim,
        high_dim_d=config.high_dim_d,
        heteroscedastic_eps=config.heteroscedastic_eps,
        alpha_scale=config.alpha_scale,
        beta_scale=config.beta_scale,
        interaction_scale=config.interaction_scale,
        heteroscedastic_scale=config.heteroscedastic_scale,
        nonlinearity_strength=config.nonlinearity_strength,
        seed=seed,
    )
    X, T, Y, eta_sample, _m_X, _g_X, _alpha = generate_nonlinear_data(cfg)

    if use_oracle_moments:
        eta_second_moment, eta_third_cumulant = eta_moments_from_config(cfg)
    else:
        eta_second_moment, eta_third_cumulant = empirical_eta_moments(eta_sample)

    mt, mo = _make_nuisance_models(nuisance)
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
        model_treatment=mt,
        model_outcome=mo,
    )

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
    matching_value = float(matching_baseline(X, T, Y, treatment_kind="continuous"))

    return (
        float(ortho_ml),
        float(robust_ortho_ml),
        float(robust_ortho_est_ml),
        float(robust_ortho_est_split_ml),
        ica_value,
        ols_value,
        matching_value,
    )


def run_nonlinear_experiments(
    config: NonlinearDGPConfig,
    n_experiments: int = 50,
    base_seed: int = 13337,
    n_jobs: int = -1,
    nuisance: str = "linear",
    use_oracle_moments: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Run ``n_experiments`` Monte Carlo replications of the nonlinear DGP.

    Each replication uses ``base_seed + i`` as the DGP seed so the runs are
    independent yet fully reproducible.

    Parameters
    ----------
    config : NonlinearDGPConfig
        DGP configuration (difficulty toggles + sample size etc.).
    n_experiments : int
        Number of Monte Carlo replications.
    base_seed : int
        First seed; replication i uses seed ``base_seed + i``.
    n_jobs : int
        Joblib parallelism (-1 = all cores, 1 = serial).
    nuisance : str
        Nuisance model for OML/HOML. One of ``"linear"``, ``"gbm"``, ``"poly"``.
    use_oracle_moments : bool
        If True, feed HOML the population eta moments. If False, use empirical
        sample moments from the observed treatment residuals.
    verbose : bool
        Print a formatted results table.

    Returns
    -------
    dict with keys:
        method_names, estimates, estimates_finite, biases, sigmas, rmse,
        n_experiments, finite_per_method, n_attempted, treatment_effect,
        n_samples, n_covariates, nuisance, nonlinear_confounding,
        heavy_tail_eta, eta_beta, high_dim, heteroscedastic_eps.
    """
    seeds = [base_seed + i for i in range(n_experiments)]

    results: List[Tuple[float, ...]] = Parallel(n_jobs=n_jobs)(
        delayed(_single_nonlinear_run)(seed, config, nuisance, use_oracle_moments) for seed in seeds
    )

    estimates = np.array(results, dtype=float)

    finite_per_col = np.isfinite(estimates).sum(axis=0)
    if verbose:
        for name, n_finite in zip(METHOD_NAMES, finite_per_col):
            n_non = len(seeds) - int(n_finite)
            if n_non:
                print(
                    f"[nonlinear_runner] {name}: {n_non}/{len(seeds)} "
                    "replications produced non-finite estimates and were dropped."
                )

    finite_mask = np.all(np.isfinite(estimates), axis=1)
    estimates_finite = estimates[finite_mask]

    biases = np.nanmean(estimates - config.treatment_effect, axis=0)
    sigmas = np.nanstd(estimates, axis=0)
    rmse = np.sqrt(biases**2 + sigmas**2)

    d_eff = config.high_dim_d if config.high_dim else config.n_covariates

    if verbose:
        axes = []
        if config.nonlinear_confounding:
            axes.append("nonlinear")
        if config.heavy_tail_eta:
            axes.append(f"heavy-tail(beta={config.eta_beta})")
        if config.high_dim:
            axes.append(f"high-dim(d={d_eff})")
        if config.heteroscedastic_eps:
            axes.append("heteroscedastic")
        axes_str = "+".join(axes) if axes else "linear (all off)"

        print(f"\nNonlinear DGP results -- axes: {axes_str}, nuisance: {nuisance}")
        print(f"n={config.n_samples}, d={d_eff}, theta={config.treatment_effect}")
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
        "n_covariates": d_eff,
        "support_size": config.support_size,
        "nuisance": nuisance,
        "nonlinear_confounding": config.nonlinear_confounding,
        "heavy_tail_eta": config.heavy_tail_eta,
        "eta_beta": config.eta_beta,
        "high_dim": config.high_dim,
        "heteroscedastic_eps": config.heteroscedastic_eps,
        "sigma_eta": config.sigma_eta,
        "sigma_outcome": config.sigma_outcome,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_covariates", type=int, default=10)
    parser.add_argument("--support_size", type=int, default=5)
    parser.add_argument("--treatment_effect", type=float, default=1.5)
    parser.add_argument("--sigma_eta", type=float, default=1.0)
    parser.add_argument("--sigma_outcome", type=float, default=1.0)
    parser.add_argument("--alpha_scale", type=float, default=1.0)
    parser.add_argument("--beta_scale", type=float, default=1.0)
    parser.add_argument("--interaction_scale", type=float, default=0.3)
    parser.add_argument("--heteroscedastic_scale", type=float, default=0.5)
    parser.add_argument("--nonlinear_confounding", action="store_true")
    parser.add_argument("--heavy_tail_eta", action="store_true")
    parser.add_argument("--eta_beta", type=float, default=1.0)
    parser.add_argument("--high_dim", action="store_true")
    parser.add_argument("--high_dim_d", type=int, default=50)
    parser.add_argument("--heteroscedastic_eps", action="store_true")
    parser.add_argument("--n_experiments", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=13337)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--nonlinearity_strength", type=float, default=1.0)
    parser.add_argument("--nuisance", choices=["linear", "gbm", "poly", "rf"], default="linear")
    parser.add_argument("--no_oracle_moments", action="store_true")
    parser.add_argument("--output_dir", default="figures/nonlinear")
    parser.add_argument("--results_file", default=None)
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    config = NonlinearDGPConfig(
        n_samples=opts.n_samples,
        n_covariates=opts.n_covariates,
        support_size=opts.support_size,
        treatment_effect=opts.treatment_effect,
        sigma_eta=opts.sigma_eta,
        sigma_outcome=opts.sigma_outcome,
        nonlinear_confounding=opts.nonlinear_confounding,
        heavy_tail_eta=opts.heavy_tail_eta,
        eta_beta=opts.eta_beta,
        high_dim=opts.high_dim,
        high_dim_d=opts.high_dim_d,
        heteroscedastic_eps=opts.heteroscedastic_eps,
        alpha_scale=opts.alpha_scale,
        beta_scale=opts.beta_scale,
        interaction_scale=opts.interaction_scale,
        heteroscedastic_scale=opts.heteroscedastic_scale,
        nonlinearity_strength=opts.nonlinearity_strength,
    )

    results = run_nonlinear_experiments(
        config=config,
        n_experiments=opts.n_experiments,
        base_seed=opts.base_seed,
        n_jobs=opts.n_jobs,
        nuisance=opts.nuisance,
        use_oracle_moments=not opts.no_oracle_moments,
        verbose=True,
    )

    if opts.results_file is None:
        nl_tag = "nl" if opts.nonlinear_confounding else "lin"
        ht_tag = f"ht{opts.eta_beta}" if opts.heavy_tail_eta else "gauss"
        hd_tag = f"hd{opts.high_dim_d}" if opts.high_dim else f"d{opts.n_covariates}"
        he_tag = "heps" if opts.heteroscedastic_eps else "homoeps"
        results_file = f"nonlinear_results_n{opts.n_samples}_{nl_tag}_{ht_tag}_{hd_tag}_{he_tag}_{opts.nuisance}.npy"
    else:
        results_file = opts.results_file

    out_path = os.path.join(opts.output_dir, results_file)
    np.save(out_path, results)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
