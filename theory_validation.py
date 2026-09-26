"""E2: validate treatment-effect asymptotic variances by simulation.

The experiment compares the population-whitened one-unit estimator from the
paper with two readouts from symmetric sample-whitened FastICA, OLS, and HOML.
All estimators in a replication use the same generated data.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import gennorm
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from baselines import ols_baseline
from ica import _fit_fastica_with_retries
from main_estimation import all_together_cross_fitting
from mcc import calc_disent_metrics
from oml_runner import DEFAULT_TREATMENT_DISCOUNTS, DEFAULT_TREATMENT_PROBS

DEFAULT_SEED = 20260926
PAPER_COVARIATE_BETA = 4.0
ESTIMATORS = (
    "A_paper_cube_population",
    "B_logcosh_eta_column",
    "B_cube_eta_column",
    "C_logcosh_eps_row",
    "C_cube_eps_row",
    "D_OLS",
    "E_HOML_known",
)
COEFFICIENT_GRID = ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (2.0, -2.0, 1.0), (0.5, 0.05, 1.0))
ETA_LAWS = ("discrete", "laplace", "uniform", "rademacher")
EPS_LAWS = ("paper_default", "laplace", "gaussian", "uniform")


@dataclass(frozen=True)
class Law:
    """A centered, unit-variance source law."""

    name: str
    sampler: Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]
    m3: float
    m4: float
    m6: float
    support: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    pdf: Callable[[float], float] | None = None

    @property
    def kappa(self) -> float:
        return self.m4 - 3.0

    @property
    def var_cube(self) -> float:
        return self.m6 - self.m3**2


@dataclass(frozen=True)
class Cell:
    """One simulation cell."""

    cell_id: str
    n: int
    replicates: int
    d: int
    a: float
    b: float
    theta: float
    eta_law: str
    eps_law: str
    covariate_law: str = "paper_gennorm_beta4"
    sigma_xi: float = 1.0
    sigma_eta: float = 1.0
    sigma_eps: float = 1.0


def _discrete_law() -> Law:
    centered = DEFAULT_TREATMENT_DISCOUNTS - np.dot(DEFAULT_TREATMENT_DISCOUNTS, DEFAULT_TREATMENT_PROBS)
    sd = math.sqrt(float(np.dot(centered**2, DEFAULT_TREATMENT_PROBS)))
    support = centered / sd
    probs = DEFAULT_TREATMENT_PROBS.copy()

    def sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        return rng.choice(support, size=size, p=probs)

    return Law(
        "discrete",
        sample,
        float(np.dot(support**3, probs)),
        float(np.dot(support**4, probs)),
        float(np.dot(support**6, probs)),
        support=support,
        probabilities=probs,
    )


def _symmetric_continuous_law(name: str) -> Law:
    if name in {"paper_default", "uniform"}:
        half_width = math.sqrt(3.0)

        def uniform_sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
            return rng.uniform(-half_width, half_width, size=size)

        def uniform_pdf(x: float) -> float:
            return 1.0 / (2.0 * half_width) if abs(x) <= half_width else 0.0

        return Law(name, uniform_sample, 0.0, 9.0 / 5.0, 27.0 / 7.0, pdf=uniform_pdf)

    if name == "laplace":
        scale = 1.0 / math.sqrt(2.0)

        def laplace_sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
            return rng.laplace(0.0, scale, size=size)

        def laplace_pdf(x: float) -> float:
            return math.exp(-abs(x) / scale) / (2.0 * scale)

        return Law(name, laplace_sample, 0.0, 6.0, 90.0, pdf=laplace_pdf)

    if name == "gaussian":

        def gaussian_sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
            return rng.standard_normal(size=size)

        def gaussian_pdf(x: float) -> float:
            return math.exp(-(x**2) / 2.0) / math.sqrt(2.0 * math.pi)

        return Law(name, gaussian_sample, 0.0, 3.0, 15.0, pdf=gaussian_pdf)

    if name == "rademacher":
        support = np.array([-1.0, 1.0])
        probs = np.array([0.5, 0.5])

        def rademacher_sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
            return rng.choice(support, size=size, p=probs)

        return Law(name, rademacher_sample, 0.0, 1.0, 1.0, support=support, probabilities=probs)

    raise ValueError(f"Unknown source law: {name}")


def _gennorm_law(name: str, beta: float) -> Law:
    raw_variance = gamma(3.0 / beta) / gamma(1.0 / beta)
    scale = 1.0 / math.sqrt(raw_variance)
    m4 = float(scale**4 * gamma(5.0 / beta) / gamma(1.0 / beta))
    m6 = float(scale**6 * gamma(7.0 / beta) / gamma(1.0 / beta))
    normalizer = beta / (2.0 * scale * gamma(1.0 / beta))

    def sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        return gennorm.rvs(beta, scale=scale, size=size, random_state=rng)

    def pdf(x: float) -> float:
        return float(normalizer * math.exp(-((abs(x) / scale) ** beta)))

    return Law(name, sample, 0.0, m4, m6, pdf=pdf)


def get_law(name: str) -> Law:
    """Return the standardized source law used by an E2 cell."""
    if name == "discrete":
        return _discrete_law()
    if name == "paper_gennorm_beta4":
        return _gennorm_law(name, PAPER_COVARIATE_BETA)
    if name == "laplace_covariate":
        return replace(_symmetric_continuous_law("laplace"), name=name)
    return _symmetric_continuous_law(name)


def contrast_statistics(law: Law, contrast: str) -> dict[str, float]:
    """Return h, m, q, tau, and nu for a standardized source."""
    if contrast == "cube":
        h = law.m3
        m = law.m4
        q = 3.0
        return {"h": h, "m": m, "q": q, "tau": abs(m - q), "nu": law.m6 - h**2 - m**2}
    if contrast != "logcosh":
        raise ValueError(f"Unknown contrast: {contrast}")

    def values(x: float) -> tuple[float, float, float, float]:
        tanh_x = math.tanh(x)
        return tanh_x, x * tanh_x, 1.0 - tanh_x**2, tanh_x**2

    if law.support is not None and law.probabilities is not None:
        vals = np.array([values(float(x)) for x in law.support])
        h, m, q, eg2 = law.probabilities @ vals
    else:
        if law.pdf is None:
            raise ValueError(f"No integration rule for {law.name}")

        def expectation(index: int) -> float:
            def integrand(x: float) -> float:
                return values(x)[index] * law.pdf(x)

            return float(quad(integrand, -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11, limit=300)[0])

        h, m, q, eg2 = (expectation(index) for index in range(4))
    return {"h": h, "m": m, "q": q, "tau": abs(m - q), "nu": eg2 - h**2 - m**2}


def pairwise_fastica_variance(first: Law, second: Law, contrast: str) -> float:
    """Pairwise symmetric-FastICA variance V(first, second)."""
    first_stats = contrast_statistics(first, contrast)
    second_stats = contrast_statistics(second, contrast)
    denominator = (first_stats["tau"] + second_stats["tau"]) ** 2
    if denominator <= 1e-24:
        return math.inf
    return float((first_stats["nu"] + second_stats["nu"] + second_stats["tau"] ** 2) / denominator)


def _mixing_matrix(cell: Cell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the source-to-observation mixing matrix for a cell."""
    a_vec = np.full(cell.d, cell.a)
    b_vec = np.full(cell.d, cell.b)
    mixing = np.zeros((cell.d + 2, cell.d + 2))
    mixing[: cell.d, : cell.d] = np.eye(cell.d) * cell.sigma_xi
    mixing[cell.d, : cell.d] = a_vec * cell.sigma_xi
    mixing[cell.d, cell.d] = cell.sigma_eta
    mixing[-1, : cell.d] = (b_vec + cell.theta * a_vec) * cell.sigma_xi
    mixing[-1, cell.d] = cell.theta * cell.sigma_eta
    mixing[-1, -1] = cell.sigma_eps
    return mixing, a_vec, b_vec


def predicted_variances(cell: Cell) -> tuple[dict[str, float], float]:
    """Compute every requested asymptotic variance for a cell."""
    eta = get_law(cell.eta_law)
    eps = get_law(cell.eps_law)
    xi = get_law(cell.covariate_law)
    _, a_vec, b_vec = _mixing_matrix(cell)
    eta_scale = cell.sigma_eta**2
    eps_scale = cell.sigma_eps**2
    xi_scale = cell.sigma_xi**2
    total_x_loading = float(np.sum((b_vec + cell.theta * a_vec) ** 2)) * xi_scale
    b_loading = float(np.sum(b_vec**2)) * xi_scale

    a_variance = (total_x_loading + eps_scale) / eta_scale * eta.var_cube / eta.kappa**2
    predictions: dict[str, float] = {
        "A_paper_cube_population": a_variance,
        "D_OLS": eps_scale / eta_scale,
        "E_HOML_known": eps_scale / eta_scale * (eta.var_cube - 6.0 * eta.kappa - 9.0) / eta.kappa**2,
    }
    for contrast in ("logcosh", "cube"):
        v_xi_eta = pairwise_fastica_variance(xi, eta, contrast)
        v_eps_eta = pairwise_fastica_variance(eps, eta, contrast)
        predictions[f"B_{contrast}_eta_column"] = b_loading / eta_scale * v_xi_eta + eps_scale / eta_scale * v_eps_eta
        predictions[f"C_{contrast}_eps_row"] = eps_scale / eta_scale * v_eps_eta
    comparison_b_cube = predictions["B_cube_eta_column"]
    return predictions, comparison_b_cube


def _paper_one_unit_estimate(
    observations: np.ndarray, mixing: np.ndarray, eta_index: int, seed: int
) -> tuple[float, bool]:
    """Population-whitened one-unit cubic FastICA initialized near eta."""
    covariance = mixing @ mixing.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    whitened = observations @ whitening.T
    true_direction = whitening @ mixing[:, eta_index]
    true_direction /= np.linalg.norm(true_direction)
    rng = np.random.default_rng(seed)
    direction = true_direction + 0.05 * rng.standard_normal(true_direction.shape)
    direction /= np.linalg.norm(direction)
    converged = False
    for _ in range(1000):
        projection = whitened @ direction
        updated = np.mean(whitened * projection[:, None] ** 3, axis=0) - 3.0 * direction
        norm = np.linalg.norm(updated)
        if not np.isfinite(norm) or norm < 1e-12:
            return math.nan, False
        updated /= norm
        if np.dot(updated, true_direction) < 0:
            updated = -updated
        if 1.0 - abs(float(np.dot(updated, direction))) < 1e-9:
            direction = updated
            converged = True
            break
        direction = updated
    if not converged:
        return math.nan, False
    unmixing_direction = whitening.T @ direction
    eta_mixing = covariance @ unmixing_direction
    estimate = eta_mixing[-1] / mixing[-2, eta_index]
    return float(estimate), True


def _symmetric_fastica_readouts(
    observations: np.ndarray,
    sources: np.ndarray,
    d: int,
    contrast: str,
    seed: int,
) -> tuple[float, float, bool]:
    """Fit symmetric FastICA once and return eta-column and eps-row readouts."""
    convergence_info: dict[str, float | int | bool] = {}
    ica, recovered, converged = _fit_fastica_with_retries(
        observations,
        random_state=seed,
        whiten="unit-variance",
        fun=contrast,
        verbose=False,
        convergence_info=convergence_info,
    )
    if not converged:
        return math.nan, math.nan, False
    permutation = calc_disent_metrics(sources, recovered)["munkres_sort_idx"].astype(int)
    permuted_mixing = ica.mixing_[:, permutation]
    scaled_mixing = permuted_mixing / np.diag(permuted_mixing)
    eta_column = float(scaled_mixing[-1, d])
    eps_row = ica.components_[permutation[-1], :]
    eps_readout = float(-eps_row[d] / eps_row[-1])
    return eta_column, eps_readout, True


def _simulate_replicate(cell: Cell, replicate: int, base_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one shared dataset and calculate all seven estimates."""
    seed_sequence = np.random.SeedSequence([base_seed, int(cell.cell_id[1:]), replicate, cell.n])
    data_seed, a_seed, logcosh_seed, cube_seed = seed_sequence.spawn(4)
    rng = np.random.default_rng(data_seed)
    xi_law = get_law(cell.covariate_law)
    eta_law = get_law(cell.eta_law)
    eps_law = get_law(cell.eps_law)
    standardized_sources = np.column_stack(
        (
            xi_law.sampler(rng, (cell.n, cell.d)),
            eta_law.sampler(rng, (cell.n,)),
            eps_law.sampler(rng, (cell.n,)),
        )
    )
    mixing, _, _ = _mixing_matrix(cell)
    observations = standardized_sources @ mixing.T
    covariates = observations[:, : cell.d]
    treatment = observations[:, cell.d]
    outcome = observations[:, -1]

    estimates = np.full(len(ESTIMATORS), np.nan)
    converged = np.ones(len(ESTIMATORS), dtype=bool)
    estimates[0], converged[0] = _paper_one_unit_estimate(
        observations, mixing, cell.d, int(a_seed.generate_state(1)[0])
    )
    for contrast, seed, b_index, c_index in (
        ("logcosh", logcosh_seed, 1, 3),
        ("cube", cube_seed, 2, 4),
    ):
        estimates[b_index], estimates[c_index], fit_converged = _symmetric_fastica_readouts(
            observations,
            standardized_sources,
            cell.d,
            contrast,
            int(seed.generate_state(1)[0]),
        )
        converged[b_index] = fit_converged
        converged[c_index] = fit_converged
    estimates[5] = float(ols_baseline(covariates, treatment, outcome)[0])
    try:
        homl = all_together_cross_fitting(
            covariates,
            treatment,
            outcome,
            cell.sigma_eta**2,
            eta_law.m3 * cell.sigma_eta**3,
            model_treatment=LinearRegression(),
            model_outcome=LinearRegression(),
        )[1]
        estimates[6] = float(homl)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        converged[6] = False
    converged &= np.isfinite(estimates)
    return estimates, converged


def _run_cell(cell: Cell, base_seed: int) -> tuple[Cell, np.ndarray, np.ndarray]:
    results = [_simulate_replicate(cell, replicate, base_seed) for replicate in range(cell.replicates)]
    estimates = np.vstack([result[0] for result in results])
    converged = np.vstack([result[1] for result in results])
    return cell, estimates, converged


def _bootstrap_variance_ci(values: np.ndarray, n: int, seed: int, draws: int) -> tuple[float, float]:
    if values.size < 3:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    bootstrap_variances = np.empty(draws)
    for draw in range(draws):
        sample = values[rng.integers(0, values.size, size=values.size)]
        bootstrap_variances[draw] = n * np.var(sample, ddof=1)
    low, high = np.quantile(bootstrap_variances, [0.025, 0.975])
    return float(low), float(high)


def summarize_cell(
    cell: Cell,
    estimates: np.ndarray,
    converged: np.ndarray,
    base_seed: int,
    bootstrap_draws: int,
) -> list[dict[str, object]]:
    """Turn raw replication estimates into one CSV row per estimator."""
    predictions, comparison_b_cube = predicted_variances(cell)
    rows = []
    for estimator_index, estimator in enumerate(ESTIMATORS):
        good = converged[:, estimator_index] & np.isfinite(estimates[:, estimator_index])
        values = estimates[good, estimator_index]
        observed_var = float(cell.n * np.var(values, ddof=1)) if values.size >= 2 else math.nan
        observed_mse = float(cell.n * np.mean((values - cell.theta) ** 2)) if values.size else math.nan
        ci_low, ci_high = _bootstrap_variance_ci(
            values,
            cell.n,
            base_seed + 1009 * int(cell.cell_id[1:]) + estimator_index,
            bootstrap_draws,
        )
        predicted = predictions[estimator]
        ratio = observed_var / predicted
        rows.append(
            {
                "cell_id": cell.cell_id,
                "n": cell.n,
                "replicates": cell.replicates,
                "d": cell.d,
                "a": cell.a,
                "b": cell.b,
                "theta": cell.theta,
                "eta_law": cell.eta_law,
                "eps_law": cell.eps_law,
                "covariate_law": cell.covariate_law,
                "estimator": estimator,
                "predicted_av": predicted,
                "comparison_b_cube_av": comparison_b_cube if estimator == ESTIMATORS[0] else math.nan,
                "observed_n_var": observed_var,
                "observed_n_var_ci_low": ci_low,
                "observed_n_var_ci_high": ci_high,
                "observed_n_mse": observed_mse,
                "ratio_observed_predicted": ratio,
                "ratio_mse_predicted": observed_mse / predicted,
                "predicted_in_bootstrap_ci": bool(ci_low <= predicted <= ci_high),
                "gross_mismatch": bool(ratio < 0.5 or ratio > 2.0),
                "mean_estimate": float(np.mean(values)) if values.size else math.nan,
                "bias": float(np.mean(values) - cell.theta) if values.size else math.nan,
                "n_success": int(values.size),
                "convergence_failure_count": int(cell.replicates - values.size),
            }
        )
    return rows


def build_cells(quick: bool = False) -> list[Cell]:
    """Construct the E2 grid, its two checks, and six large-n representatives."""
    if quick:
        return [
            Cell(
                cell_id="c000",
                n=300,
                replicates=6,
                d=1,
                a=0.5,
                b=0.05,
                theta=1.0,
                eta_law="discrete",
                eps_law="paper_default",
            )
        ]

    cells: list[Cell] = []

    def add(**kwargs: object) -> None:
        cells.append(Cell(cell_id=f"c{len(cells):03d}", **kwargs))

    for a, b, theta in COEFFICIENT_GRID:
        for eta_law in ETA_LAWS:
            for eps_law in EPS_LAWS:
                add(
                    n=5000,
                    replicates=400,
                    d=1,
                    a=a,
                    b=b,
                    theta=theta,
                    eta_law=eta_law,
                    eps_law=eps_law,
                )
    add(
        n=5000,
        replicates=400,
        d=1,
        a=0.5,
        b=0.05,
        theta=1.0,
        eta_law="discrete",
        eps_law="paper_default",
        covariate_law="laplace_covariate",
    )
    add(
        n=5000,
        replicates=400,
        d=3,
        a=0.5,
        b=0.05,
        theta=1.0,
        eta_law="discrete",
        eps_law="paper_default",
    )

    representatives = (
        (1, 0.0, 0.0, "discrete", "paper_default", "paper_gennorm_beta4", 1),
        (1, 2.0, 0.0, "laplace", "gaussian", "paper_gennorm_beta4", 1),
        (1, 0.0, 2.0, "uniform", "laplace", "paper_gennorm_beta4", 1),
        (1, 2.0, -2.0, "rademacher", "uniform", "paper_gennorm_beta4", 1),
        (1, 0.5, 0.05, "discrete", "paper_default", "laplace_covariate", 1),
        (3, 0.5, 0.05, "laplace", "paper_default", "paper_gennorm_beta4", 1),
    )
    for d, a, b, eta_law, eps_law, covariate_law, theta in representatives:
        add(
            n=20000,
            replicates=200,
            d=d,
            a=a,
            b=b,
            theta=float(theta),
            eta_law=eta_law,
            eps_law=eps_law,
            covariate_law=covariate_law,
        )
    return cells


def _format_number(value: object, digits: int = 3) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    by_estimator = {estimator: [row for row in rows if row["estimator"] == estimator] for estimator in ESTIMATORS}
    lines = [
        "# E2: theory versus simulation",
        "",
        "The paper-demand defaults are standardized generalized-normal covariates with shape "
        f"beta={PAPER_COVARIATE_BETA:g}, standardized discrete treatment noise with support inherited from "
        "`DEFAULT_TREATMENT_DISCOUNTS/PROBS`, and unit-variance uniform outcome noise. "
        "Accordingly, `paper_default` epsilon and the separately requested `uniform` epsilon are aliases; "
        "they remain separate seeded cells in the grid.",
        "",
        "For d=3, every covariate is independent with the named law and the scalar a and b are broadcast to "
        "all three coordinates (L=I). The report distinguishes gross finite-sample mismatches (observed/predicted "
        "outside [0.5, 2]) from cells where the predicted AV lies outside the percentile bootstrap 95% interval "
        "for observed n Var. Failed fits are excluded from variance and MSE calculations and reported separately.",
        "",
        "## Formula agreement",
        "",
        "| Estimator | Median obs./pred. | Range | Gross mismatches | 95% exclusions | Fit failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for estimator, estimator_rows in by_estimator.items():
        finite = [row for row in estimator_rows if np.isfinite(float(row["ratio_observed_predicted"]))]
        ratios = np.array([float(row["ratio_observed_predicted"]) for row in finite])
        interval_exclusions = [
            row
            for row in finite
            if not float(row["observed_n_var_ci_low"])
            <= float(row["predicted_av"])
            <= float(row["observed_n_var_ci_high"])
        ]
        gross_mismatches = [
            row
            for row in finite
            if float(row["ratio_observed_predicted"]) < 0.5 or float(row["ratio_observed_predicted"]) > 2.0
        ]
        fit_failures = sum(int(row["convergence_failure_count"]) for row in estimator_rows)
        lines.append(
            f"| {estimator} | {_format_number(np.median(ratios))} | "
            f"{_format_number(np.min(ratios))}-{_format_number(np.max(ratios))} | "
            f"{len(gross_mismatches)}/{len(finite)} | {len(interval_exclusions)}/{len(finite)} | {fit_failures} |"
        )

    a_rows = by_estimator[ESTIMATORS[0]]
    paper_ratios = np.array([float(row["ratio_observed_predicted"]) for row in a_rows])
    b_comparison_ratios = np.array(
        [float(row["observed_n_var"]) / float(row["comparison_b_cube_av"]) for row in a_rows]
    )
    lines.extend(
        [
            "",
            "## Estimator A comparison",
            "",
            "Estimator A's observed variance divided by the paper formula has median "
            f"{_format_number(np.median(paper_ratios))} (range {_format_number(np.min(paper_ratios))}-"
            f"{_format_number(np.max(paper_ratios))}). Dividing the same observations by formula B with the "
            f"cube contrast gives median {_format_number(np.median(b_comparison_ratios))} "
            f"(range {_format_number(np.min(b_comparison_ratios))}-"
            f"{_format_number(np.max(b_comparison_ratios))}).",
            "",
            "## ICA win/loss pattern",
            "",
            "A win means lower AV (predicted comparison) or lower n MSE (observed comparison) within the same cell.",
            "",
            "| ICA estimator | Pred. vs HOML W/L | Obs. vs HOML W/L | Pred. vs OLS W/L | Obs. vs OLS W/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    row_lookup = {(row["cell_id"], row["estimator"]): row for row in rows}
    for estimator in ESTIMATORS[1:5]:
        counts = []
        for baseline in ("E_HOML_known", "D_OLS"):
            predicted_wins = 0
            observed_wins = 0
            total = 0
            for row in by_estimator[estimator]:
                baseline_row = row_lookup[(row["cell_id"], baseline)]
                if np.isfinite(float(row["predicted_av"])) and np.isfinite(float(row["observed_n_mse"])):
                    total += 1
                    predicted_wins += float(row["predicted_av"]) < float(baseline_row["predicted_av"])
                    observed_wins += float(row["observed_n_mse"]) < float(baseline_row["observed_n_mse"])
            counts.extend(((predicted_wins, total - predicted_wins), (observed_wins, total - observed_wins)))
        lines.append(
            f"| {estimator} | {counts[0][0]}/{counts[0][1]} | {counts[1][0]}/{counts[1][1]} | "
            f"{counts[2][0]}/{counts[2][1]} | {counts[3][0]}/{counts[3][1]} |"
        )

    lines.extend(
        [
            "",
            "## Gross finite-sample mismatches",
            "",
            "These are the cells with observed/predicted outside [0.5, 2]. All such cells below are symmetric "
            "FastICA readouts that formally converged; the pairwise AV describes the regular local mode but does "
            "not capture their finite-sample tail events.",
            "",
            "| Estimator | Cell | n | d | a | b | eta | eps | covariate | Obs./pred. |",
            "|---|---|---:|---:|---:|---:|---|---|---|---:|",
        ]
    )
    gross_count = 0
    for row in rows:
        ratio = float(row["ratio_observed_predicted"])
        if np.isfinite(ratio) and (ratio < 0.5 or ratio > 2.0):
            gross_count += 1
            lines.append(
                f"| {row['estimator']} | {row['cell_id']} | {row['n']} | {row['d']} | "
                f"{_format_number(row['a'])} | {_format_number(row['b'])} | {row['eta_law']} | "
                f"{row['eps_law']} | {row['covariate_law']} | {_format_number(ratio)} |"
            )
    if gross_count == 0:
        lines.append("| None | - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Bootstrap-interval exclusions",
            "",
            "These are the cells where the predicted AV lies outside the percentile bootstrap 95% interval for "
            "observed n Var. With 88 cells per estimator, some exclusions are expected from Monte Carlo error.",
            "",
            "| Estimator | Cell | n | d | a | b | eta | eps | covariate | Obs./pred. | 95% CI for n Var | Pred. AV |",
            "|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|",
        ]
    )
    exclusion_count = 0
    for row in rows:
        predicted = float(row["predicted_av"])
        low = float(row["observed_n_var_ci_low"])
        high = float(row["observed_n_var_ci_high"])
        if np.isfinite(predicted) and np.isfinite(low) and not low <= predicted <= high:
            exclusion_count += 1
            lines.append(
                f"| {row['estimator']} | {row['cell_id']} | {row['n']} | {row['d']} | "
                f"{_format_number(row['a'])} | {_format_number(row['b'])} | {row['eta_law']} | "
                f"{row['eps_law']} | {row['covariate_law']} | "
                f"{_format_number(row['ratio_observed_predicted'])} | "
                f"[{_format_number(low)}, {_format_number(high)}] | {_format_number(predicted)} |"
            )
    if exclusion_count == 0:
        lines.append("| None | - | - | - | - | - | - | - | - | - | - | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(rows: list[dict[str, object]], csv_path: Path, markdown_path: Path) -> None:
    """Write the requested CSV and human-readable summary."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, markdown_path)


def run_experiment(
    quick: bool = False,
    n_jobs: int = 6,
    base_seed: int = DEFAULT_SEED,
    bootstrap_draws: int = 2000,
    cells: Iterable[Cell] | None = None,
) -> list[dict[str, object]]:
    """Run E2 and return aggregated output rows."""
    selected_cells = list(cells) if cells is not None else build_cells(quick=quick)
    if not 1 <= n_jobs <= 6:
        raise ValueError("n_jobs must be between 1 and 6")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\s*Found Intel OpenMP .* LLVM OpenMP", category=RuntimeWarning)
        with threadpool_limits(limits=1, user_api="blas"), parallel_backend("threading"):
            raw_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_run_cell)(cell, base_seed) for cell in selected_cells
            )
    rows: list[dict[str, object]] = []
    for cell, estimates, converged in raw_results:
        rows.extend(summarize_cell(cell, estimates, converged, base_seed, bootstrap_draws))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run one six-replicate smoke cell.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel workers (maximum: 6).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("autoresearch/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = "_quick" if args.quick else ""
    rows = run_experiment(
        quick=args.quick,
        n_jobs=args.n_jobs,
        base_seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    csv_path = args.output_dir / f"e2_theory_validation{suffix}.csv"
    markdown_path = args.output_dir / f"e2_theory_validation{suffix}.md"
    write_results(rows, csv_path, markdown_path)
    print(f"Wrote {len(rows)} rows to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
