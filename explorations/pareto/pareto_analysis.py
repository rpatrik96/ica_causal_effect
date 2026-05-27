"""
Pareto frontier analysis: ICA vs OML vs HOML variants across regimes.

Answers: Is there a single method (or cheap selector) dominating the per-regime
upper envelope once all HOML variants are included?

Key additions over the prior version
-------------------------------------
* All four OML estimators tracked: OML, HOML-known, HOML-est, HOML-split.
* Near-Gaussian blow-up sweep: gennorm beta from 1.6 → 2.1 (fine grid) to
  characterise and quantify HOML divergence as excess kurtosis → 0.
* All analysis functions are importable (no side effects at module level)
  so they can be called from tests.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gamma

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
OUT_DIR = os.path.join(REPO, "explorations", "pareto")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Method indices (must match ablation_utils.py) ─────────────────────────────
HOML_KNOWN_IDX = 1  # Robust Ortho ML with known moments
HOML_EST_IDX = 2  # Robust Ortho ML with estimated moments
HOML_SPLIT_IDX = 3  # Robust Ortho ML with nested split
ICA_IDX = 4
OML_IDX = 0

# All HOML variant indices and their short names
HOML_VARIANTS: List[Tuple[int, str]] = [
    (HOML_KNOWN_IDX, "HOML-known"),
    (HOML_EST_IDX, "HOML-est"),
    (HOML_SPLIT_IDX, "HOML-split"),
]

# Kurtosis threshold used by the 3-way selector
_SELECTOR_EK_HIGH = 3.0  # above → use ICA
_SELECTOR_EK_LOW = 0.0  # below → use OML (HOML too risky near-Gaussian)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Theoretical helpers
# ──────────────────────────────────────────────────────────────────────────────


def gennorm_excess_kurtosis(beta: float) -> float:
    """Return the analytical excess kurtosis of a zero-mean gennorm(beta).

    For GN(beta) normalised to unit variance:
        excess_kurtosis = Gamma(5/b)*Gamma(1/b) / Gamma(3/b)^2 - 3
    """
    return float(gamma(5 / beta) * gamma(1 / beta) / gamma(3 / beta) ** 2 - 3)


def homl_asymptotic_var_from_ek_and_eta2(
    eta_excess_kurtosis: float,
    eta_second_moment: float,
    eta_cubed_variance: float,
) -> float:
    """Return the HOML asymptotic variance given distributional parameters.

    HOML avar ∝ Var[η³] / (E[η³]/√Var[η³])²  =  Var[η³]² / E[η³]²

    For symmetric η (E[η³]=0) the denominator is 0, so HOML avar = ∞.
    Var[η³] = E[η⁴]·E[η²] - (E[η³])²  =  (ek+3)·E[η²]²·E[η²] - 0
              (symmetric case, E[η³]=0) = (ek+3)·E[η²]³
    so for symmetric eta:  HOML avar = ∞  always.

    This function is kept for completeness; for gennorm (symmetric) use
    `homl_asymptotic_var_gennorm_symmetric`.
    """
    del eta_excess_kurtosis, eta_second_moment, eta_cubed_variance
    return np.inf  # gennorm is always symmetric → HOML inapplicable


def estimate_excess_kurtosis(residuals: np.ndarray) -> float:
    """Estimate excess kurtosis from a 1-D residual array.

    Uses the standard (n-corrected) Fisher definition: μ₄/σ⁴ − 3.
    Returns np.nan for arrays shorter than 4.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if n < 4:
        return np.nan
    mu = residuals.mean()
    sigma2 = np.mean((residuals - mu) ** 2)
    if sigma2 < 1e-15:
        return 0.0
    m4 = np.mean((residuals - mu) ** 4)
    return float(m4 / sigma2**2 - 3)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Selector rule
# ──────────────────────────────────────────────────────────────────────────────


def kurtosis_selector(
    excess_kurtosis: float,
    ek_high: float = _SELECTOR_EK_HIGH,
    ek_low: float = _SELECTOR_EK_LOW,
) -> str:
    """Deterministic 3-way selector based on excess kurtosis.

    Parameters
    ----------
    excess_kurtosis : float
        Estimated excess kurtosis of the first-stage treatment residual.
    ek_high : float
        Threshold above which ICA is preferred (default 3.0).
    ek_low : float
        Threshold below which OML is preferred (default 0.0).
        Between ek_low and ek_high the selector picks OML as the safe option.

    Returns
    -------
    str
        One of "ICA", "OML".  HOML is never selected by default because its
        RMSE diverges for symmetric η with near-zero kurtosis.
    """
    if not np.isfinite(excess_kurtosis):
        return "OML"
    if excess_kurtosis > ek_high:
        return "ICA"
    return "OML"


def apply_selector(
    rows: List[Dict],
    ek_high: float = _SELECTOR_EK_HIGH,
    ek_low: float = _SELECTOR_EK_LOW,
) -> np.ndarray:
    """Apply kurtosis_selector to each row and return selected RMSE array."""
    selected = np.empty(len(rows))
    for i, r in enumerate(rows):
        ek = r.get("eta_excess_kurtosis", np.nan)
        choice = kurtosis_selector(float(ek), ek_high, ek_low)
        if choice == "ICA":
            selected[i] = r["rmse_ica"]
        else:
            selected[i] = r["rmse_oml"]
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# 3. Oracle envelope
# ──────────────────────────────────────────────────────────────────────────────


def oracle_envelope(rows: List[Dict], include_homl: bool = True) -> np.ndarray:
    """Per-row minimum RMSE over the candidate method set.

    Parameters
    ----------
    rows : list of dict
        Each dict must have keys ``rmse_oml`` and ``rmse_ica``, and
        optionally ``rmse_homl_known``, ``rmse_homl_est``, ``rmse_homl_split``.
    include_homl : bool
        If True, include all available HOML variants in the minimum.
        If False, oracle is min(ICA, OML) only.

    Returns
    -------
    np.ndarray
        Per-row oracle RMSE.
    """
    candidates = []
    for r in rows:
        vals = [r["rmse_oml"], r["rmse_ica"]]
        if include_homl:
            for key in ("rmse_homl_known", "rmse_homl_est", "rmse_homl_split"):
                v = r.get(key, np.nan)
                if np.isfinite(v):
                    vals.append(v)
        candidates.append(np.nanmin(vals))
    return np.array(candidates)


def oracle_envelope_is_lower_bound(rows: List[Dict]) -> bool:
    """Return True if oracle_envelope(rows) ≤ every fixed-method RMSE."""
    env = oracle_envelope(rows, include_homl=True)
    for key in ("rmse_oml", "rmse_ica", "rmse_homl_known", "rmse_homl_est", "rmse_homl_split"):
        arr = np.array([r.get(key, np.nan) for r in rows])
        finite = np.isfinite(arr)
        if finite.any():
            if not np.all(env[finite] <= arr[finite] + 1e-12):
                return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 4. HOML blow-up detection
# ──────────────────────────────────────────────────────────────────────────────


def homl_blowup_detected(rows: List[Dict], threshold: float = 10.0) -> bool:
    """Return True if any row with |excess_kurtosis| < 0.5 has HOML RMSE ≥ threshold × OML RMSE.

    This encodes the regression test: when eta is near-Gaussian, HOML-known
    RMSE should dwarf OML RMSE by a large factor.
    """
    for r in rows:
        ek = r.get("eta_excess_kurtosis", np.nan)
        if not np.isfinite(ek):
            continue
        if abs(ek) < 0.5:
            homl_rmse = r.get("rmse_homl_known", np.nan)
            oml_rmse = r.get("rmse_oml", np.nan)
            if np.isfinite(homl_rmse) and np.isfinite(oml_rmse) and oml_rmse > 1e-10:
                if homl_rmse / oml_rmse >= threshold:
                    return True
    return False


def classify_blowup_rows(rows: List[Dict]) -> List[Dict]:
    """Return rows where HOML-known RMSE is finite and > 5× OML RMSE."""
    out = []
    for r in rows:
        h = r.get("rmse_homl_known", np.nan)
        o = r.get("rmse_oml", np.nan)
        if np.isfinite(h) and np.isfinite(o) and o > 1e-10 and h / o > 5.0:
            out.append(r)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. Load Bernoulli ablation (legacy — only OML / HOML-known / ICA)
# ──────────────────────────────────────────────────────────────────────────────


def load_bernoulli_results() -> List[Dict]:
    """Load the pre-computed Bernoulli ablation results."""
    path = os.path.join(REPO, "figures", "noise_ablation_results_bernoulli.npy")
    d = np.load(path, allow_pickle=True).item()
    rows = []
    for dist_key, v in d.items():
        for cfg in v["config_results"]:
            cfg_rmse = np.array(cfg["rmse"])
            rows.append(
                dict(
                    source="bernoulli_ablation",
                    dist=dist_key,
                    n_samples=v["n_samples"],
                    eta_excess_kurtosis=float(v["eta_excess_kurtosis"]),
                    eta_skewness_squared=float(v["eta_skewness_squared"]),
                    ica_asymptotic_var=(
                        float(v["ica_asymptotic_var"]) if np.isfinite(v["ica_asymptotic_var"]) else np.nan
                    ),
                    homl_asymptotic_var=(
                        float(v["homl_asymptotic_var"]) if np.isfinite(v["homl_asymptotic_var"]) else np.nan
                    ),
                    ica_var_coeff=float(cfg.get("ica_var_coeff", v["ica_var_coeff"])),
                    treatment_effect=float(cfg.get("treatment_effect", np.nan)),
                    treatment_coef=float(cfg.get("treatment_coef_scalar", np.nan)),
                    outcome_coef=float(cfg.get("outcome_coef_scalar", np.nan)),
                    rmse_oml=float(cfg_rmse[OML_IDX]),
                    rmse_homl_known=float(cfg_rmse[HOML_KNOWN_IDX]),
                    rmse_homl_est=float(cfg_rmse[HOML_EST_IDX]),
                    rmse_homl_split=float(cfg_rmse[HOML_SPLIT_IDX]),
                    rmse_ica=float(cfg_rmse[ICA_IDX]),
                )
            )
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 6. Grid experiment (all HOML variants)
# ──────────────────────────────────────────────────────────────────────────────


def run_grid_experiment(
    beta_values: List[float],
    sample_sizes: List[int],
    n_experiments: int = 30,
    seed: int = 42,
    cache_path: Optional[str] = None,
    n_jobs: int = 4,
) -> List[Dict]:
    """Run the beta × n grid experiment tracking all HOML variants.

    Results are cached at ``cache_path`` (a .npy file).  If the file exists
    the grid is skipped and the cached data is returned.

    Each returned dict contains keys:
        source, dist, n_samples, beta, eta_excess_kurtosis, eta_skewness_squared,
        ica_asymptotic_var, homl_asymptotic_var, ica_var_coeff,
        treatment_effect, treatment_coef, outcome_coef,
        rmse_oml, rmse_homl_known, rmse_homl_est, rmse_homl_split, rmse_ica.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[grid] Loading cached results from {cache_path}")
        return list(np.load(cache_path, allow_pickle=True))

    from ablation_utils import HOML_IDX as H
    from ablation_utils import ICA_IDX as I
    from ablation_utils import ORTHO_ML_IDX as O
    from ablation_utils import ROBUST_ORTHO_EST_IDX as RE
    from ablation_utils import ROBUST_ORTHO_SPLIT_IDX as RS
    from ablation_utils import (
        compute_estimation_statistics,
        create_covariate_sampler,
        create_outcome_noise_sampler,
        extract_treatment_estimates,
        run_parallel_experiments,
    )
    from oml_runner import setup_treatment_noise
    from oml_utils import AsymptoticVarianceCalculator

    np.random.seed(seed)

    support_size = 10
    treatment_effect = 1.0
    tc_scalar = 0.5
    oc_scalar = 0.0
    treatment_coef = np.zeros(support_size)
    treatment_coef[0] = tc_scalar
    outcome_coef = np.zeros(support_size)
    outcome_coef[0] = oc_scalar
    treatment_support = outcome_support = np.arange(support_size)
    x_sample = create_covariate_sampler("gennorm", 1.0)
    eps_sample = create_outcome_noise_sampler(np.sqrt(3.0))
    var_calc = AsymptoticVarianceCalculator()

    rows = []
    total = len(beta_values) * len(sample_sizes)
    done = 0
    for beta in beta_values:
        params_or_discounts, eta_sample, mean_discount, probs = setup_treatment_noise("gennorm", gennorm_beta=beta)
        eta_cv, eta_4m, _, eta_2m, eta_3m, homl_av, _ = var_calc.calc_homl_asymptotic_var_from_distribution(
            "gennorm", params_or_discounts, probs
        )
        ek, sk2, ica_av, _, _, _ = var_calc.calc_ica_asymptotic_var_from_distribution(
            treatment_coef, outcome_coef, treatment_effect, "gennorm", params_or_discounts, probs, eta_cv
        )
        ek_true = gennorm_excess_kurtosis(beta)

        for n in sample_sizes:
            done += 1
            print(f"[grid] {done}/{total}: beta={beta}, n={n}  (ek_true={ek_true:.3f})")
            lambda_reg = np.sqrt(np.log(support_size) / n)
            results = run_parallel_experiments(
                n_experiments=n_experiments,
                x_sample=x_sample,
                eta_sample=eta_sample,
                epsilon_sample=eps_sample,
                n_samples=n,
                cov_dim_max=support_size,
                treatment_effect=treatment_effect,
                treatment_support=treatment_support,
                treatment_coef=treatment_coef,
                outcome_support=outcome_support,
                outcome_coef=outcome_coef,
                eta_second_moment=float(eta_2m),
                eta_third_moment=float(eta_3m),
                lambda_reg=lambda_reg,
                check_convergence=False,
                verbose=False,
                oracle_support=True,
                disable_baselines=True,
            )
            if not results:
                print(f"  [WARN] No results for beta={beta}, n={n}")
                continue
            ortho_rec_tau = extract_treatment_estimates(results)
            biases, sigmas, rmse_arr = compute_estimation_statistics(ortho_rec_tau, treatment_effect)
            rows.append(
                dict(
                    source="grid",
                    dist=f"gennorm:{beta}",
                    n_samples=n,
                    beta=beta,
                    eta_excess_kurtosis=ek_true,
                    eta_skewness_squared=0.0,
                    ica_asymptotic_var=float(ica_av),
                    homl_asymptotic_var=float(homl_av),
                    ica_var_coeff=float(1 + (oc_scalar + tc_scalar * treatment_effect) ** 2),
                    treatment_effect=treatment_effect,
                    treatment_coef=tc_scalar,
                    outcome_coef=oc_scalar,
                    rmse_oml=float(rmse_arr[O]),
                    rmse_homl_known=float(rmse_arr[H]),
                    rmse_homl_est=float(rmse_arr[RE]),
                    rmse_homl_split=float(rmse_arr[RS]),
                    rmse_ica=float(rmse_arr[I]),
                )
            )
    if cache_path:
        np.save(cache_path, rows)
        print(f"[grid] Saved to {cache_path}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 7. Near-Gaussian blow-up sweep
# ──────────────────────────────────────────────────────────────────────────────


def run_blowup_sweep(
    beta_values: Optional[List[float]] = None,
    n: int = 2000,
    n_experiments: int = 30,
    seed: int = 99,
    cache_path: Optional[str] = None,
    n_jobs: int = 4,
) -> List[Dict]:
    """Sweep beta near 2.0 (near-Gaussian) to quantify HOML blow-up.

    Returns rows with the same schema as ``run_grid_experiment``.
    """
    if beta_values is None:
        # Fine grid approaching Gaussian from both sides
        beta_values = [1.6, 1.7, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.2, 2.4, 2.6, 3.0]

    if cache_path and os.path.exists(cache_path):
        print(f"[blowup] Loading cached results from {cache_path}")
        return list(np.load(cache_path, allow_pickle=True))

    rows = run_grid_experiment(
        beta_values=beta_values,
        sample_sizes=[n],
        n_experiments=n_experiments,
        seed=seed,
        cache_path=None,  # don't cache sub-call with grid cache
        n_jobs=n_jobs,
    )
    if cache_path:
        np.save(cache_path, rows)
        print(f"[blowup] Saved to {cache_path}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 8. Frontier / selector analysis
# ──────────────────────────────────────────────────────────────────────────────


def compute_oracle_selector_stats(
    rows: List[Dict],
    ek_high: float = _SELECTOR_EK_HIGH,
    ek_low: float = _SELECTOR_EK_LOW,
    include_homl_in_oracle: bool = True,
) -> Dict:
    """Compute mean RMSE for all strategies and the kurtosis selector.

    Returns a dict with keys:
        mean_oracle, mean_oml, mean_ica, mean_homl_known, mean_homl_est,
        mean_homl_split, mean_selector, selector_gap_recovery_pct,
        best_fixed_method, best_fixed_rmse, n_rows.
    """
    finite = [r for r in rows if np.isfinite(r["rmse_ica"]) and np.isfinite(r["rmse_oml"])]
    if not finite:
        return {}

    ri = np.array([r["rmse_ica"] for r in finite])
    ro = np.array([r["rmse_oml"] for r in finite])
    rh_k = np.array([r.get("rmse_homl_known", np.nan) for r in finite])
    rh_e = np.array([r.get("rmse_homl_est", np.nan) for r in finite])
    rh_s = np.array([r.get("rmse_homl_split", np.nan) for r in finite])

    oracle = oracle_envelope(finite, include_homl=include_homl_in_oracle)
    selector_rmse = apply_selector(finite, ek_high, ek_low)

    method_means = {
        "ICA": ri.mean(),
        "OML": ro.mean(),
        "HOML-known": np.nanmean(rh_k),
        "HOML-est": np.nanmean(rh_e),
        "HOML-split": np.nanmean(rh_s),
    }
    best_fixed_name = min(method_means, key=method_means.get)
    best_fixed_rmse = method_means[best_fixed_name]

    oracle_mean = oracle.mean()
    selector_mean = selector_rmse.mean()
    total_gap = best_fixed_rmse - oracle_mean
    selector_gap = selector_mean - oracle_mean
    recovery = 100.0 * (1 - selector_gap / total_gap) if total_gap > 1e-10 else 100.0

    return dict(
        n_rows=len(finite),
        mean_oracle=float(oracle_mean),
        mean_oml=float(ro.mean()),
        mean_ica=float(ri.mean()),
        mean_homl_known=float(np.nanmean(rh_k)),
        mean_homl_est=float(np.nanmean(rh_e)),
        mean_homl_split=float(np.nanmean(rh_s)),
        mean_selector=float(selector_mean),
        best_fixed_method=best_fixed_name,
        best_fixed_rmse=float(best_fixed_rmse),
        oracle_gap_abs=float(total_gap),
        oracle_gap_pct=100.0 * total_gap / best_fixed_rmse if best_fixed_rmse > 0 else 0.0,
        selector_gap_recovery_pct=float(recovery),
    )


def analyze_frontier(rows: List[Dict], label: str = "") -> Dict:
    """Print a detailed frontier analysis and return summary stats.

    This is the printable wrapper around ``compute_oracle_selector_stats``.
    """
    finite = [r for r in rows if np.isfinite(r["rmse_ica"]) and np.isfinite(r["rmse_oml"])]
    if not finite:
        print(f"[{label}] No finite rows.")
        return {}

    stats = compute_oracle_selector_stats(finite)
    n = stats["n_rows"]

    ri = np.array([r["rmse_ica"] for r in finite])
    ro = np.array([r["rmse_oml"] for r in finite])
    rh_k = np.array([r.get("rmse_homl_known", np.nan) for r in finite])
    rh_e = np.array([r.get("rmse_homl_est", np.nan) for r in finite])
    rh_s = np.array([r.get("rmse_homl_split", np.nan) for r in finite])

    oracle = oracle_envelope(finite, include_homl=True)
    oracle_no_homl = oracle_envelope(finite, include_homl=False)

    print(f"\n{'='*70}")
    print(f"FRONTIER ANALYSIS: {label}  (n={n})")
    print(f"{'='*70}")
    print("  Mean RMSE per method:")
    print(f"    OML:         {ro.mean():.5f}")
    print(f"    HOML-known:  {np.nanmean(rh_k):.5f}")
    print(f"    HOML-est:    {np.nanmean(rh_e):.5f}")
    print(f"    HOML-split:  {np.nanmean(rh_s):.5f}")
    print(f"    ICA:         {ri.mean():.5f}")
    print(f"    Oracle(+HOML):    {oracle.mean():.5f}")
    print(f"    Oracle(OML+ICA):  {oracle_no_homl.mean():.5f}")
    print(f"  Best fixed method: {stats['best_fixed_method']}  ({stats['best_fixed_rmse']:.5f})")
    print(f"  Oracle gap: {stats['oracle_gap_abs']:.5f}  ({stats['oracle_gap_pct']:.1f}%)")
    print(f"  Kurtosis selector: {stats['mean_selector']:.5f}")
    print(f"  Gap recovery: {stats['selector_gap_recovery_pct']:.1f}%")

    # ICA wins fraction vs each HOML variant
    ica_vs_oml = (ri < ro).mean()
    ica_vs_homl_k = (ri < rh_k)[np.isfinite(rh_k)].mean() if np.isfinite(rh_k).any() else np.nan
    print(f"\n  ICA wins vs OML:        {100*ica_vs_oml:.1f}%")
    print(f"  ICA wins vs HOML-known: {100*ica_vs_homl_k:.1f}%")

    # Kurtosis-regime breakdown
    ek = np.array([r.get("eta_excess_kurtosis", np.nan) for r in finite])
    has_ek = np.isfinite(ek)
    if has_ek.sum() >= 3:
        print("  RMSE by excess kurtosis regime:")
        regimes = [
            ("ek < -1  (sub-Gaussian/bounded)", ek < -1),
            ("-1 ≤ ek < 0  (mildly sub-G)", (ek >= -1) & (ek < 0)),
            ("0 ≤ ek < 3  (near-Gaussian)", (ek >= 0) & (ek < 3)),
            ("ek ≥ 3  (super-Gaussian/heavy)", ek >= 3),
        ]
        for desc, mask in regimes:
            m = mask & has_ek
            if m.sum() == 0:
                continue
            winner_val = min(ri[m].mean(), ro[m].mean(), np.nanmean(rh_k[m]))
            winner_name = "ICA" if ri[m].mean() == winner_val else "OML" if ro[m].mean() == winner_val else "HOML-k"
            print(
                f"    {desc}  n={m.sum()}"
                f"  OML={ro[m].mean():.5f}  HOML-k={np.nanmean(rh_k[m]):.5f}"
                f"  ICA={ri[m].mean():.5f}  → {winner_name}"
            )

    return stats


def analyze_blowup_sweep(rows: List[Dict]) -> List[Dict]:
    """Print HOML blow-up table and return rows with HOML/OML RMSE ratios."""
    print(f"\n{'='*70}")
    print("NEAR-GAUSSIAN BLOW-UP SWEEP (beta → 2.0, n=fixed)")
    print(f"{'='*70}")
    hdr = f"  {'beta':>6}  {'ek':>7}  {'OML':>9}  {'HOML-k':>9}  {'HOML-e':>9}  {'HOML-s':>9}  {'ICA':>9}  {'H/O':>9}"
    print(hdr)
    augmented = []
    for r in sorted(rows, key=lambda x: x.get("beta", 0)):
        beta = r.get("beta", np.nan)
        ek = r.get("eta_excess_kurtosis", np.nan)
        ro = r["rmse_oml"]
        rh_k = r.get("rmse_homl_known", np.nan)
        rh_e = r.get("rmse_homl_est", np.nan)
        rh_s = r.get("rmse_homl_split", np.nan)
        ri = r["rmse_ica"]
        ratio = rh_k / ro if (np.isfinite(rh_k) and ro > 1e-10) else np.nan
        flag = " *** BLOW-UP" if (np.isfinite(ratio) and ratio > 10) else ""
        print(
            f"  {beta:>6.2f}  {ek:>7.3f}  {ro:>9.5f}  {rh_k:>9.5f}  "
            f"{rh_e:>9.5f}  {rh_s:>9.5f}  {ri:>9.5f}  {ratio:>10.2f}{flag}"
        )
        aug = dict(r)
        aug["homl_oml_ratio"] = float(ratio) if np.isfinite(ratio) else np.nan
        augmented.append(aug)
    return augmented


def analyze_grid_crossover(rows: List[Dict]) -> None:
    """Print the beta × n crossover table for all HOML variants."""
    grid = [r for r in rows if r.get("source") == "grid" and np.isfinite(r["rmse_ica"]) and np.isfinite(r["rmse_oml"])]
    if not grid:
        return
    print("\n" + "=" * 70)
    print("GRID CROSSOVER: gennorm beta x n_samples (all HOML variants)")
    print(
        f"{'beta':>6} {'n':>6} {'ek':>7} {'OML':>9} {'HOML-k':>9} "
        f"{'HOML-e':>9} {'HOML-s':>9} {'ICA':>9} {'winner':>8}"
    )
    for r in sorted(grid, key=lambda x: (x.get("beta", 0), x["n_samples"])):
        beta = r.get("beta", np.nan)
        n = r["n_samples"]
        ek = r["eta_excess_kurtosis"]
        ro = r["rmse_oml"]
        rh_k = r.get("rmse_homl_known", np.nan)
        rh_e = r.get("rmse_homl_est", np.nan)
        rh_s = r.get("rmse_homl_split", np.nan)
        ri = r["rmse_ica"]
        candidates = {"OML": ro, "HOML-k": rh_k, "HOML-e": rh_e, "HOML-s": rh_s, "ICA": ri}
        finite_cands = {k: v for k, v in candidates.items() if np.isfinite(v)}
        winner = min(finite_cands, key=finite_cands.get) if finite_cands else "N/A"
        print(
            f"  {beta:>6.1f} {n:>6d} {ek:>7.3f} {ro:>9.5f} "
            f"{rh_k:>9.5f} {rh_e:>9.5f} {rh_s:>9.5f} {ri:>9.5f} {winner:>8}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── 9a. Load Bernoulli ablation ──────────────────────────────────────────
    print("Loading existing Bernoulli ablation results...")
    try:
        bern_rows = load_bernoulli_results()
        print(f"  Bernoulli ablation: {len(bern_rows)} configs")
    except FileNotFoundError as e:
        print(f"  WARNING: {e} — skipping Bernoulli ablation")
        bern_rows = []

    # ── 9b. Main beta × n grid (all HOML variants) ───────────────────────────
    grid_cache = os.path.join(OUT_DIR, "grid_results_v2.npy")
    print(f"\nRunning / loading beta x n grid experiment (cache: {grid_cache})...")
    grid_rows = run_grid_experiment(
        beta_values=[0.5, 1.0, 1.5, 2.0, 3.0],
        sample_sizes=[500, 1000, 2000, 5000],
        n_experiments=30,
        seed=42,
        cache_path=grid_cache,
        n_jobs=4,
    )
    print(f"  Grid: {len(grid_rows)} configs (5 beta x 4 n)")

    # ── 9c. Near-Gaussian blow-up sweep ─────────────────────────────────────
    blowup_cache = os.path.join(OUT_DIR, "blowup_results.npy")
    print(f"\nRunning / loading near-Gaussian blow-up sweep (cache: {blowup_cache})...")
    blowup_rows = run_blowup_sweep(
        beta_values=[1.6, 1.7, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.2, 2.4, 2.6, 3.0],
        n=2000,
        n_experiments=30,
        seed=99,
        cache_path=blowup_cache,
        n_jobs=4,
    )
    print(f"  Blow-up sweep: {len(blowup_rows)} configs")

    # ── 9d. Frontier analyses ────────────────────────────────────────────────
    if bern_rows:
        analyze_frontier(bern_rows, "Bernoulli ablation (eta dist)")
    analyze_frontier(grid_rows, "Grid (gennorm beta x n_samples)")
    if bern_rows:
        analyze_frontier(bern_rows + grid_rows, "All combined")
    analyze_grid_crossover(grid_rows)
    analyze_blowup_sweep(blowup_rows)

    # ── 9e. Summary stats ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: oracle + selector stats (grid only)")
    print(f"{'='*70}")
    stats = compute_oracle_selector_stats(grid_rows)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Blow-up quantification
    print(f"\n{'='*70}")
    print("HOML NEAR-GAUSSIAN BLOW-UP QUANTIFICATION")
    print(f"{'='*70}")
    blowup_detected = homl_blowup_detected(blowup_rows + grid_rows, threshold=10.0)
    print(f"  Blow-up detected (HOML/OML ratio ≥ 10× near ek≈0): {blowup_detected}")
    near_gauss = [r for r in (blowup_rows + grid_rows) if abs(r.get("eta_excess_kurtosis", 99)) < 0.5]
    if near_gauss:
        ratios = [
            r["rmse_homl_known"] / r["rmse_oml"]
            for r in near_gauss
            if np.isfinite(r.get("rmse_homl_known", np.nan)) and r["rmse_oml"] > 1e-10
        ]
        if ratios:
            print(f"  Near-Gaussian rows (|ek|<0.5): {len(near_gauss)}")
            print(
                f"  HOML-known/OML ratio: min={min(ratios):.1f}x"
                f"  max={max(ratios):.1f}x  mean={np.mean(ratios):.1f}x"
            )

    # Asymptotic variance ratio prediction accuracy
    print(f"\n{'='*70}")
    print("ASYMPTOTIC VARIANCE RATIO AS CROSSOVER PREDICTOR (ICA vs OML)")
    print(f"{'='*70}")
    correct, total = 0, 0
    for r in grid_rows:
        ia = r.get("ica_asymptotic_var", np.nan)
        homl_av = r.get("homl_asymptotic_var", np.nan)
        # use HOML avar as OML proxy (HOML avar → ∞ near Gauss, OML avar is finite)
        if not (np.isfinite(r["rmse_ica"]) and np.isfinite(r["rmse_oml"])):
            continue
        if not (np.isfinite(ia) and np.isfinite(r["rmse_oml"])):
            continue
        # Prediction: if ica_avar < homl_avar → ICA better (if both finite)
        if np.isfinite(homl_av) and homl_av > 0:
            theory = ia < homl_av
            empirical = r["rmse_ica"] < r["rmse_oml"]
            correct += int(theory == empirical)
            total += 1
    if total:
        print(
            f"  Prediction accuracy (ica_avar < homl_avar → ICA empirically better): "
            f"{correct}/{total} = {100*correct/total:.1f}%"
        )

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
