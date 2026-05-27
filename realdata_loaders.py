"""Loaders for two canonical causal-inference benchmarks: IHDP and Jobs (LaLonde).

IHDP (Infant Health and Development Program)
--------------------------------------------
Semi-synthetic benchmark from Hill (2011) "Bayesian Nonparametric Modeling for
Causal Inference". Covariates are real (n=747, 25 covariates: 5 continuous +
20 binary), outcomes are simulated ("setting B": log(bw+0.5) surface). The 1000
NPCI replications each resimulate Y(0), Y(1) from the real covariates, so
ground-truth ATE/ATT are *known per replication*.

Data source: CEVAE repository (AMLab-Amsterdam), individual CSV files:
  https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{i}.csv

Column layout (no header row):
  col 0      : treatment T ∈ {0, 1}
  cols 1–10  : 10 continuous covariates (standardised)
  cols 11–25 : 15 binary covariates
  col 26     : y_factual   (observed outcome under assigned treatment)
  col 27     : y_cfactual  (counterfactual outcome)
  col 28     : mu_0        (E[Y(0)|X], noiseless)
  col 29     : mu_1        (E[Y(1)|X], noiseless)

True ATE  = mean(mu_1 - mu_0)
True ATT  = mean(mu_1 - mu_0 | T=1)

Jobs (LaLonde / Dehejia–Wahba NSW experimental data)
-----------------------------------------------------
The canonical LaLonde (1986) / Dehejia–Wahba (1999) NSW job-training experiment.
Binary treatment = job-training programme assignment. Outcome = real earnings 1978
(re78, USD). No ground-truth potential outcomes; the experimental estimate of
ATT ≈ $1,794 (Dehejia & Wahba 1999, Table 2) serves as the benchmark.

Variables: treat, age, educ, black, hisp, married, nodegree, re74, re75, re78
(standard 10-column Dehejia–Wahba format).

Primary data source: NBER Stata file at
  https://users.nber.org/~rdehejia/data/nsw_dw.dta
which is downloaded via urllib (SSL verification disabled for NBER) and
converted to CSV (``data/lalonde_nsw_dw.csv``) on first use. The ``is_real``
key in the returned metadata dict is True when this file is used.

CPS and PSID control groups (for observational bias comparison) are loaded
from ``data/cps_controls.csv`` / ``data/psid_controls.csv`` if available;
these are the standard Dehejia–Wahba CPS-1 and PSID-1 comparison samples.

Usage
-----
::

    from realdata_loaders import load_ihdp, load_jobs, load_jobs_observational

    # IHDP: single replication
    X, T, Y, true_ate = load_ihdp(replication=1)

    # IHDP: multiple replications stacked
    X, T, Y, true_ate_arr = load_ihdp(replication=None, n_replications=10)

    # Jobs (experimental NSW only)
    X, T, Y, meta = load_jobs()
    print(meta["att_benchmark"], meta["is_real"])

    # Jobs with CPS/PSID observational controls appended
    X_obs, T_obs, Y_obs, meta_obs = load_jobs_observational(comparison="cps")
"""

from __future__ import annotations

import os
import ssl
import urllib.request
from typing import Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# IHDP
# ---------------------------------------------------------------------------

_IHDP_BASE_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/"
_IHDP_N_COVARIATES = 25  # 10 continuous + 15 binary (cols 1–25)


def _ihdp_csv_path(replication: int) -> str:
    return os.path.join(DATA_DIR, f"ihdp_npci_{replication}.csv")


def _download_ihdp(replication: int, dest: Optional[str] = None) -> bool:
    """Download a single IHDP NPCI replication CSV.  Returns True on success."""
    url = _IHDP_BASE_URL + f"ihdp_npci_{replication}.csv"
    if dest is None:
        dest = _ihdp_csv_path(replication)
    if os.path.exists(dest):
        return True
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.urlopen(url, context=ctx, timeout=20)
        with open(dest, "wb") as fh:
            fh.write(req.read())
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _parse_ihdp_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Parse a (747, 30) IHDP NPCI array into (X, T, Y, ate, att).

    The returned Y is y_factual (the *observed* outcome under the assigned
    treatment). Ground-truth ATE and ATT come from mu_0 / mu_1.
    """
    t = arr[:, 0].astype(float)  # binary treatment
    x = arr[:, 1:26].astype(float)  # 25 covariates
    y_factual = arr[:, 26].astype(float)
    mu0 = arr[:, 28].astype(float)
    mu1 = arr[:, 29].astype(float)
    ate = float(np.mean(mu1 - mu0))
    att = float(np.mean((mu1 - mu0)[t == 1]))
    return x, t, y_factual, ate, att


def _ihdp_synthetic_fixture(
    n_samples: int = 747,
    n_covariates: int = 25,
    seed: int = 0,
) -> np.ndarray:
    """Return a synthetic IHDP-shaped array (n_samples, 30) for offline testing.

    The fixture preserves the essential schema properties:
      - col 0: binary T with ~19% treated (matching real IHDP)
      - cols 1–10: continuous, standardised N(0,1) roughly
      - cols 11–25: binary (Bernoulli ~0.5)
      - cols 26, 27: y_factual, y_cfactual (continuous, positive)
      - cols 28–29: mu_0, mu_1 (noiseless potential outcomes)

    The true ATE of the fixture is ~0.4 by construction.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_samples, 30))
    # Treatment: ~19% treated
    propensity = 0.19 * np.ones(n_samples)
    out[:, 0] = (rng.uniform(size=n_samples) < propensity).astype(float)
    # Continuous covariates (5 groups of 2)
    out[:, 1:11] = rng.standard_normal((n_samples, 10))
    # Binary covariates
    out[:, 11:26] = (rng.uniform(size=(n_samples, 15)) < 0.5).astype(float)
    # Potential outcomes: mu_0 = 0.5 + linear combo, mu_1 = mu_0 + 0.4 + noise
    mu0 = 0.5 + 0.1 * out[:, 1] + 0.05 * out[:, 2]
    mu0 = np.clip(mu0, 0.0, 1.0)
    mu1 = np.clip(mu0 + 0.4, 0.0, 1.0)
    t = out[:, 0]
    eps = rng.standard_normal(n_samples) * 0.05
    y_factual = np.where(t == 1, mu1, mu0) + eps
    y_factual = np.clip(y_factual, 0.0, 1.0)
    y_cfactual = np.where(t == 1, mu0, mu1) + rng.standard_normal(n_samples) * 0.05
    y_cfactual = np.clip(y_cfactual, 0.0, 1.0)
    out[:, 26] = y_factual
    out[:, 27] = y_cfactual
    out[:, 28] = mu0
    out[:, 29] = mu1
    return out


def load_ihdp(
    replication: Optional[int] = 1,
    n_replications: Optional[int] = None,
    data_dir: Optional[str] = None,
    use_fixture_on_failure: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """Load one or several IHDP NPCI replications.

    Parameters
    ----------
    replication : int or None
        1-indexed replication number. If ``None``, loads the first
        ``n_replications`` replications and stacks them.
    n_replications : int or None
        Number of replications to load when ``replication=None``.
        Defaults to 10.
    data_dir : str or None
        Override the default ``data/`` cache directory.
    use_fixture_on_failure : bool
        If True, return a synthetic fixture when download fails. If False,
        raise ``FileNotFoundError``.

    Returns
    -------
    X : np.ndarray
        Covariate matrix (n_samples, 25) or (n_rep * n_samples, 25) when
        stacking.
    T : np.ndarray
        Binary treatment vector (n_samples,) or stacked.
    Y : np.ndarray
        Observed outcome (y_factual) vector, same length.
    true_effect : float or np.ndarray
        When loading a single replication: scalar ATT.
        When stacking: 1-D array of per-replication ATTs, length n_rep.
    """
    if data_dir is not None:
        cache = data_dir
    else:
        cache = DATA_DIR

    if replication is not None:
        # Single replication
        dest = os.path.join(cache, f"ihdp_npci_{replication}.csv")
        if not os.path.exists(dest):
            ok = _download_ihdp(replication, dest=dest)
            if not ok:
                if use_fixture_on_failure:
                    arr = _ihdp_synthetic_fixture(seed=replication)
                    x, t, y, ate, att = _parse_ihdp_array(arr)
                    return x, t, y, att
                raise FileNotFoundError(
                    f"Could not download IHDP replication {replication} and " f"use_fixture_on_failure=False."
                )
        arr = np.loadtxt(dest, delimiter=",")
        x, t, y, ate, att = _parse_ihdp_array(arr)
        return x, t, y, att

    # Multiple replications stacked
    n_rep = n_replications if n_replications is not None else 10
    xs, ts, ys, atts = [], [], [], []
    for rep in range(1, n_rep + 1):
        x_i, t_i, y_i, att_i = load_ihdp(
            replication=rep,
            data_dir=cache,
            use_fixture_on_failure=use_fixture_on_failure,
        )
        xs.append(x_i)
        ts.append(t_i)
        ys.append(y_i)
        atts.append(att_i)
    return (
        np.vstack(xs),
        np.concatenate(ts),
        np.concatenate(ys),
        np.array(atts),
    )


# ---------------------------------------------------------------------------
# Jobs (LaLonde / Dehejia–Wahba)
# ---------------------------------------------------------------------------

_JOBS_COLUMNS = ["treat", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "re78"]

# Experimental ATT benchmark from Dehejia & Wahba (1999), Table 2
# NSW randomised experiment: effect of job training on 1978 earnings
_JOBS_ATT_BENCHMARK = 1794.0  # USD

# Primary source: NBER Stata file (Dehejia–Wahba sample)
_JOBS_NBER_DTA_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
_JOBS_NBER_CSV_NAME = "lalonde_nsw_dw.csv"  # cached CSV in data/

# CPS/PSID control files for observational comparison
_CPS_NBER_DTA_URL = "https://users.nber.org/~rdehejia/data/cps_controls.dta"
_PSID_NBER_DTA_URL = "https://users.nber.org/~rdehejia/data/psid_controls.dta"


def _jobs_synthetic_fixture(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic LaLonde/NSW fixture matching the known dataset statistics.

    The NSW experimental sample has n≈445 (185 treated, 260 control).
    Marginal statistics are drawn to match Table 2 of Dehejia & Wahba (1999):
      - age: mean~26, sd~7
      - educ: mean~10.3, sd~2
      - black: ~84%
      - hisp: ~6%
      - married: ~19%
      - nodegree: ~71%
      - re74: ~$2,096 (many zeros, right-skewed)
      - re75: ~$1,532 (many zeros, right-skewed)
      - re78 control: ~$4,555; treated: ~$6,349 → ATT ~$1,794

    The fixture yields ATT ≈ 1794 by construction so tests are meaningful.
    """
    rng = np.random.default_rng(seed)
    n_treat, n_ctrl = 185, 260
    n = n_treat + n_ctrl
    treat = np.concatenate([np.ones(n_treat), np.zeros(n_ctrl)])

    age = rng.normal(26, 7, n).clip(16, 55).round()
    educ = rng.normal(10.3, 2, n).clip(0, 18).round()
    black = (rng.uniform(size=n) < 0.84).astype(float)
    hisp = (rng.uniform(size=n) < 0.06).astype(float)
    married = (rng.uniform(size=n) < 0.19).astype(float)
    nodegree = (rng.uniform(size=n) < 0.71).astype(float)

    # Pre-treatment earnings: mixture of zeros and log-normal
    def _earn(n_obs, zero_prob, mu_log, sd_log):
        earn = np.zeros(n_obs)
        nonzero = rng.uniform(size=n_obs) > zero_prob
        earn[nonzero] = np.exp(rng.normal(mu_log, sd_log, nonzero.sum()))
        return earn

    re74 = _earn(n, 0.70, 7.5, 1.0)
    re75 = _earn(n, 0.60, 7.3, 1.0)

    # re78: control mean ≈ 4555, treated mean ≈ 6349 (ATT ~ 1794)
    re78_ctrl = _earn(n_ctrl, 0.45, 8.0, 1.2)
    re78_treat = _earn(n_treat, 0.35, 8.2, 1.1)
    # Rescale so the ATT is exactly 1794
    actual_att = re78_treat.mean() - re78_ctrl.mean()
    re78_treat = re78_treat + (_JOBS_ATT_BENCHMARK - actual_att)
    re78 = np.concatenate([re78_treat, re78_ctrl])

    X = np.column_stack([age, educ, black, hisp, married, nodegree, re74, re75])
    T = treat
    Y = re78
    return X, T, Y


def _download_nber_dta(url: str, dest: str) -> bool:
    """Download a Stata .dta file from NBER via urllib (SSL verification off).

    Returns True on success.
    """
    if os.path.exists(dest):
        return True
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.urlopen(url, context=ctx, timeout=30)
        raw = req.read()
        with open(dest, "wb") as fh:
            fh.write(raw)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _dta_to_csv(dta_path: str, csv_path: str) -> bool:
    """Convert a Stata .dta file to a normalised CSV using pandas.

    Column name normalisations applied:
      - ``education`` → ``educ``
      - ``hispanic`` → ``hisp``
      - ``data_id`` column dropped

    Returns True on success.
    """
    try:
        import pandas as pd  # pylint: disable=import-outside-toplevel

        df = pd.read_stata(dta_path)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.rename(columns={"education": "educ", "hispanic": "hisp"})
        if "data_id" in df.columns:
            df = df.drop(columns=["data_id"])
        df.to_csv(csv_path, index=False)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _try_download_jobs_csv(cache: str) -> Tuple[bool, str]:
    """Attempt to produce ``lalonde_nsw_dw.csv`` in *cache*.

    Strategy (in order):
    1. CSV already cached — done.
    2. DTA already cached — convert to CSV.
    3. Download DTA from NBER, then convert.

    Returns (success: bool, csv_path: str).
    """
    csv_path = os.path.join(cache, _JOBS_NBER_CSV_NAME)
    if os.path.exists(csv_path):
        return True, csv_path

    dta_path = os.path.join(cache, "nsw_dw.dta")
    if not os.path.exists(dta_path):
        ok = _download_nber_dta(_JOBS_NBER_DTA_URL, dta_path)
        if not ok:
            return False, csv_path

    ok = _dta_to_csv(dta_path, csv_path)
    return ok, csv_path


def load_jobs(
    data_dir: Optional[str] = None,
    use_fixture_on_failure: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the LaLonde / Dehejia–Wahba NSW job-training dataset.

    The loader first looks for ``data/lalonde_nsw_dw.csv`` (the normalised
    CSV derived from the NBER Stata file).  If absent it attempts to download
    ``nsw_dw.dta`` from ``https://users.nber.org/~rdehejia/data/`` and
    converts it to CSV using pandas.  On failure (no internet, pandas
    unavailable, etc.) the behaviour is controlled by ``use_fixture_on_failure``.

    Parameters
    ----------
    data_dir : str or None
        Override the default ``data/`` cache directory.
    use_fixture_on_failure : bool
        If True, return a synthetic fixture when the real data cannot be
        obtained (clearly flagged in the returned metadata dict).

    Returns
    -------
    X : np.ndarray
        Covariate matrix (n_samples, 8).
        Columns: age, educ, black, hisp, married, nodegree, re74, re75.
    T : np.ndarray
        Binary treatment vector (n_samples,).
    Y : np.ndarray
        Outcome = real earnings 1978 (re78), shape (n_samples,).
    meta : dict
        ``att_benchmark``   : float — experimental ATT benchmark (~1794 USD)
        ``att_experimental``: float — naive difference-in-means from this data
        ``is_real``         : bool  — True if real data was loaded
        ``n_treated``       : int
        ``n_control``       : int
        ``source``          : str   — description of data provenance
        ``covariate_names`` : list[str]
    """
    cache = data_dir if data_dir is not None else DATA_DIR
    ok, csv_path = _try_download_jobs_csv(cache)

    if ok:
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel

            df = pd.read_csv(csv_path)
            df.columns = [c.lower().strip() for c in df.columns]
            if df.columns[0] in ("", "unnamed: 0"):
                df = df.iloc[:, 1:]
            # Locate standard columns
            t_col = next(c for c in df.columns if c in ("treat", "treatment"))
            y_col = next(c for c in df.columns if c in ("re78", "re_78", "earnings78"))
            covariate_cols = [c for c in df.columns if c not in (t_col, y_col)]
            T = df[t_col].values.astype(float)
            Y = df[y_col].values.astype(float)
            X = df[covariate_cols].values.astype(float)
            att_experimental = float(Y[T == 1].mean() - Y[T == 0].mean())
            meta = {
                "att_benchmark": _JOBS_ATT_BENCHMARK,
                "att_experimental": att_experimental,
                "is_real": True,
                "n_treated": int((T == 1).sum()),
                "n_control": int((T == 0).sum()),
                "source": (
                    f"NBER Dehejia–Wahba NSW sample (nsw_dw.dta) downloaded from "
                    f"{_JOBS_NBER_DTA_URL}, cached at {csv_path}. "
                    f"Naive experimental ATT = ${att_experimental:,.0f} "
                    f"(benchmark ${_JOBS_ATT_BENCHMARK:,.0f})."
                ),
                "covariate_names": covariate_cols,
            }
            return X, T, Y, meta
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Fall through to fixture

    if not use_fixture_on_failure:
        raise FileNotFoundError(
            "Could not download or parse the LaLonde Jobs dataset and " "use_fixture_on_failure=False."
        )

    X, T, Y = _jobs_synthetic_fixture()
    meta = {
        "att_benchmark": _JOBS_ATT_BENCHMARK,
        "att_experimental": float(Y[T == 1].mean() - Y[T == 0].mean()),
        "is_real": False,
        "n_treated": int((T == 1).sum()),
        "n_control": int((T == 0).sum()),
        "source": (
            "SYNTHETIC FIXTURE — real data unavailable. Schema matches "
            "Dehejia & Wahba (1999) NSW experimental sample. "
            "ATT is set to the benchmark value by construction."
        ),
        "covariate_names": ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"],
    }
    return X, T, Y, meta


def load_jobs_observational(
    comparison: str = "cps",
    data_dir: Optional[str] = None,
    use_fixture_on_failure: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load NSW treated units combined with CPS or PSID observational controls.

    This is the standard observational comparison from LaLonde (1986) and
    Dehejia & Wahba (1999): treated group comes from the NSW experiment,
    control group comes from the CPS-1 or PSID-1 comparison sample.  The
    purpose is to study how well estimators recover the experimental ATT
    (~$1,794) when the controls are non-experimental.

    Parameters
    ----------
    comparison : str
        ``"cps"``  — CPS-1 comparison group (n=15,992 controls)
        ``"psid"`` — PSID-1 comparison group (n=2,490 controls)
    data_dir : str or None
        Override the default ``data/`` cache directory.
    use_fixture_on_failure : bool
        If True, fall back to fixture controls when real CPS/PSID are missing.

    Returns
    -------
    X, T, Y, meta : same layout as ``load_jobs``.
        ``meta["comparison"]`` — ``"cps"`` or ``"psid"``
        ``meta["is_real"]``    — True only when both NSW and comparison are real
    """
    cache = data_dir if data_dir is not None else DATA_DIR
    if comparison not in ("cps", "psid"):
        raise ValueError(f"comparison must be 'cps' or 'psid'; got '{comparison}'")

    # Load NSW treated units from real data
    X_nsw, T_nsw, Y_nsw, meta_nsw = load_jobs(data_dir=cache, use_fixture_on_failure=use_fixture_on_failure)
    treated_mask = T_nsw == 1
    X_treat = X_nsw[treated_mask]
    T_treat = T_nsw[treated_mask]
    Y_treat = Y_nsw[treated_mask]

    # Load observational controls
    ctrl_url = _CPS_NBER_DTA_URL if comparison == "cps" else _PSID_NBER_DTA_URL
    ctrl_csv_name = f"{comparison}_controls.csv"
    ctrl_csv_path = os.path.join(cache, ctrl_csv_name)
    ctrl_dta_name = f"{comparison}_controls.dta"
    ctrl_dta_path = os.path.join(cache, ctrl_dta_name)

    ctrl_is_real = False
    X_ctrl = Y_ctrl = T_ctrl = None

    if not os.path.exists(ctrl_csv_path):
        if not os.path.exists(ctrl_dta_path):
            _download_nber_dta(ctrl_url, ctrl_dta_path)
        if os.path.exists(ctrl_dta_path):
            _dta_to_csv(ctrl_dta_path, ctrl_csv_path)

    if os.path.exists(ctrl_csv_path):
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel

            df_ctrl = pd.read_csv(ctrl_csv_path)
            df_ctrl.columns = [c.lower().strip() for c in df_ctrl.columns]
            if df_ctrl.columns[0] in ("", "unnamed: 0"):
                df_ctrl = df_ctrl.iloc[:, 1:]
            t_col = next(c for c in df_ctrl.columns if c in ("treat", "treatment"))
            y_col = next(c for c in df_ctrl.columns if c in ("re78", "re_78", "earnings78"))
            covariate_cols = [c for c in df_ctrl.columns if c not in (t_col, y_col)]
            # Keep only the same covariates as the treated sample
            if meta_nsw.get("covariate_names"):
                covariate_cols = [c for c in covariate_cols if c in meta_nsw["covariate_names"]]
            X_ctrl = df_ctrl[covariate_cols].values.astype(float)
            T_ctrl = df_ctrl[t_col].values.astype(float)
            Y_ctrl = df_ctrl[y_col].values.astype(float)
            ctrl_is_real = True
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    if X_ctrl is None:
        if not use_fixture_on_failure:
            raise FileNotFoundError(f"Could not load {comparison.upper()} controls.")
        # Synthetic controls: same fixture as before but only the control rows
        _, T_fix, Y_fix, _ = load_jobs(data_dir=cache, use_fixture_on_failure=True)
        ctrl_mask = T_fix == 0
        X_ctrl_full, _, _ = _jobs_synthetic_fixture()
        X_ctrl = X_ctrl_full[ctrl_mask]
        T_ctrl = np.zeros(X_ctrl.shape[0])
        Y_ctrl = Y_fix[ctrl_mask]

    X_obs = np.vstack([X_treat, X_ctrl])
    T_obs = np.concatenate([T_treat, T_ctrl])
    Y_obs = np.concatenate([Y_treat, Y_ctrl])

    meta_obs = {
        "att_benchmark": _JOBS_ATT_BENCHMARK,
        "att_experimental": meta_nsw.get("att_experimental", _JOBS_ATT_BENCHMARK),
        "is_real": meta_nsw["is_real"] and ctrl_is_real,
        "n_treated": int(T_treat.shape[0]),
        "n_control": int(T_ctrl.shape[0]),
        "comparison": comparison.upper(),
        "source": (
            f"NSW treated (n={T_treat.shape[0]}) + "
            f"{comparison.upper()} observational controls (n={T_ctrl.shape[0]}). "
            f"Real data: {meta_nsw['is_real'] and ctrl_is_real}."
        ),
        "covariate_names": meta_nsw.get("covariate_names", []),
    }
    return X_obs, T_obs, Y_obs, meta_obs
