"""Semi-synthetic data generation for partially-linear model experiments.

Provides a loader for real covariate matrices and a DGP that imposes the
partially-linear regression (PLR) model on top of them:

    T = nonlinearity(X[:, support] @ treatment_coef) + eta
    Y = theta * T + nonlinearity(X[:, support] @ outcome_coef) + eps

This is the entry point for the rebuttal's semi-synthetic experiment
(Section 4 of the UAI 2026 rebuttal plan).
"""

from __future__ import annotations

import math
import os
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_nonlinearity(x: np.ndarray, name: str) -> np.ndarray:
    """Apply an element-wise nonlinearity.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.
    name : str
        One of ``"identity"`` or ``"leaky_relu"``.

    Returns
    -------
    np.ndarray
        Transformed array, same shape as *x*.

    Raises
    ------
    ValueError
        If *name* is not a recognised nonlinearity.
    """
    if name == "identity":
        return x
    elif name == "leaky_relu":
        return np.where(x >= 0, x, 0.01 * x)
    else:
        raise ValueError(f"Unknown nonlinearity '{name}'. Choose 'identity' or 'leaky_relu'.")


def _make_eta_sampler(distribution: str, rng: np.random.Generator) -> Callable[[int], np.ndarray]:
    """Return a zero-mean eta sampler for the given distribution name.

    Tries to delegate to ``oml_runner.setup_treatment_noise`` so that the
    noise is identical to the fully-synthetic experiments.  Falls back to a
    small in-module dispatcher for distributions not handled there.

    Parameters
    ----------
    distribution : str
        One of the distributions supported by ``oml_runner.setup_treatment_noise``
        (``"discrete"``, ``"laplace"``, ``"uniform"``, ``"rademacher"``) or any
        distribution understood by the fallback dispatcher.
    rng : np.random.Generator
        Generator used **only** by the fallback path; the ``oml_runner``
        path uses ``np.random`` global state (matching existing experiments).

    Returns
    -------
    Callable[[int], np.ndarray]
        Function ``eta_sample(n) -> np.ndarray`` of shape ``(n,)``.
    """
    try:
        from oml_runner import setup_treatment_noise

        _, eta_sample, _, _ = setup_treatment_noise(distribution=distribution)
        return eta_sample
    except (ImportError, ValueError):
        pass

    # Fallback dispatcher — matches distributions used in eta_noise_ablation.py
    if distribution == "discrete":
        discounts = np.array([0, -0.5, -2.0, -4.0])
        probs = np.array([0.65, 0.2, 0.1, 0.05])
        mean_d = float(np.dot(discounts, probs))

        def eta_sample(n: int) -> np.ndarray:
            idx = np.argmax(rng.multinomial(1, probs, n), axis=1)
            return np.array([discounts[i] - mean_d for i in idx])

    elif distribution == "laplace":
        laplace_scale = 1.0 / math.sqrt(2)

        def eta_sample(n: int) -> np.ndarray:
            return rng.laplace(loc=0.0, scale=laplace_scale, size=n)

    elif distribution == "rademacher":

        def eta_sample(n: int) -> np.ndarray:
            return rng.choice(np.array([1.0, -1.0]), size=n, p=np.array([0.5, 0.5]))

    elif distribution == "uniform":
        half_width = math.sqrt(3)

        def eta_sample(n: int) -> np.ndarray:
            return rng.uniform(low=-half_width, high=half_width, size=n)

    else:
        raise ValueError(
            f"Unknown eta distribution '{distribution}'. " "Supported: 'discrete', 'laplace', 'rademacher', 'uniform'."
        )

    return eta_sample


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_real_covariates(
    name: str = "california_housing",
    n_samples: int | None = None,
    standardize: bool = True,
    seed: int = 12143,
) -> np.ndarray:
    """Load a real covariate matrix from a public dataset.

    Parameters
    ----------
    name : str
        Dataset to load.  One of:

        * ``"california_housing"`` — fetched via
          ``sklearn.datasets.fetch_california_housing``.  Returns all 8
          feature columns (the target/house-price column is discarded).
        * ``"ihdp"`` — loaded from ``data/ihdp_npci_1.npz`` relative to the
          repository root.  The file must be vendored manually; no remote
          download is attempted.

    n_samples : int or None
        If given, deterministically subsample to exactly *n_samples* rows
        (without replacement) using ``np.random.default_rng(seed)``.

    standardize : bool
        If ``True`` (default), centre and scale each column to zero mean and
        unit variance.

    seed : int
        RNG seed used for subsampling.  Ignored when *n_samples* is ``None``.

    Returns
    -------
    np.ndarray
        Covariate matrix of shape ``(n, d)``.

    Raises
    ------
    FileNotFoundError
        When ``name="ihdp"`` and the NPZ file is not present.
    ValueError
        When *name* is not a recognised dataset.
    """
    if name == "california_housing":
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=False)
        X = data.data.astype(np.float64)

    elif name == "ihdp":
        npz_path = os.path.join(_REPO_ROOT, "data", "ihdp_npci_1.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"IHDP file not found at '{npz_path}'. "
                "Please vendor the file manually: download ihdp_npci_1.npz and place it at "
                f"'{npz_path}'.  Remote download is intentionally disabled to avoid cluster "
                "network issues."
            )
        npz = np.load(npz_path, allow_pickle=False)
        # Convention: covariates are stored under key 'x' or 'X'
        key = "x" if "x" in npz else "X"
        X = npz[key].astype(np.float64)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'california_housing' or 'ihdp'.")

    if n_samples is not None:
        rng = np.random.default_rng(seed)
        if n_samples > X.shape[0]:
            raise ValueError(f"n_samples={n_samples} exceeds dataset size {X.shape[0]} for '{name}'.")
        idx = rng.choice(X.shape[0], size=n_samples, replace=False)
        X = X[idx]

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Avoid division by zero for constant columns
        std = np.where(std == 0.0, 1.0, std)
        X = (X - mean) / std

    return X


def generate_semi_synthetic_pl(
    X: np.ndarray,
    treatment_effect: float = 1.0,
    eta_distribution: str = "discrete",
    treatment_coef_scalar: float = 1.0,
    outcome_coef_scalar: float = 1.0,
    sigma_outcome: float = math.sqrt(3.0),
    nonlinearity: str = "identity",
    support_size: int | None = None,
    seed: int = 12143,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate semi-synthetic data by imposing a PLR mechanism on real covariates.

    The partially-linear regression DGP is:

    .. code-block:: text

        T = nonlinearity(X[:, support] @ treatment_coef) + eta
        Y = treatment_effect * T + nonlinearity(X[:, support] @ outcome_coef) + eps

    where ``eta`` is drawn from *eta_distribution* (delegated to
    ``oml_runner.setup_treatment_noise`` so the noise is identical to the
    fully-synthetic experiments) and ``eps ~ N(0, sigma_outcome²)``.

    Parameters
    ----------
    X : np.ndarray
        Real covariate matrix of shape ``(n, d)``.
    treatment_effect : float
        True causal effect ``theta``.
    eta_distribution : str
        Treatment noise distribution.  Passed directly to
        ``oml_runner.setup_treatment_noise``; supported values are
        ``"discrete"``, ``"laplace"``, ``"uniform"``, ``"rademacher"``.
    treatment_coef_scalar : float
        Scale applied to the randomly-drawn treatment coefficient before
        L2 normalisation.  After normalisation the coefficient vector always
        has unit L2 norm (to prevent blowup with real-X scale).
    outcome_coef_scalar : float
        Same as *treatment_coef_scalar* but for the outcome coefficient.
    sigma_outcome : float
        Standard deviation of the additive outcome noise ``eps``.
    nonlinearity : str
        Element-wise nonlinearity applied to the linear index
        ``X[:, support] @ coef``.  One of ``"identity"`` (pass-through) or
        ``"leaky_relu"`` (``max(x, 0.01 x)``).
    support_size : int or None
        Number of covariate columns to include in the linear index.  If
        ``None``, all *d* columns are used.
    seed : int
        RNG seed controlling coefficient draws, support selection, and
        outcome noise.

    Returns
    -------
    X : np.ndarray
        Original covariate matrix ``(n, d)`` (returned unchanged for
        pipeline convenience).
    T : np.ndarray
        Treatment vector of shape ``(n,)``.
    Y : np.ndarray
        Outcome vector of shape ``(n,)``.
    metadata : dict
        Ground-truth information with keys:

        * ``"treatment_effect"`` — float, the true ``theta``
        * ``"treatment_coef"`` — np.ndarray, shape ``(support_size,)``
        * ``"outcome_coef"`` — np.ndarray, shape ``(support_size,)``
        * ``"eta_distribution"`` — str
        * ``"support_indices"`` — np.ndarray of int, shape ``(support_size,)``
        * ``"nonlinearity"`` — str
        * ``"sigma_outcome"`` — float
        * ``"seed"`` — int
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)

    # --- Support selection ---
    if support_size is None:
        support_size = d
        support_indices = np.arange(d)
    else:
        if support_size > d:
            raise ValueError(f"support_size={support_size} exceeds X.shape[1]={d}.")
        support_indices = rng.choice(d, size=support_size, replace=False)

    X_sup = X[:, support_indices]  # (n, support_size)

    # --- Coefficient draws (unit L2 norm after scaling) ---
    raw_tc = treatment_coef_scalar * rng.standard_normal(support_size)
    tc_norm = np.linalg.norm(raw_tc)
    treatment_coef = raw_tc / (tc_norm if tc_norm > 0 else 1.0)

    raw_oc = outcome_coef_scalar * rng.standard_normal(support_size)
    oc_norm = np.linalg.norm(raw_oc)
    outcome_coef = raw_oc / (oc_norm if oc_norm > 0 else 1.0)

    # --- Noise samplers ---
    eta_sample = _make_eta_sampler(eta_distribution, rng)
    # oml_runner.setup_treatment_noise uses the legacy np.random global state.
    # Seed it deterministically so repeated calls with the same seed reproduce.
    np.random.seed(seed ^ 0xDEADBEEF & 0xFFFFFFFF)
    eta = eta_sample(n)

    eps = rng.normal(loc=0.0, scale=sigma_outcome, size=n)

    # --- PLR mechanism ---
    T = _apply_nonlinearity(X_sup @ treatment_coef, nonlinearity) + eta
    Y = treatment_effect * T + _apply_nonlinearity(X_sup @ outcome_coef, nonlinearity) + eps

    metadata = {
        "treatment_effect": treatment_effect,
        "treatment_coef": treatment_coef,
        "outcome_coef": outcome_coef,
        "eta_distribution": eta_distribution,
        "support_indices": support_indices,
        "nonlinearity": nonlinearity,
        "sigma_outcome": sigma_outcome,
        "seed": seed,
    }

    return X, T, Y, metadata
