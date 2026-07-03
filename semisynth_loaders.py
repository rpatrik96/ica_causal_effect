#!/usr/bin/env python3
"""Semi-synthetic covariate loaders + synthetic-on-real-X PLR recast (WS2).

Implements the WS2 design of ``docs/dataset-research/DATASETS.md``: keep *real*
covariates, discard any native treatment/outcome, pre-disentangle the correlated
covariates (PCA for dense, TruncatedSVD for sparse), then impose a partially
linear model ``T = m(X)+η``, ``Y = θT + g(X)+ε`` with a controllable ``η`` so
that ``θ`` is exact by construction and ICA is *in regime* while stressed on real
covariate structure.

Two real covariate sources, both fetchable via scikit-learn and cacheable to
shared storage (compute nodes have no network — the loader caches the processed
matrix under ``semisynth_data/`` so condor jobs read the cache):

- **housing**  — California Housing (20640×8), dense, strongly correlated
  (off-diagonal |corr| up to 0.93). PCA-whitening pre-disentangle. This is the
  housing strawman done *right*: the settled failure mode is raw features into
  FastICA (``docs/research-memory/housing-semisynthetic-deferred.md``); the
  pre-disentangle step is exactly its committed fix.
- **news20**  — 20 Newsgroups TF-IDF bag-of-words, sparse high-dim text
  (the ``News`` dataset's structure), TruncatedSVD pre-disentangle. The
  sparse-X contrast to housing's dense matrix.

Mirrors the conventions of ``realdata_loaders.py`` (module-level DATA_DIR cache,
``use_fixture_on_failure`` fallback to a schema-matched synthetic fixture).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.stats import gennorm

DATA_DIR = Path(__file__).resolve().parent / "semisynth_data"


# --------------------------------------------------------------------------- #
# Real covariate loaders (cache the processed matrix; fixture fallback).       #
# --------------------------------------------------------------------------- #
def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}_X.npy"


def _correlated_fixture(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Schema-matched synthetic fallback: a correlated Gaussian covariate block
    (so the pipeline runs offline; NOT a real-data substitute for grading)."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, max(2, d // 3)))
    mixing = rng.standard_normal((latent.shape[1], d))
    X = latent @ mixing + 0.3 * rng.standard_normal((n, d))
    return X.astype(np.float64)


def load_housing_covariates(data_dir=None, use_fixture_on_failure=True) -> np.ndarray:
    """California Housing covariates (n, 8), standardized. Cached to DATA_DIR."""
    base = Path(data_dir) if data_dir else DATA_DIR
    cache = base / "housing_X.npy"
    if cache.exists():
        return np.load(cache)
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.preprocessing import StandardScaler

        X = fetch_california_housing().data.astype(np.float64)
        X = StandardScaler().fit_transform(X)
        base.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
        return X
    except Exception as exc:  # noqa: BLE001  pylint: disable=broad-exception-caught
        if not use_fixture_on_failure:
            raise
        print(f"[semisynth] housing fetch failed ({exc}); using fixture")
        return _correlated_fixture(4096, 8, seed=1)


def load_news20_covariates(
    data_dir=None, use_fixture_on_failure=True, max_features=2000, categories=None
) -> np.ndarray:
    """20 Newsgroups TF-IDF covariates (n, max_features), sparse->dense. Cached."""
    base = Path(data_dir) if data_dir else DATA_DIR
    cache = base / "news20_X.npy"
    if cache.exists():
        return np.load(cache)
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer

        cats = categories or ["sci.med", "sci.space", "rec.autos", "comp.graphics"]
        ng = fetch_20newsgroups(
            subset="train", categories=cats, remove=("headers", "footers", "quotes")
        )
        Xs = TfidfVectorizer(max_features=max_features, min_df=5).fit_transform(ng.data)
        X = np.asarray(Xs.todense(), dtype=np.float64)
        base.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
        return X
    except Exception as exc:  # noqa: BLE001  pylint: disable=broad-exception-caught
        if not use_fixture_on_failure:
            raise
        print(f"[semisynth] news20 fetch failed ({exc}); using fixture")
        return _correlated_fixture(1800, 2000, seed=2)


def load_synthetic_covariates(data_dir=None, use_fixture_on_failure=True,
                              n_rows=60000, d=10) -> np.ndarray:
    """Large iid-Gaussian covariate matrix (n_rows, d), cached. The synthetic
    control for frontier-robustness checks: confirms an effect is intrinsic to the
    estimators, not a real-covariate artifact, and supplies unlimited real (non-
    bootstrapped) rows for the n>>d regime."""
    base = Path(data_dir) if data_dir else DATA_DIR
    cache = base / "synthetic_X.npy"
    if cache.exists():
        return np.load(cache)
    rng = np.random.default_rng(20260703)
    X = rng.standard_normal((n_rows, d)).astype(np.float64)
    base.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def load_synthetic_hd_covariates(data_dir=None, use_fixture_on_failure=True,
                                 n_rows=30000, d=400) -> np.ndarray:
    """High-dimensional iid-Gaussian covariates (30000, 400), cached. Lets the
    multi-feature selector eval reach the d >= n regime (predisentangle to up to
    ~300 components, subsample down to n=200 -> d/n up to ~1.5)."""
    base = Path(data_dir) if data_dir else DATA_DIR
    cache = base / "synthetic_hd_X.npy"
    if cache.exists():
        return np.load(cache)
    rng = np.random.default_rng(20260704)
    X = rng.standard_normal((n_rows, d)).astype(np.float64)
    base.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


LOADERS = {
    "housing": load_housing_covariates,
    "news20": load_news20_covariates,
    "synthetic": load_synthetic_covariates,
    "synthetic_hd": load_synthetic_hd_covariates,
}


def load_covariates(dataset: str, **kw) -> np.ndarray:
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset '{dataset}'; choices: {sorted(LOADERS)}")
    return LOADERS[dataset](**kw)


# --------------------------------------------------------------------------- #
# Pre-disentangle + PLR recast (dataset-agnostic).                             #
# --------------------------------------------------------------------------- #
def predisentangle(X, n_components=10, method="pca", whiten=True, random_state=0) -> np.ndarray:
    """Whiten/reduce correlated real covariates before they reach FastICA.

    ``method='pca'`` (dense) or ``'svd'`` (TruncatedSVD, for sparse BoW). Returns
    an ``(n, n_components)`` float matrix, standardized to unit column variance so
    the imposed PLR coefficients act on comparable scales. This is the committed
    fix for the raw-features-into-FastICA strawman.
    """
    n_components = int(min(n_components, X.shape[1], X.shape[0] - 1))
    if method == "svd":
        from sklearn.decomposition import TruncatedSVD

        Z = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA

        Z = PCA(n_components=n_components, whiten=whiten, random_state=random_state).fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'svd'")
    Z = Z - Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (Z / std).astype(np.float64)


def _gennorm_eta(beta, size, rng) -> np.ndarray:
    """Zero-mean gennorm(beta) noise standardized to unit variance (β=1 Laplace,
    β=2 Gaussian, β<2 heavy-tailed). ICA's identifiability lever is β≠2."""
    scale = 1.0 / float(np.sqrt(gennorm.var(beta)))
    eta = gennorm.rvs(beta, scale=scale, size=size, random_state=rng)
    return eta - eta.mean()


def _plr_mean(X, coef, quad_coef, nonlinear):
    """Conditional mean m(X)/g(X). Linear ``X@coef``, or (nonlinear) a
    sin + quadratic map mirroring ``nonlinear_dgp``'s
    ``g(X)=Σ sin(π X_j)·coef + 0.5·Σ X_j²·quad`` — a form linear OLS/OML with a
    linear first stage cannot represent."""
    if not nonlinear:
        return X @ coef
    out = np.sin(np.pi * X) @ coef
    if quad_coef is not None:
        out = out + 0.5 * (X**2) @ quad_coef
    return out


def impose_plr(
    X, theta, treatment_coef, outcome_coef, sigma_eps=1.0, eta_beta=1.0, seed=0,
    nonlinear=False, treatment_quad=None, outcome_quad=None, eps_beta=None,
):
    """Impose ``T = m(X) + η``, ``Y = θT + g(X) + ε`` on pre-disentangled ``X``.

    ``θ`` is exact by construction. ``η ~ gennorm(eta_beta)`` (unit variance).
    ``ε ~ N(0, sigma_eps²)`` by default, or ``sigma_eps·gennorm(eps_beta)`` when
    ``eps_beta`` is set (heavy-tailed outcome noise — a second non-Gaussian source
    for the ICA-edge probe). ``nonlinear=True`` makes m(X), g(X) nonlinear
    (carries the r04/r06 misspecification onto real X). Returns
    ``(T, Y, theta, eta_second_moment, eta_third_cumulant)``; the two η moments
    are what the HOML/robust estimators consume as "known" moments.
    """
    rng = np.random.default_rng(seed)
    eta = _gennorm_eta(eta_beta, X.shape[0], rng)
    if eps_beta is None:
        eps = rng.normal(0.0, sigma_eps, size=X.shape[0])
    else:
        eps = sigma_eps * _gennorm_eta(eps_beta, X.shape[0], rng)
    T = _plr_mean(X, treatment_coef, treatment_quad, nonlinear) + eta
    Y = theta * T + _plr_mean(X, outcome_coef, outcome_quad, nonlinear) + eps
    eta_second_moment = float(np.mean(eta**2))
    eta_third_cumulant = float(np.mean(eta**3))  # ~0 for symmetric gennorm
    return T, Y, float(theta), eta_second_moment, eta_third_cumulant


def make_coefficients(d, seed=0, scale=1.0):
    """Fixed (dataset-level) PLR coefficients on the ``d`` pre-disentangled
    components, drawn once so only the noise varies across MC experiments. Returns
    ``(m, g, m_quad, g_quad)``: linear treatment/outcome vectors plus the quadratic
    coefficients used only in the nonlinear recast."""
    rng = np.random.default_rng(seed)
    m = scale * rng.standard_normal(d) / np.sqrt(d)
    g = scale * rng.standard_normal(d) / np.sqrt(d)
    m_quad = scale * rng.standard_normal(d) / np.sqrt(d)
    g_quad = scale * rng.standard_normal(d) / np.sqrt(d)
    return (m.astype(np.float64), g.astype(np.float64),
            m_quad.astype(np.float64), g_quad.astype(np.float64))


if __name__ == "__main__":  # smoke: cache both datasets, print shapes
    for name in ("housing", "news20"):
        X = load_covariates(name)
        method = "svd" if name == "news20" else "pca"
        Z = predisentangle(X, n_components=10, method=method)
        m, g, mq, gq = make_coefficients(Z.shape[1], seed=0)
        T, Y, th, m2, m3 = impose_plr(Z[:500], 1.0, m, g, eta_beta=1.0, seed=0)
        print(f"{name}: X{X.shape} -> Z{Z.shape}  T{T.shape} Y{Y.shape} theta={th} "
              f"eta_var={m2:.3f} eta_m3={m3:.3f}")
