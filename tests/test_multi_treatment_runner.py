"""Tests for multi_treatment_runner.py — Fig 4 / Fig E.15 producer."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# The producer drives ica.generate_ica_data which depends on real PyTorch
# (torch.bernoulli, torch.randn, torch.manual_seed). The repo's conftest
# injects a minimal stub when torch is unavailable (Python 3.13 + macOS has
# no torch wheel). Skip the runner tests in that environment — same policy
# the existing test_ica.py implicitly follows (it fails environmentally and
# is excluded from the "must pass" set per CLAUDE.md).
try:
    import torch  # noqa: F401

    _HAS_REAL_TORCH = hasattr(torch, "bernoulli") and hasattr(torch, "manual_seed")
except ImportError:
    _HAS_REAL_TORCH = False

pytestmark = pytest.mark.skipif(
    not _HAS_REAL_TORCH,
    reason="Real torch (with bernoulli/manual_seed) is required; conftest stub is insufficient.",
)

from multi_treatment_runner import run_multi_treatment_experiment  # noqa: E402

TINY_KW = dict(
    sample_sizes=[200],
    n_treatments_grid=[1, 2],
    n_covariates_grid=[5],
    n_experiments=2,
    nonlinearity="identity",
    eta_distribution="discrete",
    include_baselines=True,
    seed=12143,
    n_jobs=1,
)


REQUIRED_LEGACY_KEYS = {
    "sample_sizes",
    "n_treatments",
    "n_covariates",
    "true_params",
    "treatment_effects",
    "treatment_effects_iv",
}

REQUIRED_NEW_KEYS = {
    "treatment_effects_ica_eps_row",
    "treatment_effects_ols",
    "treatment_effects_homl",
}


def _run_tiny(output_dir: str) -> dict:
    """Helper that runs ``run_multi_treatment_experiment`` against a tiny grid."""
    output_path = os.path.join(output_dir, "results_multi_treatment.npy")
    return run_multi_treatment_experiment(output_path=output_path, **TINY_KW)


def test_runner_produces_legacy_npy_schema() -> None:
    """The output dict carries every legacy + new key the consumer reads."""
    with tempfile.TemporaryDirectory() as tmp:
        results = _run_tiny(tmp)

    keys = set(results.keys())
    missing = (REQUIRED_LEGACY_KEYS | REQUIRED_NEW_KEYS) - keys
    assert not missing, f"Missing keys in results dict: {missing}"


def test_runner_baselines_finite() -> None:
    """OLS and HOML estimates are finite where ``m`` matches the configuration.

    NaN-padded entries (``m < m_max``) are excluded from the finiteness check.
    """
    with tempfile.TemporaryDirectory() as tmp:
        results = _run_tiny(tmp)

    n_treatments = results["n_treatments"]
    for cfg_idx, m in enumerate(n_treatments):
        ols = results["treatment_effects_ols"][cfg_idx]
        homl = results["treatment_effects_homl"][cfg_idx]

        # Active (non-padded) columns are the first m.
        ols_active = ols[:, :m]
        homl_active = homl[:, :m]

        # Allow occasional non-finite entries from a single failed rep, but
        # at least one rep should be finite for a tiny well-posed config.
        assert np.isfinite(ols_active).any(), f"OLS active block all-NaN at cfg {cfg_idx}"
        assert np.isfinite(homl_active).any(), f"HOML active block all-NaN at cfg {cfg_idx}"


def test_runner_consistent_across_m() -> None:
    """m=1 and m=2 configs return arrays of correct (n_experiments, m_max) shape."""
    with tempfile.TemporaryDirectory() as tmp:
        results = _run_tiny(tmp)

    m_max = max(TINY_KW["n_treatments_grid"])
    n_experiments = TINY_KW["n_experiments"]

    for cfg_idx, m in enumerate(results["n_treatments"]):
        for key in [
            "treatment_effects",
            "treatment_effects_iv",
            "treatment_effects_ica_eps_row",
            "treatment_effects_ols",
            "treatment_effects_homl",
        ]:
            arr = results[key][cfg_idx]
            assert arr.shape == (n_experiments, m_max), (
                f"{key}[{cfg_idx}] has shape {arr.shape}, expected " f"({n_experiments}, {m_max})"
            )

        # Padded columns (m..m_max) must be NaN.
        if m < m_max:
            for key in (
                "treatment_effects_ols",
                "treatment_effects_homl",
            ):
                pad = results[key][cfg_idx][:, m:]
                assert np.all(np.isnan(pad)), f"{key}[{cfg_idx}] padding not NaN"


def test_runner_npy_round_trip() -> None:
    """Saved .npy round-trips via ``np.load(..., allow_pickle=True).item()``."""
    with tempfile.TemporaryDirectory() as tmp:
        results_in_memory = _run_tiny(tmp)

        output_path = os.path.join(tmp, "results_multi_treatment.npy")
        assert os.path.exists(output_path)

        loaded = np.load(output_path, allow_pickle=True).item()

    assert set(loaded.keys()) == set(results_in_memory.keys())
    for key in REQUIRED_LEGACY_KEYS | REQUIRED_NEW_KEYS:
        assert key in loaded
