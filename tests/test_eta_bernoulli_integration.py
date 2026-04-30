"""Integration tests for the Bernoulli noise experiment in eta_noise_ablation.py.

Covers:
- CLI flag --bernoulli_ps dispatches correctly to argparse Namespace
- --bernoulli_only mode creates noise_ablation_results_bernoulli.npy
- --add_rademacher_baseline injects 'rademacher' into the distribution list
"""

import os
import tempfile

import numpy as np
import pytest

from eta_noise_ablation import build_parser, run

# ---------------------------------------------------------------------------
# Helper: build a minimal Namespace for run() without touching real output dirs
# ---------------------------------------------------------------------------


def _tiny_opts(tmp_dir: str, **overrides):
    """Return a Namespace with tiny config pointing at tmp_dir."""
    parser = build_parser()
    base_args = [
        "--n_experiments",
        "2",
        "--n_samples",
        "300",
        "--output_dir",
        tmp_dir,
    ]
    opts = parser.parse_args(base_args)
    for k, v in overrides.items():
        setattr(opts, k, v)
    return opts


# ---------------------------------------------------------------------------
# 1. --bernoulli_ps flag dispatches correctly
# ---------------------------------------------------------------------------


def test_bernoulli_ps_flag_dispatches_to_parse_distribution_spec():
    """Namespace produced by --bernoulli_ps 0.3 0.5 has bernoulli_ps == [0.3, 0.5]."""
    parser = build_parser()
    opts = parser.parse_args(["--bernoulli_ps", "0.3", "0.5"])
    assert opts.bernoulli_ps == pytest.approx([0.3, 0.5]), f"Expected [0.3, 0.5], got {opts.bernoulli_ps}"


# ---------------------------------------------------------------------------
# 2. --bernoulli_only mode creates the dedicated output file
# ---------------------------------------------------------------------------


def test_bernoulli_only_mode_creates_outputs():
    """run() with bernoulli_only creates noise_ablation_results_bernoulli.npy."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        opts = _tiny_opts(
            tmp_dir,
            bernoulli_only=True,
            bernoulli_ps=[0.3],
            add_rademacher_baseline=False,
            # ensure randomize_coeffs is True (bernoulli_only default)
            randomize_coeffs=False,
        )
        run(opts)

        expected = os.path.join(tmp_dir, "noise_ablation_results_bernoulli.npy")
        assert os.path.exists(expected), (
            f"Expected output file not found: {expected}\n" f"Directory contents: {os.listdir(tmp_dir)}"
        )

        # Sanity: loadable dict with at least the bernoulli:0.3 key
        data = np.load(expected, allow_pickle=True).item()
        assert "bernoulli:0.3" in data, f"Key 'bernoulli:0.3' missing from results: {list(data.keys())}"


# ---------------------------------------------------------------------------
# 3. --add_rademacher_baseline injects 'rademacher' into distribution list
# ---------------------------------------------------------------------------


def test_distribution_list_with_rademacher_baseline():
    """bernoulli_only + add_rademacher_baseline=True puts 'rademacher' in results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        opts = _tiny_opts(
            tmp_dir,
            bernoulli_only=True,
            bernoulli_ps=[0.3],
            add_rademacher_baseline=True,
            randomize_coeffs=False,
        )
        run(opts)

        expected = os.path.join(tmp_dir, "noise_ablation_results_bernoulli.npy")
        assert os.path.exists(expected)
        data = np.load(expected, allow_pickle=True).item()
        assert "rademacher" in data, f"'rademacher' key missing from results: {list(data.keys())}"
