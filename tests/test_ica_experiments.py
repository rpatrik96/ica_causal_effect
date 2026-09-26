"""Smoke tests for the restored ICA-only experiment drivers."""

from pathlib import Path

import pytest

from ica_experiments import main


@pytest.mark.parametrize(
    ("experiment", "results_file", "figure_file"),
    [
        ("gennorm", "results_main_gennorm.npy", "ica_mse_vs_beta_n100.svg"),
        (
            "nonlinear_noise_split",
            "results_main_gennorm_nonlinear_noise_split.npy",
            "ica_mse_vs_noise_split_nonlinear_n100.svg",
        ),
        ("loc_scale", "results_main_loc_scale.npy", "ica_mse_heatmap_loc_scale_n100.svg"),
    ],
)
def test_quick_experiment_smoke(tmp_path, monkeypatch, experiment, results_file, figure_file):
    """Each quick driver writes both its cached data and expected figure."""
    run_dir = tmp_path / experiment
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)

    main([experiment, "--quick", "--n_seeds", "1"])

    assert Path(results_file).is_file()
    assert (Path("figures/ica") / figure_file).is_file()
