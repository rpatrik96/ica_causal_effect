"""Smoke tests for the r17 TMLR critical-eval reruns (E1, E3, E4).

Covers the new flags (ICA readouts, explicit oracle switch, robust baseline,
extra covariates), per-seed estimate storage, the r17 round YAMLs and
``autoresearch/analyze_e1.py``. joblib runs with the sequential backend so the
tests stay in-process and fast.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from joblib import parallel_backend

import eta_noise_ablation
import monte_carlo_single_instance
from ablation_utils import (
    ICA_READOUTS,
    extract_treatment_estimates,
    is_extended_mode,
    parse_ica_readouts,
    run_parallel_experiments,
    run_single_experiment,
)
from autoresearch import analyze_e1
from cluster.sweep_runner import expand_grid, load_spec
from eta_ablation_experiments import run_coefficient_ablation_experiments

REPO = Path(__file__).resolve().parents[1]
ROUNDS = REPO / "autoresearch" / "rounds"
EXPECTED_JOBS = {
    "r17_e1_coef.yaml": 16,
    "r17_e1_heatmaps.yaml": 8,
    "r17_e3_reconcile.yaml": 8,
    "r17_e3_reconcile_ablation.yaml": 8,
    "r17_e4_variance.yaml": 1,
}
NAMED_METHODS = {"oml", "homl_known", "homl_estimated", "homl_estimated_split", "ols", "matching"}
READOUT_ARGS = ["--ica_readouts", "munkres_oracle", "eps_row", "--robust_baseline"]


@pytest.fixture(autouse=True)
def _sequential_joblib():
    with parallel_backend("sequential"):
        yield


def _single_inputs(seed=0, n=300, d=4):
    rng = np.random.default_rng(seed)
    support = np.arange(d)
    coef = np.zeros(d)
    coef[0] = 1.0
    return dict(
        x=rng.laplace(size=(n, d)),
        eta=rng.choice([-1.0, 2.0], size=n, p=[2 / 3, 1 / 3]),
        epsilon=rng.uniform(-np.sqrt(3), np.sqrt(3), size=n),
        treatment_effect=1.0,
        treatment_support=support,
        treatment_coef=coef,
        outcome_support=support,
        outcome_coef=coef.copy(),
        eta_second_moment=2.0,
        eta_third_moment=2.0,
        lambda_reg=0.05,
    )


def _load_list(output_dir: Path) -> list:
    files = sorted(output_dir.rglob("*.npy"))
    assert len(files) == 1, files
    data = np.load(files[0], allow_pickle=True)
    return data.item() if data.shape == () else list(data)


def test_parse_ica_readouts():
    assert parse_ica_readouts("munkres_oracle,eps_row") == ("munkres_oracle", "eps_row")
    assert parse_ica_readouts(["munkres_oracle", "eps_row"]) == ("munkres_oracle", "eps_row")
    assert set(ICA_READOUTS) == {"munkres_oracle", "eps_row"}
    with pytest.raises(ValueError):
        parse_ica_readouts(["munkres"])


def test_default_path_keeps_legacy_tuple():
    result = run_single_experiment(**_single_inputs())
    assert len(result) == 10
    assert not isinstance(result[6], dict)
    assert not is_extended_mode(("munkres_oracle",), False)


def test_single_non_default_readout_is_labelled():
    result = run_single_experiment(**_single_inputs(), ica_readouts=("eps_row",))
    assert isinstance(result[6], dict) and set(result[6]) == {"ica_eps_row"}
    assert set(result[7]["ica_eps_row"]) >= {"converged", "attempts", "mcc"}


def test_extended_mode_computes_all_readouts_and_huber_on_same_draw():
    inputs = _single_inputs()
    both = run_single_experiment(**inputs, ica_readouts=("munkres_oracle", "eps_row"), robust_baseline=True)
    legacy = run_single_experiment(**inputs)
    assert set(both[6]) == {"ica_munkres_oracle", "ica_eps_row", "huber"}
    # Same draw: the Munkres readout and the OML/HOML estimates match the default path.
    np.testing.assert_allclose(both[6]["ica_munkres_oracle"], legacy[6])
    np.testing.assert_allclose(both[:4], legacy[:4])
    assert extract_treatment_estimates([both])[0][4] == pytest.approx(float(legacy[6][0]))
    assert np.isfinite(both[6]["huber"]).all()


def test_run_parallel_redraws_data_per_seed():
    calls = []

    def x_sample(n, d):
        calls.append(1)
        return np.random.laplace(size=(n, d))

    np.random.seed(0)
    coef = np.array([1.0, 0.0, 0.0])
    results = run_parallel_experiments(
        n_experiments=3,
        x_sample=x_sample,
        eta_sample=lambda n: np.random.choice([-1.0, 2.0], size=n, p=[2 / 3, 1 / 3]),
        epsilon_sample=lambda n: np.random.uniform(-1, 1, size=n),
        n_samples=300,
        cov_dim_max=3,
        treatment_effect=1.0,
        treatment_support=np.arange(3),
        treatment_coef=coef,
        outcome_support=np.arange(3),
        outcome_coef=coef.copy(),
        eta_second_moment=2.0,
        eta_third_moment=2.0,
        lambda_reg=0.05,
        ica_readouts=("munkres_oracle", "eps_row"),
    )
    assert len(calls) == 3 and len(results) == 3
    assert len({round(float(r[8][0]), 12) for r in results}) == 3  # OLS differs per draw


def test_coefficient_ablation_cli_saves_per_seed_estimates(tmp_path):
    eta_noise_ablation.main(
        [
            "--coefficient_ablation",
            "--no_plot",
            "--n_samples",
            "300",
            "--n_experiments",
            "3",
            "--support_size",
            "3",
            "--treatment_coef_values",
            "1.56",
            "--outcome_coef_values",
            "-1.45",
            "--treatment_effect_values",
            "1.0",
            "--oracle_support",
            "false",
            "--extra_covariates",
            "2",
            "--output_dir",
            str(tmp_path),
            *READOUT_ARGS,
        ]
    )
    (res,) = _load_list(tmp_path)
    assert set(res["theta_hats"]) == NAMED_METHODS | {"ica_munkres_oracle", "ica_eps_row", "huber"}
    assert all(v.shape == (3,) for v in res["theta_hats"].values())
    assert res["c_ica"] == pytest.approx(1 + (-1.45 + 1.56) ** 2)
    assert res["c_b"] == pytest.approx(1 + 1.45**2 * 2.0)  # Var(gennorm(beta=1)) = 2
    assert res["oracle_support"] is False and res["cov_dim_max"] == 5
    assert set(res["failure_counts"]) == set(res["theta_hats"])
    assert set(res["convergence_counts"]) == {"ica_munkres_oracle", "ica_eps_row"}


def test_oracle_switch_is_inert_without_extra_covariates_and_active_with_them():
    kwargs = dict(
        treatment_coef_values=[1.56],
        outcome_coef_values=[-1.45],
        treatment_effect_values=[1.0],
        n_samples=300,
        n_experiments=2,
        support_size=3,
        ica_readouts=("munkres_oracle", "eps_row"),
    )

    def hats(oracle, extra):
        (res,) = run_coefficient_ablation_experiments(oracle_support=oracle, extra_covariates=extra, **kwargs)
        return res["theta_hats"]

    same_a, same_b = hats(True, 0), hats(False, 0)
    for key in same_a:
        np.testing.assert_allclose(same_a[key], same_b[key], rtol=1e-9, atol=1e-12)
    diff_a, diff_b = hats(True, 3), hats(False, 3)
    assert not np.allclose(diff_a["ols"], diff_b["ols"])


def test_filtered_heatmap_and_variance_cli_paths(tmp_path):
    common = ["--no_plot", "--n_experiments", "2", *READOUT_ARGS]
    eta_noise_ablation.main(
        [
            "--filtered_heatmap",
            "--heatmap_axis_mode",
            "d_vs_n",
            "--heatmap_sample_sizes",
            "300",
            "--heatmap_dimensions",
            "3",
            "--heatmap_outcome_coef",
            "0.05",
            "--output_dir",
            str(tmp_path / "hm"),
            *common,
        ]
    )
    (cell,) = _load_list(tmp_path / "hm")
    assert cell["c_ica"] == pytest.approx(1 + (0.05 + 0.5) ** 2)
    assert cell["theta_hats"]["ica_eps_row"].shape == (2,)

    eta_noise_ablation.main(
        [
            "--variance_ablation",
            "--n_samples",
            "300",
            "--support_size",
            "3",
            "--variance_beta_values",
            "1.0",
            "--variance_values",
            "1.0",
            "--output_dir",
            str(tmp_path / "var"),
            *common,
        ]
    )
    var = _load_list(tmp_path / "var")
    grid_cell = var["grid_results"][(1.0, 1.0)]
    assert grid_cell["theta_hats"]["ols"].shape == (2,)
    assert var["c_ica"] == pytest.approx(5.0)


def test_extended_flags_rejected_where_not_implemented(tmp_path):
    with pytest.raises(ValueError):
        eta_noise_ablation.main(["--output_dir", str(tmp_path), "--ica_readouts", "eps_row"])


@pytest.mark.parametrize("scalar, expected_nonzero", [("true", 1), ("false", 10)])
def test_monte_carlo_single_config_flags(tmp_path, scalar, expected_nonzero):
    with matplotlib.rc_context():
        rc = monte_carlo_single_instance.main(
            [
                "--single_config",
                "--no_plot",
                "--n_samples",
                "300",
                "--n_experiments",
                "2",
                "--beta",
                "1.0",
                "--scalar_coeffs",
                scalar,
                "--oracle_support",
                "false",
                "--output_dir",
                str(tmp_path),
                *READOUT_ARGS,
            ]
        )
    assert rc == 0
    (res,) = _load_list(tmp_path)
    assert res["oracle_support"] is False and res["beta"] == 1.0
    assert int(np.count_nonzero(res["a"])) == expected_nonzero
    if scalar == "true":
        assert res["c_ica"] == pytest.approx(1 + (1.56 - 1.45) ** 2)
    assert res["theta_hats"]["ica_munkres_oracle"].shape == (2,)


@pytest.mark.parametrize("name", sorted(EXPECTED_JOBS))
def test_round_yamls_generate_jobs(name):
    spec = load_spec(ROUNDS / name)
    assert spec["bid"] == 43
    jobs = expand_grid(spec["grid"])
    assert len(jobs) == EXPECTED_JOBS[name]
    for job in jobs:
        args = f"{spec['base_args']} {job} --output_dir /tmp/x".split()
        assert "," not in " ".join(args)
        if spec["script"] == "eta_noise_ablation.py":
            opts = eta_noise_ablation.build_parser().parse_args(args)
            assert parse_ica_readouts(opts.ica_readouts) == ("munkres_oracle", "eps_row")
            assert opts.n_experiments == 50
        else:
            source = (REPO / spec["script"]).read_text(encoding="utf-8")
            for flag in {a for a in args if a.startswith("--")}:
                assert re.search(rf'"{flag}"', source), flag


def _synthetic_record(theta, ica_err, ols_err, oracle, c_ica):
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(20)
    return {
        "pipeline": "coefficient_ablation",
        "oracle_support": oracle,
        "theta": theta,
        "a": 0.5,
        "b": 0.05,
        "c_ica": c_ica,
        "c_b": 1.005,
        "n_samples": 100,
        "support_size": 10,
        "cov_dim_max": 10,
        "theta_hats": {
            "ica_munkres_oracle": theta + ica_err * noise,
            "ica_eps_row": theta + ica_err * noise[::-1],
            "ols": theta + ols_err * noise,
            "homl_known": theta + ols_err * np.abs(noise),
        },
    }


def test_analyze_e1_on_synthetic_results(tmp_path):
    records = [
        _synthetic_record(1.0, 0.01, 0.1, True, 1.2),
        _synthetic_record(1.0, 0.2, 0.1, False, 7.0),
    ]
    np.save(tmp_path / "coefficient_ablation_results.npy", np.array(records, dtype=object), allow_pickle=True)
    assert analyze_e1.main([str(tmp_path), "--out", str(tmp_path / "out"), "--n_boot", "200"]) == 0
    with open(tmp_path / "out" / "e1_paired_per_config.csv", encoding="utf-8") as f:
        paired = list(csv.DictReader(f))
    row = next(
        r
        for r in paired
        if r["readout"] == "ica_munkres_oracle" and r["comparator"] == "ols" and r["oracle_support"] == "True"
    )
    assert float(row["rmse_diff"]) < 0 and float(row["ci_high"]) < 0
    assert int(row["n_r_closer"]) == 20 and float(row["sign_p"]) < 1e-4
    with open(tmp_path / "out" / "e1_win_rates.csv", encoding="utf-8") as f:
        wins = list(csv.DictReader(f))
    strata = {(r["stratifier"], r["stratum"], r["oracle_support"]) for r in wins}
    assert ("c_ica_bin", "<1.5", "True") in strata and ("c_ica_bin", ">=5", "False") in strata
    assert ("c_b_bin", "<1.5", "True") in strata
    summary = (tmp_path / "out" / "e1_summary.md").read_text(encoding="utf-8")
    assert "Win rates stratified by c_ica" in summary
