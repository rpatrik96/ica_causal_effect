from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
from sweep_runner import aggregate, expand_grid, generate_sub, load_spec, pending_jobs, write_tsv  # noqa: E402

SPEC = {
    "round": "r_test",
    "script": "monte_carlo_single_instance.py",
    "base_args": "--n_experiments 2",
    "grid": {"n_samples": [500, 1000], "covariate_pdf": ["gennorm"]},
}


def _write_spec(tmp_path, overrides=None):
    spec = dict(SPEC)
    if overrides:
        spec.update(overrides)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


def test_load_spec_defaults(tmp_path):
    spec = load_spec(_write_spec(tmp_path))
    assert spec["bid"] == 43
    assert spec["resources"]["max_time_s"] == 86400
    assert spec["output_flag"] == "--output_dir"


def test_load_spec_missing_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"round": "x", "script": "y.py"}))
    with pytest.raises(ValueError, match="grid"):
        load_spec(p)


def test_expand_grid_cartesian_deterministic():
    jobs = expand_grid({"b": [1, 2], "a": ["x"]})
    assert jobs == ["--a x --b 1", "--a x --b 2"]  # keys sorted, product ordered


def test_expand_grid_bool_and_list():
    jobs = expand_grid({"flag": [True, False], "dims": [[5, 10]]})
    assert jobs == ["--dims 5 10 --flag", "--dims 5 10"]


def test_expand_grid_rejects_comma():
    with pytest.raises(ValueError, match="comma"):
        expand_grid({"a": ["1,2"]})


def test_pending_jobs_skips_done(tmp_path):
    jobs = ["--a 1", "--a 2"]
    done_dir = tmp_path / "job_0000"
    done_dir.mkdir(parents=True)
    (done_dir / "DONE").touch()
    assert pending_jobs(jobs, tmp_path) == [(1, "--a 2")]
    assert len(pending_jobs(jobs, tmp_path, force=True)) == 2


def test_generate_sub_maxtime_equal(tmp_path):
    spec = load_spec(_write_spec(tmp_path))
    sub = generate_sub(spec, Path("/proj"), tmp_path, tmp_path / "pending.txt")
    assert "MaxTime = 86400" in sub
    assert "+MaxRuntime = 86400" in sub
    assert "queue jobidx,jobscript,joboutdir,jobargs from" in sub
    assert "request_cpus = 4" in sub


def test_aggregate_scalars_and_shapes(tmp_path):
    job = tmp_path / "job_0000"
    job.mkdir(parents=True)
    (job / "DONE").touch()
    np.save(job / "res.npy", {"rmse": 0.5, "arr": np.zeros((3, 2))})
    rows = aggregate(tmp_path)
    assert rows[0]["done"] is True
    assert rows[0]["res.rmse"] == 0.5
    assert rows[0]["res.arr"] == "shape=(3, 2)"


def test_write_tsv_union_of_keys(tmp_path):
    out = tmp_path / "s.tsv"
    write_tsv([{"a": 1}, {"b": 2}], out)
    header = out.read_text().splitlines()[0].split("\t")
    assert set(header) == {"a", "b"}


def test_dry_mode_end_to_end(tmp_path, capsys, monkeypatch):
    import sweep_runner

    monkeypatch.setattr(
        sweep_runner.subprocess,
        "run",
        lambda *a, **k: pytest.fail("dry mode must not call subprocess"),
    )
    spec_path = _write_spec(tmp_path)
    monkeypatch.chdir(tmp_path)
    sweep_runner.main([str(spec_path), "--mode", "dry"])
    out = capsys.readouterr().out
    assert "2 job(s)" in out and "DRY" in out
