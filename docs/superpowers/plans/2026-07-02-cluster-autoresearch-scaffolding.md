# Cluster Autoresearch Scaffolding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship everything the cluster Claude Code instance needs to run the autoresearch campaign: portable context, research program, deterministic sweep runner, bootstrap docs, and the vetted semi-synthetic dataset shortlist.

**Architecture:** Phase 1 scaffolding (Tasks S1–S8) + Phase 0 dataset deep-research (Task R1) per the approved spec `docs/superpowers/specs/2026-07-02-cluster-autoresearch-design.md`. Tasks S1–S7 and R1 are independent and run as parallel agents writing disjoint paths; agents do NOT run `git add/commit` (index contention) — the orchestrator commits per task after review. Task S8 integrates, verifies, and pushes.

**Tech Stack:** Python ≥3.9 (`from __future__ import annotations`; cluster python may be old), PyYAML, NumPy (<2), pytest, HTCondor (`condor_submit_bid`), bash.

## Global Constraints

- `numpy<2` everywhere (torch/sklearn compatibility — see `docs/research-memory/`).
- No secrets, API keys, or machine-specific absolute paths in any committed file (portable-context requirement). Exception: `docs/CLUSTER_BOOTSTRAP.md` may name the cluster path `/is/cluster/fast/preizinger/ica_causal_effect` — it is documentation for that specific host.
- Default condor bid **43**; `MaxTime` and `+MaxRuntime` must always be equal.
- Concurrent-job cap ~**200** enforced in code (refuse larger submissions without `--yes-many`).
- Never force-push; commits use conventional prefixes (`feat:`, `docs:`, `chore:`).
- Repo remote: `https://github.com/rpatrik96/ica_causal_effect` (NOT `double_orthogonal_ml` — stale name in old docs).
- Agents write files only; the orchestrator commits. Every commit message ends with the Claude co-author trailer.

---

### Task S1: Portable permission allowlist

**Files:**
- Create: `.claude/settings.json`

**Interfaces:**
- Produces: permission allowlist consumed by the cluster Claude Code instance at session start.

- [ ] **Step 1: Write the file** (adapted from `causal-ica@4a889013`, plus `uv`, read utilities, and `rsync`):

```json
{
  "permissions": {
    "allow": [
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git pull:*)",
      "Bash(git fetch:*)",
      "Bash(git checkout:*)",
      "Bash(git merge:*)",
      "Bash(git stash:*)",
      "Bash(git remote get-url:*)",
      "Bash(gh pr:*)",
      "Bash(pre-commit:*)",
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(pytest:*)",
      "Bash(pip show:*)",
      "Bash(pip install:*)",
      "Bash(uv:*)",
      "Bash(condor_submit_bid:*)",
      "Bash(condor_submit:*)",
      "Bash(condor_q:*)",
      "Bash(condor_status:*)",
      "Bash(condor_rm:*)",
      "Bash(condor_wait:*)",
      "Bash(condor_history:*)",
      "Bash(find:*)",
      "Bash(grep:*)",
      "Bash(ls:*)",
      "Bash(tree:*)",
      "Bash(xargs:*)",
      "Bash(awk:*)",
      "Bash(wc:*)",
      "Bash(cat:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(mkdir:*)",
      "Bash(rsync:*)",
      "Bash(tmux:*)",
      "WebSearch",
      "WebFetch(domain:github.com)",
      "WebFetch(domain:arxiv.org)"
    ],
    "deny": [],
    "ask": []
  }
}
```

Deliberately absent (deny-by-prompt): `git reset` (in causal-ica's list, but this repo's global rules hard-deny destructive git), `rm`, `curl`.

- [ ] **Step 2: Validate**

Run: `python -c "import json; json.load(open('.claude/settings.json'))" && echo OK`
Expected: `OK`

- [ ] **Step 3 (orchestrator): Commit** — `chore: portable permission allowlist for cluster Claude Code`

---

### Task S2: Research-memory snapshot

**Files:**
- Create: `docs/research-memory/README.md`, `docs/research-memory/MEMORY.md`, `docs/research-memory/ihdp-ica-binary-finding.md`, `docs/research-memory/housing-semisynthetic-deferred.md`, `docs/research-memory/venv-setup.md`

**Interfaces:**
- Consumes: local auto-memory at `/Users/patrik.reizinger/.claude/projects/-Users-patrik-reizinger-Documents-GitHub-ica-causal-effect/memory/` (read each file verbatim).
- Produces: `docs/research-memory/` read by the cluster instance every round (referenced from `PROGRAM.md` and `CLAUDE.md`).

- [ ] **Step 1: Copy the three memory files + index verbatim** from the local memory dir (paths above). Keep frontmatter; add no edits beyond stripping nothing — verbatim snapshot.

- [ ] **Step 2: Write `README.md`** explaining the pattern (mirrors `causal-ica/docs/research-memory/README.md`):

```markdown
# Research memory snapshot

Snapshot of the local Claude Code auto-memory for this project, committed so
accumulated findings travel with the code to the cluster. The cluster
instance MUST read `MEMORY.md` (the index) at the start of every round and
the linked entries before designing experiments that touch their topics.

Source of truth: the local machine's
`~/.claude/projects/-Users-patrik-reizinger-Documents-GitHub-ica-causal-effect/memory/`.
Re-snapshot when local memory gains campaign-relevant entries. Cluster-side
findings do NOT go here — they go in `autoresearch/rounds/` findings docs.

Key settled results (do not re-derive):
- IHDP-100/Jobs (binary treatment): OLS 0.60 / OML 0.56 RMSE win, ICA 3.99 —
  binary T violates ICA's continuous non-Gaussian noise model. Scope, not bug.
- California Housing raw features into FastICA = method-misuse strawman;
  only the pre-disentangled design (whiten/ICA on X first) is admissible.
- Env: `pip install -r requirements.txt "numpy<2"` — numpy 2.x breaks torch.
```

- [ ] **Step 3: Verify** — `ls docs/research-memory/` shows 5 files; `grep -l "binary" docs/research-memory/*.md` finds the IHDP entry.

- [ ] **Step 4 (orchestrator): Commit** — `docs: snapshot research memory for cluster Claude Code`

---

### Task S3: Reviews + meeting-notes snapshot

**Files:**
- Create: `docs/reviews/uai2026_reviews_summary.md`, `docs/reviews/uai2026_reviews_with_responses.md`, `docs/reviews/meeting_tmlr_notes.md`
- Delete: `Meeting_tmlr.md` (root, currently untracked — content moves into `docs/reviews/meeting_tmlr_notes.md`)

**Interfaces:**
- Consumes: `~/Documents/GitHub/overlap-ica/reviews/uai2026_reviews_summary.md` and `uai2026_reviews_with_responses.md`; root `Meeting_tmlr.md`.
- Produces: `docs/reviews/` referenced by `PROGRAM.md`'s traceability table.

- [ ] **Step 1: Copy files**

```bash
mkdir -p docs/reviews
cp ~/Documents/GitHub/overlap-ica/reviews/uai2026_reviews_summary.md docs/reviews/
cp ~/Documents/GitHub/overlap-ica/reviews/uai2026_reviews_with_responses.md docs/reviews/
cp Meeting_tmlr.md docs/reviews/meeting_tmlr_notes.md
rm Meeting_tmlr.md
```

- [ ] **Step 2: Prepend provenance header** to `meeting_tmlr_notes.md` (one line): `> Notes from the TMLR-planning meeting (2026-06/07); moved from repo root.` Keep the rest verbatim.

- [ ] **Step 3: Verify** — all three files exist, root `Meeting_tmlr.md` gone.

- [ ] **Step 4 (orchestrator): Commit** — `docs: snapshot UAI2026 reviews + TMLR meeting notes for cluster context`

---

### Task S4: Research program + log

**Files:**
- Create: `autoresearch/PROGRAM.md`, `autoresearch/RESEARCH_LOG.md`, `autoresearch/rounds/.gitkeep`, `autoresearch/results/.gitkeep`

**Interfaces:**
- Consumes: spec §Workstreams/§Guardrails/§Round protocol; `docs/reviews/uai2026_reviews_summary.md`; `docs/research-memory/`.
- Produces: `autoresearch/PROGRAM.md` — the single source of truth the cluster instance re-reads every round; grid-spec YAML schema shared with Task S5 (`round`, `script`, `base_args`, `output_flag`, `grid`, `resources{cpus,memory_gb,disk_gb,max_time_s}`, `bid`).

- [ ] **Step 1: Write `autoresearch/PROGRAM.md`** with exactly these sections (compose prose from the spec — no invented facts; all numbers below are settled results from `docs/research-memory/`):

1. **Mission** — produce the experimental evidence for the TMLR resubmission of arXiv:2507.16467, addressing the UAI 2026 reviews (`docs/reviews/uai2026_reviews_summary.md`). Rounds continue until stopping criteria, no calendar deadline.
2. **Traceability table** — columns: reviewer concern → workstream → deliverable. Rows: OLS baseline everywhere (NeBn) → WS1 → regime-map figs + "OLS breaks when…" claim; real/semi-synthetic data (NbV6, iuAn) → WS2 → benchmark table ≥2 datasets; diverse/harder experiments (iuAn, LRoS) → WS3 → high-dim + ML-nuisance comparisons; sensitivity to violations (iuAn, NeBn, NbV6) → WS3 → sensitivity curves; estimator selection Pareto (meeting/Rahul) → WS4; projection pursuit (meeting/Lester) → WS4.
3. **Round protocol** — the 7 numbered steps from the spec verbatim (pull & re-read; hypothesis+grid logged BEFORE running; tiny local sanity run; submit via `cluster/sweep_runner.py`; write up previous round while jobs run; grade evidence honestly incl. failures; commit+push on `autoresearch/cluster-rounds`; propose next round).
4. **Workstreams WS1–WS4** — each with Hypothesis, Knobs/grid dimensions, Success criteria, Stopping criteria, and an initial `- [ ]` task queue. Copy the content from spec §Workstreams; make WS1's initial queue concrete: (a) smoke round `r01_smoke` (tiny `monte_carlo_single_instance.py` grid, 4–8 jobs); (b) coarse regime grid over η gennorm β ∈ {0.5,1,2,3,4} × n ∈ {500,1k,2k,5k,10k} × d ∈ {5,10,20,50} with all five estimators; (c) adaptive refinement near boundaries; (d) OLS-breakdown hunt (nonlinear g(X), d≳n).
5. **Guardrails** — spec §Guardrails verbatim (binary IHDP/Jobs settled; no raw-feature housing; numpy<2; per-round result namespacing; MaxTime==+MaxRuntime; no login-node compute; ≤200 jobs; bid 43; never force-push; report failures faithfully).
6. **File map** — where things live: `cluster/sweep_runner.py` (+ YAML schema example), `autoresearch/results/round_<tag>/` (aggregates committed, raw `.npy` not), `autoresearch/rounds/findings_roundNN.md`, `RESEARCH_LOG.md`, `docs/research-memory/`, `docs/dataset-research/DATASETS.md`, `docs/CLUSTER_BOOTSTRAP.md`.
7. **Findings-doc template** — heading skeleton: Hypothesis / Grid / Jobs (submitted, completed, failed — with reasons) / Results (table + figures) / Evidence grade (confirmed / suggestive / negative / broken) / Implications for the paper / Proposed next round.

- [ ] **Step 2: Write `autoresearch/RESEARCH_LOG.md`** seed:

```markdown
# Research log — cluster autoresearch campaign

One entry per round, newest first. Format:
`## Round NN (date) — <one-line hypothesis>` then Status / Outcome / Link to findings doc.

## Round 00 (2026-07-02) — scaffolding
Status: complete (local). Scaffolding + program committed; campaign not yet started.
Next: bootstrap cluster session (docs/CLUSTER_BOOTSTRAP.md), then WS1 r01_smoke.
```

- [ ] **Step 3: Verify** — `PROGRAM.md` contains all 7 sections, the YAML schema example, and no "TBD".

- [ ] **Step 4 (orchestrator): Commit** — `feat: autoresearch program, log, and round scaffolding`

---

### Task S5: Deterministic sweep runner (TDD)

**Files:**
- Create: `cluster/sweep_runner.py`, `cluster/run_autoresearch_job.sh`, `tests/test_sweep_runner.py`
- Modify: `requirements.txt` (add `pyyaml`)

**Interfaces:**
- Consumes: grid-spec YAML schema (see Task S4).
- Produces:
  - CLI: `python cluster/sweep_runner.py <spec.yaml> [--mode condor|dry|local] [--force] [--limit N] [--aggregate-only] [--yes-many]`
  - Functions (used by tests): `load_spec(path) -> dict`, `expand_grid(grid: dict) -> list[str]`, `pending_jobs(jobs, results_root, force=False) -> list[tuple[int, str]]`, `generate_sub(spec, project_dir, results_root, pending_file) -> str`, `aggregate(results_root) -> list[dict]`, `write_tsv(rows, path)`.
  - On-disk contract: per-job dir `autoresearch/results/<round>/job_<idx:04d>/` with `DONE` marker on success; `pending.txt` (comma-separated: `idx,script,outdir,args`); generated `sweep.sub`; `jobs_manifest.tsv`; `summary.tsv`.

- [ ] **Step 1: Write the failing tests** — `tests/test_sweep_runner.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
from sweep_runner import (  # noqa: E402
    aggregate,
    expand_grid,
    generate_sub,
    load_spec,
    pending_jobs,
    write_tsv,
)

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
        sweep_runner.subprocess, "run",
        lambda *a, **k: pytest.fail("dry mode must not call subprocess"),
    )
    spec_path = _write_spec(tmp_path)
    monkeypatch.chdir(tmp_path)
    sweep_runner.main([str(spec_path), "--mode", "dry"])
    out = capsys.readouterr().out
    assert "2 job(s)" in out and "DRY" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_sweep_runner.py -q` (or `python -m pytest` if on PATH)
Expected: collection error `ModuleNotFoundError: No module named 'sweep_runner'`

- [ ] **Step 3: Write `cluster/sweep_runner.py`**

```python
#!/usr/bin/env python3
"""Deterministic HTCondor sweep runner for the autoresearch campaign.

Expands a YAML grid spec into HTCondor jobs, submits via condor_submit_bid,
waits, and aggregates per-job .npy results into TSV tables.

No LLM calls, no git operations, no adaptive logic — judgment belongs to the
orchestrating Claude instance (see autoresearch/PROGRAM.md).

Usage:
    python cluster/sweep_runner.py grid.yaml                # submit + wait + aggregate
    python cluster/sweep_runner.py grid.yaml --mode dry     # expand + print only
    python cluster/sweep_runner.py grid.yaml --mode local --limit 1  # sanity run
    python cluster/sweep_runner.py grid.yaml --aggregate-only        # re-aggregate
"""
from __future__ import annotations

import argparse
import itertools
import re
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import yaml

REQUIRED_KEYS = {"round", "script", "grid"}
DEFAULT_RESOURCES = {"cpus": 4, "memory_gb": 16, "disk_gb": 10, "max_time_s": 86400}
DEFAULT_BID = 43
MAX_JOBS_WITHOUT_CONFIRM = 200

SUB_TEMPLATE = """\
# Auto-generated by sweep_runner.py for round {round} — regenerate, don't edit.
universe = vanilla
executable = cluster/run_autoresearch_job.sh
initialdir = {project_dir}
getenv = True

request_cpus = {cpus}
request_memory = {memory_gb} GB
request_disk = {disk_gb} GB

MaxTime = {max_time_s}
+MaxRuntime = {max_time_s}
periodic_remove = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))

log = {condor_dir}/sweep.log
output = {condor_dir}/job_$(jobidx).out
error = {condor_dir}/job_$(jobidx).err

arguments = "$(jobidx) $(jobscript) $(joboutdir) $(jobargs)"
queue jobidx,jobscript,joboutdir,jobargs from {pending_file}
"""


def load_spec(path: str | Path) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f)
    missing = REQUIRED_KEYS - set(spec)
    if missing:
        raise ValueError(f"grid spec missing required keys: {sorted(missing)}")
    if not isinstance(spec["grid"], dict) or not spec["grid"]:
        raise ValueError("grid must be a non-empty mapping of flag -> list of values")
    spec.setdefault("base_args", "")
    spec.setdefault("output_flag", "--output_dir")
    spec.setdefault("bid", DEFAULT_BID)
    resources = dict(DEFAULT_RESOURCES)
    resources.update(spec.get("resources", {}))
    spec["resources"] = resources
    return spec


def expand_grid(grid: dict) -> list[str]:
    """Cartesian product of grid values -> arg strings, deterministic order."""
    keys = sorted(grid)
    jobs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        parts = []
        for key, value in zip(keys, combo):
            flag = key if key.startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    parts.append(flag)
            elif isinstance(value, (list, tuple)):
                parts.append(f"{flag} {' '.join(str(v) for v in value)}")
            else:
                parts.append(f"{flag} {value}")
        args = " ".join(parts)
        if "," in args:
            raise ValueError(f"comma not allowed in args (breaks condor queue-from): {args}")
        jobs.append(args)
    return jobs


def pending_jobs(jobs: list[str], results_root: Path, force: bool = False) -> list[tuple[int, str]]:
    out = []
    for idx, args in enumerate(jobs):
        if not force and (results_root / f"job_{idx:04d}" / "DONE").exists():
            continue
        out.append((idx, args))
    return out


def generate_sub(spec: dict, project_dir: Path, results_root: Path, pending_file: Path) -> str:
    condor_dir = results_root / "condor"
    return SUB_TEMPLATE.format(
        round=spec["round"],
        project_dir=project_dir,
        condor_dir=condor_dir,
        pending_file=pending_file,
        **spec["resources"],
    )


def aggregate(results_root: Path) -> list[dict]:
    rows = []
    for job_dir in sorted(results_root.glob("job_*")):
        row: dict = {"job": job_dir.name, "done": (job_dir / "DONE").exists()}
        for npy in sorted(job_dir.rglob("*.npy")):
            stem = npy.stem
            try:
                # allow_pickle: these .npy files are dict payloads written by our
                # own experiment scripts on our own cluster (repo result format),
                # never third-party input.
                data = np.load(npy, allow_pickle=True)
            except Exception as exc:  # noqa: BLE001 — record, don't crash the sweep
                row[f"{stem}:load_error"] = str(exc)
                continue
            payload = data.item() if getattr(data, "shape", None) == () else None
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if np.isscalar(value) or (hasattr(value, "shape") and value.shape == ()):
                        try:
                            row[f"{stem}.{key}"] = float(value)
                        except (TypeError, ValueError):
                            row[f"{stem}.{key}"] = str(value)
                    else:
                        row[f"{stem}.{key}"] = f"shape={getattr(value, 'shape', type(value).__name__)}"
            else:
                row[stem] = f"shape={getattr(data, 'shape', '?')}"
        rows.append(row)
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")


def _build_lines(spec: dict, pend: list[tuple[int, str]], results_root: Path) -> list[str]:
    lines = []
    for idx, args in pend:
        outdir = results_root / f"job_{idx:04d}"
        full_args = " ".join(x for x in [spec["base_args"], args] if x)
        if spec["output_flag"]:
            full_args = f"{full_args} {spec['output_flag']} {outdir}"
        lines.append(f"{idx:04d},{spec['script']},{outdir},{full_args}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec", help="YAML grid spec (see autoresearch/PROGRAM.md)")
    ap.add_argument("--mode", choices=["condor", "dry", "local"], default="condor")
    ap.add_argument("--force", action="store_true", help="re-run jobs with DONE markers")
    ap.add_argument("--limit", type=int, default=None, help="only first N pending jobs")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--yes-many", action="store_true", help=f"allow >{MAX_JOBS_WITHOUT_CONFIRM} jobs")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    project_dir = Path.cwd()
    results_root = project_dir / "autoresearch" / "results" / spec["round"]

    if args.aggregate_only:
        rows = aggregate(results_root)
        write_tsv(rows, results_root / "summary.tsv")
        n_done = sum(bool(r["done"]) for r in rows)
        print(f"aggregated {len(rows)} job(s), {n_done} DONE -> {results_root / 'summary.tsv'}")
        return 0

    jobs = expand_grid(spec["grid"])
    pend = pending_jobs(jobs, results_root, force=args.force)
    if args.limit is not None:
        pend = pend[: args.limit]
    print(f"{len(jobs)} job(s) in grid, {len(pend)} pending [{args.mode.upper()}]")
    if not pend:
        print("nothing to do (all DONE; use --force to re-run)")
        return 0
    if len(pend) > MAX_JOBS_WITHOUT_CONFIRM and not args.yes_many:
        print(f"refusing to submit {len(pend)} > {MAX_JOBS_WITHOUT_CONFIRM} jobs without --yes-many")
        return 2

    lines = _build_lines(spec, pend, results_root)
    if args.mode == "dry":
        for line in lines:
            print(f"  {line}")
        return 0

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "condor").mkdir(exist_ok=True)
    manifest_rows = [{"idx": i, "args": a, "status": "submitted"} for i, a in pend]

    if args.mode == "local":
        for line in lines:
            idx, script, outdir, job_args = line.split(",", 3)
            Path(outdir).mkdir(parents=True, exist_ok=True)
            print(f"[local] job {idx}: python {script} {job_args}")
            result = subprocess.run(  # nosec B603
                [sys.executable, script] + job_args.split(), cwd=project_dir
            )
            status = "done" if result.returncode == 0 else f"failed({result.returncode})"
            if result.returncode == 0:
                (Path(outdir) / "DONE").touch()
            manifest_rows[[r["idx"] for r in manifest_rows].index(int(idx))]["status"] = status
    else:  # condor
        pending_file = results_root / "pending.txt"
        pending_file.write_text("\n".join(lines) + "\n")
        sub_file = results_root / "sweep.sub"
        sub_file.write_text(generate_sub(spec, project_dir, results_root, pending_file))
        submit = subprocess.run(  # nosec B603 B607
            ["condor_submit_bid", str(spec["bid"]), str(sub_file)],
            capture_output=True, text=True, check=True,
        )
        print(submit.stdout.strip())
        match = re.search(r"cluster (\d+)", submit.stdout)
        cluster_id = match.group(1) if match else None
        log_file = results_root / "condor" / "sweep.log"
        timeout = spec["resources"]["max_time_s"] + 3600
        wait_cmd = ["condor_wait", "-wait", str(timeout), str(log_file)]
        if cluster_id:
            wait_cmd.append(cluster_id)
        print(f"waiting on {log_file} (timeout {timeout}s) ...")
        subprocess.run(wait_cmd, check=False)  # nosec B603 B607
        for row in manifest_rows:
            done = (results_root / f"job_{row['idx']:04d}" / "DONE").exists()
            row["status"] = "done" if done else "failed-or-held"

    write_tsv(manifest_rows, results_root / "jobs_manifest.tsv")
    rows = aggregate(results_root)
    write_tsv(rows, results_root / "summary.tsv")
    n_done = sum(bool(r["done"]) for r in rows)
    n_fail = len(manifest_rows) - sum(r["status"] == "done" for r in manifest_rows)
    print(f"summary: {n_done}/{len(rows)} job dirs DONE, {n_fail} failed/held")
    print(f"-> {results_root / 'summary.tsv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `cluster/run_autoresearch_job.sh`** (executable):

```bash
#!/bin/bash
# HTCondor job wrapper for sweep_runner.py.
# Usage: run_autoresearch_job.sh <idx> <script> <outdir> <args...>
# Resolves the project dir from its own location — no hardcoded cluster paths.
set -uo pipefail

IDX="$1"; SCRIPT="$2"; OUTDIR="$3"; shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

mkdir -p "${OUTDIR}"
echo "[job ${IDX}] python ${SCRIPT} $*"
python "${SCRIPT}" "$@"
STATUS=$?

if [ ${STATUS} -eq 0 ]; then
    touch "${OUTDIR}/DONE"
fi
exit ${STATUS}
```

Run: `chmod +x cluster/run_autoresearch_job.sh`

- [ ] **Step 5: Add `pyyaml` to `requirements.txt`** (append line `pyyaml`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_sweep_runner.py -q`
Expected: `10 passed`

- [ ] **Step 7: Lint** — `pre-commit run --files cluster/sweep_runner.py tests/test_sweep_runner.py` (repo has `.pre-commit-config.yaml`; fix silently and retry until green).

- [ ] **Step 8 (orchestrator): Commit** — `feat: deterministic HTCondor sweep runner for autoresearch rounds`

---

### Task S6: Cluster bootstrap doc

**Files:**
- Create: `docs/CLUSTER_BOOTSTRAP.md`

**Interfaces:**
- Consumes: sweep-runner CLI (Task S5), `PROGRAM.md` (Task S4).
- Produces: the exact login-node procedure; the first-prompt text the user pastes into `claude`.

- [ ] **Step 1: Write the doc** with exactly these sections:

1. **One-time setup** (login node):

```bash
cd /is/cluster/fast/preizinger
git clone https://github.com/rpatrik96/ica_causal_effect.git
cd ica_causal_effect
python3 -m venv .venv            # use `uv venv` if uv is available
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt "numpy<2"
python -m pytest tests/test_sweep_runner.py -q   # smoke: 10 passed
mkdir -p ~/jobs
chmod +x cluster/run_autoresearch_job.sh cluster/run_experiment.sh
```

2. **Claude Code session** — `tmux new -s autoresearch`, `cd /is/cluster/fast/preizinger/ica_causal_effect`, `claude` (auth reuses the subscription login already working for causal-ica on this host; verify with a trivial prompt). Detach `Ctrl+B d`, resume `tmux attach -t autoresearch`; after a session dies, `claude --continue`.
3. **First prompt** (verbatim, in a fenced block):

```text
Read autoresearch/PROGRAM.md, autoresearch/RESEARCH_LOG.md, and
docs/research-memory/MEMORY.md (follow its links). You are the campaign
orchestrator on the MPI-IS cluster. Work in rounds per the Round protocol
in PROGRAM.md, on branch autoresearch/cluster-rounds. Start with WS1
round r01_smoke: validate cluster/sweep_runner.py end-to-end with the
smoke grid described in WS1's task queue, then write
autoresearch/rounds/findings_round01.md, commit, push, and continue to
the next round. Never run heavy compute on the login node — always go
through sweep_runner.py. Report failures faithfully.
```

4. **Round cadence & monitoring** — `condor_q` / `condor_watch_q`; results under `autoresearch/results/<round>/`; findings in `autoresearch/rounds/`; the local user steers by editing `PROGRAM.md` and pushing.
5. **Troubleshooting** — held jobs (`condor_q -hold`, `condor_release`), `MaxTime`/`+MaxRuntime` both set by the runner, venv recreation, stale DONE markers (`--force`).

- [ ] **Step 2: Verify** — doc contains the clone URL, the `numpy<2` install line, the tmux commands, and the first prompt block.

- [ ] **Step 3 (orchestrator): Commit** — `docs: cluster bootstrap procedure for the autoresearch campaign`

---

### Task S7: README fixes + CLAUDE.md addendum

**Files:**
- Modify: `cluster/README.md` (stale repo name/paths), `CLAUDE.md` (repo root)

**Interfaces:**
- Consumes: file names/paths produced by S4–S6.

- [ ] **Step 1: Fix `cluster/README.md`:** replace every `double_orthogonal_ml` with `ica_causal_effect` and the clone URL with `https://github.com/rpatrik96/ica_causal_effect.git`; in "Setup", note the recommended cluster location `/is/cluster/fast/preizinger/ica_causal_effect` and `pip install -r requirements.txt "numpy<2"`; add a short "Autoresearch sweeps" section after "Submit Files" pointing to `sweep_runner.py`:

```markdown
## Autoresearch sweeps

Round-based campaign sweeps go through `cluster/sweep_runner.py` (YAML grid
spec → condor_submit_bid → condor_wait → aggregated TSV); see
`autoresearch/PROGRAM.md` for the schema and `docs/CLUSTER_BOOTSTRAP.md`
for the session setup. The `sweep_*.sub` files below remain for manual
one-off submissions. On the MPI-IS cluster always submit with
`condor_submit_bid <bid>` (default bid 43).
```

- [ ] **Step 2: Append to root `CLAUDE.md`** (new section before "Testing and Code Quality"):

```markdown
## Autoresearch campaign (cluster)

This repo hosts an autonomous experiment campaign for the TMLR resubmission
of arXiv:2507.16467, run by a Claude Code instance on the MPI-IS HTCondor
cluster.

- **Program & protocol**: `autoresearch/PROGRAM.md` — re-read every round.
- **Log & findings**: `autoresearch/RESEARCH_LOG.md`, `autoresearch/rounds/`.
- **Settled findings**: `docs/research-memory/` — read before designing
  experiments (binary-treatment IHDP/Jobs and raw-feature housing are
  settled; do not re-run).
- **Reviewer context**: `docs/reviews/`.
- **Job submission**: `cluster/sweep_runner.py` (never heavy compute on the
  login node); bootstrap in `docs/CLUSTER_BOOTSTRAP.md`.
- **Branch**: campaign commits go to `autoresearch/cluster-rounds`.
```

- [ ] **Step 3: Verify** — `grep -c double_orthogonal_ml cluster/README.md` returns 0.

- [ ] **Step 4 (orchestrator): Commit** — `docs: point cluster instance at autoresearch program; fix stale repo paths`

---

### Task R1: Dataset deep-research (Phase 0)

**Files:**
- Create: `docs/dataset-research/DATASETS.md`

**Interfaces:**
- Consumes: web (WebSearch/WebFetch); spec §Phase 0 vetting criteria.
- Produces: ranked, vetted shortlist consumed by WS2 on the cluster.

**Execution note:** runs as parallel research agents (one per dataset family) + adversarial verification + synthesis — not a single coding agent.

- [ ] **Step 1: Research each family** against the five hard criteria (continuous/dose treatment; ground-truth effects; public + feasible on CPU; real covariates amenable to pre-disentangling; DML/TE-literature precedent). Families: (a) ACIC 2016/2017/2018; (b) dose-response benchmarks — TCGA dosage + News as used by SCIGAN/DRNet/VCNet; (c) Twins & IHDP continuous-treatment adaptations; (d) DML empirical datasets with continuous treatment (e.g., 401(k) eligibility is binary — check; PennML/OpenML regression tables as X for synthetic-on-real-X); (e) benchmark conventions in recent continuous-treatment papers (what would TMLR reviewers recognize?).
- [ ] **Step 2: Adversarially verify** every kept candidate: URL resolves today, license permits use, treatment variable really is continuous (or the dose design makes it so), size fits CPU cluster, the "ground truth" mechanism is exact (simulated) not estimated.
- [ ] **Step 3: Write `DATASETS.md`:** per dataset — source URL, license, n, d, treatment type, ground-truth mechanism, loader spec (file format, preprocessing incl. the pre-disentangle step where needed, target API mirroring `realdata_loaders.py`), expected ICA regime fit (η distribution the design induces), citations (verified — no fabrication; unverifiable → `[CITE-CHECK]`). End with a **ranked pick of 2–4** to implement in WS2 and an explicit "rejected + why" list.
- [ ] **Step 4 (orchestrator): Commit** — `docs: vetted semi-synthetic dataset shortlist for WS2 (Phase 0 deep-research)`

---

### Task S8: Integration, verification, push

**Files:**
- Modify: none new — verifies and lands everything.

- [ ] **Step 1: Full test suite** — `./.venv/bin/python -m pytest --tb=short -q`. Expected: all green (existing suite + 10 new).
- [ ] **Step 2: Pre-commit over changed files** — fix silently, retry until green.
- [ ] **Step 3: Cross-reference check** — every path referenced in `PROGRAM.md`, `CLUSTER_BOOTSTRAP.md`, `CLAUDE.md` addendum, and `cluster/README.md` exists in the tree (`grep -o` the paths, test `-e`).
- [ ] **Step 4: Push master** — `git push origin master` (scaffolding lands on master; the campaign itself uses `autoresearch/cluster-rounds`).
- [ ] **Step 5: Report** — commits landed, test counts, what the user does next (bootstrap per `docs/CLUSTER_BOOTSTRAP.md`).

---

## Self-Review (completed)

- **Spec coverage:** every spec §Components row maps to a task (S1 settings, S2 memory, S3 reviews, S4 program/log/rounds, S5 runner+sub+wrapper+tests, S6 bootstrap, S7 README/CLAUDE.md, R1 datasets); spec §Testing → S5 tests + S8; §Error handling → runner exit-status handling + bootstrap troubleshooting.
- **Placeholder scan:** no TBDs; doc tasks carry complete content or verbatim-copy commands with sources.
- **Type consistency:** `load_spec/expand_grid/pending_jobs/generate_sub/aggregate/write_tsv` signatures match between S5 code and tests; YAML schema in S4 matches `load_spec` keys; `pending.txt` 4-field format matches wrapper's `$1..$3 + shift 3`.
