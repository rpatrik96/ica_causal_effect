# Cluster bootstrap: autoresearch campaign on the MPI-IS cluster

Login-node procedure to set up and launch the autonomous experiment campaign
(see `autoresearch/PROGRAM.md` for the round protocol and
`cluster/README.md` for HTCondor details).

## 1. One-time setup (login node)

```bash
cd /is/cluster/fast/preizinger
git clone https://github.com/rpatrik96/ica_causal_effect.git double_orthogonal_ml
cd double_orthogonal_ml         # repo `ica_causal_effect`; on-disk dir is double_orthogonal_ml
python3 -m venv .venv            # use `uv venv` if uv is available
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt "numpy<2"
python -m pytest tests/test_sweep_runner.py -q   # smoke: 10 passed
mkdir -p ~/jobs
chmod +x cluster/run_autoresearch_job.sh cluster/run_experiment.sh
```

## 2. Claude Code session

Run the orchestrator inside tmux (survives SSH disconnects) **and** under
`claude-guard` (survives the login-node OOM killer — see §6):

```bash
tmux new -s autoresearch
cd /is/cluster/fast/preizinger/double_orthogonal_ml
claude-guard                     # NOT plain `claude`; auto-resumes on OOM kill
```

`claude-guard` (in `~/.local/bin/`) caps the V8 heap
(`NODE_OPTIONS=--max-old-space-size=3072`) so claude GCs instead of ballooning
past the per-user RAM cap, logs each kill to `~/.claude/claude-guard-kills.log`,
and auto-restarts with `--continue` on exit 137/143. Auth reuses the
subscription login already working for causal-ica on this host; verify with a
trivial prompt before pasting the first prompt below.

- Detach: `Ctrl+B d`
- Resume: `tmux attach -t autoresearch`
- After a session dies (and claude-guard is not up): `claude-guard --continue`
  (inside the tmux session, from the repo root)
- Watch RAM headroom in a second pane: `claude-memwatch` (live used/cap %, peak,
  failcnt against the cgroup limit).

**Do NOT** run 1M-context or a second heavy `claude` (e.g. an interactive
session plus a headless `claude -p`) on the login node concurrently — two fat
node processes ≈ the RAM cap → OOM. Serialize heavy claude work; keep the
compute in condor.

## 3. First prompt (paste verbatim)

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

## 4. Round cadence & monitoring

- Watch the queue with `condor_q` (or `condor_watch_q` for a live view).
- Sweep results land under `autoresearch/results/<round>/`; per-round
  findings under `autoresearch/rounds/`.
- The cluster instance commits and pushes to `autoresearch/cluster-rounds`
  each round; the local user steers the campaign by editing
  `autoresearch/PROGRAM.md` and pushing — the orchestrator re-reads it at
  the start of every round.

## 5. Troubleshooting

- **Local sanity run killed with `failed(-9)` / `-137`**: the global OOM killer
  took it. The experiment scripts fan out with joblib `Parallel(n_jobs=-1)`,
  which on a 32-core login node spawns one heavy numpy/torch/FastICA worker per
  core and blows the per-user RAM cap. `sweep_runner.py --mode local` now caps
  that fan-out automatically (`LOKY_MAX_CPU_COUNT=2`, single-threaded BLAS); tune
  with `SWEEP_LOCAL_MAX_JOBS` (worker cap) and, if a run still balloons,
  `SWEEP_LOCAL_MEM_GB=6` (bounds the child's address space so it dies with a
  clean MemoryError instead of the kernel killing the whole session). Keep local
  runs to `--limit 1`; they validate config, not compute.
- **Held jobs**: inspect with `condor_q -hold` (the HoldReason column says
  why); release with `condor_release <cluster_id>` once fixed. Runtime
  limits are not the usual cause — the runner sets both `MaxTime` and
  `+MaxRuntime` on every submission.
- **Broken venv** (e.g., after a Python module change on the cluster):
  delete and recreate it with the one-time-setup block above; the
  `"numpy<2"` pin is required or torch imports break.
- **Stale DONE markers**: if a round aborted mid-sweep and
  `sweep_runner.py` skips configs as already done, re-run the sweep with
  `--force` to ignore the markers and recompute.

## 6. Login-node OOM survival (why claude keeps getting killed)

Diagnosed 2026-07-02 on the MPI-IS login node. Each user is capped at a few GiB
of **physical RAM** by a cgroup memory limit (login3: 4 GiB; login1: 8 GiB),
while the shared node's **swap is chronically exhausted**. When a process grows
past the cap it cannot be swapped out, so the **global OOM killer** terminates
the largest-RSS process — usually `claude` (node/V8) or a fat local experiment
run — with **SIGKILL (exit 137, or `-9` from a subprocess)**. The admin cap
cannot be raised; the only lever is **absolute RAM — stay well under it**.

Two fat processes cause this here:

1. **`claude` itself** — mitigated by launching under **`claude-guard`** (heap
   cap + auto-`--continue`) in tmux, running only one heavy claude at a time,
   and avoiding 1M-context for the orchestrator loop (§2). Watch headroom with
   `claude-memwatch`.
2. **Local sanity runs** — mitigated by the `--mode local` fan-out cap in
   `sweep_runner.py` (§5). Real compute always goes to condor, where each job
   gets its own node and memory request.

Both are the login-node analogue of the OOM hardening in the causal-ica
orchestrator; the guards live in `~/.local/bin/` (`claude-guard`,
`claude-memwatch`) and are documented in claude-config's `CLAUDE.md`.
