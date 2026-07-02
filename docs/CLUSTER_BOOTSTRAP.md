# Cluster bootstrap: autoresearch campaign on the MPI-IS cluster

Login-node procedure to set up and launch the autonomous experiment campaign
(see `autoresearch/PROGRAM.md` for the round protocol and
`cluster/README.md` for HTCondor details).

## 1. One-time setup (login node)

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

## 2. Claude Code session

Run the orchestrator inside tmux so it survives SSH disconnects:

```bash
tmux new -s autoresearch
cd /is/cluster/fast/preizinger/ica_causal_effect
claude
```

Auth reuses the subscription login already working for causal-ica on this
host; verify with a trivial prompt before pasting the first prompt below.

- Detach: `Ctrl+B d`
- Resume: `tmux attach -t autoresearch`
- After a session dies: `claude --continue` (inside the tmux session, from
  the repo root)

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
