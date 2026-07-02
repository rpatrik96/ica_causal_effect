# Cluster Autoresearch Campaign — Design Spec

**Date:** 2026-07-02
**Goal:** Address the UAI 2026 reviewer concerns (and the TMLR meeting feedback) for the paper in `overlap-ica` (arXiv:2507.16467) by running an autonomous, Claude-Code-driven experiment campaign on the MPI-IS HTCondor cluster.
**Target:** TMLR resubmission — no hard deadline; optimize for thoroughness over speed.

## Decisions (locked with user, 2026-07-02)

| Decision | Choice |
|----------|--------|
| Venue | TMLR resubmission (per `Meeting_tmlr.md`) |
| Cluster execution | Interactive Claude Code in tmux on the MPI-IS login node, dispatching jobs via `condor_submit_bid` |
| Dataset research | Locally first: deep-research produces a vetted shortlist; the cluster implements it |
| Scope | All four workstreams (regime map + OLS, semi-synthetic, harder synthetics, stretch goals) |
| Orchestration | Claude-driven rounds + deterministic `sweep_runner.py` batch runner (adapted from `causal-ica/autoresearch/htcondor/orchestrator.py`) |
| Review surface | Single long-running branch `autoresearch/cluster-rounds` |

## Architecture

Two Claude instances hand off through git.

- **Local (this machine):** dataset deep-research (Phase 0), repo scaffolding (Phase 1), and between-round supervision. Steering = editing `autoresearch/PROGRAM.md` and pushing.
- **Cluster (MPI-IS login node, tmux):** the campaign orchestrator. Reads `PROGRAM.md` + `docs/research-memory/`, plans a round, submits jobs through `cluster/sweep_runner.py`, analyzes aggregated results, writes findings, commits, pushes, iterates.

```
local Claude ──(datasets doc, scaffolding, PROGRAM.md)──▶ git push
                                                             │
cluster Claude (tmux) ◀── git pull ──────────────────────────┘
   │ round plan (hypothesis + grid.yaml)
   ▼
cluster/sweep_runner.py ──▶ condor_submit_bid 43 ──▶ .npy on shared FS
   │                                                   │
   ◀── aggregate ── results table (tsv/parquet) ◀──────┘
   ▼
autoresearch/rounds/findings_roundN.md + RESEARCH_LOG.md ──▶ commit + push
```

Raw `.npy` arrays stay on the cluster filesystem. Only aggregated tables, plots, and findings docs are committed (`rsync` raw arrays on demand).

Auth: the cluster instance reuses the Claude Code subscription login already working for the `causal-ica` project on the same login node. No API keys enter the repo.

## Components

### 1. Portable context (committed to this repo)

Mirrors `causal-ica` commit `4a889013` ("ship research context + portable permissions for cluster Claude Code").

| Path | Contents |
|------|----------|
| `.claude/settings.json` | Permission allowlist with no machine-specific paths: `git`, `gh pr`, `python*`, `pytest`, `pip`, `uv`, `condor_submit_bid`, `condor_submit`, `condor_q`, `condor_status`, `condor_rm`, `condor_wait`, file utilities, `WebSearch`, `WebFetch(domain:github.com)` |
| `docs/research-memory/MEMORY.md` + entries | Snapshot of the local auto-memory: IHDP-binary finding, housing-strawman lesson, venv/numpy<2 pin; README explaining the snapshot pattern |
| `docs/reviews/` | Snapshots of `uai2026_reviews_summary.md`, `uai2026_reviews_with_responses.md` (from `overlap-ica`), and `Meeting_tmlr.md` (moved from repo root) |
| `autoresearch/PROGRAM.md` | The research program (see below) |
| `autoresearch/RESEARCH_LOG.md` | Seeded running log; one entry per round |
| `autoresearch/rounds/` | Per-round findings docs (`findings_round01.md`, …) |
| `docs/CLUSTER_BOOTSTRAP.md` | Exact login-node commands: clone to `/is/cluster/fast/preizinger/ica_causal_effect`, venv creation with `numpy<2` pin, `pytest --tb=short -q` smoke test, tmux session, first prompt to paste into `claude` |
| `docs/dataset-research/DATASETS.md` | Phase 0 output: vetted semi-synthetic shortlist with loader specs |
| `CLAUDE.md` addendum | Points the cluster instance at `PROGRAM.md`, the bootstrap doc, and the research memory |
| `cluster/README.md` fixes | Replace stale `double_orthogonal_ml` name/paths; document `condor_submit_bid` |

### 2. `cluster/sweep_runner.py` + `cluster/autoresearch.sub`

Deterministic batch runner so the orchestrator spends context on judgment, not condor polling. Adapted from `causal-ica`'s `orchestrator.py` (submit/wait/parse machinery only — the LLM-proposal loop is replaced by Claude Code itself).

- **Input:** a YAML grid spec: script + argument grid, resources (CPUs, memory, `MaxTime`/`+MaxRuntime` — kept equal), bid, round tag.
- **Behavior:** expand grid → write per-job params → queue through the generic `autoresearch.sub` template → `condor_submit_bid` → `condor_wait` on the event log → parse per-job exit status → aggregate result `.npy` files into a single table under `autoresearch/results/round_<tag>/` (aggregate tables are committed; raw `.npy` is not) → print a one-screen summary (completed / failed / held, headline metrics).
- **Idempotent:** skips configs whose result file already exists (`--force` overrides). Output dirs are namespaced per round, which also avoids the known existence-check short-circuit in `eta_noise_ablation.py`.
- **Non-goals:** no LLM calls, no git operations, no adaptive logic — those belong to the orchestrating Claude.

### 3. `autoresearch/PROGRAM.md`

The single source of truth the cluster instance re-reads every round. Contains:

- **Mission:** produce the evidence the TMLR resubmission needs, mapped to reviewer concerns (traceability table: concern → workstream → deliverable).
- **Workstreams** with hypotheses, success criteria, and per-round task queues (checkboxes).
- **Round protocol** (below).
- **Guardrails** (below).
- **Stopping criteria** per workstream — rounds continue until criteria are met, not until a date.

## Round protocol

1. Pull; re-read `PROGRAM.md`, `RESEARCH_LOG.md`, and the latest findings doc.
2. Pick the highest-value open question; state the hypothesis and the grid in the log *before* running.
3. Validate the config with one tiny local run (sanity, not compute — login-node etiquette).
4. Submit via `sweep_runner.py`; while jobs run, write up the previous round.
5. Aggregate; grade the evidence honestly in `findings_roundN.md`. Failed/NaN/non-converged runs are documented, never silently dropped (FastICA convergence failures are themselves data).
6. Commit + push on `autoresearch/cluster-rounds`; update `RESEARCH_LOG.md` with the proposed next round.
7. Sessions are disposable: state lives in the committed docs. Resume with `tmux attach` or `claude --continue`.

## Workstreams

### WS1 — Regime map + OLS everywhere (priority 1)

Chart win/lose regions for **ICA vs. OML vs. HOML vs. OLS vs. matching** over: η non-Gaussianity (gennorm β, Bernoulli, Student-t), confounding strength, covariate dimension d, sample size n, outcome noise, and nuisance complexity. Coarse grid first; refine adaptively near decision boundaries.

- Reviewer basis: NeBn ("natural baseline is OLS"), Meeting notes ("OLS should break down somewhere").
- Success criteria: (a) regime-map figures with all five estimators on every panel; (b) an evidenced "OLS breaks down when…" claim — expected at nonlinear nuisance misspecification and large d/n; (c) the empirical companion to the (locally proved, out-of-scope-here) OLS analogue of Theorem 3.1.

### WS2 — Semi-synthetic benchmarks (priority 2)

Real covariates + simulated PLR treatment/outcome with controllable η, so ground truth is exact and treatment stays continuous — ICA's actual domain. Implements the loaders specced in `DATASETS.md`, including the committed pre-disentangle design (PCA-whitening or FastICA on X before estimation) that fixes the housing strawman.

- Reviewer basis: NbV6, iuAn (real/semi-synthetic data).
- Success criteria: benchmark table on ≥2 vetted datasets; binary-treatment IHDP/Jobs results retained and framed as scope honesty (documented failure mode), not as a win.

### WS3 — Harder synthetics + sensitivity (priority 3)

Nonlinear confounding (random MLPs/GPs), ML nuisances (lasso/RF/GBM), d up to a few hundred including d ≳ n, and the reviewer-requested sensitivity sweeps: gradually Gaussianizing η, injecting η–ε dependence.

- Reviewer basis: iuAn, LRoS (diverse experiments, ML nuisances); iuAn, NeBn, NbV6 (sensitivity to assumption violations).
- Success criteria: sensitivity curves showing graceful vs. abrupt degradation per assumption, with the violation magnitude quantified.

### WS4 — Stretch goals (priority 4)

- **Estimator selection / Pareto (Rahul):** can a data-driven selector (estimated η kurtosis, n, d) match the better of ICA/OML everywhere? Extends the estimator-selection studies from PR #39.
- **Projection pursuit (Lester):** estimate only the treatment-relevant direction instead of the full unmixing matrix; prototype + empirical comparison against full ICA.
- Success criteria: a selector that is never worse than the per-regime best by more than a stated margin; a working projection-pursuit prototype with a cost/accuracy comparison.

**Out of scope:** paper writing (edits to `overlap-ica` stay local/manual) and theory (nonlinear identifiability, the OLS analogue of Theorem 3.1 as a proof — the cluster provides only the empirical companion).

## Phase 0 — dataset deep-research (local, first implementation step)

Deep-research pass over semi-synthetic causal-inference benchmarks. Hard vetting criteria:

1. Continuous (or dose) treatment — binary-treatment datasets are admitted only as documented failure-mode cases.
2. Ground-truth effects (simulated response surface on real covariates).
3. Public download at CPU-cluster-feasible size; license permits use.
4. Real covariates amenable to the pre-disentangle design.
5. Citable precedent in the DML/treatment-effect literature.

Candidates to vet: ACIC 2016–2018, TCGA dosage (SCIGAN/GANITE dose-response), News (Johansson et al.), Twins, California Housing (pre-disentangled design only), and generic synthetic-on-real-X designs. Output: `docs/dataset-research/DATASETS.md` with per-dataset source URL, license, n/d, treatment type, ground-truth mechanism, loader spec, expected ICA regime, citations — and a ranked pick of 2–4 to implement.

## Guardrails (encoded in PROGRAM.md)

- Do **not** re-run binary-treatment IHDP/Jobs expecting ICA to win — the honest result (OLS 0.60 / OML 0.56 / ICA 3.99 RMSE on IHDP-100) is settled; cite it as scope.
- Never run raw-feature California Housing — pre-disentangled design only.
- Pin `numpy<2` (torch/sklearn compatibility on this codebase).
- Namespace result dirs per round; delete or rename stale `.npy` before re-running an existing config.
- `MaxTime` and `+MaxRuntime` must match in every submit file.
- No heavy compute on the login node — everything goes through condor; cap concurrent jobs at ~200; default bid 43.
- Never force-push; never rewrite history on `autoresearch/cluster-rounds`.
- Report failures faithfully: a round whose jobs all died is a finding, not an embarrassment to hide.

## Monitoring and steering

The user pulls and reads `autoresearch/rounds/findings_*.md` + `RESEARCH_LOG.md`; steering means editing `PROGRAM.md` priorities and pushing. No always-on local monitor (16 GB local RAM constraint); the local Claude runs on-demand progress checks when asked.

## Implementation phases

| Phase | Where | Deliverable |
|-------|-------|-------------|
| 0 | Local | `DATASETS.md` from deep-research |
| 1 | Local | Scaffolding commit: portable context, `PROGRAM.md`, `sweep_runner.py` + submit template, bootstrap doc, README fixes |
| 2 | Cluster | Bootstrap + smoke round: tiny WS1 grid end-to-end (submit → aggregate → findings → push) |
| 3+ | Cluster | Workstream rounds WS1 → WS2 → WS3 → WS4 until stopping criteria are met |

## Error handling

- `sweep_runner.py` treats held/removed/non-zero-exit jobs as failures, lists them in the round summary, and never blocks the aggregate on stragglers beyond `MaxTime`.
- FastICA non-convergence: keep the existing retry-with-increasing-tolerance mechanism; filtered runs are counted and reported per config.
- Cluster-session death: tmux + committed state make every round resumable; the bootstrap doc includes the resume procedure.
- Network loss to GitHub from the login node: rounds queue locally (commits succeed, push retries next round).

## Testing

- `sweep_runner.py` gets unit tests (grid expansion, idempotency, result aggregation from fixture `.npy` files) in `tests/`, runnable without condor via a `--mode dry` flag (mirrors `orchestrator.py`).
- Phase 2's smoke round is the end-to-end integration test on the real cluster.
- Dataset loaders (Phase 0 outputs implemented in WS2) get the same test treatment as `realdata_loaders.py` (37-test suite precedent).
