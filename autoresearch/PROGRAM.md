# Research program — cluster autoresearch campaign

Single source of truth for the campaign orchestrator (the Claude Code instance
on the MPI-IS cluster). **Re-read this file at the start of every round**,
together with `RESEARCH_LOG.md`, the latest findings doc, and
`docs/research-memory/MEMORY.md` (follow its links). The local user steers by
editing this file and pushing.

## 1. Mission

Produce the experimental evidence for the TMLR resubmission of
[arXiv:2507.16467](https://arxiv.org/abs/2507.16467), addressing the UAI 2026
reviews (`docs/reviews/uai2026_reviews_summary.md`) and the TMLR meeting
feedback (`docs/reviews/meeting_tmlr_notes.md`). Rounds continue until each
workstream's stopping criteria are met — there is no calendar deadline;
optimize for thoroughness over speed.

Out of scope: paper writing (edits to `overlap-ica` stay local/manual) and
theory (nonlinear identifiability; the OLS analogue of Theorem 3.1 as a proof —
the cluster provides only the empirical companion).

## 2. Traceability: reviewer concern → workstream → deliverable

| Reviewer concern | Workstream | Deliverable |
|---|---|---|
| OLS baseline everywhere (NeBn: "natural baseline is OLS") | WS1 | Regime-map figures with all five estimators; an evidenced "OLS breaks down when…" claim |
| Real / semi-synthetic data (NbV6, iuAn) | WS2 | Benchmark table on ≥2 vetted datasets |
| More diverse / harder experiments, ML nuisances (iuAn, LRoS) | WS3 | High-dim + ML-nuisance comparisons |
| Sensitivity to assumption violations (iuAn, NeBn, NbV6) | WS3 | Sensitivity curves per assumption |
| Estimator-selection Pareto frontier (TMLR meeting / Rahul) | WS4 | Data-driven selector vs. per-regime best |
| Projection pursuit as selective estimation (TMLR meeting / Lester) | WS4 | Prototype + cost/accuracy comparison vs. full ICA |
| Headline comparison uses a source-oracle ICA readout, oracle support and no OLS/OML (TMLR critical evaluation) | WS5 | Paired-test reruns of Fig. 2/3 (E1), r02-vs-Fig. 2 reconciliation (E3), variance-ablation rerun (E4) |

## 3. Round protocol

1. Pull; re-read `PROGRAM.md`, `RESEARCH_LOG.md`, and the latest findings doc.
2. Pick the highest-value open question; state the hypothesis and the grid in the log *before* running.
3. Validate the config with one tiny local run (sanity, not compute — login-node etiquette).
4. Submit via `sweep_runner.py`; while jobs run, write up the previous round.
5. Aggregate; grade the evidence honestly in `findings_roundN.md`. Failed/NaN/non-converged runs are documented, never silently dropped (FastICA convergence failures are themselves data).
6. Commit + push on `autoresearch/cluster-rounds`; update `RESEARCH_LOG.md` with the proposed next round.
7. Sessions are disposable: state lives in the committed docs. Resume with `tmux attach` or `claude --continue`.

## 4. Workstreams

### WS1 — Regime map + OLS everywhere (priority 1)

**Hypothesis.** Win/lose regions for ICA vs. OML vs. HOML vs. OLS vs. matching
are chartable and interpretable: ICA is competitive under weak confounding,
small samples, and strongly non-Gaussian η; OLS breaks down somewhere —
expected at nonlinear nuisance misspecification and large d/n.

**Knobs / grid dimensions.** η non-Gaussianity (gennorm β, Bernoulli,
Student-t), confounding strength, covariate dimension d, sample size n,
outcome noise, nuisance complexity. Coarse grid first; refine adaptively near
decision boundaries.

**Success criteria.** (a) Regime-map figures with all five estimators on every
panel; (b) an evidenced "OLS breaks down when…" claim — expected at nonlinear
nuisance misspecification and large d/n; (c) the empirical companion to the
(locally proved, out-of-scope-here) OLS analogue of Theorem 3.1.

**Stopping criteria.** All three success criteria met and adaptive refinement
near the decision boundaries no longer moves them; or a criterion is graded
negative with the evidence documented in a findings doc.

**Initial task queue.**
- [ ] (a) Smoke round `r01_smoke`: tiny `monte_carlo_single_instance.py` grid, 4–8 jobs — validates `sweep_runner.py` end-to-end.
- [ ] (b) Coarse regime grid: η gennorm β ∈ {0.5, 1, 2, 3, 4} × n ∈ {500, 1k, 2k, 5k, 10k} × d ∈ {5, 10, 20, 50}, all five estimators.
- [ ] (c) Adaptive refinement near the win/lose boundaries found in (b).
- [ ] (d) OLS-breakdown hunt: nonlinear g(X), d ≳ n.

### WS2 — Semi-synthetic benchmarks (priority 2)

**Hypothesis.** Real covariates + simulated PLR treatment/outcome with
controllable η — so ground truth is exact and treatment stays continuous,
ICA's actual domain — reproduce the synthetic regime findings on realistic
covariate structure.

**Knobs / grid dimensions.** Dataset (per `docs/dataset-research/DATASETS.md`
ranked picks), η distribution, pre-disentangle variant (PCA whitening or
FastICA on X before estimation — the committed fix for the housing strawman),
treatment/outcome coefficient scales.

**Success criteria.** Benchmark table on ≥2 vetted datasets; binary-treatment
IHDP/Jobs results retained and framed as scope honesty (documented failure
mode), not as a win.

**Stopping criteria.** Benchmark table complete on ≥2 vetted datasets with the
scope-honesty framing written; or the vetted datasets are exhausted with the
negatives documented.

**Initial task queue.**
- [ ] Implement loaders for the ranked picks in `DATASETS.md` (same test treatment as `realdata_loaders.py`).
- [ ] Run the benchmark grid on ≥2 datasets, including the pre-disentangle step.
- [ ] Write the scope-honesty framing for the settled binary-treatment IHDP/Jobs results (cite `docs/research-memory/ihdp-ica-binary-finding.md`; do not re-run).

### WS3 — Harder synthetics + sensitivity (priority 3)

**Hypothesis.** Estimator rankings under nonlinear confounding, ML nuisances,
and high dimension differ from the linear low-dim setting, and degradation
under assumption violations is quantifiable — graceful for some assumptions,
abrupt for others.

**Knobs / grid dimensions.** Nonlinear confounding (random MLPs/GPs), ML
nuisances (lasso/RF/GBM), d up to a few hundred including d ≳ n, gradually
Gaussianizing η, injecting η–ε dependence.

**Success criteria.** Sensitivity curves showing graceful vs. abrupt
degradation per assumption, with the violation magnitude quantified.

**Stopping criteria.** Sensitivity curves cover both reviewer-requested
violations (Gaussianization of η, η–ε dependence) plus the nonlinear/ML-nuisance
comparisons, each with quantified violation magnitude; or a sweep is graded
negative/broken with the reason documented.

**Initial task queue.**
- [ ] Nonlinear-confounding grid (random MLPs/GPs) with ML nuisances (lasso/RF/GBM).
- [ ] High-dimension sweep: d up to a few hundred, including d ≳ n.
- [ ] Sensitivity sweep: gradually Gaussianize η.
- [ ] Sensitivity sweep: inject η–ε dependence.

### WS4 — Stretch goals (priority 4)

**Hypothesis.** (Rahul) A data-driven selector on estimated η kurtosis, n, and
d can match the better of ICA/OML everywhere. (Lester) Projection pursuit can
estimate only the treatment-relevant direction instead of the full unmixing
matrix, at lower cost.

**Knobs / grid dimensions.** Selector features (estimated η kurtosis, n, d)
and decision rule; projection-pursuit prototype vs. full ICA across the WS1
regime grid.

**Success criteria.** A selector that is never worse than the per-regime best
by more than a stated margin; a working projection-pursuit prototype with a
cost/accuracy comparison.

**Stopping criteria.** Both success criteria met (or graded negative with
evidence); WS4 only starts once WS1–WS3 stopping criteria are met or blocked.

**Initial task queue.**
- [ ] Extend the estimator-selection studies from PR #39 into a selector evaluated on the WS1 regime grid.
- [ ] Prototype projection pursuit for the treatment-relevant direction; compare cost/accuracy against full ICA.

### WS5 — TMLR critical-eval reruns (r17: E1, E3, E4)

**Purpose.** A TMLR critical evaluation of the paper found that the headline
linear-PLR comparison (Fig. 2 coefficient ablation; Fig. 3 filtered heatmaps
at a=0.5, b=0.05, θ=1) resolves the ICA permutation by Hungarian matching to
the true sources (`ica.py:ica_treatment_effect_estimation`), passes oracle
support to both methods by default, and reports neither OLS nor first-order
OML. It also found that the variance of the implemented η-column ratio readout
depends on b rather than on c_ICA = 1 + ‖b + aθ‖². WS5 reruns those
experiments with the oracle-free eps-row readout
(`ica_treatment_effect_estimation_eps_row`) next to the Munkres-oracle one,
with OLS, OML, HOML (known and estimated moments), matching and Huber
regression on the cross-fitted residuals computed on the same draws, with
per-seed estimates saved for paired tests, and with the oracle-support switch
exercised. In the paper's pipelines X consists of exactly the support columns,
so `--oracle_support` has no effect unless `--extra_covariates` appends
irrelevant columns; the E1 and E3 grids therefore cross oracle support with
`--extra_covariates {0, 40}`, where 0 is the paper setting. WS5 also reconciles
campaign round r02 with Fig. 2 (E3) and reruns the variance ablation on the
fixed HOML moments (E4).

**Rounds.** All use 50 seeds, `--ica_readouts munkres_oracle eps_row`,
`--robust_baseline` and bid 43. Each YAML header gives the settings, the job
split and the cost estimate with its derivation (timed local runs at reduced
seeds, scaled to 50 seeds; cluster cores assumed up to 2× slower, which doubles
the CPU-h column at most).

| Round (`autoresearch/rounds/`) | Script | Grid | Jobs | Est. CPU-h (local) |
|---|---|---|---:|---:|
| `r17_e3_reconcile.yaml` | `monte_carlo_single_instance.py --single_config` | covariate β {1, 4} × `scalar_coeffs` {true, false} × oracle {true, false} | 8 | 0.3 |
| `r17_e3_reconcile_ablation.yaml` | `eta_noise_ablation.py --coefficient_ablation`, a=1.56, b=−1.45, θ=1 | covariate β {1, 4} × `extra_covariates` {0, 40} × oracle | 8 | 0.2 |
| `r17_e4_variance.yaml` | `eta_noise_ablation.py --variance_ablation` | η shape {0.5, 1, 1.5, 2.5, 3, 4} × σ² {0.25, 0.5, 1, 2, 4} in one job | 1 | 0.25 |
| `r17_e1_coef.yaml` | `eta_noise_ablation.py --coefficient_ablation` | a (4) × `extra_covariates` {0, 40} × oracle; each job runs 4 b × 6 θ | 16 | 3.8 |
| `r17_e1_heatmaps.yaml` | `eta_noise_ablation.py --filtered_heatmap` | axis mode {d_vs_n, beta_vs_n} × `extra_covariates` {0, 40} × oracle; each job is one full heatmap | 8 | 5.8 |

The `r17_e3_reconcile.yaml` header lists the differences found between the
r02 job and the Fig. 2 setting.

**Launch order.**
1. E3 first, as a cheap check: both E3 YAMLs (16 jobs). Run the protocol-step-3
   local sanity check on one E3 job only (`--limit 1`); the E1 jobs are too
   heavy for the login node.
2. Then E4 (1 job) and E1 (`r17_e1_coef`, then `r17_e1_heatmaps`; 24 jobs).

**Deliverables.**
- The `autoresearch/analyze_e1.py` summary over `r17_e1_coef` and
  `r17_e1_heatmaps` (`e1_summary.md` plus the per-config RMSE, paired-test and
  win-rate CSVs), committed under `autoresearch/results/r17_e1_analysis/`.
- The E3 difference table: per job, RMSE per method (from `analyze_e1.py` over
  both E3 rounds) set against the difference list in the
  `r17_e3_reconcile.yaml` header, stating which differences the grid isolates.
- The E4 heatmaps regenerated with the existing plotting: re-run the job's
  command without `--no_plot` (it reloads the saved `.npy`), plus
  `analyze_e1.py` over `r17_e4_variance` for the paired statistics.
- `findings_round17.md` per §7, with failure and FastICA-convergence counts per
  method from the `failure_counts` and `convergence_counts` fields.

**Stopping criteria.** All 41 jobs are DONE, or every failed job is documented
with its reason (§3 step 5); the deliverables above are written; each E3
difference is either isolated by the grid or listed as not isolated. WS5 has
no adaptive refinement: follow-up grids are proposed in the findings doc, not
run.

**Hand-back.** Results come back as a pull request against this repo
(`ica_causal_effect`) containing the findings doc, the analysis outputs and
the aggregated tables; raw `.npy` stays on the cluster (§6).

## 5. Guardrails

- Do **not** re-run binary-treatment IHDP/Jobs expecting ICA to win — the honest result (OLS 0.60 / OML 0.56 / ICA 3.99 RMSE on IHDP-100) is settled; cite it as scope.
- Never run raw-feature California Housing — pre-disentangled design only.
- Pin `numpy<2` (torch/sklearn compatibility on this codebase).
- Namespace result dirs per round; delete or rename stale `.npy` before re-running an existing config.
- `MaxTime` and `+MaxRuntime` must match in every submit file.
- No heavy compute on the login node — everything goes through condor; cap concurrent jobs at ~200; default bid 43.
- **Login-node OOM survival.** The login node caps per-user RAM against exhausted swap, so a fat process is SIGKILLed (exit 137 / `-9`) by the global OOM killer. Launch the orchestrator under `claude-guard` in tmux (not plain `claude`), run only one heavy claude at a time, and skip 1M-context for the loop. Keep the local sanity run (protocol step 3) to `--limit 1`; `sweep_runner.py --mode local` already caps joblib fan-out (`SWEEP_LOCAL_MAX_JOBS`, default 2) so it can't OOM the session. Full rationale + knobs in `docs/CLUSTER_BOOTSTRAP.md` §6.
- Never force-push; never rewrite history on `autoresearch/cluster-rounds`.
- Report failures faithfully: a round whose jobs all died is a finding, not an embarrassment to hide.

## 6. File map

| Path | Contents |
|---|---|
| `cluster/sweep_runner.py` | Deterministic sweep runner: YAML grid spec → `condor_submit_bid` → `condor_wait` → aggregated TSV. No LLM calls, no git, no adaptive logic — judgment stays with the orchestrator. |
| `autoresearch/results/<round>/` | Per-round job outputs (`job_<idx:04d>/` with `DONE` markers), `jobs_manifest.tsv`, `summary.tsv`. Aggregate tables are committed; raw `.npy` is not (`rsync` on demand). |
| `autoresearch/rounds/findings_roundNN.md` | Per-round findings docs (template in §7). |
| `autoresearch/RESEARCH_LOG.md` | Running log, one entry per round, newest first. |
| `docs/research-memory/` | Settled findings snapshot — read `MEMORY.md` before designing experiments that touch its topics. |
| `docs/dataset-research/DATASETS.md` | Vetted semi-synthetic dataset shortlist with loader specs (WS2 input). |
| `docs/CLUSTER_BOOTSTRAP.md` | Login-node setup, tmux session, first prompt, troubleshooting. |

### Grid-spec YAML schema

Consumed by `cluster/sweep_runner.py`. Required keys: `round`, `script`,
`grid`. Defaults: `base_args: ""`, `output_flag: "--output_dir"`, `bid: 43`,
`resources: {cpus: 4, memory_gb: 16, disk_gb: 10, max_time_s: 86400}`.
No commas anywhere in expanded args (breaks condor `queue ... from`).

```yaml
round: r01_smoke                      # names autoresearch/results/<round>/
script: monte_carlo_single_instance.py
base_args: "--n_experiments 20 --covariate_pdf gennorm"
output_flag: "--output_dir"           # per-job output dir flag; "" to omit
grid:                                 # flag -> list of values; Cartesian product
  n_samples: [500, 1000]
  sigma_outcome: [1.732]
bid: 43
resources:
  cpus: 4
  memory_gb: 16
  disk_gb: 10
  max_time_s: 86400                   # MaxTime == +MaxRuntime, set by the runner
```

Grid values: booleans emit/omit the bare flag; lists emit space-separated
multi-value flags (e.g. `dims: [[5, 10]]` → `--dims 5 10`).

## 7. Findings-doc template

Every round writes `autoresearch/rounds/findings_roundNN.md` with this skeleton:

```markdown
# Round NN — <one-line hypothesis>

## Hypothesis

## Grid
(spec YAML or a pointer to it)

## Jobs
Submitted / completed / failed — with reasons (held, non-zero exit,
FastICA non-convergence counts per config).

## Results
Table + figures.

## Evidence grade
One of: confirmed / suggestive / negative / broken — with justification.

## Implications for the paper

## Proposed next round
```
