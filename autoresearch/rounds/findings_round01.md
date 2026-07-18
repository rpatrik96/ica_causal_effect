# Round 01 (r01_smoke) — validate cluster/sweep_runner.py end-to-end

## Hypothesis

Operational, not scientific: the autoresearch pipeline
(`grid spec → sweep_runner.py → condor_submit_bid → run_autoresearch_job.sh →
monte_carlo_single_instance.py → .npy + DONE → aggregate`) runs end-to-end on real
compute nodes, with the `.venv` symlink resolving to the working `care` environment.
Secondary: a tiny single-regime n-scaling sanity that all five estimators are wired
correctly (RMSE should fall monotonically-ish with n).

## Grid

`autoresearch/rounds/r01_smoke.yaml`:

```yaml
round: r01_smoke
script: monte_carlo_single_instance.py
base_args: "--n_experiments 10 --single_config --no_plot --covariate_pdf gennorm"
grid: { n_samples: [200, 500, 1000, 2000] }   # 4 jobs
resources: { cpus: 4, memory_gb: 16, disk_gb: 10, max_time_s: 3600 }
```

`--single_config` collapses `OMLParameterGrid` to one point per job:
covariate pdf `gennorm(β=4)`, support size `d=10`, treatment effect `θ=1`,
treatment coef `1.56`, outcome coef `-1.45`, oracle support ON. The only varied
axis is `n_samples`.

**Regime caveat (drives round 02).** The single varied covariate shape is the
*covariate* `gennorm β=4`; the **treatment noise η is `discrete`** (the default
`--eta_noise_dist`, excess kurtosis ≈ 5.05), *not* a swept η. The paper's ICA
identifiability story is about **η** non-Gaussianity, which is controlled by
`--eta_noise_dist` (`gennorm_heavy`/`gennorm_light`/`laplace`/`gennorm`), a knob
this smoke did not exercise. So this round says nothing about ICA's favourable
regime — that is round 02's job.

## Jobs

Submitted 4, completed 4, failed/held 0.

- `job_0000` (n=200) ran **locally on the login node** as the one-tiny-run sanity
  (PROGRAM §3.3), 1 config × 10 exp, ~30 s.
- `job_0001`–`0003` (n=500/1000/2000) ran on **compute nodes** via
  `condor_submit_bid 43` (cluster 17380709), all reached `DONE`.
- FastICA convergence: 0 non-converged / 0 NaN across all 40 experiments
  (`n_exp_kept = 10/10` every config).

Enabling fix required first: `monte_carlo_single_instance.py:675` called
`OMLParameterGrid.create_from_config(config)`, which had been removed from
`oml_utils.py` by the earlier "chore: gitignore" commit `a114722` (a stale call
site on `master` — the script did not run at all before this). Restored the
classmethod verbatim from `a114722^`. Caught by the PROGRAM §3.3 local sanity run
*before* any condor submission.

## Results

Per-estimator RMSE vs true θ=1 (`autoresearch/results/r01_smoke/metrics.tsv`),
covariate `gennorm β=4`, η `discrete`, d=10, 10 experiments each:

| n | OLS | OML | HOML | ICA | matching |
|------|--------|--------|--------|---------|----------|
| 200 | 0.0614 | 0.2257 | 0.1402 | **8.769** | 0.2325 |
| 500 | 0.0400 | 0.1316 | 0.0431 | 0.3391 | 0.1311 |
| 1000 | 0.0326 | 0.0677 | 0.0680 | 0.2352 | 0.0896 |
| 2000 | 0.0279 | 0.0358 | 0.0367 | 0.1577 | 0.0578 |

All estimators improve with n (pipeline + DGP + estimators wired correctly). OLS
is best throughout and ICA worst at this single regime point; the n=200 ICA
figure (RMSE 8.77) is a single high-variance FastICA draw (bias +2.74, std 8.33
over 10 seeds) that damps out by n=500. This is one point in a discrete-η,
oracle-support setting — **not** evidence about ICA's merits; it is consistent
only with the prior that discrete/two-point η is off-regime for ICA (cf. the
settled binary-treatment finding, `docs/research-memory/ihdp-ica-binary-finding.md`).

## Evidence grade

**confirmed** (operational). The end-to-end pipeline works on real compute nodes:
grid expansion, condor submission/bidding, `.venv`→`care` activation, per-job
`.npy` output, `DONE` markers, and aggregation all validated; 4/4 jobs DONE,
0 failed/held. The scientific content is deliberately negligible (single regime)
and is graded only as a wiring sanity, which passed.

## Implications for the paper

None directly. Establishes the machinery for the WS1 regime map and confirms the
five-estimator return path (`ortho_rec_tau` columns: 0 OML, 1 HOML, 2 OML-Est,
3 OML-Split, 4 ICA, 5 OLS, 6 matching — note OLS/matching are **not** in the
`biases`/`sigmas` summary and must be aggregated from `ortho_rec_tau` directly,
which `autoresearch/analyze_round.py` now does).

## Proposed next round

**r02_regime_eta** (WS1 coarse regime map, priority-1). Vary the axis that
actually drives the ICA story — **η** — plus n, at a small fixed d, all five
estimators:

- η: `--eta_noise_dist ∈ {gennorm_heavy (Laplace-tailed, β=1), laplace,
  gennorm_light (β=4), discrete}` to bracket heavy-tailed → light-tailed → two-point.
- n ∈ {500, 1000, 2000, 5000}; d fixed at 10; drop `--single_config` so the full
  internal (θ, coef) grid runs — but first confirm the per-job cost and whether
  the swept covariate `beta` should also move, then size the grid to stay under
  ~50 jobs. Resolve one open code question before submitting: whether the
  loop `beta` is forwarded to η when `--eta_noise_dist gennorm` (it is not, for
  the fixed-β shortcuts `gennorm_heavy/light`), so the η-β sweep must use the
  shortcut distributions or a dedicated `eta_noise_dist=gennorm` + `beta` path.
- Expectation: ICA competitive (≤ OML) under heavy-tailed η at moderate n;
  OLS/OML best under light-tailed/discrete η. This is the first real regime-map panel.
