# Round 03 — r03_oracle_nuisance: does removing the oracle break OLS or favour ICA?

## Hypothesis

r02 mapped the OLS-favorable corner (oracle support ON, well-specified low-d):
OLS won all 25 configs, ICA never competitive. The proposed route to an
ICA-favorable / "OLS breaks down when…" region (WS1 criterion b) was to **remove
the oracle**: with `--no_oracle_support`, every estimator must fit the full
covariate vector (cov_dim_max = **50**) instead of the true 10-dim support — 40
nuisance covariates. Hypothesis: OLS/OML degrade under nuisance-selection
pressure while ICA's structural identification holds → ICA closes/reverses the
gap at small n.

## Grid

`autoresearch/rounds/r03_oracle_nuisance.yaml`. `oracle ∈ {ON, OFF}` ×
`eta ∈ {gennorm_heavy, discrete}` × `n ∈ {500,1000,2000,5000,10000}` = 20 jobs,
`--single_config` (covariate gennorm β=4, true support d=10 within D=50
covariates, θ=1), 25 experiments each, via `cluster/sweep_runner.py`.

The oracle flag is toggled through the grid's bare-flag mechanism
(`no_oracle_support: [false, true]`). Verified `--no_oracle_support` is a real
change here (D=50 ≠ d=10). This round also added `oracle_support` to the result
payload and an `oracle`/`D` column to `analyze_round.py` so the panel is
self-describing (the flag was previously unrecorded).

## Jobs

Submitted 20 / completed **20** / **0** failed in the final tally, but honestly:
the first two condor submissions each lost **2–3 jobs to transient node
evictions** (exit 0, no `.npy`) on a heavily-loaded pool (562 held jobs
cluster-wide at submit time). The failing set *changed between runs*
(→ transient, not a config bug — confirmed by reproducing every "failed" config
successfully in a local sanity run), so the fix was a plain re-submit of the
missing indices. One stale pre-relabel `.npy` from a late-executing run-1 job
had to be pruned (it lacked the `oracle_support` key and produced a phantom
duplicate row before removal). All 20 kept 25/25 experiments (no FastICA
non-convergence drops even oracle-OFF at D=50). Determinism cross-check: the ten
oracle-ON configs reproduce r02's values **exactly**.

## Results

Oracle ON→OFF effect (RMSE; `×` = OFF/ON ratio):

| η | n | OLS on | OLS off | × | ICA on | ICA off | × |
|---|---:|---:|---:|:--:|---:|---:|:--:|
| gennorm_heavy | 500 | 0.049 | 0.051 | **1.0** | 0.338 | 1.709 | **5.1** |
| gennorm_heavy | 1000 | 0.023 | 0.021 | **0.9** | 0.232 | 2.118 | **9.1** |
| gennorm_heavy | 2000 | 0.023 | 0.022 | 1.0 | 0.221 | 0.417 | 1.9 |
| gennorm_heavy | 5000 | 0.014 | 0.014 | 1.0 | 0.106 | 0.108 | 1.0 |
| gennorm_heavy | 10000 | 0.008 | 0.009 | 1.0 | 0.075 | 0.077 | 1.0 |
| discrete | 500 | 0.049 | 0.048 | 1.0 | 0.372 | 1.079 | 2.9 |
| discrete | 1000 | 0.029 | 0.030 | 1.1 | 0.229 | 0.348 | 1.5 |
| discrete | 2000 | 0.026 | 0.025 | 1.0 | 0.153 | 0.196 | 1.3 |
| discrete | 5000 | 0.013 | 0.013 | 1.0 | 0.071 | 0.077 | 1.1 |
| discrete | 10000 | 0.010 | 0.010 | 1.0 | 0.060 | 0.060 | 1.0 |

Full five-estimator table: `autoresearch/results/r03_oracle_nuisance/metrics.tsv`.

**The hypothesis is refuted — the effect runs the opposite way.**

1. **OLS and OML are essentially invariant to the oracle** (× ≈ 1.0 at every n
   and both η). Their ℓ1-regularized nuisance fit (`λ = √(log D / n)`) absorbs 40
   zero-coefficient covariates at no measurable cost. OLS does **not** break down
   here.
2. **ICA is the estimator that breaks down under nuisance covariates**, severely
   at small n: RMSE inflates **×5.1 (n=500) and ×9.1 (n=1000)** for
   gennorm_heavy, ×2.9 for discrete at n=500. ICA must unmix a (D+2)=52-dim
   system; at small n FastICA + Munkres matching degrade badly. The penalty
   **vanishes by n≈5000** (× ≈ 1.0), i.e. it is a small-sample dimensionality
   cost, not an asymptotic one.
3. **Matching also degrades oracle-OFF** (×1.5–4), and unlike ICA the penalty
   *persists* to n=10000 (curse of dimensionality in 50-d covariate matching).
4. **HOML is oracle-insensitive** (moment-based on the treatment residual).

## Evidence grade

**Negative** (for the hypothesis / criterion b on this axis), and informative.
Removing the oracle does not expose an OLS breakdown or an ICA-favorable region;
it does the reverse. Combined with r02, the picture sharpens: **ICA's viable
region is the low-nominal-dimension, oracle/known-support setting** — adding
nuisance covariates hurts ICA (and matching) far more than the regularized
first/second-order methods. This is a real constraint on where ICA can win, and
directly relevant to the WS3 "high-dim / ML-nuisance" workstream (expect ICA to
lose there unless dimension is controlled).

## Implications for the paper

- Strong, honest answer to "why not just OLS": in the linear well-specified
  model OLS is **remarkably robust** — invariant to 40 nuisance covariates and
  best-in-class across r02+r03's 45 configs. The paper should concede this corner
  explicitly and locate ICA's advantage precisely, not broadly.
- ICA's dimensionality fragility (×9 at n=1000) is a caveat to state plainly and
  a design guide: pre-dimension-reduction (the committed PCA/FastICA-on-X
  pre-disentangle step for WS2) is not optional for ICA at realistic D.
- Criterion (b) "OLS breaks down when…" is **not** on the oracle/linear-nuisance
  axis. The remaining candidates are **model misspecification** (nonlinear g(X)
  that OLS cannot represent) and **true d ≳ n** (here D=50 < n; not yet tested).

## Proposed next round

**r04_nonlinear_breakdown** — the misspecification route to criterion (b). Use
the nonlinear DGP (`nonlinear_dgp.py` / `nonlinear_runner.py`): nonlinear outcome
g(X) (random MLP / GP) with η non-Gaussian, oracle support ON (isolate
misspecification from dimensionality, given r03's dimensionality finding),
n ∈ {500, 2000, 10000}. Hypothesis: linear OLS/OML are biased by the unmodeled
nonlinearity while ICA — which identifies via η's non-Gaussianity rather than a
correctly specified conditional mean — degrades more gracefully. This is the
first setting where OLS is *structurally* wrong rather than merely
higher-variance. If ICA still loses, that is itself a strong (if negative)
scope statement for the resubmission.
