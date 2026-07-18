# Round 02 — r02_eta_regime: first real WS1 regime panel (η shape × n)

## Hypothesis

Along the axis that drives the ICA identifiability story — the treatment-noise
distribution η — the five estimators (OLS, OML, HOML, ICA, matching) separate
into chartable win/lose regions. Two concrete a-priori predictions from the
asymptotic-variance analysis (encoded in `r02_eta_regime.yaml`):

1. **HOML** exploits skewness `E[η³]`, so its asymptotic variance is finite only
   for the skewed (`discrete`) η and **degenerate for every symmetric η**
   (`gennorm_heavy`, `rademacher`, `uniform`, `gennorm_light`) → HOML should be
   high-variance on the symmetric shapes, worst at small n.
2. **ICA** asymptotic variance is finite for all shapes, ordered
   `discrete < rademacher < heavy < uniform < gennorm_light`; ICA's finite-sample
   RMSE should approach this ordering as n grows.

## Grid

`autoresearch/rounds/r02_eta_regime.yaml`. One fixed operating point
(`--single_config`: covariate gennorm β=4, d=10, θ=1, treatment_coef=1.56,
outcome_coef=−1.45, **oracle support ON**), 25 Monte-Carlo experiments per job,
swept over:

- `eta_noise_dist ∈ {gennorm_heavy, discrete, rademacher, uniform, gennorm_light}`
- `n_samples ∈ {500, 1000, 2000, 5000, 10000}`

5 × 5 = 25 jobs, all on GPU/compute nodes via `cluster/sweep_runner.py` condor
cluster `17381305`.

## Jobs

Submitted 25 / completed **25** / failed-or-held **0**. Every config kept 25/25
experiments (no FastICA non-convergence drops at this operating point). Metrics
in `autoresearch/results/r02_eta_regime/metrics.tsv` (extracted with
`autoresearch/analyze_round.py`).

Operational note: the local sanity run for this round was previously killed by
the login-node OOM killer (`failed(-9)`) because `monte_carlo_single_instance.py`
fans out with joblib `Parallel(n_jobs=-1)` — one heavy worker per core (32 on
login1). `sweep_runner.py --mode local` now caps that fan-out; the compute-node
jobs are unaffected (each gets its own node). See `docs/CLUSTER_BOOTSTRAP.md` §6.

## Results

Per-estimator RMSE (lower is better; **bold** = best in row):

| η | n | OLS | OML | HOML | ICA | matching |
|---|---:|---:|---:|---:|---:|---:|
| discrete | 500 | **0.049** | 0.126 | 0.072 | 0.372 | 0.139 |
| discrete | 1000 | **0.029** | 0.065 | 0.065 | 0.229 | 0.085 |
| discrete | 2000 | **0.026** | 0.043 | 0.037 | 0.153 | 0.048 |
| discrete | 5000 | **0.013** | 0.019 | 0.019 | 0.071 | 0.033 |
| discrete | 10000 | **0.010** | 0.011 | 0.014 | 0.060 | 0.026 |
| gennorm_heavy | 500 | **0.049** | 0.109 | 0.090 | 0.338 | 0.121 |
| gennorm_heavy | 1000 | **0.023** | 0.060 | 0.104 | 0.232 | 0.083 |
| gennorm_heavy | 2000 | **0.023** | 0.041 | 0.056 | 0.221 | 0.051 |
| gennorm_heavy | 5000 | **0.014** | 0.020 | 0.035 | 0.106 | 0.036 |
| gennorm_heavy | 10000 | **0.008** | 0.010 | 0.025 | 0.075 | 0.024 |
| rademacher | 500 | **0.037** | 0.107 | 0.244 | 0.248 | 0.124 |
| rademacher | 1000 | **0.029** | 0.066 | 0.037 | 0.168 | 0.094 |
| rademacher | 2000 | **0.024** | 0.035 | 0.024 | 0.143 | 0.070 |
| rademacher | 5000 | **0.013** | 0.020 | 0.013 | 0.069 | 0.032 |
| rademacher | 10000 | **0.009** | 0.010 | 0.010 | 0.051 | 0.026 |
| uniform | 500 | **0.039** | 0.104 | **28.1** | 0.600 | 0.145 |
| uniform | 1000 | **0.028** | 0.065 | 0.080 | 0.195 | 0.093 |
| uniform | 2000 | **0.026** | 0.049 | 0.041 | 0.183 | 0.066 |
| uniform | 5000 | **0.015** | 0.018 | 0.016 | 0.089 | 0.037 |
| uniform | 10000 | **0.009** | 0.013 | 0.013 | 0.052 | 0.025 |
| gennorm_light | 500 | **0.039** | 0.121 | **5.67** | 0.714 | 0.113 |
| gennorm_light | 1000 | **0.036** | 0.064 | 0.285 | 0.236 | 0.076 |
| gennorm_light | 2000 | **0.018** | 0.032 | 0.047 | 0.183 | 0.061 |
| gennorm_light | 5000 | **0.014** | 0.017 | 0.045 | 0.116 | 0.032 |
| gennorm_light | 10000 | **0.007** | 0.010 | 0.021 | 0.076 | 0.020 |

**1. OLS wins every one of the 25 configs**, OML second (~2–3× OLS), matching
third. At this operating point — oracle nuisance support, low dimension (d=10),
well-specified linear DGP — OLS is essentially optimal and ICA's identifiability
machinery buys nothing. This is *not* a bug; it is the corner of the regime map
where the "natural baseline is OLS" reviewer point holds hardest.

**2. HOML's symmetric-η breakdown is confirmed, and it is catastrophic at small
n.** The two bounded symmetric-light shapes blow up at n=500 — `uniform` RMSE
**28.1**, `gennorm_light` **5.67** — versus 0.04–0.10 for the other estimators.
This is exactly the skewness-degeneracy prediction: with `E[η³]=0` the HOML
moment estimator divides by a near-zero quantity and explodes. The blowup is
worst for bounded light-tailed symmetric η, milder for `rademacher` (0.24) and
`gennorm_heavy` (0.09), and absent for skewed `discrete` (0.07). It shrinks
quickly with n (uniform: 28.1 → 0.08 by n=1000), so it is a small-sample
degeneracy, not an asymptotic one — the finite-sample face of the "avar is ∞
under symmetry" statement.

**3. ICA is not competitive at this operating point** — worst or near-worst
everywhere (RMSE 0.05–0.71), only beating the HOML small-n blowups. It does
improve monotonically with n (e.g. gennorm_light 0.71 → 0.076), confirming
round 01's finite-sample-vs-asymptotic gap. The predicted asymptotic ordering
`discrete < rademacher < heavy < uniform < light` has **not** emerged even at
n=10000 (observed: rademacher 0.051 ≈ uniform 0.052 < discrete 0.060 < heavy
0.075 ≈ light 0.076) — `light` highest as predicted, but `discrete` is mid, not
lowest. ICA is converging to its asymptotics far more slowly than OML/OLS here.

## Evidence grade

**Suggestive.** One prediction confirmed cleanly (HOML symmetric-η small-n
breakdown), one partially (ICA's slow, not-yet-asymptotic ordering). The panel
is a valid regime-map slice but maps only the OLS-favorable corner: with oracle
support and a well-specified low-d linear DGP there is **no ICA-competitive
region on this axis** — consistent with the hypothesis that ICA needs weak/absent
oracle knowledge, higher d, or harder nuisances to express its advantage. WS1
success criterion (a) (all five estimators on every panel) is met for this
slice; criterion (b) ("OLS breaks down when…") is **not** yet evidenced here —
OLS never breaks down at this operating point.

## Implications for the paper

- Direct answer to the "OLS is the natural baseline" reviews: **quantify the
  corner where OLS wins** (oracle nuisance, low-d, well-specified) rather than
  hide it — this panel is that evidence, and it is honest scope-setting.
- The HOML uniform/gennorm_light n=500 blowup is a crisp, citable illustration
  that the second-order (skewness) estimator is fragile under symmetric η — a
  point in favor of ICA/first-order methods when skewness is unavailable.
- ICA's slow finite-sample convergence is a caveat to state plainly: its low
  *asymptotic* variance under non-Gaussian η does not translate to small-sample
  wins in the well-specified low-d setting.

## Proposed next round

**r03_confounding_oracle** — turn off the oracle and add nuisance difficulty,
the most likely axis to reveal an ICA-favorable / OLS-breakdown region:
`oracle_support` OFF × d ∈ {10, 50, 100} × n ∈ {500, 2000, 10000}, at a
non-Gaussian η where ICA is theoretically strong (`gennorm_heavy` or `discrete`).
Hypothesis: as d grows toward/past n without oracle support, OLS and OML degrade
(nuisance misspecification / variance) while ICA's structural identification
holds up — the first candidate for criterion (b)'s "OLS breaks down when…" claim.
