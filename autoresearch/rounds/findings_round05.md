# Round 05 — r05_nonlinear_eta_regime: is the nonlinear-regime story η-robust?

## Hypothesis

r04 (at η=Laplace, eta_beta=1) found under nonlinear confounding: OLS and
weak-nuisance OML break down (RMSE flat in n ≈0.42); only nonlinear+gbm OML
recovers (decreasing in n); ICA worst and flat (~0.55). This round sweeps the
treatment-noise non-Gaussianity to test whether that story is η-robust.

The nonlinear DGP draws η ~ gennorm(β = eta_beta): β=0.5 super-heavy, β=1
Laplace, **β=2 Gaussian (ICA non-identifiable)**, β=4 sub-Gaussian light. Tested:
- **H1 (rescue η-robust):** gbm-OML decreases with n at every β.
- **H2 (breakdown η-robust):** OLS and linear-nuisance OML flat/high at every β.
- **H3 (ICA never recovers):** ICA high/flat at all β, and — sub-prediction —
  worst at β=2 (Gaussian).

## Grid

`autoresearch/rounds/r05_nonlinear_eta_regime.yaml`. Nonlinear confounding ON,
`eta_beta ∈ {0.5, 1.0, 2.0, 4.0}` × `nuisance ∈ {linear, gbm}` × `n ∈ {2000,
10000}` = 16 jobs, heavy-tailed η, low-dim, oracle moments ON, `--n_jobs 4`,
θ=1.5. All 16 DONE, 0 failed, 25/25 experiments finite. Linear-confounding
control is r04. Scope: this DGP has only gennorm η (no discrete/rademacher/uniform
— those would need a `nonlinear_dgp` extension). Metrics:
`autoresearch/results/r05_nonlinear_eta_regime/metrics.tsv`.

## Results

RMSE under nonlinear confounding (θ=1.5):

| β | n | nuis | OLS | OML | HOML | ICA | matching |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.5 | 2000 | gbm | 0.411 | **0.179** | 0.178 | 0.548 | 0.339 |
| 0.5 | 10000 | gbm | 0.417 | **0.136** | 0.106 | 0.571 | 0.320 |
| 0.5 | 2000 | linear | 0.411 | 0.414 | 0.399 | 0.548 | 0.339 |
| 0.5 | 10000 | linear | 0.417 | 0.418 | 0.425 | 0.571 | 0.320 |
| 1.0 | 10000 | gbm | 0.418 | **0.132** | 0.272 | 0.588 | 0.323 |
| 1.0 | 10000 | linear | 0.418 | 0.418 | 0.460 | 0.588 | 0.323 |
| 2.0 | 10000 | gbm | 0.417 | **0.125** | 0.403 | 0.564 | 0.319 |
| 2.0 | 10000 | linear | 0.417 | 0.417 | 0.463 | 0.567 | 0.319 |
| 4.0 | 10000 | gbm | 0.416 | **0.127** | 0.872 | 0.588 | 0.320 |
| 4.0 | 10000 | linear | 0.416 | 0.416 | 0.464 | 0.589 | 0.320 |

(Full 16 rows in metrics.tsv; n=2000 rows omitted above for the light betas.)

**H1 confirmed — the rescue is fully η-robust.** gbm-OML RMSE decreases with n at
*every* β: 0.179→0.136 (β0.5), 0.175→0.132 (β1), 0.178→0.125 (β2), 0.168→0.127
(β4). At n=10000 it sits at ≈0.13 regardless of β — a flat 3.2× improvement over
OLS. The double-orthogonal rescue does not depend on η (as expected: OML is an
orthogonal-moment method, η-agnostic).

**H2 confirmed — the breakdown is fully η-robust.** OLS is flat at **0.411–0.418
across the entire grid**, and linear-nuisance OML flat at ≈0.42 for every β.
Misspecification bias is η-independent.

**H3 confirmed on the main claim, and the sub-prediction is refuted informatively.**
ICA is ≈0.55 at every β (means 0.559 / 0.559 / 0.552 / 0.557 for β=0.5/1/2/4) and
does **not** improve with n. Two things:
- ICA fails **even at β=0.5**, its most-favorable strongly-non-Gaussian η. So
  ICA's collapse under nonlinear confounding is **not** a non-Gaussianity
  deficit — it is the nonlinear confounding itself.
- The sub-prediction "worst at β=2 (Gaussian)" is **wrong**: ICA is essentially
  *flat* across β, β=2 (0.552) marginally the *best* of the four. The
  interpretation is the finding: nonlinear confounding so thoroughly breaks ICA's
  linear-mixing identification that η's non-Gaussianity — the quantity ICA exploits
  — becomes **inert**. The Gaussian/non-Gaussian axis, which drove the whole ICA
  story in the linear model (r02), carries no signal once the mixing is nonlinear.

Aside (honest): HOML is erratic under nonlinear confounding — HOML+gbm ranges
0.106 (β0.5,n10k) to **0.872** (β4,n10k), occasionally worse than OLS. Its
higher-order-moment estimator is unstable here; OML (first-order orthogonal) is
the reliable performer.

## Evidence grade

**Confirmed.** r04's three nonlinear-regime findings — OLS/weak-OML breakdown,
gbm-OML rescue, ICA failure — are each η-robust across the gennorm-β axis. The new
insight is that under nonlinear confounding the η non-Gaussianity axis is **inert
for ICA**: it cannot rescue ICA even at β=0.5, confirming that ICA's domain is the
*linear*-mixing model.

## Implications for the paper

- The nonlinear-regime map is now closed and robust: **flexible-nuisance OML is
  the estimator of record under nonlinear confounding**, uniformly in η and
  improving with n; OLS/weak-OML have irreducible bias; ICA and matching are
  biased and n-insensitive.
- Sharpens the ICA scope claim to a testable dichotomy: **η non-Gaussianity is
  ICA's lever only when the mixing is linear** (r02). Break linearity and the
  lever disconnects (r05). This is a clean, defensible boundary for the
  contribution.
- HOML instability under nonlinear confounding is worth a footnote: prefer the
  first-order orthogonal OML with an ML nuisance over the higher-order variant
  outside the moment-friendly (skewed-η, linear) regime.

## Proposed next round

WS1 is now well-evidenced: (a) five-estimator panels (r02), (b) OLS breakdown +
its cause and cure (r04) and η-robustness (r05), plus ICA's win-region (r02) and
loss-regions (r03 dimensionality, r04/r05 nonlinearity). Recommend **pivoting to
WS2 (semi-synthetic benchmarks)**: build the ranked-dataset loaders from
`docs/dataset-research/DATASETS.md` with the pre-disentangle step, and run the
benchmark grid on ≥2 vetted datasets. That moves the evidence from synthetic to
realistic covariate structure, the other half of the reviewer ask. A short
WS1 capstone (r06: nonlinear + true d≳n, the one untested breakdown axis) is
optional if reviewers press on high dimension.
