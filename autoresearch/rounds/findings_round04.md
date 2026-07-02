# Round 04 — r04_nonlinear_breakdown: the evidenced OLS breakdown (criterion b)

## Hypothesis

r02/r03 established that OLS never breaks down in the *linear* model (invariant
even to 40 nuisance covariates) and that ICA is the fragile estimator. So "OLS
breaks down when…" (WS1 success criterion b) must come from OLS being
*structurally* wrong. This round uses the nonlinear-confounding DGP
(`nonlinear_runner.py` / `nonlinear_dgp.py`): with nonlinear g(X), m(X) linear
OLS cannot represent the conditional means and is asymptotically biased; the
double-orthogonal OML/HOML survive **only** with a flexible first-stage nuisance.

Four hypotheses, tested by a 2×2×3 difference-in-differences:
- **H1 (control):** confounding=linear → OLS competitive at all n.
- **H2 (breakdown):** nonlinear + weak (linear/LassoCV) nuisance → OLS **and** OML
  biased; RMSE plateaus, does not →0 with n.
- **H3 (rescue):** nonlinear + flexible (gbm) nuisance → OML/HOML recover.
- **H4 (ICA):** does ICA — identifying via η's non-Gaussianity rather than a
  correct conditional mean — degrade gracefully under nonlinear confounding?

## Grid

`autoresearch/rounds/r04_nonlinear_breakdown.yaml`. `confounding ∈ {linear,
nonlinear}` × `nuisance ∈ {linear, gbm}` × `n ∈ {500, 2000, 10000}` = 12 jobs,
η heavy-tailed throughout (`--heavy_tail_eta --eta_beta 1.0`), low nominal dim
(n_covariates=10, support_size=5) to isolate nonlinearity from r03's
dimensionality effect, oracle moments ON, `--n_jobs 4`, θ=1.5. All 12 jobs DONE,
0 failed/held, 25/25 experiments finite per config. Metrics via the new
`autoresearch/analyze_nonlinear_round.py` (the nonlinear payload schema differs
from the linear rounds) → `autoresearch/results/r04_nonlinear_breakdown/metrics.tsv`.

## Results

RMSE (θ=1.5; **bold** = best in row):

| confound | nuisance | n | OLS | OML | HOML | ICA | matching |
|---|---|---:|---:|---:|---:|---:|---:|
| linear | linear | 500 | **0.048** | 0.055 | 0.111 | 0.176 | 0.084 |
| linear | linear | 2000 | **0.025** | 0.027 | 0.060 | 0.092 | 0.039 |
| linear | linear | 10000 | 0.0083 | **0.0082** | 0.030 | 0.035 | 0.015 |
| linear | gbm | 500 | **0.048** | 0.188 | 0.171 | 0.176 | 0.084 |
| linear | gbm | 2000 | **0.025** | 0.116 | 0.089 | 0.092 | 0.039 |
| linear | gbm | 10000 | **0.0083** | 0.060 | 0.032 | 0.033 | 0.015 |
| **nonlin** | **linear** | 500 | 0.408 | 0.409 | 0.456 | 0.560 | **0.359** |
| **nonlin** | **linear** | 2000 | 0.418 | 0.420 | 0.422 | 0.530 | **0.347** |
| **nonlin** | **linear** | 10000 | 0.418 | 0.418 | 0.460 | 0.588 | **0.323** |
| **nonlin** | **gbm** | 500 | 0.408 | **0.217** | 0.405 | 0.560 | 0.359 |
| **nonlin** | **gbm** | 2000 | 0.418 | **0.175** | 0.271 | 0.530 | 0.347 |
| **nonlin** | **gbm** | 10000 | 0.418 | **0.132** | 0.272 | 0.590 | 0.323 |

**H1 confirmed.** Linear confounding: OLS is best/near-best and →0 with n
(0.048 → 0.0083). The control behaves like r02/r03.

**H2 confirmed — this is the evidenced OLS breakdown.** Under nonlinear
confounding, **OLS RMSE is pinned at ≈0.42 at every n** (0.408 → 0.418 → 0.418):
it does **not** shrink with sample size — the signature of asymptotic bias from
misspecification (≈28% relative bias at θ=1.5). Crucially, **OML with the weak
linear nuisance breaks down identically** (0.409 → 0.420 → 0.418): orthogonality
does not save you when the first stage cannot represent the nonlinear g(X).

**H3 confirmed — the rescue is specific to flexible nuisances.** Nonlinear
confounding + **gbm** nuisance is the **only** configuration whose OML RMSE
*decreases* with n (0.217 → 0.175 → 0.132), a 3.2× improvement over the
linear-nuisance OML (0.418) at n=10000, gap widening with n. HOML+gbm partially
recovers (0.41 → 0.27). So the double-orthogonal method's value is real but
**conditional on a first stage flexible enough to fit the confounding** —
precisely the ML-nuisance story the reviewers (iuAn, LRoS) asked for.

**H4 answered — ICA does not benefit; it is the worst estimator here.** Under
nonlinear confounding ICA is flat and highest (0.56 → 0.53 → 0.59): its
identification rests on the *linear*-ICA mixing structure, which nonlinear
confounding violates, and — unlike OML — it has no flexible-nuisance lever to
pull. Matching is biased but lowest among the non-gbm estimators (≈0.33),
because its local comparisons tolerate mild nonlinearity better than a global
linear fit, though it too plateaus.

Secondary finding: with **linear** confounding, the gbm nuisance *hurts*
(OML 0.060 vs 0.0082 at n=10000) — flexibility costs variance when the truth is
linear. Nuisance flexibility should match the DGP, not be maximal by default.

## Evidence grade

**Confirmed** for WS1 criterion (b): an evidenced, isolated "OLS breaks down
when the confounding is nonlinear" claim, with the linear-confounding column as
the control and the weak-vs-flexible nuisance columns pinning the cause to
first-stage misspecification. Simultaneously a **negative for ICA**: it is the
worst estimator under nonlinear confounding and gains nothing from n — the
nonlinear regime is an OML-with-ML-nuisance story, not an ICA story.

## Implications for the paper

- **Criterion (b) is met.** The headline figure writes itself: RMSE-vs-n curves
  under nonlinear confounding — OLS and linear-nuisance-OML flat at ≈0.42, gbm-OML
  the lone curve bending toward zero. This is the crisp answer to "why not just
  OLS": OLS is exactly right until the confounding is nonlinear, then only a
  *flexible-nuisance* orthogonal method recovers.
- **Frames the ICA contribution honestly.** ICA's regime is low-dim, linear-mixing,
  strongly-non-Gaussian η (r02) — it is *not* the tool for nonlinear confounding.
  The paper should claim ICA where it wins and defer to ML-nuisance OML where it
  does not, rather than over-claim.
- Pair with r02 (η regime, HOML symmetric breakdown) and r03 (dimensionality
  fragility of ICA) for a complete "where each estimator wins/loses" map.

## Proposed next round

WS1 criteria (a) all-five-estimators panels and (b) OLS-breakdown are now both
evidenced (r02 + r04). Two directions:
- **r05_nonlinear_eta_regime** — repeat r04's nonlinear+gbm winner across the η
  shapes of r02 (gennorm β, discrete, rademacher, uniform) to check whether the
  gbm-OML rescue and ICA's failure are η-robust, completing the regime map with a
  nonlinear axis.
- Or pivot to **WS2** (semi-synthetic benchmarks, `docs/dataset-research/DATASETS.md`)
  now that the synthetic OLS-breakdown evidence is in hand — the pre-disentangle
  loaders are the next build.
Recommend r05 first (cheap, closes WS1's regime map), then WS2.
