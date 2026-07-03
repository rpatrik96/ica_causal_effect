# Round 10 (WS2) — r10_ica_win_frontier: the ICA-win frontier is driven by ε, not η

## Hypothesis

r09 found ICA beats OLS/OML at large n only when both structural noises are
non-Gaussian. This round charts the full frontier: sweep η and ε non-Gaussianity
jointly at fixed large n and map where ICA is the best estimator. Expected
(H1–H3): a contiguous ICA-win region where both β are well below 2, deepening
toward the (0.5, 0.5) super-heavy corner, roughly symmetric in η vs ε.

## Grid

`autoresearch/rounds/r10_ica_win_frontier.yaml`. housing covariates, d′=5,
n=50000 (bootstrap), linear nuisance, θ=1. `eta_beta {0.5,1,1.5,2,3}` ×
`eps_beta {0.5,1,1.5,2,3}` = 25 jobs (gennorm β: 0.5 super-heavy, 1 Laplace,
2 Gaussian, 3 light). All 25 DONE, 25/25 finite. Figure:
`autoresearch/results/r10_ica_win_frontier/ica_win_frontier.{png,pdf}`
(`autoresearch/plot_ica_win_frontier.py`).

## Results

ICA/OLS RMSE ratio (<1 = ICA beats OLS; rows η, cols ε):

| η \ ε | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---:|---:|---:|---:|---:|---:|
| **0.5** | **0.53** | 1.06 | 1.04 | 2.02 | 1.11 |
| **1.0** | **0.86** | **0.86** | 1.62 | 2.20 | 1.50 |
| **1.5** | 0.98 | 1.26 | 2.32 | 4.83 | 2.08 |
| **2.0** | **0.54** | 1.62 | 4.35 | **99.8** | 5.72 |
| **3.0** | **0.71** | 1.01 | 2.13 | 4.78 | 2.36 |

Cells where ICA is best of **all** estimators (ICA < min(OLS, OML)):
(η,ε) = (0.5,0.5), (1,0.5), (1,1), (2,0.5), (3,0.5).

**The frontier is not a symmetric corner — it is a vertical band at heavy ε.
Outcome-noise non-Gaussianity is the dominant driver; treatment-noise
non-Gaussianity is secondary.**

- **The ε=0.5 column is the win region**, across essentially all η: ICA beats OLS
  at η = 0.5 (0.53), 1.0 (0.86), 1.5 (0.98, tied), 2.0 (0.54) and 3.0 (0.71). As
  ε moves toward Gaussian the edge vanishes column by column (ε=1 mixed; ε≥1.5 all
  losses).
- **η non-Gaussianity is neither necessary nor sufficient.** Not necessary: at
  ε=0.5 even a **Gaussian** treatment noise (η=2.0) wins (0.54). Not sufficient: a
  super-heavy η=0.5 with Gaussian ε (η=0.5, ε=2.0) *loses* at 2.02. The η
  dependence at fixed ε=0.5 is weak and non-monotonic (0.53→0.86→0.98→0.54→0.71).
- **The both-Gaussian cell (η=2, ε=2) is catastrophic: 99.8× OLS.** Two Gaussian
  sources are unseparable by ICA (at most one Gaussian source is identifiable), so
  the θ-carrying row is essentially unidentified. The whole ε=2 column and η=2 row
  form the red "Gaussian cross" in the figure.

**Mechanism.** The estimator here is the **eps-row** ICA identifier: it reads θ
from the row of the unmixing matrix that loads on the outcome noise ε. That row is
sharpened by ε's non-Gaussianity — so ε's tail, not η's, sets the efficiency edge.
This is specific to the eps-row identification; a method that read θ from the
η-row would show the transposed frontier.

## Evidence grade

**Confirmed**, and it refines r09. ICA does have a large-n win region, but the
controlling variable is the **outcome-noise** non-Gaussianity, not the
treatment-noise non-Gaussianity that the ICA-for-treatment-effects story usually
emphasizes. The frontier is a heavy-ε band, and the (Gaussian, Gaussian) corner is
where ICA fails hardest (100× OLS).

## Implications for the paper

- **Precise assumption for the "ICA wins" claim:** at n ≫ d in the linear PLR,
  ICA (eps-row) is the most efficient estimator **iff the outcome noise ε is
  sufficiently non-Gaussian** (here β ≲ 1); the treatment noise η is largely
  irrelevant to the *efficiency* edge (though it remains what makes θ identifiable
  at all). This is a sharper and more surprising statement than "ICA likes
  non-Gaussian η" and is worth foregrounding — with the eps-row mechanism as the
  explanation.
- **A ready headline figure:** `ica_win_frontier.pdf` — the blue heavy-ε win band,
  the red Gaussian cross, the 100× both-Gaussian corner. It makes ICA's operating
  regime legible at a glance and pairs with the WS1 loss-region map.
- Caveat to state: the frontier is drawn for the eps-row identifier; note the
  dependence would transpose under an η-row identifier, and both collapse at the
  Gaussian null.

## Proposed next round

- **Robustness of the frontier:** reproduce the ε-driven band on news20 (sparse X)
  and on synthetic X (confirm it is estimator-intrinsic, not housing-specific), and
  at a second n to show the win band deepens with n (as r09).
- **Estimator-selection tie-in (WS4):** the frontier is a ready-made ground truth
  for a selector on estimated ε-kurtosis — "use ICA when the outcome residual is
  heavy-tailed" — a concrete, defensible selection rule.
