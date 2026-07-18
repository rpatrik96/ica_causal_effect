# Round 09 (WS2) — r09_ica_edge_nlarged: ICA gets a large-n edge, but only with non-Gaussian ε

## Hypothesis

Across the campaign ICA was competitive but **never best** in the linear PLR
(r02, r07): at every n it trailed OLS/OML. Theory says ICA exploits η's
non-Gaussianity through higher moments, so any advantage should be an asymptotic
(large-n) effect, plausibly visible only when n ≫ d and η is strongly
non-Gaussian. This round pushes into that corner on real covariates to test
whether ICA ever wins (ICA RMSE < min(OLS, OML)).

## Grid

`autoresearch/rounds/r09_ica_edge_nlarged.yaml`. housing covariates, d′=5
(small d), η gennorm β=0.5 (super-heavy), linear (well-specified) nuisance,
`--bootstrap` so n exceeds the 20640 rows. Axes:
`eps_beta {2.0 (Gaussian), 0.5 (heavy-tailed)}` × `n {2000, 10000, 50000}`
(n/d′ up to 10000). θ=1, 25 experiments each. All 6 jobs DONE, 25/25 finite.

The `eps_beta` axis is the new lever: the earlier rounds always used **Gaussian**
outcome noise ε, so only η was non-Gaussian. Here ε can be heavy-tailed too — a
*second* non-Gaussian source for ICA to exploit.

## Results

RMSE and the decisive ICA/OLS ratio (θ=1):

| ε | n | OLS | OML | ICA | ICA/OLS | ICA best? |
|---|---:|---:|---:|---:|:--:|:--:|
| Gaussian (β=2) | 2000 | 0.0220 | 0.0237 | 0.0239 | 1.09 | no |
| Gaussian (β=2) | 10000 | 0.0100 | 0.0099 | 0.0133 | 1.32 | no |
| Gaussian (β=2) | 50000 | 0.0025 | 0.0026 | 0.0051 | 2.02 | no |
| **heavy (β=0.5)** | 2000 | 0.0203 | 0.0207 | **0.0140** | **0.69** | **YES** |
| **heavy (β=0.5)** | 10000 | 0.0104 | 0.0102 | **0.0067** | **0.64** | **YES** |
| **heavy (β=0.5)** | 50000 | 0.0046 | 0.0044 | **0.0024** | **0.53** | **YES** |

**ICA has a genuine large-n efficiency edge — conditional on a non-Gaussian ε.**

- With **heavy-tailed ε**, ICA is the **best** estimator at every n, and the edge
  **grows with n**: ICA/OLS = 0.69 → 0.64 → **0.53** (≈2× lower RMSE than OLS at
  n=50000). The advantage strengthening with n is the signature of an *asymptotic
  efficiency* gain, exactly what the higher-moment identification predicts.
- With **Gaussian ε**, there is **no edge** and it gets *worse* with n:
  ICA/OLS = 1.09 → 1.32 → **2.02**. ICA converges but to a higher-variance limit
  than the second-moment estimators when the outcome noise is Gaussian.

**This resolves the campaign-long puzzle.** ICA never won in r02/r07 **because ε
was Gaussian there** — only η carried non-Gaussian signal. The ICA estimator reads
θ from the ε-source row of the unmixing matrix; a Gaussian ε is the one source ICA
cannot sharpen on, so it forfeits efficiency to OLS/OML. Make **both** PLR noises
non-Gaussian and ICA's higher-moment machinery finally beats the second-moment
estimators — increasingly so as n grows.

## Evidence grade

**Confirmed (positive).** ICA does get an edge when n ≫ d — a real, growing-with-n
efficiency advantage — but it requires a non-Gaussian outcome noise ε, not just a
non-Gaussian treatment noise η. The two ε columns are a clean controlled contrast
(same η, d, n; only ε's tail changes), and the opposite n-trends (edge widening vs
disadvantage widening) make the mechanism unambiguous.

## Implications for the paper

- **A concrete "ICA wins here" result**, the positive counterpart to the WS1
  loss-regions. Headline: in the fully non-Gaussian linear PLR with n ≫ d, ICA's
  RMSE is ~½ that of OLS/OML and the gap widens with n.
- **Sharpens the assumption statement.** The paper should make explicit that ICA's
  efficiency edge needs non-Gaussianity in *both* structural noises; with a
  Gaussian outcome noise ICA is only competitive, not superior. This reframes the
  earlier "competitive but not best" observations as a direct consequence of the
  Gaussian-ε design, not a weakness of ICA.
- Combine with r02 (η-shape regime), r04/r05 (nonlinear-confounding losses), r06
  (dimension), r07 (real-X reproduction): the complete map now has an explicit
  ICA-wins cell (n ≫ d, linear mixing, both noises non-Gaussian).

## Proposed next round

- **r10**: sweep η kurtosis × ε kurtosis at fixed large n to chart the ICA-win
  frontier (how non-Gaussian must each noise be?), and confirm the edge on news20
  (sparse X) — does the ε-driven edge reproduce across covariate structure?
- Fold the ε non-Gaussianity control back into a WS1 synthetic run to confirm the
  effect is not covariate-specific (quick verification, high value).
