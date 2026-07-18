# Round 16 (WS3) — sensitivity to assumption violations: η-Gaussianisation & η–ε dependence

## Hypothesis

The reviewer ask (iuAn/NeBn/NbV6): quantify sensitivity to assumption violations,
showing graceful vs abrupt degradation per assumption. Two sweeps at a fixed
otherwise-benign operating point (synthetic X, d′=5, n=10000, linear PLR, θ=1):
1. **Gaussianise η** — eta_beta 0.5→2.0 (violation magnitude = η excess kurtosis:
   ~25 at β=0.5, 3 at β=1, 0 at β=2), crossed with ε heavy (β=0.5) vs Gaussian (β=2).
2. **Inject η–ε dependence** — ρ = Corr(η,ε) from 0 to 0.8 (violation magnitude = ρ),
   violating the PLM exogeneity assumption Cov(η,ε)=0.

## Results

### Sweep 1 — Gaussianising η (`r16_gauss_eta`, 14 cells)

RMSE vs eta_beta (figure `ws3_gauss_eta.{png,pdf}`):

| | β=0.5 | β=1 | β=1.5 | β=1.75 | β=2 (Gaussian) |
|---|---:|---:|---:|---:|---:|
| **heavy ε** OLS/OML | ~0.011 | ~0.009 | ~0.008 | ~0.009 | ~0.011 |
| heavy ε **ICA** | 0.006 | 0.009 | 0.009 | 0.009 | 0.008 |
| heavy ε **HOML** | 0.023 | 0.016 | 0.034 | 0.134 | **22.3** |
| Gaussian ε **ICA** | 0.019 | 0.037 | 0.056 | 0.204 | **0.530** |
| Gaussian ε **HOML** | 0.022 | 0.022 | 0.042 | 0.109 | **40.8** |

- **OLS / OML / matching are flat (robust)** across the whole η sweep, at both ε
  levels (~0.01). They do not depend on η's distribution — η-Gaussianisation is a
  non-issue for them.
- **ICA degrades gracefully iff ε is non-Gaussian.** With heavy ε, ICA is flat
  (~0.008) even at fully Gaussian η — it identifies θ through the ε source, so
  Gaussianising η barely matters. With Gaussian ε, ICA climbs steadily and then
  **abruptly hits the non-identifiability wall** (0.019 → 0.204 → 0.530 as η→
  Gaussian): two Gaussian sources cannot be separated. So ICA's sensitivity to η
  is entirely conditional on ε carrying non-Gaussian signal — the r10 law, seen as
  a degradation curve.
- **HOML fails abruptly at Gaussian η**, at both ε levels (22–41× at β=2, rising
  from ~0.02): its higher-order-moment estimator degenerates as η's excess kurtosis
  → 0. HOML is the estimator most sensitive to η-Gaussianisation.

### Sweep 2 — η–ε dependence (`r16_eta_eps_dep`, 8 cells)

RMSE vs ρ (figure `ws3_eta_eps_dep.{png,pdf}`):

| ρ | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| OLS | 0.011 | 0.100 | 0.200 | 0.300 | 0.500 | 0.800 |
| OML | 0.011 | 0.100 | 0.199 | 0.299 | 0.499 | 0.799 |
| HOML | 0.023 | 0.112 | 0.211 | 0.310 | 0.509 | 0.806 |
| ICA | 0.006 | 0.101 | 0.201 | 0.301 | 0.501 | 0.801 |
| matching | 0.013 | 0.102 | 0.202 | 0.302 | 0.502 | 0.801 |

- **Every estimator biases by exactly ρ** (RMSE ≈ ρ, all five curves collapse onto
  the RMSE = ρ reference line). No method is robust — OLS, OML, HOML, ICA, matching
  all fail identically and linearly. This is not estimator error: a non-zero
  Cov(η,ε) enters the treatment-effect estimand directly (θ̂ → θ + ρ under unit-
  variance noises), so **η–ε dependence breaks identification itself**, and no
  PLM-based method can escape it.

## Evidence grade

**Confirmed.** The two assumptions have opposite robustness profiles, quantified:
- **η non-Gaussianity is a soft, estimator-specific assumption.** OLS/OML/matching
  are fully robust; ICA is robust whenever ε is non-Gaussian and only fails at the
  all-Gaussian wall; HOML is the one abruptly sensitive method (needs η non-Gaussian).
  Degradation is graceful except at the exact Gaussian point.
- **η–ε independence (exogeneity) is a hard, universal assumption.** Violating it
  biases every estimator linearly in ρ; there is no robust method because the
  estimand changes. Abrupt and uniform.

## Implications for the paper

- **A clean sensitivity section.** The two curves are the reviewer-requested
  "graceful vs abrupt" evidence: distributional assumptions (η shape) degrade
  gracefully and only for the methods that rely on them, while the structural
  exogeneity assumption (η⊥ε) degrades everything at once. This positions ICA
  honestly — its extra assumption (non-Gaussian η) is mild and, when ε is
  non-Gaussian, costless — while flagging η⊥ε as the assumption that matters for
  all estimators alike.
- Pairs with the selector: the selector already routes away from ICA/HOML when the
  estimated kurtoses are low (the graceful-degradation regimes), but η–ε dependence
  is undetectable from residual marginals and must be argued from design, not
  diagnosed — a caveat worth stating.

## Proposed next round

WS3's two headline sensitivity sweeps are done. Optional extensions: heteroscedastic
ε(X) (already a DGP flag), and repeating both curves on real X to confirm the
degradation profiles are covariate-independent (expected from r11). Otherwise the
campaign now covers WS1–WS4 reviewer asks; next is assembling the paper figure set.
