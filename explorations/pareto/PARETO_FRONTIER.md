# Pareto Frontier Memo: ICA vs OML vs HOML — Is There a Dominating Method?

**Question (Reviewer Rahul):** Is there a Pareto frontier of algorithms that does at least as well *everywhere* as the better of {OML, ICA}? i.e. is there a single method, or a cheap data-driven selector, that dominates the per-regime upper envelope?

---

## What the Reviewer Really Wants

The reviewer asks whether the paper's regime-dependent story reduces to a useless "it depends" answer. The challenge: if no single method dominates, does the practitioner need oracle knowledge of the regime, or can they run a cheap pre-test?

---

## Methods Compared

Four estimators are now included in the full comparison:

| Index | Method | Description |
|-------|--------|-------------|
| OML | Ortho ML | Standard first-order orthogonal (Double ML), always valid |
| HOML-known | Robust Ortho ML (Known) | Second-order with oracle moments |
| HOML-est | Robust Ortho ML (Est.) | Second-order with moments estimated from training fold |
| HOML-split | Robust Ortho ML (Split) | Second-order with nested split moment estimation |
| ICA | FastICA | Treatment effect via ICA on the joint residual |

**Key result**: OML dominates all three HOML variants on average across our grid. HOML is not a viable safe default.

---

## Evidence: Who Wins Where

### Setup

- **Data sources**: Fresh beta × n grid — gennorm(beta) × sample size, 30 experiments per cell (n=500/1000/2000/5000, beta=0.5/1.0/1.5/2.0/3.0).
- **Scripts**: `explorations/pareto/pareto_analysis.py`, results in `explorations/pareto/grid_results_v2.npy`.

### gennorm(beta) × n Grid — All Methods

| beta | excess kurtosis | OML | HOML-known | HOML-est | HOML-split | ICA | Winner |
|------|----------------|-----|------------|----------|------------|-----|--------|
| 0.5 | +22.2 (very heavy) | 0.053 | 0.079 | 0.079 | 0.080 | 0.055 | OML/ICA (tied, n-dep) |
| 0.5 | +22.2, n=2000 | 0.020 | 0.044 | 0.044 | 0.044 | **0.019** | ICA |
| 0.5 | +22.2, n=5000 | 0.013 | 0.027 | 0.027 | 0.027 | **0.011** | ICA |
| 1.0 | +3.0 (Laplace) | 0.026–0.046 | 0.035–0.123 | 0.034–0.104 | 0.034–0.103 | 0.012–0.046 | ICA (large n) / OML (small n) |
| 1.5 | +0.76 (mild) | 0.013–0.052 | 0.062–0.449 | 0.059–0.268 | 0.060–1.562 | 0.015–0.063 | OML |
| 2.0 | 0.0 (Gaussian) | **0.015–0.037** | 0.963–30.8 | 0.751–3.646 | 1.195–221.8 | 0.035–0.072 | OML |
| 3.0 | −0.58 (light) | **0.013–0.046** | 0.039–0.159 | 0.071–0.185 | 0.068–0.145 | 0.012–0.066 | OML/ICA |

**Key pattern**: ICA wins when beta < ~1.5 (excess kurtosis > 0, heavy-tailed eta) at large n. OML wins everywhere else. **HOML never wins any single cell** — OML consistently beats all HOML variants across the entire grid.

### Mean RMSE Summary (grid, 20 regimes)

| Strategy | Mean RMSE | vs Oracle |
|----------|-----------|-----------|
| Always HOML-split | 11.53 | +43200% |
| Always HOML-known | 1.86 | +6873% |
| Always HOML-est | 0.496 | +1758% |
| Always ICA | 0.0340 | +27.5% |
| **Always OML** | **0.0275** | **+2.8%** |
| **Kurtosis selector** | **0.0275** | **+3.1%** |
| Oracle (per-regime best) | 0.0267 | baseline |

---

## HOML Near-Gaussian Blow-up: Quantified

HOML's score denominator contains E[η³]/√Var[η³], which → 0 as η → Gaussian. The asymptotic variance diverges, and this translates directly to finite-sample RMSE explosion.

### Fine Sweep: beta → 2.0 (n=2000, 30 experiments)

| beta | excess kurtosis | OML | HOML-known | HOML-est | HOML-split | HOML-k/OML ratio |
|------|----------------|-----|------------|----------|------------|-----------------|
| 1.60 | +0.553 | 0.025 | 0.246 | 0.141 | 0.144 | **9.9×** |
| 1.70 | +0.379 | 0.022 | 0.310 | 1.675 | 0.993 | **14.1×** |
| 1.80 | +0.232 | 0.022 | 0.878 | 0.308 | 0.508 | **40.6×** |
| 1.90 | +0.108 | 0.025 | 4.391 | 1.039 | 0.971 | **176×** |
| 1.95 | +0.052 | 0.028 | 1.011 | 13.96 | 2.901 | **36×** |
| 2.00 | 0.000 | 0.023 | 0.657 | 8.129 | 2.019 | **29×** |
| 2.05 | −0.048 | 0.019 | 2.392 | 13.51 | 4.385 | **128×** |
| 2.10 | −0.093 | 0.023 | 10.40 | 11.99 | 1.105 | **445×** |
| 2.20 | −0.175 | 0.023 | 2.175 | 1.153 | 3.690 | **94×** |
| 2.40 | −0.312 | 0.019 | 0.138 | 0.148 | 0.124 | 7.4× |
| 2.60 | −0.420 | 0.018 | 0.078 | 0.071 | 0.063 | 4.4× |
| 3.00 | −0.582 | 0.030 | 0.074 | 0.075 | 0.071 | 2.5× |

**HOML blow-up is extreme and non-monotone.** The instability zone spans |excess kurtosis| < ~0.4 (roughly beta ∈ [1.6, 2.5]). Within this zone, HOML-known/OML ratios range from **4× to 841×** (mean ~150× across all near-Gaussian grid cells). OML RMSE stays below 0.037 everywhere.

All three HOML variants blow up — HOML-split is the worst (RMSE up to 222 at n=500), confirming that the instability is not an artifact of moment estimation; it is intrinsic to the second-order score denominator.

---

## Is There "No Free Lunch"?

**Yes, strictly**: no single method universally dominates. ICA loses badly when eta is Gaussian (up to OML winning by ~50% RMSE). OML loses to ICA at large n with very heavy-tailed eta (~16% at n=5000, beta=0.5).

However, the asymmetry is now clearer and more important:

- **HOML is dominated by OML everywhere in our grid.** Adding HOML to the oracle candidate set improves mean oracle RMSE by 0.00% — HOML wins zero cells.
- **The correct k-way comparison is ICA vs OML** (not ICA vs HOML). OML is the safe, always-valid baseline.
- **The oracle gap is small**: "always OML" is only **2.8%** worse than the per-regime best of {OML, ICA}.

---

## Oracle Envelope: Does HOML Help?

Oracle(+HOML) = Oracle(OML+ICA only) = **0.02668** mean RMSE — identical to 5 decimal places. HOML contributes nothing to the oracle because it never achieves the lowest RMSE in any (beta, n) cell. The practical frontier is spanned by {OML, ICA}.

---

## Feasible Selector

### Grid results (20 regimes: 5 beta × 4 n)

| Strategy | Mean RMSE | vs Oracle |
|----------|-----------|-----------|
| Always ICA | 0.03402 | +27.5% |
| Always OML | 0.02746 | +2.8% |
| **Kurtosis selector** | **0.02751** | **+3.1%** |
| Oracle (OML+ICA) | 0.02668 | baseline |

**Feasible kurtosis selector**: estimate excess kurtosis of the treatment residual `T - hat_m(X)`. Then:
- If excess kurtosis > 3.0: **use ICA**
- Otherwise: **use OML**

**Gap recovery: −5.4%** — the selector slightly *hurts* relative to "always OML" because it occasionally picks ICA in ambiguous regimes (beta=1.0–1.5). "Always OML" already recovers 97.2% of the oracle at negligible cost.

**Revised conclusion**: In this grid, the kurtosis selector does not improve over "always OML". The oracle gap is small enough (2.8%) that "always OML" is already near-optimal. The selector becomes worthwhile only when very heavy-tailed treatment noise (ek ≫ 3) is reliably present at large n, which the practitioner can verify by inspecting the residual kurtosis.

### Why HOML is Excluded from the Selector

The theoretical asymptotic variance of HOML ∝ Var[η³] / E[η³]², and E[η³] = 0 for any symmetric η (including gennorm). This makes HOML inapplicable to the entire gennorm family and to all symmetric discrete distributions (Rademacher, uniform). In practice, HOML blows up for |excess_kurtosis| < ~0.4. A practitioner cannot safely use HOML without strong prior knowledge that η is both asymmetric and substantially non-Gaussian.

---

## Binary Treatment: Separate Issue

For genuinely binary T (Bernoulli propensity model), ICA diverges (RMSE is O(100–1000×) larger than OML). This is model misspecification. **For binary T: always use OML/HOML.** ICA's non-Gaussianity assumption requires continuous treatment noise η with non-zero higher moments.

---

## Recommended Rebuttal Stance

**Honest position**: There is no single dominating method. The paper correctly identifies a regime-dependent tradeoff. However:

1. **HOML is not a competitive alternative.** All three HOML variants are dominated by OML across the entire gennorm × n grid. HOML-known/OML RMSE ratios reach **841×** at beta=2.0. The denominator instability (E[η³] → 0 for symmetric η) is an intrinsic limitation, not a finite-sample artifact.

2. **The practical frontier is {OML, ICA}.** "Always OML" is only **2.8%** worse than the per-regime best of {OML, ICA}. The oracle gap has shrunk compared to the prior analysis (was 4.1% when comparing ICA vs OML alone) because OML now includes the regime where ICA formerly won against HOML.

3. **The correct comparison is ICA vs OML**, not ICA vs HOML. HOML's asymptotic variance also diverges near Gaussian η. OML is the safe non-parametric baseline; ICA is the upgrade when non-Gaussianity is strong.

4. **The kurtosis selector is a "tie-break" optimization**, not a necessity. In this grid, "always OML" already recovers 97% of the oracle. The selector is useful when the practitioner knows they are in a heavy-tail regime (ek > 3) and wants the ~25% ICA advantage at large n.

5. **For binary T**: use OML. ICA's model assumptions are violated.

**One-sentence rebuttal**: "Adding all HOML variants to the oracle candidate set does not improve the oracle (HOML never wins any regime in our grid, with HOML-known/OML ratios reaching 841× near-Gaussian); the practical frontier is {OML, ICA}, and 'always OML' is within 2.8% of the per-regime oracle, with ICA worthwhile only when treatment residuals exhibit strong non-Gaussianity (estimated excess kurtosis > 3)."

---

## Files

- `explorations/pareto/pareto_analysis.py` — analysis script with importable functions
- `explorations/pareto/grid_results_v2.npy` — grid results with all HOML variants (5 beta × 4 n × 30 reps)
- `explorations/pareto/blowup_results.npy` — near-Gaussian blow-up sweep (12 beta values, n=2000, 30 reps)
- `explorations/pareto/grid_results.npy` — legacy cache (OML/HOML-known/ICA only, superseded)
- `tests/test_pareto.py` — 60 pytest tests for frontier/selector logic
