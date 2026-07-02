# UAI 2026 — Submission 897 Review Summary

**Legend:** 🟢 Strengths · 🔴 Weaknesses · 🔵 TODOs

## Score Overview

| Reviewer | Rating | Conf. | Novelty | Correct. | Evidence | Reprod. | Clarity |
|----------|--------|-------|---------|----------|----------|---------|---------|
| LRoS | 5 — Borderline Accept | 3 | 3 | 2 | 2 | 2 | 2 |
| iuAn | 4 — Borderline Reject | 3 | 2 | 3 | 2 | 3 | 3 |
| NeBn | 3 — Reject | **4** | 2 | 3 | 3 | 3 | 4 |
| NbV6 | 3 — Reject | 3 | 2 | 2 | 2 | 2 | 3 |

Mean rating 3.75. Two borderlines + two rejects. Highest-confidence reviewer (NeBn) recommends Reject.


Our rebuttal overall message/framing strategy is: My idea is to, whenever we discuss our contributions (for example in the intro and abstract), lead with a statement like “we show that, surprisingly, FastICA works out of the box for treatment effect estimation” and then follow with  “then we explain why, drawing connections between ICA and double machine learning / orthogonal machine learning and proving that ICA is more sample efficient than the state-of-the-art higher-order OML approaches in certain regimes.”  This way we’re being upfront about the fact that we’re using fastica out of the box and highlighting it as a strength of the paper.
---

## Cross-Cutting Themes

### 🟢 Recurring Strengths
- **Elegant ICA↔OML bridge** — all four reviewers.
- **Linear-case theory** is clear; finite-sample asymptotics for ICA estimator atypical for ICA literature (NbV6).
- **Regime identification** where ICA beats OML (weak confounding, small samples, higher-order vs. ICA).
- **Computational simplicity** — TE falls out of the unmixing matrix.
- **Writing/notation** quality (NbV6, NeBn).

### 🔴 Recurring Weaknesses
- **Nonlinear PLR is heuristic** (LRoS, iuAn, NbV6) — no identifiability proof; success may be a simulation artifact. -> we did a lot of ablations and also highlighted theoretical results suggesting that the additive PLR model is a contributing factor
- **Wrong / missing baselines** — OLS in linear SEM (NeBn); matching (NbV6); missing baseline in Fig. 4 (NbV6). -> Response: We would be happy to add OLS as an additional baseline, but we believe it is important to clarify that higher-order OML was designed for high-dimensional linear nuisance. In fact, all of the experiments in \citet{mackeyetal} demonstrating the value of higher-order OML used linear nuisance functions, and we have replicated the same experiments in this work.

- **Limited novelty** (iuAn, NbV6) — reads as direct FastICA application without algorithmic modification.
- **Restrictive, uncheckable assumptions** (NeBn, iuAn, NbV6).
- **Synthetic-only experiments** (iuAn, NbV6); low-dim avoids OML's regime (LRoS).
- **Continuous treatment only** (NbV6).

### 🔵 Consolidated TODO List
- [ ] Identifiability for nonlinear PLR, **or** explicit scope/failure modes (LRoS, iuAn).
- [ ] OLS analogue of Theorem 3.1 in linear SEM (NeBn).
- [ ] State whether Theorem 3.1 covers multivariate treatment (NeBn).
- [ ] Add baselines: OLS, matching, baseline for Fig. 4 (NeBn, NbV6).
- [ ] More diverse experiments: higher-dim, varying non-Gaussianity, complex confounding, ML nuisances (iuAn, LRoS). -> many of these are already addressed, see appendix
- [ ] Real or semi-synthetic data (NbV6, iuAn).
- [ ] Sensitivity analysis to assumption violations (iuAn, NeBn, NbV6).
- [ ] Reconcile FastICA-logcosh vs. kurtosis-based theory (NbV6).
- [ ] Explain Fig. 2 (Right) high-vs-medium c_ICA inconsistency with Thm. 3.1 (NbV6). p> not encessarily inconsistent, but we added nuanced explanation. Thm 3.1. has factors beyond c_ICA (kurtosis), which also affects teh relationship, also see appdx ablations
- [ ] Discuss extension to binary treatment (NbV6).
- [ ] Discuss when independence + non-Gaussianity are realistic (iuAn). -> common even in nonlienar ICA and has many applications (cite from 3.4 in https://arxiv.org/pdf/2504.13101)
- [ ] Minor: "COROLLARY"→"PROPOSITION" in C.2/C.4; typos "indetereminacies", "cannot not" (NbV6).

---

## Per-Reviewer Breakdown

### Reviewer LRoS — Rating 5 (Borderline Accept), Conf. 3
*Novelty 3, Correct. 2, Evidence 2, Reprod. 2, Clarity 2.*

**Summary:** ICA for TE estimation in PLR via shared non-Gaussian moment conditions with OML.

🟢 **Strengths**
- Very elegant ICA↔OML bridge for TE estimation.

🔴 **Weaknesses**
- Linear FastICA on nonlinearly-generated PLR data: no rigorous identifiability proof; empirical success may be a simulation artifact.
- Linear ICA on low-dim, fully linear DGPs avoids OML's core advantage (flexible high-dim nuisances). -> Response: Clarify that we demonstrated improvement on high-dimensional problems as well with references to paper figures


🔵 **Implied TODOs**
- Rigorous identifiability for linear-unmixing-on-nonlinear-PLR.
- High-dim experiments with flexible ML nuisances.

**Justification:** "New perspective, but fails to adequately support it."

---

### Reviewer iuAn — Rating 4 (Borderline Reject), Conf. 3
*Novelty 2, Correct. 3, Evidence 2, Reprod. 3, Clarity 3.*

**Summary:** ICA recovers TE from unmixing matrix under non-Gaussian noise; clarifies link to OML; provides theory and empirics.

🟢 **Strengths**
- Conceptually interesting connection ICA↔TE.
- Simple, computationally efficient method.
- Clear linear-case theory; useful insight on non-Gaussianity and ICA↔OML.
- Identifies regimes (weak confounding, small samples) where ICA is competitive.
- May stimulate causal-inference / rep-learning research.

🔴 **Weaknesses**
- Limited novelty — reinterpretation of existing non-Gaussian ID ideas. -> this is the contribution, limited novelty is a relative assessment, as you also acknowledged the interesting connection (See also other reviewes)
- Theory restricted to linear; nonlinear extension heuristic.
- Limited experiments — simple synthetic, no realistic benchmarks.

🔵 **TODOs**
- Section 3.5: rigorous guarantees, or explicit scope and failure modes.
- Vary non-Gaussianity, raise dimensionality, use complex confounding, more baselines.
- Discuss when independence + non-Gaussianity are realistic; sensitivity to violations.

**Justification:** Falls short due to limited novelty, missing nonlinear theory, limited experiments.

---

### Reviewer NeBn — Rating 3 (Reject), Conf. 4 *(highest)*
*Novelty 2, Correct. 3, Evidence 3, Reprod. 3, Clarity 4.*

**Summary:** ICA for TE estimation under confounding; in linear SEM ICA can beat OML; nonlinear extension empirical only.

🟢 **Strengths**
- Theorem 3.1 (ICA vs. OML) is a nice connection.
- Empirical comparison across a, b, … is helpful.
- Nonlinear extension interesting.

🔴 **Weaknesses**
- OML's strength is complex nuisances; in linear SEM the natural baseline is **OLS**, not OML.
- Required assumptions are quite restrictive and uncheckable from data.

🔵 **TODOs / Comments**
- Provide an OLS-based analogue of Theorem 3.1.
- Try to relax the strong assumptions; provide empirical / sensitivity tests.
- Clarify whether Theorem 3.1 covers univariate or multivariate treatment.

**Justification:** Unclear whether ICA empirically beats OLS; theoretical guarantees too strong / uncheckable for practitioners.

---

### Reviewer NbV6 — Rating 3 (Reject), Conf. 3
*Novelty 2, Correct. 2, Evidence 2, Reprod. 2, Clarity 3.*

**Summary:** Underexplored ICA↔TE connection; ICA indeterminacy fixed by causal graph; finite-sample asymptotics. Main contribution: bridging source separation and effect modification.

🟢 **Strengths**
- Using ICA demixing for causal effect estimation in PLR is novel.
- Finite-sample asymptotic results for ICA — atypical for prior ICA work.
- ICA can be more sample-efficient than higher-order OML — interesting.
- Well written, clear notation, sound theory with consistent proofs.

🔴 **Weaknesses**
- Method handles continuous, not the more common binary, treatment. ->This seems like a misunderstanding that can be corrected (as we do handle binary and any other form of treatment).

- Restricted to linear additive PLR; nonlinear application heuristic.
- Novelty: applies FastICA without significant modification.

🔵 **TODOs / Comments**
- Fig. 2 (Right): explain high vs. medium c_ICA inconsistency with Thm. 3.1.
- Fig. 4: add baseline for linear ICA on nonlinear PLR.
- Add real / semi-synthetic data experiments.
- Add other causal estimators as baselines (e.g., matching).
- Theory uses kurtosis but experiments use FastICA with logcosh — justify or rerun. -> See Hyvarinen's ICA book, 8.3.3 Approximating negentropy :"Thus we obtain approximations of negentropy that give a very good compromise between the properties of the two classic nongaussianity measures given by kurtosis and negentropy. They are conceptually simple, fast to compute, yet have appealing statistical properties, especially robustness. Therefore, we shall use these objective functions in our ICA methods. Interestingly, kurtosis can be expressed in this same framework" + also our loss function ablation in fig TBD in appdx
- Discuss extension to binary treatment. -> note extension, see comment above
- Minor: "COROLLARY"→"PROPOSITION" in C.2 and C.4.
- Typos: "indetereminacies" (p. 6), "cannot not" (Appendix B.1).

**Justification:** Limited novelty, no binary-treatment support, lack of real applications.

---

## Suggested Rebuttal Priorities

1. 🔴 Defend / scope the nonlinear-PLR claim (3 reviewers).
2. 🔴 Add OLS baseline (highest-confidence reviewer) and matching (NbV6).
3. 🔴 Reframe novelty — emphasize finite-sample asymptotics, OML connection, regime characterization.
4. 🔴 Sensitivity analysis to non-Gaussianity / independence violations.
5. 🔴 Clarify Fig. 2 (Right) anomaly and Fig. 4 missing baseline.
6. 🔴 Justify logcosh-vs-kurtosis (or rerun with kurtosis).
7. 🔴 Address binary-treatment obstruction (continuous-noise requirement).
8. Fix minor typos and COROLLARY→PROPOSITION labels.
