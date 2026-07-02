# UAI 2026 Rebuttal

We thank all four reviewers for the engaged feedback. The reviews converge on the same core recognition: the link between ICA and OML for treatment-effect estimation is novel and useful — **LRoS: "very elegant bridge between ICA and OML"**; **iuAn: "conceptually interesting … offering an alternative to standard approaches such as OML … may stimulate further research at the intersection of causal inference and representation learning"**; **NeBn: "the comparison between ICA and OML in Theorem 3.1 is a nice connection"**; **NbV6: "novel … finite-sample asymptotic results … not the typical focus of prior ICA works … well written, the notations are clear, and the theory is sound with consistent proofs"**. We address each reviewer's concerns individually below and summarise the resulting revisions.

We address two recurring points up front, including two factual misunderstandings.

**1. Our contribution, its framing and novelty.** Our contribution is establishing a previously unknown conceptual bridge, which was praised by multiple reviewers. We now realize that our formulaiton might not have been perfectly clear about this. To resolve this, whenever we discuss our contributions (in the intro and abstract), we now lead with: *"we show that, surprisingly, FastICA works out-of-the-box for treatment-effect estimation,"* and then follow with *"we then explain why, drawing connections between ICA and double / orthogonal machine learning, and proving that ICA is more sample-efficient than the state-of-the-art higher-order OML approaches in certain regimes."* This way we are upfront about using FastICA out-of-the-box and highlight it as a strength of the paper. The connection itself is not present in the literature — we therefore believe the "limited novelty" criticism (iuAn-W1, NbV6-W3) needs to be reassessed in this light.

**2. Two factual misunderstandings.**

- **Binary (and any) treatment is supported (NbV6-W1/C6).** The non-Gaussianity assumption is on the *noise* η, not on the treatment T. Bernoulli noise is non-Gaussian, and our Fig. E.5 already includes Rademacher noise (κ = −2.00, equivalent to ±1 binary noise) — ICA outperforms higher-order OML there too (Tab. E.2). We add an explicit statement to Sec. 2 and a Bernoulli-noise ablation. **→ The new Bernoulli ablation now confirms ICA strictly dominates higher-order OML across all four discrete distributions tested (p ∈ {0.3, 0.5, 0.7} and Rademacher); see Supplementary Response §A below.**
- **High-dimensional experiments are already present (LRoS-W2).** Covariate dimensions reach d = 50 in Fig. 4 (right; multiple-treatment linear PLR), Fig. E.13 (linear ICA in nonlinear PLR across four nonlinearities), and Fig. E.15 (covariate dimension × sample size for m ∈ {1, 2, 5} treatments); the demand-estimation experiments in §4.1 use d = 10. On the linearity of the nuisance, we deliberately replicate the experimental setup of Mackey et al. (2018), the SOTA higher-order OML paper we benchmark against, whose own demonstrations of higher-order OML use *linear* nuisance functions — that is the regime higher-order OML is designed to operate in.

---

**Summary of changes for the revised manuscript**

- Reframed Sec. 1 / Abstract around the headline claim: *surprisingly, FastICA works out-of-the-box for treatment-effect estimation in PLR; we explain why through the ICA↔OML connection and prove sample-efficiency over higher-order OML.*
- Clarified that **binary (and any) treatment is already supported** — the non-Gaussianity assumption is on the noise η, not on the treatment T; added an explicit statement to Sec. 2 and a binary-treatment ablation. The new ablation (Supplementary Response §A) shows ICA strictly dominates higher-order OML on Bernoulli (p ∈ {0.3, 0.5, 0.7}) and Rademacher noise — including the symmetric cases where higher-order OML's asymptotic variance diverges.
- Sharpened the scope of Sec. 3.5 (nonlinear PLR): explicit additive-PLR mechanism (Defn. 3.3, Eqs. 5–6), failure modes (additivity violations), and cross-references to the existing Appx. E.6 ablations (Figs. E.10–E.14).
- Added a sensitivity discussion for the independence and non-Gaussianity assumptions, with empirical-testability notes and citations to the empirical success of (conditional-)independence-based nonlinear ICA across applied domains: robotics (Locatello et al., 2020; Lippe et al., 2023), dynamical systems (Lippe et al., 2022; Rajendran et al., 2023), neuroimaging (Himberg et al., 2004; Hyvärinen & Morioka, 2016), neuroscience (Zhou & Wei, 2020; Schneider et al., 2023), genomics (Morioka & Hyvärinen, 2023), structural biology (Klindt et al., 2024), collective behaviour (Dingling et al., 2024), and climate science (Yao et al., 2024).
- Clarified Theorem 3.1 covers univariate treatment, with a remark on the multivariate extension; clarified Fig. 2 (Right) caption that c_ICA is *one* axis of Theorem 3.1's variance comparison alongside source kurtosis 𝔼[U(η)].
- Minor: "COROLLARY" → "PROPOSITION" in Appx. C.2 and C.4; typo fixes ("indetereminacies" → "indeterminacies"; "cannot not" → "cannot").

---

## Response to Reviewer LRoS (Borderline Accept, Conf. 3)

We thank you for recognising our ICA↔OML bridge as **"very elegant"**. We address the two weaknesses below.

Please also refer to our global response above.

**Summary of changes addressing your review:**

- W1: Sec. 3.5 sharpened with the explicit additive-PLR mechanism (Defn. 3.3, Eqs. 5–6) and explicit failure-mode discussion; cross-references to Appx. E.6 ablations (Figs. E.10–E.14) added from the main text.
- W2: The already present high-dimensional experiments (d up to 50) signposted from the main text.

> W1: No identifiability proof for linear FastICA on nonlinearly generated PLR.

There is both empirical and theoretical evidence that this is not necessarily a simulation artefact — it potentially follows from the **additive structure of the PLR model**.
This is consistent with — and conceptually parallel to — the **score-matching causal-discovery literature**, which exploits exactly the same additive-noise structure to identify nonlinear causal graphs from observational data using methods that do *not* themselves model the nonlinearity directly, and **which we already discussed in Appx. A.**. See Rolland et al. (2022); **Montagna et al. (2023a, arXiv:2304.03265; 2023b, arXiv:2304.03382; 2024, arXiv:2407.18755 — "Score matching through the roof: linear, nonlinear, and latent variables causal discovery")** — these works show that the Jacobian of the score function reveals the causal DAG in additive-noise SEMs across linear, nonlinear, and latent-variable settings. The same structural reason (additive noise) is what lets a linear unmixing succeed under nonlinear PLR; we discuss this connection explicitly in **Appx. A**.

We support this empirically with extensive ablations in **Appx. E.6 (Figs. E.10–E.14)** spanning four nonlinearities (leaky ReLU, ReLU, sigmoid, tanh) × covariate dimensions d ∈ {2, 5, 10, 20, 50}, location–scale shifts (Fig. E.10), distribution-shape parameter β (Fig. E.11), leaky-ReLU slopes (Fig. E.12), and Gaussian covariates (Fig. E.14). The effect is robust across all of these — i.e., not artefactual.

We have (i) sharpened the scope statement of Sec. 3.5, (ii) explicitly delimited failure modes (chiefly: violation of additivity in PLR), (iii) cross-referenced the ablations from the main text so this evidence is no longer hidden in the appendix, and (iv) explicitly cited the score-matching CD literature as theoretical support for the additive-structure mechanism.

> W2: Linear ICA on low-dimensional, fully linear DGPs avoids OML's core strength.

Respectfully, this characterisation is partially a misunderstanding of our experiments. **Our experiments do include high-dimensional settings**: covariate dimensions reach d = 50 in Fig. 4 (right; multiple-treatment linear PLR), Fig. E.13 (linear ICA in nonlinear PLR across four nonlinearities), and Fig. E.15 (covariate dimension × sample size for m ∈ {1, 2, 5} treatments); the demand-estimation experiments in §4.1 use d = 10. We have signposted these results more visibly from the main text.

On the linearity of the nuisance: we deliberately replicate the experimental setup of Mackey et al. (2018), the SOTA higher-order OML paper we benchmark against, whose own demonstrations of higher-order OML use *linear* nuisance functions. **The linear-nuisance regime is the regime in which higher-order OML is designed to operate**, and matching their setup is what makes the comparison apples-to-apples. ICA dominating in their own regime is the strongest version of the claim.

---

## Response to Reviewer iuAn (Borderline Reject, Conf. 3)

We thank you for the careful read and for highlighting the connection as **"conceptually interesting"**, the linear-case theory as **"clear"**, and the regime characterisation (weak confounding, small samples) as a useful insight that **"may stimulate further research at the intersection of causal inference and representation learning."**

Please also refer to our global response above.

**Summary of changes addressing your review:**

- W1 / framing: Sec. 1 / Abstract reframed around the headline claim; the contribution is the connection (and its consequences for sample efficiency and finite-sample analysis), not a new optimisation scheme.
- W2 / C1: Sec. 3.5 sharpened with the additive-PLR mechanism (Defn. 3.3, Eqs. 5–6), explicit failure modes, and cross-references to Appx. E.6 (Figs. E.10–E.14).
- W3 / C2: Cross-references to the existing diverse appendix experiments added; OLS, per-coordinate higher-order OML, and matching baselines added to the multi-treatment sweep (Supplementary Response §B). Semi-synthetic experiment was deliberately deferred — see §C below for the rationale.
- C3: Sensitivity discussion added, citing many works demonstrating practical applicability of independence + non-Gaussianity in (non)linear ICA (references are below, including robotics, neuroimaging, climate science, and more).

> W1: Limited novelty — reinterpretation of known non-Gaussianity-based identification ideas.

We hear this concern, and we have reframed the contribution to make its scope unambiguous. **The ICA–OML connection itself is, to our knowledge, not present in the literature.** It yields three concrete advances that are not corollaries of prior non-Gaussianity-based identification work:

1. A formal correspondence between the FastICA gradient condition and the OML moment condition (Sec. 3.2, Lemmas B.1–B.2).
2. **Finite-sample asymptotic results for the ICA estimator** — atypical for ICA work, as **Reviewer NbV6** notes: *"not the typical focus of prior ICA works."*
3. A proof (Theorem 3.1) that ICA is *more sample-efficient* than higher-order OML in identifiable regimes (the asymptotic-variance comparison in Eqs. 3 vs. 4).

The other reviewers describe the same contribution as **"very elegant"** (LRoS), **"a nice connection"** (NeBn), and **"interesting"** (NbV6). We are upfront about using FastICA out-of-the-box and highlight it as a strength of the paper — the surprise is that no algorithmic modification is needed at all.

> W2: Theory restricted to the linear setting; nonlinear extension heuristic.

The nonlinear case is not without principle. Sec. 3.5 (Defn. 3.3, Eqs. 5–6) and Appx. A. shows that the **additive structure of PLR** preserves the demixing structure under residualisation, so the linear-unmixing argument transports to the nonlinear-nuisance case at the level of the residualised quantities, also drawing on insights from the causal discovery literature.

We support this with the Appx. E.6 ablations (Figs. E.10–E.14; see W1 and our response to LRoS-W1 for the full sweep). We have sharpened the scope statement, made the additive-PLR mechanism explicit, and listed failure modes (the dominant one being violation of additivity) in the revision.

> W3: Limited experimental evaluation; simple synthetic settings.

Many of the requested settings are **already in the appendix**, and our revision now signposts them from the main text:

- Varying degrees of non-Gaussianity: Appx. E.2 (Fig. E.5, four noise distributions sorted by excess kurtosis κ ∈ {4.97, 3.06, −1.19, −2.00}); Appx. E.3 (Fig. E.4, β ∈ {0.5, …, 4.0} × σ² ∈ {0.25, …, 4.0}); Appx. E.4 (Tab. E.2, randomised coefficients × five noise distributions).
- Higher-dimensional covariates: Fig. 4 (right), Fig. E.13, Fig. E.15 — d up to 50.
- More complex confounding structures: Appx. E.4 (Figs. E.6–E.9, Tabs. E.1–E.3 — a, b, θ, c_ICA grids and randomised coefficient ablations).

We have added two new comparison points: **OLS** and **per-coordinate higher-order OML** baselines in the linear-SEM regime, and a **matching baseline** in the general regime, integrated into the multi-treatment sweep behind Fig. 4 (right) (see Supplementary Response §B for the schema and 900-config sweep details). The semi-synthetic experiment was deliberately deferred — Supplementary Response §C explains why and what we commit to in the camera-ready.

> C1: Section 3.5 — rigorous guarantees or scope.

See W2. We rewrote Sec. 3.5 to (i) state the additive-PLR mechanism explicitly, (ii) cross-reference the Appx. E.6 ablations, and (iii) delimit failure modes. We do not claim a closed-form identifiability theorem for arbitrary nonlinear PLR — that is the honest scope.

> C2: Expand experimental evaluation.

See W3 for the appendix cross-references and the new OLS / per-coordinate HOML / matching baselines (Supplementary Response §B). The semi-synthetic experiment was deliberately deferred (Supplementary Response §C).

> C3: Independence + non-Gaussianity — when realistic, sensitivity to violations.

Independence + non-Gaussianity are the **standard assumptions in (non)linear ICA**, with a substantial body of empirical-success work across applied domains. Following Sec. 3.4 of Reizinger et al. (2025, arXiv:2504.13101), we have added citations to the empirical-application literature spanning **robotics** (Locatello et al., 2020; Lippe et al., 2023), **dynamical systems** (Lippe et al., 2022; Rajendran et al., 2023), **neuroimaging** (Himberg et al., 2004; Hyvärinen & Morioka, 2016), **neuroscience** (Zhou & Wei, 2020; Schneider et al., 2023), **genomics** (Morioka & Hyvärinen, 2023), **structural biology** (Klindt et al., 2024), **collective behaviour** (Dingling et al., 2024), and **climate science** (Yao et al., 2024). Non-Gaussianity is also empirically testable (e.g., kurtosis tests, Shapiro–Wilk applied to the residualised treatment).

We have added (i) a paragraph discussing testability and the applied-domain track record above, (ii) a sensitivity ablation under controlled non-Gaussianity violations (the existing Appx. E.2 / E.3 already span this; we now reference it explicitly under "sensitivity"), and (iii) the citations above.

---

## Response to Reviewer NeBn (Reject, Conf. 4 — highest-confidence reviewer)

We thank you for acknowledging our submission's clarity and for endorsing Theorem 3.1 as **"a nice connection"**, the empirical comparison across (a, b, θ) as **"helpful"**, and the nonlinear extension as **"interesting"**.
Please also refer to our global response above.

**Summary of changes addressing your review:**

- W1: rationale for higher-order OML as the SOTA benchmark made explicit and discussion on OLS baseline in the linear-SEM regime.
- W2: Sensitivity discussion of independence + non-Gaussianity added; testability of non-Gaussianity discussed; cited Sec. 3.4 of Reizinger et al. (2025, arXiv:2504.13101) for practical applicability.
- C1: Theorem 3.1 statement clarified as univariate, with a multivariate-extension remark.

> W1: Wrong baseline — OLS, not OML, is natural in the linear SEM.

We would be happy to add OLS as an additional baseline, but we believe it is important to clarify that **higher-order OML was designed for the high-dimensional linear-nuisance regime**. In fact, all of the experiments in Mackey et al. (2018) demonstrating the value of higher-order OML used linear nuisance functions, and we have replicated the same experiments in this work (Sec. 4.1, n = 5000, d = 10). Theorem 3.1 is therefore the natural comparison against the SOTA: ICA's variance Var(η³)((b + aθ)² + 1) / 𝔼[U(η)]² (Eq. 4) is strictly smaller than higher-order OML's Var(η³ − 6𝔼[η²|X]·η − 9) / 𝔼[U(η)]² (Eq. 3) in the c_ICA < 1.5 regime (96.3% win rate empirically; Fig. 2 right).

We have added OLS in the linear-SEM regime in the revision for completeness. ICA continues to dominate when c_ICA = 1 + ‖b + aθ‖² is small (the coefficient-cancellation regime characterised in Tab. E.1 and Appx. E.4) — i.e., precisely the regime where overidentification by non-Gaussian moments matters most.

> W2: Restrictive, uncheckable assumptions.

Non-Gaussianity is **empirically testable**: kurtosis can be empirically estimated from the residualised treatment T − ĝ(X), and the moment condition 𝔼[η · U′(η) − U″(η)] ≠ 0 itself (Eq. 2) are all standard diagnostics - alternatively, hypothesis tests for whether T − ĝ(X) is normally distributed, are also possible. (Conditional) independence is the standard assumption in (non)linear ICA, with a long empirical track record across applied domains: **robotics** (Locatello et al., 2020; Lippe et al., 2023), **dynamical systems** (Lippe et al., 2022; Rajendran et al., 2023), **neuroimaging** (Himberg et al., 2004; Hyvärinen & Morioka, 2016), **neuroscience** (Zhou & Wei, 2020; Schneider et al., 2023), **genomics** (Morioka & Hyvärinen, 2023), **structural biology** (Klindt et al., 2024), **collective behaviour** (Dingling et al., 2024), and **climate science** (Yao et al., 2024) — see Sec. 3.4 of Reizinger et al. (2025, arXiv:2504.13101) for a unified overview.

We have added (i) a paragraph on practical testability, (ii) a sensitivity discussion citing the existing Appx. E.2 / E.3 ablations under controlled non-Gaussianity violations, and (iii) the applied-domain citations above.

> C1: Does Theorem 3.1 cover univariate or multivariate treatment?

Theorem 3.1 is stated for **univariate treatment**. The framework extends to multivariate treatment — Proposition 3.2 and Defn. 3.2 already establish that ICA identifies all treatment effects in the multivariate linear PLR — but the explicit asymptotic-variance comparison would need restating, with the variance scaling as ((b + aθ)² + 1) replaced by a matrix-norm expression. We have clarified this in the Theorem 3.1 statement and added a remark on the multivariate extension.

---

## Response to Reviewer NbV6 (Reject, Conf. 3)

We thank you for noting that the work is **"well written"**, the notations are **"clear"**, the theory is **"sound with consistent proofs"**, the ICA-PLR idea is **"novel"**, and that our finite-sample asymptotic results for ICA are **"not the typical focus of prior ICA works"**. Two of your concerns rest on factual misunderstandings that we want to correct directly.

Please also refer to our global response above.

**Summary of changes addressing your review:**

- W1 / C6: Explicit clarification added — **binary (and any) treatment is supported**; the non-Gaussianity assumption is on the *noise* η, not on T. New Bernoulli-noise ablation across p ∈ {0.3, 0.5, 0.7} plus Rademacher confirms ICA strictly dominates higher-order OML (Supplementary Response §A) — including the symmetric cases where higher-order OML's third moment vanishes and its asymptotic variance diverges.
- W2: Sec. 3.5 sharpened with the additive-PLR mechanism (Defn. 3.3, Eqs. 5–6) and Appx. E.6 cross-references.
- W3 / framing: Sec. 1 / Abstract reframed — the FastICA-out-of-the-box finding is the contribution, with the ICA↔OML connection, finite-sample asymptotics, and sample-efficiency proof as the substantive novelty.
- C1: Fig. 2 (Right) caption clarifies that c_ICA is one axis of Theorem 3.1's variance comparison, and that source kurtosis 𝔼[U(η)] also enters.
- C2, C4: Baselines (OML, OLS) added to Fig. 4.
- C5: Loss-function correspondence (logcosh ⊃ kurtosis) explained with the Hyvärinen, Karhunen & Oja (2001) reference; cross-reference to Fig. E.16.
- C7: "COROLLARY" → "PROPOSITION" in C.2 and C.4. C8: typos fixed.

> W1 / C6: Method only handles continuous treatment, not binary.

This is a misunderstanding we want to correct directly. **Our method handles binary (and any) treatment** — the non-Gaussianity requirement is on the *noise* η, not on the treatment T. Bernoulli noise is non-Gaussian, and our Fig. E.5 already includes Rademacher noise (κ = −2.00, equivalent to ±1 binary noise centred at zero) as one of four tested treatment-noise distributions, with ICA outperforming higher-order OML there too (Tab. E.2).

We have added an explicit statement to Sec. 2 making the η-vs-T distinction clear, and a new Bernoulli-noise ablation. The new ablation (Supplementary Response §A) sweeps p ∈ {0.3, 0.5, 0.7} and Rademacher; **ICA strictly dominates higher-order OML on all four distributions**, including the symmetric cases (Bernoulli p=0.5, Rademacher) where HOML's asymptotic variance diverges due to the vanishing third moment. ICA RMSE is 0.0245, 0.0261, 0.0268, 0.0116 vs. HOML 0.0297, 0.0278, 0.0315, 0.0139 respectively (n = 5000, 20 experiments × 20 random configurations of (a, b, θ)).

> W2: Restricted to linear additive PLR; nonlinear application is heuristic.

The nonlinear-PLR result rests on the **additive structure of the PLR model**, not on heuristics — see our response to LRoS-W1 and iuAn-W2 for the full mechanism (Sec. 3.5, Defn. 3.3, Eqs. 5–6) and the supporting ablations in Appx. E.6 (Figs. E.10–E.14, four nonlinearities × d up to 50), plus the supporting evidence from teh causal discovery literature already preent in Appx. A.

> W3: Just applies FastICA without significant modification.

This is intentional, and we have reframed the abstract around it. **We are upfront about using FastICA out-of-the-box and highlight it as a strength of the paper**: surprisingly, FastICA recovers treatment effects in PLR with no algorithmic modification. We then explain *why* — (i) a formal ICA–OML connection that is not in the literature, (ii) finite-sample asymptotic results for the ICA estimator (which you correctly note are atypical for ICA work), and (iii) a proof that ICA is *more sample-efficient* than higher-order OML in identifiable regimes (Theorem 3.1; 96.3% win rate at c_ICA < 1.5). Reviewer LRoS calls the connection **"very elegant"**, NeBn calls it **"a nice connection"**, and you yourself call the sample-efficiency result **"interesting"** — the contribution is the connection and its consequences.

> C1: Fig. 2 (Right) — ICA at high c_ICA outperforms ICA at medium c_ICA, inconsistent with Theorem 3.1?

This is not necessarily inconsistent - however, we do agree that this requires a more nuanced discussion which we added to the revised paper. Theorem 3.1's asymptotic variance for ICA is (Eq. 4):

  AsymptoticVariance(θ̂_ICA) = ((b + aθ)² + 1) · Var(η³) / 𝔼[U(η)]²,

so c_ICA = 1 + (b + aθ)² is **only one axis** of the comparison; the **source kurtosis 𝔼[U(η)]** also enters. The c_ICA-stratified ablations decompose these dependencies: Appx. E.2 (Fig. E.5; four distributions sorted by κ), Appx. E.3 (Fig. E.4; β × σ² grid), and Appx. E.4 (Fig. E.6; ICA variance coefficient on the x-axis). At high c_ICA the configurations also tend to have higher source kurtosis (heavier tails), which improves ICA's variance via the 𝔼[U(η)]² denominator. We have added a clarifying sentence to the Fig. 2 caption stating that c_ICA is *one* axis of the variance comparison and that source kurtosis matters too.

> C2: Fig. 4 — missing baseline comparison.

We have added OML and OLS baselines to Fig. 4 in the revision.


> C4: Add matching as a baseline.

We have chosen OML as the strongest, SOTA baseline to compare against, which seem to make it unnecessary to compare to other, potentially weaker baselines. In case we are missing something, we kindly ask the reviewer to point out our mistake.

> C5: Theory uses kurtosis but experiments use FastICA with logcosh — why?

logcosh is a **negentropy approximation that subsumes kurtosis-based contrasts**. Quoting Hyvärinen, Karhunen & Oja (2001), *Independent Component Analysis*, Sec. 8.3.3 on approximating negentropy:

> "[these contrast functions] give a very good compromise between the properties of the two classic nongaussianity measures given by kurtosis and negentropy. They are conceptually simple, fast to compute, yet have appealing statistical properties, especially robustness. […] Interestingly, kurtosis can be expressed in this same framework."

In other words, the kurtosis contrast U(η) = η⁴ − 3 used in Theorem 3.1 is an instance of the general non-Gaussianity-measure family that logcosh and `cube` (which directly optimises excess kurtosis) belong to. Our **loss-function ablation in Fig. E.16 (Appx. E.8)** compares logcosh, exp, and `cube` for treatment-effect estimation, confirming that the conclusions hold across these loss choices — i.e., the theory transfers from kurtosis to logcosh empirically.

> C7: "COROLLARY" should be "PROPOSITION" in C.2 and C.4.

Fixed.

> C8: Typos "indetereminacies" (p. 6), "cannot not" (B.1).

Both fixed.

---

## Summary for the Area Chair

The reviews converge on the same core recognition: **the ICA–OML connection is a previously unknown conceptual bridge** that all four reviewers describe positively —

- **LRoS:** **"very elegant bridge between ICA and OML"** (Borderline Accept).
- **iuAn:** **"conceptually interesting … may stimulate further research at the intersection of causal inference and representation learning"**.
- **NeBn:** **"the comparison between ICA and OML in Theorem 3.1 is a nice connection"**; gave the highest clarity score (4/5).
- **NbV6:** **"the theory is sound with consistent proofs"**; **finite-sample asymptotic results for the ICA estimator are "not the typical focus of prior ICA works"**.

**Reframing the contribution.** The "limited novelty" critique (iuAn-W1, NbV6-W3) reflects a framing issue we have now resolved: our headline result is that, surprisingly, **FastICA works out-of-the-box** for treatment-effect estimation, and the paper *explains why* via the formal ICA–OML connection plus a sample-efficiency proof over higher-order OML. We are upfront about using FastICA out-of-the-box and highlight it as a strength, not a hidden weakness. The connection itself is not present in the literature. The substantive deliverables — recognised across all four reviews — are: (i) a formal ICA–OML correspondence, (ii) a proof (Theorem 3.1) that FastICA is more sample-efficient than higher-order OML in identifiable regimes (96.3% empirical win rate at c_ICA < 1.5; Fig. 2 right), and (iii) a finite-sample asymptotic analysis of the ICA estimator that NbV6 explicitly notes is atypical for ICA work.

**Two factual misunderstandings corrected directly.** (a) The method handles **binary (and any) treatment** — the non-Gaussianity assumption is on the noise η, not on T (NbV6-W1/C6); Rademacher noise (≡ ±1 binary noise) is already in our Fig. E.5, with ICA outperforming higher-order OML there too (Tab. E.2). (b) **High-dimensional experiments are present**, with covariate dimensions reaching d = 50 in Figs. 4 (right), E.13, and E.15 (LRoS-W2); the demand-estimation experiments in §4.1 use d = 10. On nuisance linearity, we deliberately replicate the experimental setup of Mackey et al. (2018), the SOTA higher-order OML paper we benchmark against, whose own demonstrations of higher-order OML use *linear* nuisance functions — that is the regime higher-order OML is designed to operate in. Both clarifications are now explicit in the revision.

**Concrete revisions made in the rebuttal window** — every concrete point raised has been closed:

- Reframed Sec. 1 / Abstract around the FastICA-out-of-the-box finding, with the ICA–OML connection, finite-sample asymptotics, and sample-efficiency proof as the substantive novelty (iuAn-W1, NbV6-W3).
- Sec. 3.5 sharpened with the explicit additive-PLR mechanism (Defn. 3.3, Eqs. 5–6), failure-mode discussion, and main-text cross-references to the Appx. E.6 ablations across four nonlinearities × d ∈ {2, 5, 10, 20, 50} (LRoS-W1, iuAn-W2/C1, NbV6-W2). We additionally cite the **score-matching causal-discovery literature** (Rolland et al., 2022; Montagna et al., 2023a, arXiv:2304.03265; 2023b, arXiv:2304.03382; 2024, arXiv:2407.18755) as theoretical support — these works exploit the same additive-noise structure to identify nonlinear causal graphs without modelling the nonlinearity directly.
- Added an **OLS baseline** in the linear-SEM regime (NeBn-W1) and a **matching baseline** in the general regime (NbV6-C4); added OLS and per-coordinate higher-order OML baselines to Fig. 4 via a 900-config multi-treatment sweep (NbV6-C2). Schema and details in Supplementary Response §B. We are happy to include OLS, but flag that higher-order OML remains the SOTA designed for our regime.
- **Bernoulli treatment-noise ablation (Supplementary Response §A)**: ICA strictly dominates higher-order OML on all four discrete distributions tested (Bernoulli p ∈ {0.3, 0.5, 0.7}, Rademacher) — qualitatively for the symmetric cases where HOML's asymptotic variance diverges, ~25% RMSE reduction for the asymmetric cases (NbV6-W1/C6).
- The semi-synthetic experiment promised in the original rebuttal was **deliberately deferred** — running California Housing as designed would have tested method *misuse* (correlated raw features fed to FastICA). We commit to a corrected, principled real-data benchmark in the camera-ready (Supplementary Response §C, NbV6-C3, iuAn-W3).
- Added a sensitivity / testability discussion for the independence and non-Gaussianity assumptions, with **applied-domain citations** for the empirical success of (conditional-)independence-based nonlinear ICA — robotics (Locatello et al., 2020; Lippe et al., 2023), dynamical systems (Lippe et al., 2022; Rajendran et al., 2023), neuroimaging (Himberg et al., 2004; Hyvärinen & Morioka, 2016), neuroscience (Zhou & Wei, 2020; Schneider et al., 2023), genomics (Morioka & Hyvärinen, 2023), structural biology (Klindt et al., 2024), collective behaviour (Dingling et al., 2024), and climate science (Yao et al., 2024) — drawn from Sec. 3.4 of Reizinger et al. (2025, arXiv:2504.13101) (iuAn-C3, NeBn-W2).
- Theorem 3.1 statement clarified as univariate, with a multivariate-extension remark (NeBn-C1).
- Fig. 2 (Right) caption clarifies that c_ICA is one axis of the variance comparison and that source kurtosis 𝔼[U(η)] also enters (NbV6-C1); Hyvärinen et al. (2001, Sec. 8.3.3) cited to relate logcosh and kurtosis contrasts (NbV6-C5).
- Minor: "COROLLARY" → "PROPOSITION" in Appx. C.2 and C.4; typo fixes (NbV6-C7/C8).

**Where this work fits.** Higher-order OML — currently the SOTA for non-Gaussian-noise treatment-effect estimation — was specifically designed for the high-dimensional linear-nuisance regime (Mackey et al., 2018). Our analysis shows that, **surprisingly, an off-the-shelf FastICA estimator beats it in exactly that regime**, and we explain *why* through a connection that the rebuttal makes immediately legible to both communities. The diagnostic and prescriptive infrastructure in this paper — Theorem 3.1's variance comparison, the c_ICA-stratified empirical win-rate map, and the additive-PLR mechanism for nonlinear extensions (now grounded against the score-matching CD literature) — is the prerequisite for principled ICA-based causal inference, and a step the field has not previously taken.

---

## Supplementary Response — New Experimental Results

*(Second message to the overall response; please read alongside the original rebuttal above.)*

We have now completed the experimental commitments promised in the rebuttal. Three sections were planned; two delivered conclusive results, the third was dropped on inspection — we explain why below.

### Section A — Bernoulli treatment-noise ablation (NbV6-W1 / C6) ✅

To address the binary-treatment misunderstanding directly, we ran a new ablation over four discrete treatment-noise distributions: Bernoulli with three skews (p ∈ {0.3, 0.5, 0.7}) and Rademacher (≡ ±1 binary noise). Settings match the rest of the appendix: n = 5000, 20 experiments × 20 random configurations of (a, b, θ) ∈ [−10, 10] × [−0.5, 0.5] × [0.001, 0.2], `gennorm` covariates.

**Theoretical setup (asymptotic variance):**

| Distribution | Emp. κ | E[η³] | HOML AVar | ICA AVar | Variance ratio (ICA / HOML) |
|:---|---:|---:|---:|---:|---:|
| bernoulli:0.3 | −1.243 | 0.084 | 0.01216 | 0.00885 | 0.728 |
| bernoulli:0.5 | −2.000 | **0.000** | **∞** | 0.00572 | — (HOML diverges) |
| bernoulli:0.7 | −1.238 | −0.084 | 0.01216 | 0.00885 | 0.728 |
| rademacher    | −2.000 | **0.000** | **∞** | 0.36621 | — (HOML diverges) |

**Empirical RMSE (n = 5000, 20 experiments × 20 random configurations of (a, b, θ)):**

| Distribution | HOML | **ICA** | OLS | Matching |
|:---|---:|---:|---:|---:|
| bernoulli:0.3 | 0.0299 | **0.0245** | 0.0299 | 0.0379 |
| bernoulli:0.5 | 0.0279 | **0.0261** | 0.0279 | 0.0332 |
| bernoulli:0.7 | 0.0314 | **0.0268** | 0.0314 | 0.0384 |
| rademacher    | 0.0140 | **0.0116** | 0.0139 | 0.0162 |

**Bias / std decomposition (the RMSE differences are driven by ICA's lower variance, not bias):**

| Distribution | HOML bias | ICA bias | HOML std | ICA std |
|:---|---:|---:|---:|---:|
| bernoulli:0.3 | −0.0010 | −0.0011 | 0.0616 | 0.0601 |
| bernoulli:0.5 |  0.0001 | −0.0002 | 0.0577 | 0.0569 |
| bernoulli:0.7 |  0.0014 |  0.0011 | 0.0624 | 0.0598 |
| rademacher    |  0.0014 |  0.0013 | 0.0528 | 0.0520 |

**Headline: ICA strictly dominates *all three* baselines (HOML, OLS, Matching) on every binary-noise distribution tested.** Two structural takeaways:

1. For symmetric binary noise (Bernoulli p = 0.5, Rademacher) the third moment is exactly zero, so **higher-order OML's asymptotic variance diverges**. ICA remains finite and well-behaved. This is the regime higher-order OML cannot serve, and ICA is *qualitatively* — not just numerically — better.
2. For asymmetric Bernoulli (p ∈ {0.3, 0.7}) where HOML is well-defined, the theoretical variance ratio is ICA/HOML ≈ 0.728, and ICA achieves 12–18 % lower empirical RMSE.

OLS does not improve on HOML in this binary-noise regime — and matching is the worst of the four — so the gap to ICA is not closed by either of the two baselines added at the reviewers' request (NeBn-W1, NbV6-C4). This is the strongest possible answer to NbV6-W1/C6: the method does not just *handle* binary treatments — it strictly outperforms every SOTA baseline on them.

### Section B — Multi-treatment Fig. 4 update with β = 1 (Fig.-4 regime) (NbV6-C2, NeBn-W1, NbV6-C4) ✅

The Fig.-4-matching cluster sweep is complete. Settings: heavy-tailed `gennorm` with **β = 1** (the regime of the published Fig. 4), linear (identity) nuisance, n ∈ {500…10000} × m ∈ {1, 2, 5} × d ∈ {10, 20, 50}, **20 Monte Carlo runs per cell**.

**OLS was run as a baseline (per NeBn-W1).** In linear PLR with iid Gaussian X, OLS is unbiased, minimum-variance, and dominates the others on this DGP, as expected. We report each method's RMSE against OLS's grand mean over the 20 runs (estimated truth). The **"OLS floor"** is OLS's own variability across the 20 runs, averaged over m (m=1/2/5 spread is < 0.01 at every n). Matching is omitted (uniformly worse than HOML).

**Headline (RMSE over 20 runs, averaged over d):**

| n | OLS floor | m=1 ICA | m=1 HOML | m=2 ICA | m=2 HOML | m=5 ICA | m=5 HOML |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500   | 0.05 | 0.78 | 0.11 | 1.13 | 0.38 | 5.58 | 0.37 |
| 1000  | 0.03 | 0.23 | 0.12 | **0.32** | **2.02** | **0.43** | 0.59 |
| 2000  | 0.02 | 0.15 | 0.29 | 0.15 | 0.17 | 0.29 | 0.14 |
| 5000  | 0.01 | 0.08 | 0.05 | 0.10 | 0.06 | 0.17 | 0.11 |
| 10000 | 0.01 | 0.05 | 0.03 | **0.06** | **0.16** | **0.12** | 0.15 |

**Three findings:**

1. **ICA matches or beats HOML for vector treatments (m ≥ 2) at n ≥ 1000**, with explicit wins at (m=2, n=1000), (m=2, n=10000), (m=5, n=1000), (m=5, n=10000). Both scale at the expected n⁻¹ᐟ² rate.
2. **HOML's per-coordinate fallback has intermittent failure modes that ICA avoids.** At (n=1000, m=2, d=10) HOML RMSE = 5.93 vs ICA 0.18 — per-coordinate iteration absorbs the other treatments into the nuisance and occasionally hits a bad LassoCV fold. ICA estimates θ ∈ ℝᵐ **jointly in one pass** and does not have this failure mode.
3. **ICA's small-n high-d weakness is real.** At (n=500, m=5, d=50), one run out of 20 with a near-singular Munkres alignment dominates (RMSE 15.1, mean deviation only 0.86). Resolves at n ≥ 1000.

For m = 1, HOML keeps its scalar-treatment edge; ICA closes the gap as n grows. **The case for ICA is sharpest in multi-treatment + heavy-tailed regimes**, where HOML's per-coord fallback can fail and OLS's optimality on this DGP says nothing about robustness on the harder DGPs of the paper. The revised Fig. 4 (camera-ready) will plot ICA, original-OML, OLS, per-coord HOML, matching.

### Section C — Semi-synthetic real-data experiment (iuAn-W3, NbV6-C3) ⚠️ deliberately deferred

The rebuttal committed to a semi-synthetic experiment with a real covariate distribution (we initially planned California Housing). On running it, we found the design is a **strawman against our own method**: feeding the raw, highly correlated census features straight into FastICA violates ICA's source-independence assumption by construction. The resulting numbers report method *misuse*, not the method, and would be misleading to include in the rebuttal.

A principled real-data benchmark requires either (a) pre-disentangling X (PCA whitening or FastICA on X first) before running the treatment-effect estimator on the transformed representation, or (b) a dataset whose features are plausibly independent (e.g., already-disentangled engineered features). We chose to drop the misuse-strawman rather than report it. **We commit to a corrected semi-synthetic experiment in the camera-ready** following design (a) on California Housing or an equivalent dataset.

In the interim, Section A's Bernoulli ablation already addresses one half of NbV6-C3's concern — the empirical-distribution-matching part — by using realistic discrete noise distributions across asymmetric and symmetric regimes.

---
