# Nonlinear Theory — Feasibility Memo

**Context.** TMLR rebuttal exploration for "Estimating Treatment Effects with Independent Component Analysis" ([arXiv:2507.16467](https://arxiv.org/abs/2507.16467)). A reviewer is expected to ask: *the partially linear model (PLR) is restrictive — does the ICA–OML connection extend to nonlinear treatment effects or a nonlinear structural model?* This memo answers as both theory reviewer and author planning the next paper. Every cited paper has been verified to exist (arXiv IDs given inline).

**Headline verdict.** "Nonlinear theory" is **future work, not a rebuttal deliverable.** A correct, novel theorem for any genuinely nonlinear extension is a paper in itself, not a two-week rebuttal addendum. The *one* extension that is plausibly within reach and worth flagging is the **varying-coefficient PLR**, $Y = \theta(X)\,T + g(X) + \varepsilon$, because it keeps the latent mixing *linear conditional on $X$* — the only structure the current proof actually needs. Everything beyond that (general $\phi(T,X)$, nonlinear mixing of the sources) collides head-on with the nonlinear-ICA non-identifiability wall and should be framed honestly as an open direction, not promised.

---

## 1. What exactly is linear, and where each linearity is used

The model is the PLR of Robinson (1988):
$$
Y = \theta^\top T + g(X) + \varepsilon, \qquad T = m(X) + \eta,
$$
with $\varepsilon \perp \eta \perp X$, $\mathbb{E}[\varepsilon\mid X]=\mathbb{E}[\eta\mid X]=0$. Both nuisances $g,m$ may be arbitrarily nonlinear; the *causal effect* $\theta$ is a **constant** entering **linearly**.

The ICA argument works by conditioning out $X$ and reading the residual system. Write the residuals $\tilde T = T-m(X)=\eta$ and $\tilde Y = Y-\mathbb{E}[Y\mid X]$. Then
$$
\begin{bmatrix}\tilde T\\ \tilde Y\end{bmatrix}
=
\underbrace{\begin{bmatrix}1 & 0\\ \theta & 1\end{bmatrix}}_{A}
\begin{bmatrix}\eta\\ \varepsilon\end{bmatrix},
$$
i.e. a **linear, square, invertible mixing** of two independent non-Gaussian sources. (In the code the observed vector is ordered $[X, T, Y]$ and the full mixing $A$ is $(d{+}2)\times(d{+}2)$ with $X$ passing through identically; see `ica.py:generate_ica_data` and the unmixing-row extraction in `ica_treatment_effect_estimation_eps_row`, where the $\varepsilon$-row of $W=A^{-1}$ is $[-b^\top, -\theta, 1]$ and $\theta$ is read off after normalizing the $Y$-loading to 1.)

Three distinct linearities are load-bearing:

**(L1) The effect $\theta$ is a constant scalar/vector.** This is what makes $A$ a *constant matrix*, so that ICA — which estimates a single global mixing matrix — has a target to estimate. Used in **identification**: $\theta = -W_{\varepsilon, T}/W_{\varepsilon,Y}$, a fixed entry of the unmixing matrix.

**(L2) The mixing of $(\eta,\varepsilon)$ into $(\tilde T,\tilde Y)$ is linear.** This follows from (L1) *plus* additivity of $\varepsilon$ and the fact that $T$ enters $Y$ only through $\theta^\top T$. It is what licenses **linear ICA identifiability** (Comon 1994; Shimizu et al. LiNGAM 2006): a linear mix of independent non-Gaussian sources is identifiable up to permutation and scaling, and the permutation/scaling are pinned here by the known triangular structure and the unit $\varepsilon\!\to\!Y$ loading.

**(L3) Linearity-in-$\eta$ of the score / cumulants.** The efficiency comparison with higher-order OML (HOML; Mackey, Syrgkanis & Zadik, [arXiv:1711.00342](https://arxiv.org/abs/1711.00342)) is a statement about *the same moment conditions*. The ICA asymptotic variance implemented in `oml_utils.py` is
$$
V_{\mathrm{ICA}} = c_{\mathrm{ICA}}\,\frac{\operatorname{Var}[\eta^3]}{\kappa(\eta)^2},\qquad
c_{\mathrm{ICA}} = 1 + \lVert b + a\,\theta\rVert^2,
$$
where $a,b$ are the covariate-loadings on $T,Y$, $\kappa$ is excess kurtosis, versus the HOML variance $V_{\mathrm{HOML}}=\operatorname{Var}[\eta^3]/(\mathbb{E}[\eta^3]/\sqrt{\operatorname{Var}[\eta^3]})^2$. Both formulas are *closed-form polynomials in the cumulants of the additive noises*. That closed form exists **only because** the estimand is a single linear coefficient of an additively-mixed system; the influence function is then a fixed quadratic/cubic moment functional. This is where (L1)+(L2) feed the efficiency story.

**Where it breaks.**

- If the effect is **nonlinear**, $Y=\phi(T,X)+\varepsilon$, then $\tilde Y$ is no longer a *linear* combination of $\eta$ and $\varepsilon$: $\tilde Y = \phi(m(X)+\eta, X) - \mathbb{E}[\phi\mid X] + \varepsilon$, and the "$\eta$-channel" enters through $\partial_T\phi$ evaluated along a random path. (L2) fails: the mixing is now a *nonlinear* function of $\eta$ (and $X$). There is no constant $A$ to estimate, hence no unmixing entry equal to $\theta$.
- If the **mixing of the latent sources is genuinely nonlinear** (sources passed through an unknown diffeomorphism), (L2) fails by construction and we are in nonlinear-ICA territory (Section 3) — *not identifiable* without auxiliary structure.
- Even keeping $\theta$ linear but letting it be **heterogeneous**, $\theta(X)$, breaks (L1) only "softly": the mixing is still linear *for each fixed $X$* but $A=A(X)$ varies. This is the salvageable case (Section 2(i), Section 4).

---

## 2. Extension targets, in increasing difficulty

**(i) Varying-coefficient PLR: $Y=\theta(X)T+g(X)+\varepsilon$.** The conditional mixing $A(X)=\begin{bsmallmatrix}1&0\\\theta(X)&1\end{bsmallmatrix}$ is still **linear given $X$**. The ICA side would need a *conditional* (covariate-indexed) linear ICA: identify $\theta(\cdot)$ as a function rather than a scalar. Two routes: (a) localize — run linear ICA in $X$-neighborhoods / on a kernel-reweighted sample and read $\theta(x)$ pointwise; (b) treat $X$ as an auxiliary variable that modulates the *mixing* (not just the source distribution) and lean on conditional-ICA machinery. The estimand of practical interest, the ATE $\mathbb{E}[\theta(X)]$ or $\mathbb{E}[\theta(X)T]$, is a low-dimensional functional, so a $\sqrt n$ efficiency statement is conceivable. **This is the realistic next theorem.**

**(ii) Nonparametric $Y=\phi(T,X)+\varepsilon$ with a low-dim functional (ATE).** Here ICA has no constant matrix to recover. The only way to keep ICA relevant is to insist on the **additive separability of $\varepsilon$**: $\varepsilon$ is still a clean independent non-Gaussian source added at the end, so a *nonlinear unmixing* that isolates $\varepsilon$ from $(T,Y,X)$ would give the structural residual, and the ATE $=\mathbb{E}[\phi(1,X)-\phi(0,X)]$ would be recovered by a separate regression of the de-noised $Y-\varepsilon$ on $(T,X)$. The ICA side would need **nonlinear ICA with $X$ as auxiliary variable** (Section 3) to even isolate $\varepsilon$ — and would identify $\varepsilon$ only up to an invertible scalar reparametrization, which is fatal unless additional anchoring (the additive, unit-loading structure of $\varepsilon$) is imposed. High risk.

**(iii) Genuinely nonlinear mixing of the latent sources.** Full nonlinear ICA. The ICA side would need the entire auxiliary-variable identifiability apparatus *and* a way to map the identified-up-to-equivalence sources back to the specific functional $\theta$. This is a research program, not an extension.

---

## 3. Connection to nonlinear-ICA identifiability theory (the crux)

The core obstruction: **linear ICA is identifiable** (up to permutation + scaling) from non-Gaussianity alone (Comon 1994; Shimizu et al. 2006), but **unconditional nonlinear ICA is fundamentally non-identifiable** — Hyvärinen & Pajunen (1999, *Neural Networks* 12(3):429–439; pre-arXiv, verified via the canonical citation in all the works below) show that for an unknown nonlinear mixing one can always construct infinitely many alternative independent-source decompositions (e.g. via Darmois/Gram–Schmidt constructions). Independence alone is not enough.

Identifiability is **restored by auxiliary structure**, and the covariates $X$ are exactly the natural auxiliary variable here:

- **Time-Contrastive Learning (TCL)** — Hyvärinen & Morioka, [arXiv:1605.06336](https://arxiv.org/abs/1605.06336): nonlinear ICA becomes identifiable when sources are conditionally exponential-family with a *segment/auxiliary* index modulating their distribution.
- **GCL / auxiliary-variable nonlinear ICA** — Hyvärinen, Sasaki & Turner, [arXiv:1805.08651](https://arxiv.org/abs/1805.08651): identifiability up to a known equivalence whenever an observed auxiliary variable $u$ renders the sources conditionally independent and *sufficiently modulates* them ("variability" / rank conditions on the sufficient statistics).
- **iVAE** — Khemakhem, Kingma, Monti & Hyvärinen, [arXiv:1907.04809](https://arxiv.org/abs/1907.04809): the same conditional exponential-family identifiability inside a VAE; auxiliary $u$ indexes a conditional prior $p(s\mid u)$.
- **ICE-BeeM** — Khemakhem, Monti, Kingma & Hyvärinen, [arXiv:2002.11537](https://arxiv.org/abs/2002.11537): extends conditional identifiability to energy-based models, broadening the family beyond exponential.
- **Linear identifiability** — Roeder, Metz & Kingma, [arXiv:2007.00810](https://arxiv.org/abs/2007.00810): under a discriminative/conditional model, representations are identifiable up to a *linear* transform — useful because it is the linear-equivalence residual that the PLR structure can then pin down.
- Broader causal-representation-learning identifiability with interventions/auxiliaries: Varıcı et al., [arXiv:2310.15450](https://arxiv.org/abs/2310.15450) and [arXiv:2402.00849](https://arxiv.org/abs/2402.00849).

**Assessment — can these be repurposed so that $\theta$ (a functional) is identified even when full source recovery is only identifiable up to equivalence?** Partially, and this is the genuinely interesting angle:

1. In the auxiliary-variable theory, $X$ plays the role of $u$: conditional on $X$, the structural noises $(\eta,\varepsilon)$ are independent. That matches the PLR assumptions exactly.
2. But the auxiliary-variable results identify the sources up to a **per-component invertible (often affine, in the conditionally-Gaussian/exp-family case) reparametrization** and a permutation. For estimating $\theta$ we do **not** need to fix that reparametrization globally — we need only **one scalar contrast**: the relative loading of the structural-$\varepsilon$ direction onto $T$ vs $Y$. This is a quotient functional that *can be invariant to the residual equivalence class* if the equivalence is restricted to per-source scaling (as in the linear case) rather than arbitrary monotone warping.
3. The catch: the GCL/iVAE equivalence is "up to component-wise *nonlinear* invertible maps" in the general case, and a nonlinear warp of $\varepsilon$ destroys the "unit additive loading onto $Y$" anchor that the linear proof uses to fix scaling. So repurposing works **only** when the extension keeps $\varepsilon$ *linearly/additively* attached to $Y$ — i.e. the varying-coefficient and additive-$\varepsilon$ cases (Section 2(i)–(ii)), **not** target (iii).

Net: nonlinear-ICA identifiability gives a *credible scaffold* with $X$ as the auxiliary variable, but it buys source recovery up to an equivalence that is too coarse for $\theta$ unless additive structure of $\varepsilon$ is retained. That retained additivity is precisely the "partially linear in $\varepsilon$" residue worth building the next theorem on.

---

## 4. Minimal theorem sketch — varying-coefficient PLR

Most promising target: $Y=\theta(X)\,T+g(X)+\varepsilon$, $T=m(X)+\eta$, estimand the ATE-type functional $\bar\theta=\mathbb{E}[\theta(X)]$ (or pointwise $\theta(x)$).

**Assumptions.**
- (A1) $\varepsilon\perp\eta\mid X$, both mean-zero given $X$, finite eighth moments.
- (A2) **Conditional non-Gaussianity:** for a.e. $x$, at least one of $\eta\mid X{=}x$, $\varepsilon\mid X{=}x$ is non-Gaussian (excess kurtosis or third cumulant nonzero). This is the *conditional* analogue of the paper's non-Gaussianity assumption and the source of identifiability.
- (A3) $\theta(\cdot), g(\cdot), m(\cdot)$ lie in a Hölder/Donsker class estimable at $o(n^{-1/4})$ rate (standard DML nuisance condition; Chernozhukov et al., [arXiv:1701.08687](https://arxiv.org/abs/1701.08687)).
- (A4) Overlap/variability: $\operatorname{Var}(\eta\mid X)$ bounded away from 0; the auxiliary modulation of $X$ on the conditional source laws satisfies the GCL "sufficient variability" rank condition ([arXiv:1805.08651](https://arxiv.org/abs/1805.08651)).

**Identification claim.** For a.e. $x$, the conditional residual system $[\tilde T,\tilde Y]^\top = A(x)[\eta,\varepsilon]^\top$ with $A(x)=\begin{bsmallmatrix}1&0\\\theta(x)&1\end{bsmallmatrix}$ is a linear non-Gaussian mixture; by conditional linear-ICA identifiability (A2), $A(x)$ — hence $\theta(x)$ — is identified up to the permutation/scaling fixed by the unit additive $\varepsilon\!\to\!Y$ loading and the triangular constraint. Therefore $\bar\theta=\mathbb{E}[\theta(X)]$ is identified.

**Efficiency statement (target form).** With cross-fitted nuisances and a Neyman-orthogonal moment for $\bar\theta$ built from the conditional cumulant conditions, the estimator is $\sqrt n$-consistent and asymptotically normal with variance
$$
V_{\mathrm{ICA}}(\bar\theta)=\mathbb{E}_X\!\Big[\big(1+\lVert b(X)+a(X)\theta(X)\rVert^2\big)\,\tfrac{\operatorname{Var}[\eta^3\mid X]}{\kappa(\eta\mid X)^2}\Big],
$$
the conditional-expectation lift of the constant-$\theta$ formula in `oml_utils.py`. The regime where this beats conditional-HOML mirrors the paper's: ICA wins when $\eta$ is **symmetric but high-kurtosis** ($\mathbb{E}[\eta^3]=0$ kills HOML's $\mathbb{E}[\eta^3]$ denominator while ICA's $\kappa^2$ survives).

**Hardest technical obstacle.** Making the *pointwise* (conditional) ICA estimate of $\theta(x)$ aggregate into a $\sqrt n$ functional. Localized/kernel ICA estimates $\theta(x)$ at a nonparametric (slower-than-$\sqrt n$) rate; the orthogonal-moment machinery must absorb that first-stage error. Concretely: writing a moment functional whose derivative w.r.t. the *function* $\theta(\cdot)$ (and w.r.t. the conditional cumulants of $\eta$) is Neyman-orthogonal, so that plug-in $o(n^{-1/4})$ nuisance errors do not bias $\bar\theta$. This is exactly the step that the constant-$\theta$ paper gets for free (one parameter, one matrix entry) and that turns the extension into real work. Secondary obstacle: the GCL "sufficient variability" condition (A4) is *not* automatically satisfied by arbitrary $X$ — it must be assumed or tested, and it is the thing a reviewer will probe.

---

## 5. Critical reviewer-lens verdict and rebuttal language

**Verdict.** Do **not** attempt a new nonlinear theorem in this rebuttal. (i) Target (iii) is blocked by nonlinear-ICA non-identifiability and is a multi-year program. (ii) Target (ii) needs nonlinear ICA that identifies $\varepsilon$ only up to an equivalence too coarse for $\theta$ unless additivity is imposed, at which point it collapses toward (i). (iii) Target (i), the varying-coefficient PLR, is the credible extension but still requires a genuine orthogonal-moment + conditional-ICA argument with a nontrivial rate-aggregation step — a paper, not a paragraph. Promising it in a rebuttal invites a reviewer to demand the proof.

**Recommended framing: a precise limitations + future-work paragraph**, written so the limitation reads as a *deliberate scope choice with a clear path forward*, not an oversight. Suggested language:

> Our identifiability and efficiency results are stated for the partially linear model, where the treatment effect $\theta$ is a constant entering linearly and the structural noises $(\eta,\varepsilon)$ are mixed *linearly* into $(T,Y)$ after conditioning on $X$. This linearity is what makes the ICA connection exact: it turns $\theta$ into a fixed entry of the unmixing matrix and yields the closed-form cumulant expression for the asymptotic variance. Two extensions preserve enough of this structure to remain in reach. **First**, the varying-coefficient model $Y=\theta(X)T+g(X)+\varepsilon$ keeps the latent mixing linear *conditional on $X$*; with covariates acting as the auxiliary variable familiar from conditional/nonlinear-ICA identifiability (Hyvärinen et al., 2018/2019; Khemakhem et al., 2020), one expects $\theta(\cdot)$ to be identifiable and the ATE $\mathbb{E}[\theta(X)]$ to admit a $\sqrt n$ orthogonal estimator. **Second**, retaining additive non-Gaussian $\varepsilon$ in an otherwise nonparametric $Y=\phi(T,X)+\varepsilon$ leaves $\varepsilon$ as a recoverable independent source. **Genuinely nonlinear mixing of the latent sources, by contrast, runs into the non-identifiability of unconditional nonlinear ICA (Hyvärinen & Pajunen, 1999), which can only be circumvented with the auxiliary-variable assumptions above.** We therefore view nonlinear treatment-effect theory as the natural next step rather than a corollary of the present results, and we make the partial-linearity assumption explicit rather than implicit.

This concedes the limitation, names the real obstruction with a correct citation, and signals a concrete program — which is what a TMLR reviewer wants to see, and it costs no new theorem.

---

## Verified citations (all confirmed to exist)

| Work | Identifier | Role |
|---|---|---|
| Reizinger, Mackey, Brendel, Krishnan — *Estimating Treatment Effects with ICA* | [arXiv:2507.16467](https://arxiv.org/abs/2507.16467) | this paper |
| Mackey, Syrgkanis, Zadik — *Orthogonal Machine Learning: Power and Limitations* | [arXiv:1711.00342](https://arxiv.org/abs/1711.00342) | HOML baseline / efficiency |
| Chernozhukov et al. — *Double/Debiased/Neyman ML of Treatment Effects* | [arXiv:1701.08687](https://arxiv.org/abs/1701.08687) | DML / orthogonal moments |
| Hyvärinen & Pajunen — *Nonlinear ICA: existence and uniqueness results* | *Neural Networks* 12(3):429–439, 1999 (pre-arXiv) | non-identifiability impossibility |
| Hyvärinen & Morioka — *Time-Contrastive Learning and Nonlinear ICA* | [arXiv:1605.06336](https://arxiv.org/abs/1605.06336) | auxiliary-variable identifiability |
| Hyvärinen, Sasaki, Turner — *Nonlinear ICA Using Auxiliary Variables and GCL* | [arXiv:1805.08651](https://arxiv.org/abs/1805.08651) | auxiliary-variable identifiability |
| Khemakhem, Kingma, Monti, Hyvärinen — *VAEs and Nonlinear ICA (iVAE)* | [arXiv:1907.04809](https://arxiv.org/abs/1907.04809) | conditional identifiability |
| Khemakhem, Monti, Kingma, Hyvärinen — *ICE-BeeM* | [arXiv:2002.11537](https://arxiv.org/abs/2002.11537) | energy-based extension |
| Roeder, Metz, Kingma — *On Linear Identifiability of Learned Representations* | [arXiv:2007.00810](https://arxiv.org/abs/2007.00810) | linear-equivalence identifiability |
| Varıcı, Acartürk, Shanmugam, Tajer — *General Identifiability and Achievability for CRL* | [arXiv:2310.15450](https://arxiv.org/abs/2310.15450) | CRL identifiability |
| Varıcı et al. — *Score-based CRL: Linear and General Transformations* | [arXiv:2402.00849](https://arxiv.org/abs/2402.00849) | CRL with interventions |

*Not on arXiv (standard references, not fabricated):* Comon (1994, *Signal Processing*); Shimizu, Hoyer, Hyvärinen, Kerminen — LiNGAM (2006, *JMLR*); Robinson (1988, *Econometrica*) for the PLR. Hyvärinen & Pajunen (1999) verified via its canonical citation in the auxiliary-variable papers above (it is the impossibility result they each cite as motivation).

---

# Part II — Score additivity as the bridge to nonlinear theory

**Context (second round).** The previous round concluded that the varying-coefficient PLR is the only credible nonlinear extension, with the rate-aggregation of a slow pointwise $\theta(x)$ into a $\sqrt n$ functional as the hard obstacle. This part chases a sharper conjecture the author flagged: that the **additivity in the score / orthogonal moment for $\theta$** is exactly what licenses a *selective* estimator (estimate $\theta$'s direction, not the whole ICA system) **and** is the structural invariant most likely to survive into a nonlinear extension. Everything below was cross-checked against the author's own draft of the journal/UAI-2026 version of this paper (private repo `rpatrik96/overlap-ica`), the selective-PP exploration in this repo, and the primary literature. Citations are verified inline.

## 6. Where additivity of the score actually lives

There are three distinct "additivities" in play, and conflating them is the main way to get this wrong. They must be kept separate.

**(S1) Structural additivity (the model).** PLR is an additive-noise model (ANM): $Y=\theta T+f(X)+\varepsilon$, $T=g(X)+\eta$, with $\varepsilon$ entering $Y$ with a *unit, additive* loading. The `overlap-ica` draft makes this explicit — the PLR is framed as an ANM with $Z=(X,T,Y)$, $S=(\xi,\eta,\varepsilon)$, and a triangular mixing $A$ (`main_text.tex` §2, the `fig:fig1` framing). This is (L1)+(L2) of Part I restated in ANM language.

**(S2) Additivity of the empirical estimating equation over samples.** Both the OML/HOML estimator (`main_estimation.py`) and the FastICA fixed point are defined by an **empirical mean of a per-sample score** set to zero. For HOML with test function $U'(\eta)=\eta^3$, the exact construction in `main_estimation.py:all_together` (lines 107–108) and `all_together_cross_fitting` (lines 221–225, 269) is
$$
\hat\theta=\frac{\sum_i \tilde Y_i\,m(\tilde T_i)}{\sum_i \tilde T_i\,m(\tilde T_i)},\qquad
m(\tilde T_i)=\tilde T_i^{\,3}-3\,\sigma_\eta^2\,\tilde T_i-\kappa_3,
$$
i.e. $\hat\theta$ solves $\frac1n\sum_i \psi(W_i;\theta,\hat g,\hat f,\hat\sigma_\eta^2,\hat\kappa_3)=0$ with the **linear-in-$\theta$, additive-over-$i$** score
$$
\psi(W;\theta,\dots)=\big(\tilde Y-\theta\,\tilde T\big)\,\underbrace{\big(\tilde T^3-3\sigma_\eta^2\tilde T-\kappa_3\big)}_{m(\tilde T)}.
$$
That this is an *additive* (sample-mean) M-estimator is what delivers a CLT and the closed-form sandwich variance in `oml_utils.py`. This is generic to all Z-estimators; it is *not* the deep fact.

**(S3) Additive separability of the population score in the structural noises — the load-bearing one.** The genuinely useful additivity is that the **log-likelihood score of the PLR factorizes**, and its *cross-partial in $(T,Y)$ returns $\theta$ directly*. The `overlap-ica` appendix (`appendix.tex` §C, `sec:app_cd`, lines 353–366) writes, for Gaussian noises (illustration only — the argument is about the *structure*, not Gaussianity):
$$
\log p(Z)=-\tfrac12\big(Y-f(X)-\theta T\big)^2-\tfrac12\big(T-g(X)\big)^2+\log p(X),
$$
$$
\partial_T\log p(Z)=g(X)-T+\theta\big(Y-f(X)-\theta T\big)=-\eta+\theta\varepsilon,
\qquad\boxed{\ \partial^2_{T,Y}\log p(Z)=\theta.\ }
$$
The mixed second derivative of the log-density **is the treatment effect**, with the nuisances $f,g$ and $\log p(X)$ annihilated by the differentiation. This is precisely the object the score-matching causal-discovery line targets: Montagna et al. (2304.03382, abstract: *"discover the whole causal graph from the second derivative of the log-likelihood in non-linear additive Gaussian noise models"*); Rolland et al. (2203.04413); Montagna et al. (2304.03265, arbitrary noise; 2407.18755, latent variables). The reason $\partial^2_{T,Y}$ isolates $\theta$ is **exactly (S1)**: additive nuisances live in the diagonal/lower-triangular Jacobian entries and have zero mixed $T\!\to\!Y$ second derivative. Equivalently, in the inference-map Jacobian (`main_text.tex` Eq. `eq:jinf_plr`),
$$
J_{f^{-1}}=c\begin{bmatrix}1&0&0\\-g'(X)&1&0\\-f'(X)&-\theta&1\end{bmatrix},
$$
the $T\!\to\!\varepsilon$ entry is $-c\,\theta$ and $\theta$ is read off by normalizing against any diagonal entry to kill the ICA scale $c$ (`reizinger_jacobian-based_2023`, TMLR 2023, OpenReview `2Yo9xqR6Ab`).

**This is the bridge.** (S3) is the statement that survives nonlinearity *as long as additive separability of $\varepsilon$ and the $\theta T$ term holds, even if $f,g$ are arbitrarily nonlinear*. The cross-partial $\partial^2_{T,Y}\log p$ equals $\theta$ pointwise regardless of how nonlinear $f,g$ are; nonlinear $f,g$ only change the *other* score entries. That is the precise sense in which "as long as the causal effect is additive, we don't need to care about (nonlinear) confounders" (the author's TL;DR in `notes.tex`).

## 7. What additivity buys: selective estimation and Neyman orthogonality

**Selective estimation.** Full FastICA fits a $(d{+}2)\times(d{+}2)$ unmixing $W$ to read one scalar from one row. Additivity (S3) says the only object you need is a **single bilinear contrast** — the $T\!\to\!\varepsilon$ direction — not the global system. Two routes make this concrete:

- *Score / cross-moment route.* Estimate $\theta$ from the single moment $\mathbb E[\tilde Y\,m(\tilde T)]/\mathbb E[\tilde T\,m(\tilde T)]$ (HOML) or from $\widehat{\partial^2_{T,Y}\log p}$. One direction, one scalar — no need to demix $X$.
- *Projection-pursuit route (the `selective_pp` exploration).* PP-2D / PP-fixedpoint run FastICA only on the 2D residual $[v,r]=[\tilde T,\tilde Y]$, which is a *sufficient statistic for $\theta$* after partialling out $X$. The `selective_pp` memo confirms identification works (low bias) and is $3\text{–}26\times$ faster than full ICA for $d\ge10$.

But the selective route has a **statistical cost that additivity does not erase**, and this is the key honest finding linking the two explorations: `selective_pp` shows PP-2D has $2\text{–}4\times$ *higher* variance than full ICA. Additivity guarantees the 2D residual space *identifies* $\theta$; it does **not** guarantee it is *efficient*. Full ICA borrows strength from the $d$ covariate rows to sharpen the mixing-matrix estimate (overidentification, IV-style), and selective PP throws that away. So additivity buys **identifiability of a selective target and a clean orthogonal moment — not minimum variance.** This is the sharp correction to a naive "selective = better" reading.

**Neyman orthogonality.** Additivity (S1) is also *why the HOML moment is second-order Neyman-orthogonal*. Mackey, Syrgkanis & Zadik (1711.00342, ICML 2018, abstract verbatim) build *"second-order orthogonal moments if and only if the treatment residual is not normally distributed,"* and *"our proof relies on Stein's lemma."* The Stein connection is exactly the author's NeurIPS note from Lester (`notes.tex`): the orthogonal moments are Stein operators, $\mathbb E[g(\eta)\eta-g'(\eta)]$-type conditions, which vanish for Gaussians. The moment in `main_estimation.py` is the $r=3$ instance: `appendix.tex` Lem. `lem:homl_cond` proves $\mathbb E[\eta^4]\ne3$ (non-zero excess kurtosis) is the non-degeneracy condition, and `appendix.tex` Lem. `lem:ica_cond` shows FastICA's optimality condition $\mathbb E[\eta U'(\eta)-U''(\eta)]\ne0$ reduces to the *identical* $\mathbb E[\eta^4]\ne3$. **Additive structure ⟹ the score's $\theta$-derivative is non-degenerate (Stein/kurtosis condition) ⟹ Neyman-orthogonal moment ⟹ plug-in $o(n^{-1/4})$ nuisance error does not bias $\theta$.** Additivity is the hinge that makes both the HOML moment and the FastICA contrast orthogonal to the *same* nuisance perturbations, which is the paper's whole "ICA = HOML" thesis.

## 8. Does score additivity give a nonlinear route the full-ICA route does not?

**Yes — and it is a genuinely better framing than Part I's full-ICA-conditional route.** The argument:

1. **The functional, not the system, is the target.** Full nonlinear-ICA recovery of $(\eta,\varepsilon)$ is only identifiable up to per-component invertible (generally nonlinear) warps (GCL/iVAE: 1805.08651, 1907.04809). A nonlinear warp of $\varepsilon$ destroys the unit-loading anchor and kills $\theta$. This is the Part I wall.
2. **But $\theta=\partial^2_{T,Y}\log p(Z)$ is a functional of the *observed-data score*, not of the recovered sources.** It does not require solving the BSS problem at all. The score $\nabla\log p(T,Y\mid X)$ is identified from the observed conditional density (estimable by score matching), and its mixed partial returns $\theta$ *whenever the structural model is additive in $T$ and $\varepsilon$* — i.e. $Y=f(X)+\theta T+\varepsilon$ with arbitrarily nonlinear $f,g$. **No source recovery, no warp ambiguity, no auxiliary-variability rank condition.**
3. **This is strictly weaker than what full-ICA needs.** The score route survives the exact regime (nonlinear $f,g$) that breaks linear-ICA-on-raw-data, because differentiation annihilates the nuisances regardless of their nonlinearity. The `overlap-ica` draft already *empirically* confirms linear FastICA on residualised PLR recovers $\theta\approx1.549$ under nonlinear nuisance (`notes.tex` colab note; `main_text.tex` §3.5, Figs. E.10–E.14), and explains *why* via the additive-noise / score-matching connection (`main_text.tex` lines 628–634).
4. **The honest boundary.** The score route requires $\theta T$ and $\varepsilon$ to enter $Y$ *additively and separably*. The dominant failure mode, stated correctly in the draft (`main_text.tex` line 634), is a non-separable interaction $Y=f(X)+\theta T+h(X,T)+\varepsilon$: then $\partial^2_{T,Y}\log p$ is no longer constant, $\theta(X)$ becomes heterogeneous (back to Part I §2(i) varying-coefficient), and if $h$ is non-additive in $T$ the cross-partial is not even a clean effect. So additivity-in-$T$ is the true frontier — *not* nonlinearity of the nuisance.

What `overlap-ica` / the notes add that Part I did not have: the realization that the bridge is the **score**, not the mixing matrix. The full-ICA route (Part I §2–4) tries to lift *linear conditional ICA* and is blocked by rate-aggregation; the **score route lifts a single cross-partial functional** that is (a) nuisance-orthogonal by construction and (b) estimable by score matching at the rates already worked out in 2304.03382/2407.18755. This is a materially cleaner attack.

## 9. Sharpened theorem / assumption sketch — the score-additivity estimator

**Target.** $Y=f(X)+\theta T+\varepsilon$, $T=g(X)+\eta$; estimand $\theta$ (constant) or its varying-coefficient lift $\bar\theta=\mathbb E[\theta(X)]$.

**Assumptions.**
- (B1) *Additive separability:* $T$ enters $Y$ only through the additive term $\theta T$ (no $h(X,T)$ interaction); $\varepsilon$ additive with unit loading. — the true frontier; $f,g$ arbitrarily nonlinear.
- (B2) *Conditional independence / non-Gaussian source:* $\eta\perp\varepsilon\mid X$, mean-zero given $X$; at least one of $\eta\mid X,\varepsilon\mid X$ non-Gaussian (Stein/kurtosis non-degeneracy, `lem:homl_cond`/`lem:ica_cond`). Needed only for the FastICA/HOML instantiation, *not* for the score-matching instantiation (which uses the observed score directly).
- (B3) *Score regularity:* $(T,Y)\mapsto\log p(T,Y\mid X)$ twice differentiable; $\partial^2_{T,Y}\log p$ estimable by score matching at $o(n^{-1/4})$ (rates: Rolland 2203.04413; Montagna 2304.03382/2407.18755).
- (B4) *Cross-fitting* of $f,g$ (or of the score) on disjoint folds, exactly as `all_together_cross_fitting`.

**Identification.** Under (B1), $\partial^2_{T,Y}\log p(T,Y\mid X)=\theta$ a.e.\ (Gaussian-noise case proven in `appendix.tex` lines 353–366; general additive-noise case is the cross-partial of $\log p_\varepsilon(Y-f-\theta T)+\log p_\eta(T-g)$, whose mixed partial is $\theta\cdot\big[-\,(\log p_\varepsilon)''\big]$ — constant in $(T,Y)$ after the standard ANM normalization, identified up to the same scale $c$ that the diagonal normalization removes, cf. `eq:jinf_plr`). For varying $\theta(X)$, the identity holds conditionally and $\bar\theta=\mathbb E_X[\partial^2_{T,Y}\log p(\cdot\mid X)]$.

**Orthogonal estimator (sketch).** Form the additive, Neyman-orthogonal moment
$$
\hat\theta=\arg\!\operatorname{zero}_\theta\ \frac1n\sum_i \psi\big(W_i;\theta,\hat f,\hat g,\hat s\big),\qquad
\psi=\big(\tilde Y-\theta\tilde T\big)\,m(\tilde T;\hat s),
$$
with $m$ the score-derived test function (the $\eta^3-3\sigma^2\eta-\kappa_3$ of `main_estimation.py`, or its general-noise analogue $-(\log p_\eta)'(\tilde T)$). Orthogonality to $(\hat f,\hat g,\hat s)$ follows from (B1)+(B2) by Stein's lemma (Mackey et al. 1711.00342). For $\bar\theta$ the moment is averaged over $X$; the **hard step remains** (Part I §4): a slow conditional-score/$\theta(x)$ estimate must be absorbed into a $\sqrt n$ orthogonal moment for $\bar\theta$ — score additivity gives the orthogonal *form* but not, for free, the *rate aggregation*. This is where the next paper's real work sits.

## 10. Verdict: rebuttal point, future-work paragraph, or next paper?

**All three, at different grain — and crucially, the rebuttal-grade version already exists in the author's draft, so it costs nothing here.**

- **Rebuttal point (safe, immediate).** The cross-partial identity $\partial^2_{T,Y}\log p=\theta$ and the additive-noise/score-matching framing are *already in* `overlap-ica` (`main_text.tex` §3.5, `appendix.tex` §C). For *this* TMLR rebuttal, the defensible sentence is: **"Our linear ICA estimator works under nonlinear nuisance because PLR is an additive-noise model; differentiating the log-density annihilates the nuisances and the mixed $T\!\to\!Y$ partial returns $\theta$ — the same additive structure that the score-matching causal-discovery literature (Rolland 2022; Montagna 2023/2024) exploits. We do not claim a closed-form identifiability theorem for arbitrary nonlinear PLR; the frontier is additive *separability in $T$*, not nonlinearity of the nuisance."** This is honest, novel-sounding, and fully backed.

- **Future-work paragraph.** The selective/score route as a *replacement* for full ICA: it identifies $\theta$ from a single nuisance-orthogonal functional, but (per `selective_pp`) at a variance cost versus full ICA. Frame as: additivity ⟹ minimal identifying object exists; closing the efficiency gap is open.

- **Next paper (the real seed).** A score-matching treatment-effect estimator: estimate $\nabla\log p(T,Y\mid X)$ by score matching, read $\hat\theta=\widehat{\partial^2_{T,Y}\log p}$, prove $\sqrt n$-normality via an orthogonal moment, and handle $\theta(X)$ heterogeneity. This unifies the paper's "ICA=HOML" thesis with score-based CD under one differential-of-the-score umbrella and *bypasses* the nonlinear-ICA non-identifiability wall entirely (you never recover sources). The bottleneck is unchanged from Part I — rate aggregation of a nonparametric conditional-score estimate into a $\sqrt n$ functional — but the *target* is now provably nuisance-robust by additivity, which the full-ICA route never cleanly delivered.

**Headline.** *Score additivity is the real bridge.* The treatment effect is a single cross-partial of the observed-data score, $\theta=\partial^2_{T,Y}\log p$; additive structure (not linearity of the nuisance) is what makes that cross-partial (i) constant, (ii) free of $f,g$, (iii) Neyman-orthogonal via Stein, and (iv) recoverable without solving the BSS problem — so it sidesteps the nonlinear-ICA non-identifiability wall that blocks the full-ICA route. Selective estimation is licensed by this additivity but pays a variance cost (confirmed empirically in `selective_pp`); efficiency, and the $\sqrt n$ aggregation for $\bar\theta=\mathbb E[\theta(X)]$, are the open problems that make this a paper rather than a paragraph.

## Verified citations (Part II — all confirmed)

| Work | Identifier | Role |
|---|---|---|
| Mackey, Syrgkanis, Zadik — *Orthogonal ML: Power and Limitations* | [arXiv:1711.00342](https://arxiv.org/abs/1711.00342); ICML 2018, PMLR 80:3375 | HOML score; *"proof relies on Stein's lemma"*; 2nd-order orthogonal iff non-Gaussian |
| Rolland et al. — *Score Matching Enables Causal Discovery of Nonlinear Additive Noise Models* | [arXiv:2203.04413](https://arxiv.org/abs/2203.04413); ICML 2022 | score $\to$ ANM graph |
| Montagna et al. — *Scalable Causal Discovery with Score Matching* | [arXiv:2304.03382](https://arxiv.org/abs/2304.03382) | *2nd derivative of log-likelihood* recovers graph |
| Montagna et al. — *Causal Discovery with Score Matching on Additive Models with Arbitrary Noise* | [arXiv:2304.03265](https://arxiv.org/abs/2304.03265) | non-Gaussian additive noise |
| Montagna et al. — *Score Matching through the Roof: linear, nonlinear, latent* | [arXiv:2407.18755](https://arxiv.org/abs/2407.18755) | latent-variable score CD |
| Reizinger, Sharma, Bethge, Schölkopf, Huszár, Brendel — *Jacobian-based Causal Discovery with Nonlinear ICA* | TMLR 2023, [OpenReview 2Yo9xqR6Ab](https://openreview.net/forum?id=2Yo9xqR6Ab) | $J_{f^{-1}}$ reads off $\theta$ up to scale; additivity remark |
| Reizinger, Balestriero, Klindt, Brendel — *Position: An Empirically Grounded Identifiability Theory…* | [arXiv:2504.13101](https://arxiv.org/abs/2504.13101) | applied-domain track record of ICA assumptions |

*Internal artifacts referenced (this repo / author's draft):* `main_estimation.py` (HOML additive score, lines 107–108, 221–225, 269); `oml_utils.py` (closed-form variance); `explorations/selective_pp/SELECTIVE_PROJECTION_PURSUIT.md` (PP-2D identifies but is higher-variance than full ICA); `rpatrik96/overlap-ica` private draft — `appendix.tex` §C `sec:app_cd` ($\partial^2_{T,Y}\log p=\theta$), Lemmas `lem:homl_cond`/`lem:ica_cond` (shared $\mathbb E[\eta^4]\ne3$ condition), `main_text.tex` §3.5 + `eq:jinf_plr`, `notes.tex` (additivity-key remark, Lester's Stein/projection-pursuit note).
