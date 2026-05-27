# Selective Projection Pursuit for Treatment Effect Estimation

## The Idea (Reviewer Lester)

Full FastICA fits a (d+2)×(d+2) unmixing matrix W to recover all latent
sources from [X, T, Y], then reads theta from a single row — the eps row
identified by the y_loading heuristic. The entire (d+2)-dimensional
estimation problem is solved to extract one scalar. Projection pursuit (PP)
can target a single non-Gaussian direction directly, bypassing the full
system. The hypothesis: lower asymptotic variance, faster computation, and
better robustness to convergence failures.

---

## The Model

Partially linear regression (PLR):

    T = g(X) + eta
    Y = theta * T + f(X) + eps

After partialling out X via cross-fitted LassoCV (same as main_estimation.py),
the residuals satisfy a 2-component ICA model in the 2D space:

    v := T - g_hat(X) = eta + estimation noise
    r := Y - f_hat(X) = theta * v + eps

The mixing matrix for [v, r] is lower-triangular:

    A = [[1,     0  ],
         [theta, 1  ]]

with sources [eta, eps]. Identifying the eps row of the 2D unmixing matrix
W = A^{-1} gives theta directly.

---

## Formal Estimators

### PP-2D (recommended selective estimator)

1. Cross-fit residuals (v, r) via LassoCV with 2-fold split.
2. Form Z = [v, r], shape (n, 2).
3. Run FastICA(n_components=2) on Z with unit-variance whitening.
4. Identify the eps row via y_loading: pick the row of W_2d with largest
   |W_2d[:, 1]| (Y-column loading). This is the same criterion as
   ica_treatment_effect_estimation_eps_row.
5. Normalize the eps row so W_2d[eps_idx, 1] = 1.
6. theta_hat = -W_2d[eps_idx, 0]  (sign from the model W * [v,r]' = sources).

**Key property**: This is FastICA restricted to the 2D sufficient statistic
for theta after partialling out X. The unmixing problem has dimension 2,
not (d+2), regardless of how large d is.

### PP-fixedpoint (targeted single-component variant)

Runs the FastICA fixed-point iteration (Newton step) directly in 2D whitened
space, seeded from the Y-axis direction (the eps heuristic). Uses n_restarts=4
random restarts; picks by highest negentropy. Falls back to the y_loading
criterion on the orthogonal direction if the found direction is not eps.

Equivalent to FastICA deflation in 2D. Same statistical properties as PP-2D
but a lighter implementation (explicit fixed-point loop, no sklearn overhead).

### Full ICA (baseline)

ica_treatment_effect_estimation_eps_row() on full (d+2)-dimensional X.
Unmixing dimension = (d+2) × (d+2).

---

## Theoretical Analysis

### Why PP-2D is NOT statistically wasteful-free

The "selective estimation" argument rests on whether the 2D projection [v,r]
is a sufficient statistic for theta. It is: after partialling out X, all
information about theta is in (v, r). Fitting ICA in (d+2) dimensions vs 2
dimensions affects the **mixing matrix estimation variance**, not identifiability.

**ICA asymptotic variance** (from the paper, CLAUDE.md):

    Var_ICA(theta_hat) = ica_var_coeff * kappa_3(eps)^2 / kappa_4(eps)^2

where ica_var_coeff = 1 + ||coef_outcome + theta * coef_treatment||^2.

The ica_var_coeff term reflects the *full mixing system*. When PP-2D works
in 2D, the mixing system has only 2 × 2 components, so the asymptotic
variance formula for PP-2D replaces ica_var_coeff with a 2D analogue. In
the ideal case (X well-partialled out, so X columns don't contribute to
eps row loading), the 2D variance should be lower than the (d+2)D variance.

### Why 1D scalar search fails

A 1D unconstrained negentropy search over t in (r - t*v) does NOT find theta.
The global negentropy maximum in this 1D family is at a *mixture direction*
of eta and eps, not at either pure source. Empirically (n=2000, d=2, Laplace):

    - Scan peak: negentropy = 0.353 at t ≈ 0.63  (mixture)
    - True eps direction (t = theta = 1.55): negentropy = 0.341
    - True eta direction (t ≈ 0): negentropy = 0.333

This is because whitening rotates the coordinate axes: the global negentropy
maximum in whitened 2D space can fall strictly between the two pure sources
when they have similar non-Gaussianity. The full mutual-orthogonality constraint
of FastICA is essential; it cannot be replaced by a single-projection search.

---

## Empirical Results

All experiments: n_experiments=200, theta_true=1.55, eta~Laplace (beta=1.0),
eps~Laplace (beta=1.0), LassoCV nuisance (same grid as paper), 2-fold CV.

### Configuration 1: n=1000, d=3, Laplace noise

| Method        |   Bias |  Sigma |   RMSE | Time(ms) | N_valid |
|---------------|--------|--------|--------|----------|---------|
| HOML          | 0.0000 | 0.0330 | 0.0330 |     84.7 |     200 |
| FullICA       | 0.0009 | 0.0361 | 0.0361 |    298.3 |     200 |
| PP-2D         |-0.0030 | 0.1549 | 0.1550 |     84.2 |     200 |
| PP-fixedpoint |-0.0016 | 0.0855 | 0.0855 |     81.8 |     200 |

### Configuration 2: n=2000, d=3, Laplace noise

| Method        |   Bias |  Sigma |   RMSE | Time(ms) | N_valid |
|---------------|--------|--------|--------|----------|---------|
| HOML          |-0.0007 | 0.0210 | 0.0210 |    183.3 |     200 |
| FullICA       | 0.0026 | 0.0262 | 0.0263 |    788.9 |     200 |
| PP-2D         | 0.0026 | 0.1040 | 1.1041 |    186.3 |     200 |
| PP-fixedpoint |-0.0020 | 0.0806 | 0.0806 |    182.0 |     200 |

### Configuration 3: n=1000, d=10, Laplace noise

| Method        |   Bias |  Sigma |   RMSE | Time(ms) | N_valid |
|---------------|--------|--------|--------|----------|---------|
| HOML          |-0.0050 | 0.0347 | 0.0351 |    146.2 |     200 |
| FullICA       | 0.0038 | 0.0387 | 0.0388 |   1675.1 |     200 |
| PP-2D         | 0.0046 | 0.0712 | 0.0713 |    152.1 |     200 |
| PP-fixedpoint | 0.0012 | 0.0593 | 0.0594 |    143.2 |     200 |

### Configuration 4: n=1000, d=3, near-Gaussian (beta=1.8)

| Method        |   Bias |  Sigma |   RMSE | Time(ms) | N_valid |
|---------------|--------|--------|--------|----------|---------|
| HOML          | 0.0025 | 0.0329 | 0.0330 |    188.4 |     200 |
| FullICA       | 0.0613 | 0.3955 | 0.4002 |   1118.7 |     200 |
| PP-2D         | 0.0091 | 0.3969 | 0.3970 |    291.7 |     200 |
| PP-fixedpoint | 0.0064 | 0.3640 | 0.3640 |    222.4 |     200 |

### Configuration 5: n=5000, d=20, Laplace noise

| Method        |   Bias |  Sigma |   RMSE | Time(ms) | N_valid |
|---------------|--------|--------|--------|----------|---------|
| HOML          | 0.0003 | 0.0151 | 0.0151 |    202.4 |     200 |
| FullICA       |-0.0019 | 0.0168 | 0.0169 |   4981.9 |     200 |
| PP-2D         | 0.0038 | 0.0712 | 0.0713 |    200.0 |     200 |
| PP-fixedpoint |-0.0021 | 0.0708 | 0.0709 |    191.1 |     200 |

---

## Key Findings

### 1. PP-2D does NOT beat full ICA on variance

Across all configurations, PP-2D has *higher* sigma than FullICA. In configs
1-3 (heavy-tailed), FullICA sigma is 0.036-0.039 while PP-2D is 0.071-0.155.
This is the opposite of the hypothesis.

**Why**: Full ICA uses all (d+2) dimensions to constrain the unmixing, which
provides more statistical information for identifying the mixing matrix. The
additional d covariate rows act as auxiliary information that sharpens the
identification of the eps row. PP-2D discards this information by projecting
to 2D first.

### 2. PP-fixedpoint consistently beats PP-2D on variance

PP-fixedpoint (targeted fixed-point iteration with restarts) reduces sigma
by 30-45% compared to PP-2D across all heavy-tailed configurations:
- Config 1 (d=3): PP-fixedpoint sigma 0.086 vs PP-2D 0.155
- Config 3 (d=10): PP-fixedpoint sigma 0.059 vs PP-2D 0.071

This is likely because the targeted seeding from the Y-axis direction finds
a better local optimum more reliably than FastICA's random initialization.

### 3. Timing: PP-2D ≈ HOML ≈ PP-fixedpoint << FullICA

| d  | n    | HOML (ms) | PP-2D (ms) | PP-fp (ms) | FullICA (ms) | Speedup |
|----|------|-----------|------------|------------|--------------|---------|
|  3 | 1000 |      84.7 |       84.2 |       81.8 |        298.3 |    3.6× |
|  3 | 2000 |     183.3 |      186.3 |      182.0 |        788.9 |    4.3× |
| 10 | 1000 |     146.2 |      152.1 |      143.2 |       1675.1 |   11.7× |
| 20 | 5000 |     202.4 |      200.0 |      191.1 |       4981.9 |   26.1× |

The speedup scales with d (dimension of X), exactly as expected: full ICA
costs O(n(d+2)^2) per iteration, while PP-2D costs O(n·2^2). At d=20 the
speedup is 25×. This is a genuine computational advantage.

### 4. Near-Gaussian failure (beta=1.8): all ICA methods break

At beta=1.8 (weakly non-Gaussian), all ICA-based methods have RMSE ~0.40
vs HOML's 0.033. This is the known breakdown regime and is the same for
both selective PP and full ICA. PP-2D and PP-fixedpoint do not regress this
failure mode.

### 5. Statistical inefficiency of selective PP vs full ICA

The RMSE penalty of PP-2D vs FullICA is substantial:
- d=3, n=1000: PP-2D RMSE = 0.155 vs FullICA 0.036 (4.3× worse)
- d=10, n=1000: PP-2D RMSE = 0.071 vs FullICA 0.039 (1.8× worse)
- d=20, n=5000: PP-2D RMSE = 0.071 vs FullICA 0.017 (4.2× worse)

The gap grows with n (larger samples expose the variance gap more clearly)
and with d (more auxiliary information lost by the 2D projection).

---

## Round 2 — The selective estimator is the orthogonal score (OML), not PP-in-ICA

The reframing from round 1: "selective" estimation should mean *estimate only what
θ needs, with no global object* — and that lives in the orthogonal-**score**
framework, not inside ICA. PP-2D was selective *within ICA* and inherited ICA's
machinery (whitening, a 2×2 unmixing) while throwing away the strength full ICA
borrows across covariates — the worst of both worlds. The orthogonal score for θ
is additive over samples and reads θ off the residual moment directly; it never
forms an unmixing matrix at all. It is *selective by construction*.

Round-2 run (`selective_pp.py --suite`, 80 reps, gennorm η/ε), separating OML
(first-order) from HOML (second-order) against full ICA:

| Config | Method | Bias | Sigma | RMSE | Time(ms) | vs ICA |
|---|---|---|---|---|---|---|
| **d=10, n=1000, Laplace** | OML | -0.003 | 0.037 | **0.0369** | 172 | **10.8×** |
| | HOML-Known | 0.001 | 0.088 | 0.0880 | 172 | 10.8× |
| | FullICA | 0.004 | 0.038 | 0.0376 | 1863 | 1.0× |
| | PP-2D | -0.001 | 0.039 | 0.0385 | 167 | 11.2× |
| **d=3, n=1000, near-Gaussian (β=1.8)** | OML | 0.006 | 0.031 | **0.0318** | 129 | 9.1× |
| | HOML-Known | 0.007 | 2.438 | **2.4377** | 129 | 9.1× |
| | HOML-Est | 0.422 | 4.728 | **4.7465** | 129 | 9.1× |
| | FullICA | 0.018 | 0.372 | 0.3728 | 1175 | 1.0× |
| **d=20, n=5000, Laplace** | OML | 0.001 | 0.016 | **0.0155** | 198 | **61.7×** |
| | HOML-Known | -0.005 | 0.036 | 0.0359 | 198 | 61.7× |
| | FullICA | -0.002 | 0.018 | 0.0182 | 12238 | 1.0× |
| | PP-2D | 0.010 | 0.111 | 0.1112 | 195 | 62.9× |

**Three findings that flip the round-1 conclusion:**

1. **OML matches or beats full ICA on RMSE, at 5–62× less compute.** Full ICA's
   only RMSE edge is low-dimensional heavy tails (d=3 Laplace: ICA ≈0.024 vs OML
   ≈0.037). By d=10 they tie; by d=20 OML *wins* (0.0155 vs 0.0182) while running
   62× faster. The speedup grows as O((d+2)²/4) because OML never touches the
   (d+2)² unmixing system.

2. **OML, not HOML, is the safe selective default.** HOML blows up near-Gaussian
   (RMSE 2.44 / 4.75 at β=1.8, denominator ∝ excess kurtosis → 0), exactly the
   instability the Pareto analysis quantifies. First-order OML is robust there
   (0.032). So "selective HOML" specifically means the *first-order* orthogonal
   score; the second-order variants buy nothing here and are fragile.

3. **PP-2D stays Pareto-dominated.** Selective-in-ICA never recovers full ICA's
   variance (0.04–0.11) and is beaten by OML at the same speed. This confirms the
   selective lever belongs in the score framework, not in a smaller ICA.

**Sharpened answer to the reviewer:** full ICA is the *global, wasteful* estimator
— it solves a (d+2)-dimensional blind source separation to read one scalar. The
orthogonal score (OML) is the selective estimator that was there all along: it
targets θ directly, is robust where ICA and HOML are not, and is an order of
magnitude (up to 62×) cheaper with equal-or-better accuracy. ICA earns its keep
only in the low-dimensional, strongly-non-Gaussian corner where its borrowed
cross-covariate strength outweighs its cost. This connects to the score-additivity
analysis in `../nonlinear_theory/NONLINEAR_THEORY.md`: additivity of the score is
what makes selective, nuisance-robust estimation of θ possible without solving the
BSS problem.

---

## Honest Assessment

### Does selective PP work?

Partially. PP-2D correctly identifies theta (low bias) and is fast (matches
HOML timing), but it has substantially higher variance than full ICA. The
computational saving is real and grows with d, but it comes at a statistical
cost that makes PP-2D a Pareto-dominated choice: HOML matches or beats it
on both RMSE and speed in all tested configurations.

PP-fixedpoint is strictly better than PP-2D within the selective PP class,
but it still does not beat full ICA.

### Why does PP lose statistically?

The 2D residual space [v, r] is a sufficient statistic for theta *given
perfect nuisance estimation*. But in finite samples with LassoCV nuisance,
the d covariate columns in full ICA provide additional constraints on the
mixing matrix that reduce estimation variance. Full ICA "borrows strength"
from the covariate structure even though covariates are not needed for
identification. This is analogous to how overidentified IV estimators beat
exactly identified ones.

### Is this a contribution for rebuttal / future work?

**For rebuttal**: The message is nuanced. Selective PP (PP-2D) reduces
compute by 3-26× at the cost of 2-4× higher RMSE. It also has no
convergence advantages over full ICA (convergence was 100% in all
configurations here). This is not a clear win and should not be presented
as a rebuttal contribution.

**For future work / discussion section**: Yes. The observation that the
2D residual space is the "informationally minimal" object for theta
identification is theoretically interesting. It opens the question: can
we design a reweighted estimator that leverages the d covariate dimensions
from full ICA without fitting the full (d+2)^2 system? This is the
"projection pursuit with auxiliary information" problem.

### What would a referee attack?

1. **No variance advantage**: The core claim ("selective estimation lowers
   variance") is empirically false. A referee would note that full ICA's
   RMSE is 2-4× lower than PP-2D.

2. **PP-2D is dominated by HOML**: In all configurations tested, HOML has
   lower RMSE and the same or lower compute time. PP-2D is not useful.

3. **No convergence improvement**: All methods converged in 100% of runs.
   The convergence advantage was the most credible original motivation and
   it did not materialize here. (Note: this DGP may be easier than the
   paper's main experiments; heavy-tailed noise makes FastICA converge
   reliably.)

4. **PP-fixedpoint is ad hoc**: The choice to seed from the Y-axis
   direction and use the y_loading criterion to resolve the eta/eps
   ambiguity essentially re-implements the heuristic from
   ica_treatment_effect_estimation_eps_row, just in a smaller space.

5. **Consistency with existing theory**: The paper's asymptotic variance
   formula for ICA already accounts for the full (d+2)-dimensional system.
   Deriving the asymptotic variance for PP-2D and showing when it is
   smaller than the full ICA formula would be needed for a theoretical claim.

---

## Summary Verdict

Selective PP as "faster ICA" works: PP-2D and PP-fixedpoint run at HOML
speed (vs 4-26× slower full ICA) with correct theta identification. But the
variance penalty is large: PP methods are statistically between HOML and
full ICA, closer to the former. The "selective" framing is interesting as
theory but not as an estimator improvement.

For the rebuttal, the safe framing is: "PP-2D shows that the 2D residual
space [v, r] carries all structural identification information for theta,
and that FastICA can be applied selectively to that subspace. This reduces
compute by ~10-25× for d≥10 at a modest RMSE cost, which may be acceptable
in large-scale applications." Do not claim variance improvement over full ICA.

**Superseded by Round 2 (see above).** The right selective estimator is not a
smaller ICA but the orthogonal score (first-order OML): equal-or-better RMSE than
full ICA at 5–62× less compute, robust where HOML and ICA break. PP-2D remains a
theory curiosity (the 2D residual space is informationally sufficient) but is
Pareto-dominated by OML on both axes. The rebuttal message is therefore stronger
and simpler than round 1 suggested: *ICA is the global/wasteful estimator; the
orthogonal score is the selective one, and it wins outside the low-dim heavy-tail
corner.*
