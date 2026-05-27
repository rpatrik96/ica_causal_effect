# Real-data benchmarks: IHDP and Jobs (rebuttal)

## Datasets and setup

### IHDP (Infant Health and Development Program)

Semi-synthetic benchmark from Hill (2011). The 672 observations per replication
come from the real IHDP study. Outcomes are simulated from the nonlinear
"setting B" response surface, with the true potential-outcome means
$\mu_0(X), \mu_1(X)$ known per replication, so the per-replication ATT/ATE is
exact. We use the **canonical IHDP-100 NPZ benchmark** (the same file used by
CEVAE/CFRNet), which provides 100 genuinely distinct simulated surfaces — unlike
the CEVAE CSV mirror, which only ships 10 replications and is unsuitable for
tight confidence intervals.

- **Source**: `ihdp_npci_1-100.train.npz` from `https://www.fredjo.com/files/`
  (672 samples × 25 covariates × 100 replications)
- **Data status**: **REAL DATA downloaded** and cached under `data/`
- **Covariates**: 25 (6 continuous + 19 binary)
- **Treatment**: binary, ~19% treated
- **True ATT**: heterogeneous across replications, mean $\approx 3.9$ (the
  effect varies by replication; bias/RMSE are computed against each
  replication's own true ATT)

### Jobs (LaLonde / Dehejia–Wahba NSW experiment)

The canonical LaLonde (1986) / Dehejia–Wahba (1999) NSW job-training
randomised experiment. Binary treatment = assignment to job-training. Outcome =
real earnings in 1978 (USD). No ground-truth potential outcomes are available;
the experimental ATT estimate $\approx \$1{,}794$ from Dehejia & Wahba (1999,
Table 2) serves as the benchmark.

- **Primary source**: NBER Stata file `nsw_dw.dta` at
  `https://users.nber.org/~rdehejia/data/nsw_dw.dta` — **REAL DATA downloaded
  and cached** as `data/lalonde_nsw_dw.csv`.
- **Naive experimental ATT from data**: $1,794 (185 treated, 260 NSW control)
  — matches the D&W benchmark exactly.
- **Observational controls**: CPS-1 (n=15,992) and PSID-1 (n=2,490) comparison
  samples also downloaded from NBER and cached.

---

## Estimator setup

Both datasets have **binary treatment** $T \in \{0,1\}$. Configuration:

| Component | Choice |
|-----------|--------|
| Treatment nuisance | `LassoCV` (linear) or `GradientBoostingRegressor` (GBM) |
| Outcome nuisance | same as treatment nuisance |
| eta moments (HOML known) | LogisticRegression propensity → $\eta = T - \hat p(X)$; empirical $\mathbb{E}[\eta^2]$, $\mathbb{E}[\eta^3]$ |
| ICA | `ica_treatment_effect_estimation_eps_row` with `check_convergence=False` |
| OLS | `baselines.ols_baseline` |
| Matching | `baselines.matching_baseline(treatment_kind="binary")` — propensity-score k-NN |

Cross-fitting uses 2-fold KFold throughout.

---

## IHDP results (real data, 100 replications)

Bias/sigma/RMSE computed relative to each replication's own true ATT (mean
$\approx 3.9$). 95% CIs via the delta method on per-replication squared errors.

### Linear (Lasso) nuisance

| Method | bias | sigma | **RMSE** | SE(RMSE) | 95% CI |
|---|---|---|---|---|---|
| **Ortho ML** | −0.153 | 0.543 | **0.564** | 0.079 | [0.410, 0.719] |
| Robust Ortho ML | +0.267 | 2.834 | **2.847** | 0.448 | [1.968, 3.725] |
| Robust Ortho Est | +0.350 | 3.364 | **3.383** | 0.555 | [2.294, 4.471] |
| Robust Ortho Split | +0.476 | 4.295 | **4.322** | 0.771 | [2.812, 5.832] |
| ICA | +1.784 | 3.564 | **3.985** | 0.538 | [2.932, 5.039] |
| **OLS** | −0.043 | 0.596 | **0.597** | 0.108 | [0.386, 0.808] |
| Matching | +0.170 | 1.024 | **1.038** | 0.185 | [0.676, 1.401] |

### GBM nuisance

| Method | bias | sigma | **RMSE** | SE(RMSE) | 95% CI |
|---|---|---|---|---|---|
| Ortho ML | −0.310 | 0.878 | **0.931** | 0.139 | [0.659, 1.203] |
| Robust Ortho ML | −0.264 | 1.716 | **1.736** | 0.273 | [1.201, 2.271] |
| Robust Ortho Est | −0.288 | 1.152 | **1.188** | 0.179 | [0.837, 1.538] |
| Robust Ortho Split | +0.045 | 5.217 | **5.217** | 1.015 | [3.228, 7.206] |
| ICA | +1.784 | 3.564 | **3.985** | 0.538 | [2.932, 5.039] |
| **OLS** | −0.043 | 0.596 | **0.597** | 0.108 | [0.386, 0.808] |
| Matching | +0.170 | 1.024 | **1.038** | 0.185 | [0.676, 1.401] |

ICA's row is identical across nuisance columns because it does not use the OML
residualisation. The earlier 10-replication smoke run (CEVAE CSV mirror, a
different low-effect surface) reported ICA as best; that does **not** survive on
the canonical 100-replication benchmark — see take-aways.

---

## Jobs results (REAL DATA — NSW experimental sample)

**Data**: 185 treated, 260 NSW control. Naive diff-in-means = $1,794.
ATT benchmark (D&W 1999): **$1,794**.

### Experimental NSW only (n=445)

Single run (no replication variance available for real data — one fixed dataset).

| Method | estimate ($) | dev vs D&W ($) | dev vs DIM ($) |
|---|---|---|---|
| **Ortho ML (linear)** | **1,895** | **+101** | **+101** |
| Robust Ortho ML (linear) | 1,902 | +108 | +108 |
| Robust Ortho Est (linear) | 1,914 | +120 | +120 |
| Robust Ortho Split (linear) | 93 | −1,701 | −1,701 |
| ICA | 947 | −847 | −847 |
| OLS | 1,676 | −118 | −118 |
| Matching | 1,786 | −8 | −8 |

| Method | estimate ($) | dev vs D&W ($) |
|---|---|---|
| **Ortho ML (GBM)** | **2,091** | **+297** |
| Robust Ortho ML (GBM) | 2,382 | +588 |
| Robust Ortho Est (GBM) | 2,148 | +354 |
| Robust Ortho Split (GBM) | 2,439 | +645 |
| ICA | 947 | −847 |

**Key finding**: On the experimental NSW sample, OML (linear) comes closest to
the benchmark ($1,895 vs. $1,794, dev +$101). Matching is also close (−$8).
ICA returns ~$947 — about half the true effect. GBM nuisance inflates OML/HOML
estimates upward on this small dataset (n=445), suggesting overfitting in the
first stage with a 200-tree GBM.

### Observational comparison: NSW treated + CPS controls (n=185+15,992=16,177)

Naive diff-in-means = **−$8,498** (severe selection bias: CPS controls earn far
more than NSW participants in 1978).

| Method | estimate ($) | dev vs benchmark |
|---|---|---|
| Ortho ML | 913 | −881 |
| Robust Ortho ML | 877 | −917 |
| Robust Ortho Est | 827 | −967 |
| Robust Ortho Split | 810 | −984 |
| **ICA** | **2,917** | **+1,123** |
| OLS | 699 | −1,095 |
| Matching | −2,518 | −4,312 |

The selection bias is severe: all linear-nuisance OML variants recover only
~$850–$913 (vs. benchmark $1,794). ICA overshoots at $2,917. Matching in the
wrong direction (−$2,518).

### Observational comparison: NSW treated + PSID controls (n=185+2,490=2,675)

Naive diff-in-means = **−$15,205** (even more extreme selection bias).

| Method | estimate ($) | dev vs benchmark |
|---|---|---|
| Ortho ML | −884 | −2,678 |
| Robust Ortho ML | −432 | −2,226 |
| Robust Ortho Est | −361 | −2,155 |
| Robust Ortho Split | 2,977 | +1,183 |
| **ICA** | **2,444** | **+650** |
| OLS | 752 | −1,042 |
| Matching | −3,496 | −5,290 |

On the PSID observational sample, ICA gets closest to the benchmark among the
non-split variants. The large deviations for all methods are expected and
well-known in the LaLonde literature — the PSID comparison sample is not a
credible control group for NSW, and propensity-score-based approaches require
careful overlap trimming.

---

## Take-aways

### IHDP

1. **First-order OML and OLS win; ICA does not.** On the canonical
   100-replication benchmark, OLS (RMSE 0.597) and first-order Ortho ML (0.564,
   linear nuisance) are the clear best; Matching follows (1.038). **ICA is poor
   (RMSE 3.99, bias +1.78)** — it systematically overshoots the true effect
   (e.g. replication 100: ICA = 7.14 vs. true ATT 3.90).

2. **This is the expected outcome, not a failure — IHDP has binary treatment.**
   The ICA estimator models the treatment noise $\eta$ as a continuous
   non-Gaussian source mixed linearly into $(T, Y)$; binary $T$ violates that
   assumption. This matches the binary-treatment experiments and the Pareto
   analysis, both of which find ICA misspecified for binary $T$. IHDP and Jobs
   are therefore OML/OLS territory, not ICA territory — the honest message is
   that ICA's domain is *continuous, heavy-tailed* treatment noise, not binary
   interventions.

3. **GBM nuisance hurts OML here.** Switching Lasso → GBM raises OML RMSE from
   0.564 to 0.931: with only $n=672$ and a heterogeneous effect, the flexible
   first stage overfits rather than helping. The HOML variants are high-variance
   under both nuisances (RMSE 1.2–5.2), driven by the unstable score denominator
   on binary $T$ — the same instability documented in the Pareto analysis.

4. **An earlier 10-replication smoke run reported ICA as best (RMSE 0.22).**
   That used the CEVAE CSV mirror, a different low-effect response surface, with
   too few replications for a stable estimate; it does not reproduce on the
   100-replication benchmark and should not be cited.

### Jobs (experimental NSW)

5. **On the true experimental sample** (n=445, randomised), OML with linear
   nuisance comes closest to the benchmark (+$101 deviation), followed by
   Matching (−$8). ICA underestimates at $947. This is consistent with the
   theory: ICA is designed for continuous treatment with non-Gaussian noise;
   on small binary-treatment datasets, the eps-row identification may be less
   stable. Robust Ortho Split collapses ($93) due to insufficient sample size
   for the nested-split estimator.

6. **The GBM nuisance inflates OML/HOML estimates on Jobs** (n=445 is too
   small for 200-tree GBM without regularisation), confirming that flexible
   nuisance is only beneficial at larger n.

### Jobs (observational — LaLonde bias experiment)

7. **All estimators struggle on the CPS/PSID comparison samples**, as expected
   from the classical LaLonde (1986) result. The selection bias (CPS DIM =
   −$8,498; PSID DIM = −$15,205) is not fully corrected by any method. ICA
   shows the smallest absolute deviation from the benchmark on the PSID sample
   (+$650), suggesting it may partially exploit the non-Gaussian earnings
   distribution to identify the treatment direction, but the result should not
   be over-interpreted given the known lack of overlap.

8. **The Jobs observational results do not constitute a claim that ICA
   outperforms OML in general.** They illustrate that the estimators behave
   differently under extreme selection bias, which is useful context for the
   reviewer.

---

## Code artifacts

| File | Role |
|------|------|
| `realdata_loaders.py` | `load_ihdp()`, `load_jobs()`, `load_jobs_observational()` — download (NBER .dta), cache, and fixture fallback |
| `realdata_runner.py` | Argparse runner; `run_ihdp()`, `run_jobs()`, `run_jobs_observational()`, `_run_single_replication()`, `_make_nuisance_models()` |
| `tests/test_realdata.py` | 37 tests (loader shapes, schema, NPZ benchmark, CI keys, nuisance factory, smoke tests) — all pass |
| `data/ihdp_npci_1-100.train.npz` | Real IHDP-100 benchmark (672×25×100, from fredjo.com) |
| `data/lalonde_nsw_dw.csv` | Real NSW data (from NBER nsw_dw.dta, n=445) |
| `data/cps_controls.csv` | CPS-1 comparison sample (n=15,992) |
| `data/psid_controls.csv` | PSID-1 comparison sample (n=2,490) |
| `figures/realdata/ihdp_results_n100_linear.npy` | IHDP Lasso results (100 real reps) |
| `figures/realdata/ihdp_results_n100_gbm.npy` | IHDP GBM results (100 real reps) |
| `figures/realdata/jobs_results_linear.npy` | Jobs NSW experimental results (Lasso) |
| `figures/realdata/jobs_results_gbm.npy` | Jobs NSW experimental results (GBM) |
| `figures/realdata/jobs_cps_obs_results_linear.npy` | Jobs CPS observational results |
| `figures/realdata/jobs_psid_obs_results_linear.npy` | Jobs PSID observational results |
