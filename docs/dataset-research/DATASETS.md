# Semi-synthetic dataset shortlist for WS2 (Phase 0 deep-research)

Vetted candidates for the semi-synthetic benchmarks workstream (WS2) of the TMLR
resubmission of arXiv:2507.16467. Each candidate was researched per dataset
family and adversarially verified (URL resolves, license permits use, treatment
is continuous or dose, size is CPU-feasible, ground truth is *exact/simulated*
not estimated). See the spec §Phase 0 for the five hard criteria.

**Bottom line for WS2.** ICA's home regime is the partially linear model (PLM)
with a *continuous* treatment carrying *heavy-tailed non-Gaussian* noise. None of
the shipped benchmarks give us that directly — they are either binary-treatment
(ACIC 2016/2018) or dose-response with a bounded, light-tailed dosage and a
nonlinear outcome (TCGA, News, VCNet-IHDP). The common route is the
**synthetic-on-real-X PLR recast**: keep the real covariates, discard the native
treatment/outcome, and impose our own `T = m(X) + η`, `Y = θ·T + g(X) + ε` with a
controllable `η` (`gennorm(β)` / Student-t). This makes `θ` exact by construction
and puts ICA in-regime while stressing it on real covariate structure. Native
DGPs are retained only as **documented failure-mode cases** (spec criterion 1),
matching the settled binary-IHDP result (`docs/research-memory/`).

---

## Shared design: synthetic-on-real-X PLR recast + pre-disentangle

This is the design philosophy behind every "keep" below (precedent: ACIC 2016
do-it-yourself DGPs; RealCause generative-on-real-X, Neal et al. 2020; Hill 2011
IHDP response surfaces). It is a one-line generalization of the current
`monte_carlo_single_instance.py` / `eta_noise_ablation.py` DGP with real `X`
substituted for Gaussian `X`.

- **Ground truth:** exact — `θ` and the surfaces are fixed in closed form, so
  ATE/CATE is known analytically. Satisfies criterion 2 without any estimated
  generative model. (RealCause's *own* "truth" is w.r.t. a fitted neural model —
  cited here as design precedent only, not as the truth mechanism we use.)
- **η control:** draw `η ~ gennorm(β)` (β<2 heavy-tailed / β=1 Laplace / β→∞
  uniform) or Student-t; keep the additive-linear treatment so PLM moment
  conditions stay exact. This is the *only* convention that lets us control η on
  real covariates.
- **Pre-disentangle (mandatory, criterion 4):** real covariates are correlated
  (co-expressed genes, sparse word counts). Feeding them raw to FastICA is the
  known California-Housing strawman (`docs/research-memory/`). Whiten/reduce
  first: `PCA` (or `TruncatedSVD` for sparse BoW) → optionally `FastICA` on `X` →
  keep `d' ≈ 10–50` components → then impose the PLR. One-hot categoricals and
  standardize before whitening.

### Loader API (mirrors `realdata_loaders.py`)

Proposed module `semisynth_loaders.py`. Every loader mirrors the existing
conventions in `realdata_loaders.py`: a module-level `DATA_DIR` cache under the
repo, `data_dir` override, `use_fixture_on_failure` fallback to a
schema-matched synthetic fixture, and a `(X, T, Y, true_effect)` return tuple
(scalar exact `θ` in the recast; `meta` dict where noted).

```python
# Covariate source — returns real X only (the WS2 primary path).
def load_<name>_covariates(data_dir=None, use_fixture_on_failure=True,
                           n_top=None) -> np.ndarray:  # (n, d) real covariates

# Native DGP — failure-mode case (binary / bounded-dose T, as shipped).
def load_<name>_native(..., data_dir=None) -> (X, T, Y, true_effect)

# Shared recast helpers (module-level, dataset-agnostic):
def predisentangle(X, n_components=None, method="pca", whiten=True,
                   random_state=0) -> np.ndarray                 # (n, d')
def impose_plr(X, theta, treatment_coef, outcome_coef, sigma_eps,
               eta_dist="gennorm", eta_beta=1.0, seed=0
               ) -> (T, Y, theta)   # T=m(X)+η, Y=θT+g(X)+ε; θ exact by construction
```

`load_<name>` (top-level) returns the recast `(X_predisentangled, T, Y, θ)` so
WS2 estimators consume it exactly like the loaders in `realdata_loaders.py`.
Native loaders exist for the failure-mode table only.

---

## Candidate summary

| # | Dataset | n | d (raw) | Native T | License | Precedent | Role |
|---|---------|---|---------|----------|---------|-----------|------|
| 1 | TCGA dose-response (SCIGAN/DRNet) | 9659 | 4000 (of 20531 genes) | dose ∈ (0,1] Beta | code MIT; TCGA GDC open-access | SCIGAN NeurIPS'20, DRNet AAAI'20 | primary high-d recast + best pre-disentangle testbed |
| 2 | ACIC 2016 (Dorie & Hill / CPP) | 4802 | 58 (mixed) | binary | `aciccomp2016` GPL≥2; causallib data CDLA-Sharing | Dorie et al. Stat.Sci. 2019; Curth et al. NeurIPS'21 | primary low-friction recast + native failure-mode case |
| 3 | News (NYTimes bag-of-words) | 3000–5000 | ~2870–3477 (of 102660 vocab) | dose | UCI BoW CC BY 4.0; DRNet code MIT | Johansson et al. ICML'16; DRNet/SCIGAN/VCNet | second pre-disentangle dataset (sparse BoW) |
| 4 | Dominick's Orange Juice (EconML) | 28947 | 15 | continuous (real log-price) | EconML MIT+BSD-3; data UChicago Booth academic | Oprescu et al. ICML'19 (ORF) | genuinely-continuous real-treatment framing (needs recast for truth) |
| 5 | VCNet continuous-treatment IHDP | 747 | 25 (5 cont + 20 disc) | dose ∈ (0,1) sigmoid | repo license=null; IHDP covariates freely redistributed | VCNet ICLR'21 oral; TransTEE, CRNet | small continuous-T sanity dataset |
| 6 | ACIC 2018 (IBM/LBIDD) | 1k–50k | ~177 (mostly binary) | binary | code Apache-2.0; data Synapse-gated | Shimoni et al. 2018; Dragonnet NeurIPS'19 | sample-size sweep (failure-mode-vs-n); Synapse friction |

---

## 1. TCGA gene-expression dose-response (SCIGAN / DRNet)

| Field | Value |
|-------|-------|
| Source URL | `https://github.com/ioanabica/SCIGAN` (code); `https://github.com/d909b/drnet` (code + `tcga.db` via S3); expression from GDC `https://portal.gdc.cancer.gov/` and `https://registry.opendata.aws/tcga/` |
| License | SCIGAN **MIT**, DRNet **MIT** (both verified). TCGA RNA-seq = **GDC Open Access** tier, redistributable for research under NIH GDS (expression is not individually attributable → no dbGaP certification). |
| n | 9659 patients (5000 in-sample + 4659 out-sample, verified) |
| d | 4000 most-variable genes (SCIGAN/TransTEE, log-normalized); 20531 raw (DRNet `tcga.db`) |
| Treatment type | Native: dose `d ∈ (0,1]`, Beta-distributed with x-dependent optimal dose (selection-bias knob `--dosage_selection_bias`). **Recast:** continuous additive `T = m(X)+η`. |
| Ground truth | Native: exact closed-form response surface (SCIGAN `data_simulation.py`, three surfaces incl. `y = C·(v0ᵀx + 12(v1ᵀx·t − v2ᵀx·t²))`, C=10) → exact ADRF. **Recast:** `θ` exact by construction. |
| Loader spec | **No data ships in either repo** — SCIGAN README instructs a separate TCGA download into `datasets/`; DRNet `tcga.db`/`news.db` is a ~10 GB S3 download. Load the gene matrix as `X`; `n_top=4000` most-variable genes; log-normalize; **PCA-whiten mandatory** (co-expressed genes → correlated). DRNet code is Py2.7 but the SQLite `.db` is readable from Py3 — we only need the covariate matrix. After reduction, 9659×4000 float64 ≈ 300 MB — CPU-trivial. |
| Expected ICA fit | Native OFF-regime (bounded light-tailed Beta dose, polynomial surface) → failure-mode case only. **Recast:** PCA-whiten → `d'≈10–50` → impose PLR with `gennorm(β)` η → in-regime with exact truth and full η control. The ideal pre-disentangle testbed of the set (4000 highly-correlated features). |
| Citations | Bica, Jordon, van der Schaar, "Estimating the Effects of Continuous-valued Interventions using Generative Adversarial Networks" (SCIGAN), NeurIPS 2020, arXiv:2002.12326 (verified). Schwab, Linhardt, Bauer, Buhmann, Karlen, "Learning Counterfactual Representations for Estimating Individual Dose-Response Curves" (DRNet), AAAI 2020, arXiv:1902.00981 (verified). TransTEE: Zhang, Zhang, Lipton, Li, Xing, arXiv:2202.01336 (verified). |

## 2. ACIC 2016 data challenge (Dorie & Hill; Collaborative Perinatal Project covariates)

| Field | Value |
|-------|-------|
| Source URL | `https://github.com/vdorie/aciccomp` (R pkg `aciccomp2016`, `dgp_2016()` for all 77×100 datasets); Python subset `https://github.com/BiomedSciAI/causallib` → `from causallib.datasets import load_acic16` (**instances 1–10**); competition page `http://jenniferhill7.wixsite.com/acic-2016/competition` |
| License | `aciccomp2016` **GPL (≥2)** (verified in DESCRIPTION). causallib repo Apache-2.0, but the ACIC16 **data** is redistributed under **CDLA-Sharing** (LICENSE in the data dir; share-alike terms apply if redistributing derived data). |
| n | 4802 units (complete cases; verified from Dorie et al.). 77 settings × 100 replications = 7700 datasets. A few MB per instance; CPU-trivial. |
| d | 58 covariates (per Dorie et al.: 3 categorical, 5 binary, 27 count, 23 continuous — split not quote-verified). One-hot the categoricals before whitening. |
| Treatment type | Native: **binary** `z ∈ {0,1}` → failure-mode case only. **Primary use:** custom continuous-T DGP on the real `x.csv` covariates. |
| Ground truth | Exact. causallib `zymu_{i}.csv` ships `z, y0, y1` **and** noiseless conditional means `mu0, mu1` per unit; `dgp_2016(input_2016, parameterNum, simulationNum)` regenerates all 77×100 parametrically. Recast `θ` exact by construction. |
| Loader spec | `pip install causallib`; `load_acic16(instance=1..10)` ships `x.csv` (4802×58) + `zymu_{i}.csv`. Full 7700: `remotes::install_github('vdorie/aciccomp/2016')` then `dgp_2016()`, or export `x.csv` once and run the Python recast. One-hot categoricals; standardize; PCA-whiten before FastICA. Lowest-friction of the set (pip-installable subset). |
| Expected ICA fit | Native binary T → two-point η residuals, outside PLM/ICA (consistent with the settled IHDP-binary finding); failure-mode case. **Recast:** real X + `T = m(X)+gennorm(β)` + `Y = θT+g(X)+ε` → exact truth, full η control — the ideal semi-synthetic design on mixed-type real covariates. |
| Citations | Dorie, Hill, Shalit, Scott, Cervone, "Automated versus Do-It-Yourself Methods for Causal Inference", Statistical Science 34(1):43–68, 2019, DOI 10.1214/18-STS667, arXiv:1707.02641 (verified). Curth, Svensson, Weatherall, van der Schaar, "Really Doing Great at Estimating CATE?", NeurIPS 2021 Datasets & Benchmarks (verified; analyzes an ACIC2016 subset in Appendix D). |

## 3. News (NYTimes bag-of-words dose-response)

| Field | Value |
|-------|-------|
| Source URL | UCI Bag of Words (NYTimes) `https://archive.ics.uci.edu/dataset/164/bag+of+words`; packaged by DRNet (`news.db`, S3) and VCNet (`news_generate_data.py`, `https://github.com/lushleaf/varying-coefficient-net-with-functional-tr`) |
| License | UCI Bag of Words = **CC BY 4.0** — attribution required (credit David Newman, DOI 10.24432/C5ZG6P). **Not** the LDC-licensed NYT Annotated Corpus (that conflation was corrected in verification). DRNet code MIT. Never commit raw article text; derived word counts are redistributable. |
| n | 3000 (SCIGAN/VCNet) – 5000 (DRNet), subsampled from ~300k UCI NYTimes docs |
| d | ~2858–3477 bag-of-words features (DRNet ~2870; VCNet ~3477), subsampled from the 102660-word UCI vocab. Verify exact `d` against the loaded file at implementation time. |
| Treatment type | Native: dose (VCNet `t ∈ [0,1]` Beta; DRNet reading-time `s ~ Exp`). **Recast:** continuous additive `T = m(X)+η`. |
| Ground truth | Native: exact closed-form sinusoidal surface `y = C·(v0ᵀx + sin(π·(v1ᵀx/v2ᵀx)·t))`. **Recast:** `θ` exact by construction. |
| Loader spec | UCI `docword.nytimes.txt.gz` (223 MB) → sparse count matrix, subsample vocab; or DRNet `news.db` (S3, ~10 GB combined w/ `tcga.db`); or VCNet `news_generate_data.py` (generator script present; cache produced by it). Sparse BoW → **TruncatedSVD / PCA-whitening mandatory** before FastICA. Lower-dim and sparser than TCGA → the natural second pre-disentangle dataset. CPU-feasible (5000×~3000). |
| Expected ICA fit | Native Beta/Exp dose + nonlinear outcome → OFF-regime. **Recast:** TruncatedSVD-whiten sparse counts → impose PLR + `gennorm(β)` η → in-regime. Different X structure (sparse text) from TCGA (dense genes) — complements it for the benchmark table. |
| Citations | Johansson, Shalit, Sontag, "Learning Representations for Counterfactual Inference", ICML 2016, PMLR v48:3020–3029, arXiv:1605.03661 (verified). Dose variants: DRNet (AAAI 2020), SCIGAN (NeurIPS 2020), VCNet — Nie, Ye, Liu, Nicolae, ICLR 2021 oral, arXiv:2103.07861 (verified). |

## 4. Dominick's Orange Juice store-level demand (EconML)

| Field | Value |
|-------|-------|
| Source URL | `https://msalicedatapublic.blob.core.windows.net/datasets/OrangeJuice/oj_large.csv` (downloaded at runtime by EconML notebooks; repo `https://github.com/py-why/EconML`) |
| License | EconML **MIT** (+ BSD-3-Clause for `econml.tree`/`econml.grf`, forked from scikit-learn). Dominick's Finer Foods data from UChicago Booth Kilts Center, free for academic use with registration; `oj_large.csv` publicly downloadable from the Azure blob. |
| n | 28947 store-week rows (verified) |
| d | 15 store-level covariates (mean age, income, education, brand, etc.; verified) |
| Treatment type | **Genuinely continuous real treatment** (log price). This is its distinctive asset. |
| Ground truth | Not native-exact (real observational data, no true `θ`). **Requires the synthetic-on-real-X recast:** keep real covariates (optionally real log-price as `m(X)`), impose `Y = θT + g(X) + ε`. Qualifies only under the recast. |
| Loader spec | Download `oj_large.csv` (28947×15, trivial). Not bundled in EconML — the notebooks fetch it from the Azure blob above. Low d → pre-disentangle optional/cheap; CPU-trivial. For exact truth and η control, simulate the outcome (and optionally the treatment) on the real covariates. |
| Expected ICA fit | Real continuous treatment is a genuine plus for the "continuous treatment is a realistic setting" framing, but native treatment noise is not the controllable heavy-tailed η ICA needs and there is no ground-truth `θ`. Under the recast with `gennorm(β)` η, ICA applies; without it, this is a demonstration, not a benchmark. |
| Citations | Oprescu, Syrgkanis, Wu, "Orthogonal Random Forest for Causal Inference", ICML 2019, PMLR v97, arXiv:1806.03467 (verified). Flagship real continuous-treatment example in EconML's LinearDML/ORF price-elasticity tutorials. |

## 5. VCNet continuous-treatment IHDP (semi-synthetic)

| Field | Value |
|-------|-------|
| Source URL | `https://github.com/lushleaf/varying-coefficient-net-with-functional-tr` (`ihdp_generate_data.py` at repo root; covariates `dataset/ihdp/ihdp.csv`); paper arXiv:2103.07861; OpenReview `RmB-88r9dL` |
| License | Repo **license=null** (GitHub API confirmed) — **do not vendor the code; reimplement the DGP equations**. The 747×25 IHDP covariate matrix (`ihdp.csv`, ~52 KB) is Hill (2011)'s standard set, widely redistributed (npci, CEVAE); this project already uses IHDP covariates (commit `fa54d11`), so no new data-license exposure. |
| n | 747 (verified: train/test 471+276) |
| d | 25 covariates — **5 continuous + 20 discrete** (paper `Scon={1,2,3,5,6}`; corrected from the "6+19" claim) |
| Treatment type | Native: continuous dose `t ∈ (0,1)` via sigmoid link. **Recast:** additive PLR. |
| Ground truth | Native: exact — `ihdp_generate_data.py` computes the ADRF `ψ(t)=(1/747)Σ_j y(t,x_j)` over the noiseless outcome; individual `y(t,x)` exact by construction. **Recast:** `θ` exact. |
| Loader spec | **Reimplement the DGP from the shipped code, not the paper appendix** — they differ (code hardcodes a "v2" factor override `factor1=1.5, factor2=0.5`; denominator `0.1+min` vs paper's `0.5+5·min`; the `sigmoid(2t)` link doubles the effective pre-sigmoid treatment-noise std to 1.0 vs paper's 0.5). Follow-ups (TransTEE, CRNet) clone this repo, so the code is the operative spec. Small → many cheap CPU replications; pre-disentangle optional at d=25 (only 5 continuous covariates — handle the 20 discrete separately or drop before FastICA). |
| Expected ICA fit | Native OFF-regime (Gaussian pre-sigmoid noise squashed to (0,1) → bounded sub-Gaussian; multiplicative-nonlinear outcome `sin(3πt)/(1.2−t)`) → failure-mode case. **Recast:** small continuous-treatment sanity dataset with imposed PLR + `gennorm(β)` η. Distinct from the settled **binary**-IHDP failure mode (OLS 0.60 / OML 0.56 / ICA 3.99) — do **not** conflate. Secondary companion (small d/n limit the pre-disentangle benefit). |
| Citations | Nie, Ye, Liu, Nicolae, "VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments", ICLR 2021 **oral**, arXiv:2103.07861 (verified). Follow-ups reusing IHDP-continuous: TransTEE (arXiv:2202.01336), CRNet (Zhu et al., AAAI 2024, arXiv:2403.14232). Note: DRNet (AAAI 2020) *predates* VCNet and is a baseline in it, not a follow-up. |

## 6. ACIC 2018 data challenge (IBM / LBIDD scaling benchmark)

| Field | Value |
|-------|-------|
| Source URL | `https://www.synapse.org/Synapse:syn11294478` (full data; **Synapse registration required**, content login-gated); code + sample `https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework`; paper arXiv:1802.05046 |
| License | Framework code **Apache-2.0** (verified). Data: Synapse platform/challenge terms, registration-gated, no SPDX license — **verify Synapse data-use terms before redistributing any derived artifact** (flag stands). |
| n | Sample sizes `n ∈ {1k, 2.5k, 5k, 10k, 25k, 50k}` (verified in paper). "63 DGPs × ~40 datasets each" is literature-corroborated, not primary-source-verified — treat as approximate. Per-file (≤50k × ~177) is tens of MB, CPU-feasible; full corpus multi-GB. |
| d | ~177 LBIDD-derived covariates (mostly binary/categorical) — literature-sourced, not in the arXiv paper text. |
| Treatment type | Native **binary** → failure-mode case. Marginal value over ACIC 2016 = the n=1k–50k **sample-size sweep** on real covariates (failure-mode-vs-n curves; convergence-rate experiments the fixed n=4802 ACIC 2016 cannot support). |
| Ground truth | Exact — per-unit counterfactual `_cf.csv` files ship simulated `y0, y1` for every unit (verified in paper); treatment + outcomes fully simulated on real LBIDD covariates. Recast `θ` exact. |
| Loader spec | Register on synapse.org, pull `syn11294478` via `synapseclient` (`syn.get`). Shared `x.csv` + per-DGP factual `(z,y)` + `_cf.csv` counterfactuals. The IBM repo ships joining code + a data sample sufficient for pipeline dev without Synapse. High-dim mostly-binary X → PCA-whiten + likely dimensionality reduction before FastICA. |
| Expected ICA fit | Native binary T → same two-point η failure mode as ACIC 2016. **Recast:** continuous-T DGP with controlled heavy-tailed η on up to 50k real covariate rows → convergence-rate experiments. Registration friction makes this lower-priority than ACIC 2016. |
| Citations | Shimoni, Yanover, Karavani, Goldschmidt, "Benchmarking Framework for Performance-Evaluation of Causal Inference Analysis", arXiv:1802.05046, 2018 (verified). Shi, Blei, Veitch, "Adapting Neural Networks for the Estimation of Treatment Effects" (Dragonnet), NeurIPS 2019 (verified; uses ACIC 2018/LBIDD via Synapse alongside IHDP). |

---

## Ranked picks for WS2 (implement 2–4)

1. **TCGA dose-response (SCIGAN/DRNet).** The best high-dimensional real-covariate
   source and the ideal pre-disentangle testbed: 4000 highly-correlated
   gene-expression features are exactly the case where feeding raw `X` to FastICA
   fails and PCA/FastICA-whitening earns its keep. MIT code + GDC open-access data,
   exact ground truth under the imposed PLR, and canonical continuous-treatment
   precedent (SCIGAN NeurIPS'20) TMLR reviewers recognize. Only friction is a
   one-time manual TCGA/`tcga.db` download.

2. **ACIC 2016.** The lowest-friction, most-recognized causal-inference benchmark
   in the set — `pip install causallib` ships a 4802×58 mixed-type covariate table
   with exact `mu0/mu1`. Serves double duty: the native binary-treatment DGP is a
   ready-made *documented failure-mode* row (mirroring the settled IHDP-binary
   result), and the real covariates drive the continuous-T recast with exact `θ`.
   Mixed-type covariates exercise the one-hot + whiten path. Pairs with TCGA to
   satisfy the "≥2 datasets" success criterion immediately.

3. **News (NYTimes bag-of-words).** The complementary second pre-disentangle
   dataset: sparse high-dim text counts (TruncatedSVD-whitening) versus TCGA's
   dense gene matrix, so the benchmark table spans two very different real-X
   structures. Clean CC BY 4.0 license, standard continuous-dose precedent
   (Johansson ICML'16 lineage), CPU-feasible at 5000×~3000.

4. **Dominick's Orange Juice (optional 4th).** The only candidate with a *genuinely
   continuous real-world treatment* (log price), which strengthens the "continuous
   treatment is a realistic setting" framing. Trivial 28947×15 download, MIT code,
   recognized DML application (ORF, ICML'19). Requires the synthetic-on-real-X
   recast for exact ground truth (no native `θ`), so it is a framing/robustness
   addition rather than a core benchmark — include if WS2 has budget after the top
   three.

**Priority pair:** TCGA + ACIC 2016 (high-d dense + mixed-type, both low-friction,
both with a native failure-mode row). Add News third for the sparse-X contrast;
OJ fourth for the real-continuous-treatment story.

---

## Rejected + why

No candidate that passed adversarial verification was rejected. The following
families were excluded upstream (spec §Phase 0 / plan Task R1 candidate lists),
with family-level reasons inferred from the vetting criteria:

| Family | Reason |
|--------|--------|
| **Twins** (binary "heavier twin" treatment, binary mortality outcome) | Fails criterion 1 (continuous/dose treatment) and ICA's continuous non-Gaussian noise model — same two-point-η failure mode as binary IHDP/Jobs, already settled. No marginal value over the retained failure-mode cases. |
| **California Housing, raw features** | Settled method-misuse strawman (`docs/research-memory/`): raw correlated features into FastICA. Admissible *only* under the pre-disentangle + imposed-PLR design, at which point it is a generic synthetic-on-real-X covariate source with no causal-inference precedent — dominated by ACIC 2016/TCGA, which offer both real X *and* citable TE-literature standing. |
| **401(k) eligibility / participation** (DML) | Treatment (eligibility/participation) is **binary** — fails criterion 1; no ground-truth continuous effect. A DML staple but off-regime for ICA. |
| **Binary IHDP / Jobs (LaLonde NSW)** — already in `realdata_loaders.py` | Binary treatment; the honest result (OLS 0.60 / OML 0.56 / ICA 3.99 RMSE on IHDP-100) is **settled** (`docs/research-memory/`). Retained in the paper as documented scope honesty, **not** re-run in WS2 expecting an ICA win (spec Guardrails). |

**Note on RealCause** (Neal, Huang, Raghupathi, arXiv:2011.15007): cited as
*design-philosophy precedent* for synthetic-on-real-X, not admitted as a dataset —
its own ground truth comes from an *estimated* neural generative model, which fails
criterion 2 (exact, not estimated). The exact-`θ` mechanism we use is the
ACIC/IHDP hand-specified closed-form DGP.
