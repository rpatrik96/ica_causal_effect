# Research log — cluster autoresearch campaign

One entry per round, newest first. Format:
`## Round NN (date) — <one-line hypothesis>` then Status / Outcome / Link to findings doc.

## Round 10 (WS2, 2026-07-03) — r10_ica_win_frontier: the ICA-win frontier is driven by ε, not η
Status: complete. Outcome: **confirmed** + charted. 5×5 sweep eta_beta × eps_beta {0.5,1,1.5,2,3} at
fixed n=50000, housing, d'=5, linear PLR. 25/25 DONE. **The ICA-win region is NOT a symmetric corner —
it's a vertical band at heavy ε.** ICA beats OLS down almost the entire ε=0.5 column (across η incl.
Gaussian η=2 → 0.54), and loses as ε→Gaussian (ε≥1.5 all losses). η non-Gaussianity is neither
necessary (Gaussian η=2 + heavy ε wins) nor sufficient (heavy η=0.5 + Gaussian ε loses 2.02×).
Both-Gaussian corner (η=2,ε=2) = **99.8× OLS** (two Gaussian sources unseparable). **Mechanism**: the
eps-row ICA estimator reads θ from the ε-source row, sharpened by ε's non-Gaussianity → ε's tail (not
η's) sets the efficiency edge (method-specific; an η-row identifier would transpose it). Refines r09:
the controlling variable is OUTCOME-noise non-Gaussianity, not the treatment-noise non-Gaussianity the
ICA story usually emphasizes. Figure: `autoresearch/results/r10_ica_win_frontier/ica_win_frontier.{png,pdf}`
(diverging heatmap, blue=win band, red Gaussian cross). Findings: `autoresearch/rounds/findings_round10.md`.
Next: reproduce the ε-band on news20 + synthetic X; feeds a WS4 selector on estimated ε-kurtosis.

## Round 09 (WS2, 2026-07-03) — r09_ica_edge_nlarged: ICA gets a large-n edge, but only with non-Gaussian ε
Status: complete. Outcome: **confirmed (positive)** — resolves the campaign-long "ICA competitive but
never best" puzzle. housing, d'=5, η β=0.5 (super-heavy), linear nuisance, bootstrap; axes
eps_beta {2.0 Gaussian, 0.5 heavy} × n {2000,10000,50000}. 6/6 DONE. **ICA WINS at every n with
heavy-tailed ε** (ICA best of all estimators), and the **edge GROWS with n**: ICA/OLS = 0.69→0.64→
**0.53** at n=50k (≈2× lower RMSE than OLS) — an asymptotic efficiency gain. With **Gaussian ε: no
edge**, and it worsens with n (ICA/OLS 1.09→1.32→2.02). Mechanism: ICA reads θ from the ε-source row;
a Gaussian ε is the one source it can't sharpen on, so it forfeits efficiency — which is exactly why
r02/r07 (Gaussian ε) never showed an ICA win. Make BOTH PLR noises non-Gaussian and ICA's
higher-moment identification beats the second-moment OLS/OML, increasingly with n. First explicit
"ICA wins" cell of the campaign. Findings: `autoresearch/rounds/findings_round09.md`.
Next: chart the η-kurtosis × ε-kurtosis win frontier; confirm on news20; verify on synthetic X.

## Round 08 (WS2, 2026-07-03) — r08_nonlinear_realX: nonlinear OLS breakdown on real covariates
Status: complete. Outcome: **confirmed** (with a dataset-dependent nuance). Carried the r04 nonlinear
recast (impose_plr nonlinear=True: m/g = sin(πX)+0.5X²) onto real X. dataset {housing,news20} ×
nuisance {linear,gbm} × n {2000,10000}, η β=1, bootstrap. 8/8 DONE. **H1 OLS breakdown transfers to
real X** — OLS biased, n-flat (housing 0.31→0.42; news20 ~0.083); magnitude covariate-dependent
(harsher on housing's dense PCA comps). **H2 gbm-OML rescue is DATASET-DEPENDENT**: rescues cleanly on
news20 (gbm-OML 0.065→**0.023**, 3.6× better than OLS at n=10k) but FAILS on housing (gbm-OML 0.51 >
OLS 0.42) — sin(πX) on 8 dense PCA comps is too oscillatory for GBM to learn at n≤10k, so residual
confounding persists + CF variance. Honest nuance: the rescue needs the nuisance to actually fit g(X).
**H3 ICA fails** under nonlinear confounding on both (0.36–0.86), n-insensitive, as r04/r05. Findings:
`autoresearch/rounds/findings_round08.md`. Pairs with r09: ICA loses under nonlinear confounding (r08),
wins under fully-non-Gaussian linear PLR at n≫d (r09) — bounds ICA's domain on real covariates.

## Round 07 (WS2, 2026-07-02) — r07_ws2_semisynth: ICA's non-Gaussianity story holds on real covariates
Status: complete. Outcome: **confirmed** — first WS2 semi-synthetic benchmark, on 2 real covariate
datasets. Synthetic-on-real-X PLR recast: real covariates → PRE-DISENTANGLE (PCA dense / TruncatedSVD
sparse) → impose T=m(X)+η, Y=θT+g(X)+ε, θ exact, η~gennorm(β). Datasets: **housing** (California
Housing 20640×8 dense, |corr|≤0.93, PCA) and **news20** (20 Newsgroups TF-IDF 2365×2000 sparse, SVD).
16/16 jobs DONE, 25/25 finite. Key result (H2): **ICA's non-Gaussianity advantage reproduces on real
X** — ICA/OLS ratio 1.3–1.6× at β=0.5 (super-heavy, ICA's ideal), degrading to **22.6× (housing) /
21.3× (news20) at the β=2 Gaussian null** where ICA collapses (non-identifiable), recovering at β=4.
Same U-shape-around-β=2 as synthetic r02, on BOTH dense and sparse real covariates. **Validates the
pre-disentangle fix**: raw features → ICA strawman failure (settled); PCA/SVD-whitened → ICA
competitive (1.3–2.5× OLS) whenever η non-Gaussian. H1: OLS/OML strong/stable (well-specified linear
PLR). H3: HOML degenerate at β=2 (symmetric → zero skewness), as in r02. Infra: new
`semisynth_loaders.py` (predisentangle + impose_plr + cached housing/news20 loaders, fixture
fallback) + `semisynth_runner.py` (reuses the estimator stack, nonlinear-schema payload); caches on
shared lustre (gitignored). Findings: `autoresearch/rounds/findings_round07.md`. WS2 ≥2-dataset table
met. Next: add binary failure-mode row (cite settled IHDP) + optional ACIC-2016 recast; nonlinear-g(X)
recast to carry r04 onto real X.

## Round 06 (WS1 capstone, 2026-07-02) — r06_nonlinear_highdim: does the gbm-OML rescue survive d≳n?
Status: complete (9/12 jobs; 3 d=500 casualties documented). Outcome: **confirmed**. Swept
n_covariates(d) {50,200,500} × nuisance {linear,gbm} × n {200,1000} under nonlinear confounding;
d/n spans 0.05–2.5. **gbm-OML rescue is dimension-robust** — best estimator at every completed d/n
incl. d/n=1.0 (0.361 vs OLS 1.125) and d/n=2.5 (0.419). **OLS shows textbook DOUBLE DESCENT** —
RMSE 0.41 (d/n≤0.2) → **1.125 peak at d/n=1.0 (interpolation threshold)** → 0.356 (d/n=2.5, ridgeless
min-norm recovers); worst exactly at d≈n. **ICA catastrophic at d≳n** — 7.42 (d/n=1.0), 3.16
(d/n=2.5), 1–2 orders worse than anything else (FastICA can't unmix a (d+2)-dim system with n≤d);
the extreme of r03's fragility. Matching stable ~0.4–0.5. Casualties: the two d=500 LINEAR-nuisance
jobs hit the 2h max_time thrashing in non-converging LassoCV (computational finding: linear nuisance
intractable at d≫n), + one d=500 gbm removed at max_time — all d=500, decisive d/n=1.0 & 2.5 points
captured via d=200/gbm jobs. Findings: `autoresearch/rounds/findings_round06.md`. WS1 fully mapped
(r02 η, r04 breakdown, r05 η-robust, r06 dimension). Next: nonlinear-g(X) recast on real X (carry
r04/r06 into WS2), or paper figure generation from the metrics.tsv tables.

## Round 05 (2026-07-02) — r05_nonlinear_eta_regime: is the nonlinear-regime story η-robust?
Status: complete. Outcome: **confirmed**. 16/16 jobs DONE, 0 failed, 25/25 finite. Swept
eta_beta {0.5,1,2(Gaussian null),4} × nuisance {linear,gbm} × n {2000,10000} under nonlinear
confounding, to test η-robustness of r04. All three r04 findings hold across η: **(H1) gbm-OML
rescue is η-robust** — decreases with n at every β (→~0.13 at n=10k, uniformly, 3.2× better than
OLS); **(H2) OLS/weak-OML breakdown is η-robust** — OLS flat 0.411–0.418 across the whole grid,
linear-nuisance OML flat ~0.42; **(H3) ICA fails at every β** (~0.55, flat, n-insensitive) —
INCLUDING β=0.5 (its most-favorable heavy tail). Key insight: the sub-prediction "ICA worst at
β=2 Gaussian" is refuted — ICA is *flat* across β (β=2 marginally best), i.e. **under nonlinear
confounding the η non-Gaussianity axis is INERT for ICA**: nonlinear mixing breaks ICA's
identification so thoroughly that non-Gaussianity (its lever in the linear model, r02) carries no
signal. Clean scope boundary: η non-Gaussianity is ICA's lever ONLY when mixing is linear. Aside:
HOML erratic under nonlinear confounding (HOML+gbm 0.106→0.872 across β, sometimes worse than OLS);
first-order OML is the reliable one. Findings: `autoresearch/rounds/findings_round05.md`.
WS1 now well-evidenced end-to-end. Next: **pivot to WS2** (semi-synthetic loaders from
docs/dataset-research/DATASETS.md + pre-disentangle, benchmark ≥2 datasets); optional WS1 capstone
r06 (nonlinear + true d≳n).

## Round 04 (2026-07-02) — r04_nonlinear_breakdown: the evidenced OLS breakdown (criterion b)
Status: complete. Outcome: **confirmed** (WS1 criterion b met) + **negative for ICA**. 12/12 jobs
DONE, 0 failed, 25/25 experiments finite. 2×2×3 diff-in-diff: confounding {linear, nonlinear} ×
nuisance {linear=LassoCV weak, gbm=flexible} × n {500,2000,10000}, η heavy-tailed, low-dim, via
`nonlinear_runner.py`. **OLS breakdown evidenced:** under nonlinear confounding OLS RMSE is pinned
at ≈0.42 at ALL n (0.408→0.418→0.418; ~28% rel. bias at θ=1.5) = asymptotic misspecification bias,
vs →0 in the linear control. OML with the weak linear nuisance breaks down identically (≈0.42) —
orthogonality doesn't save a first stage that can't fit g(X). **Rescue is nuisance-specific:** only
nonlinear+gbm OML *decreases* with n (0.217→0.175→0.132, 3.2× better than linear-nuisance OML at
n=10k) — the double-orthogonal value REQUIRES a flexible first stage (the ML-nuisance story
reviewers asked for). **ICA does not benefit** — worst estimator under nonlinear confounding
(~0.55, flat in n): its linear-mixing identification is violated and it has no flexible-nuisance
lever. Secondary: gbm nuisance HURTS under linear confounding (OML 0.060 vs 0.0082 at n=10k) — match
nuisance flexibility to the DGP. Infra: added `autoresearch/analyze_nonlinear_round.py` (nonlinear
payload schema differs — precomputed rmse array). Findings: `autoresearch/rounds/findings_round04.md`.
WS1 criteria (a)+(b) now both evidenced (r02+r04). Next: **r05_nonlinear_eta_regime** (nonlinear+gbm
across r02's η shapes — is the rescue/ICA-failure η-robust?), then pivot to WS2 semi-synthetic.

## Round 03 (2026-07-02) — r03_oracle_nuisance: does removing the oracle break OLS or favour ICA?
Status: complete. Outcome: **negative** (hypothesis refuted; informative). Crossed oracle {ON,OFF}
× η {gennorm_heavy, discrete} × n, at true support d=10 within D=50 covariates. 20/20 configs,
25/25 experiments kept; oracle-ON configs reproduce r02 exactly (determinism check). Hypothesis
was that oracle-OFF (40 nuisance covariates) would break OLS/OML and favour ICA. **The opposite
happened:** OLS and OML are essentially **invariant** to the oracle (RMSE ratio OFF/ON ≈ 1.0 at
every n — the √(logD/n) regularization absorbs the nuisances), while **ICA breaks down** under the
extra covariates, worst at small n (ICA RMSE ×5.1 at n=500, ×9.1 at n=1000 for gennorm_heavy;
×2.9 discrete n=500), recovering to parity only by n≈5000. Matching also degrades (×1.5–4) and its
penalty persists to n=10k (curse of dim in 50-d matching); HOML is oracle-insensitive. Takeaway:
ICA's viable region is **low nominal dimension / known support**; adding nuisances hurts ICA
(and matching) far more than the regularized OML/OLS — relevant caveat for WS3. Criterion (b) "OLS
breaks down when…" is NOT on the oracle/linear-nuisance axis; next candidates are nonlinear g(X)
misspecification and true d≳n. Infra this round: added `oracle_support` to the result payload +
`oracle`/`D` columns to `analyze_round.py` (the flag was unrecorded); tolerated 2–3 transient node
evictions per submit on the loaded pool by re-submitting missing indices (failing set changed
between runs → transient, verified by local repro). Findings: `autoresearch/rounds/findings_round03.md`.
Next: **r04_nonlinear_breakdown** — nonlinear DGP, oracle ON, to test the misspecification route to
criterion (b).

## Round 02 (2026-07-02) — r02_eta_regime: first real WS1 regime panel (η shape × n)
Status: complete. Outcome: **suggestive**. 25/25 jobs DONE, 0 failed/held, 25/25 experiments
kept per config (no FastICA drops). At the fixed operating point (oracle support ON, d=10,
well-specified linear DGP), **OLS wins all 25 configs**, OML second, matching third; **ICA is
not competitive** (worst/near-worst, RMSE 0.05–0.71) though it improves monotonically with n.
Two a-priori predictions tested: (1) **CONFIRMED** — HOML's skewness estimator degenerates under
symmetric η and blows up at small n (uniform n=500 RMSE **28.1**, gennorm_light **5.67**; skewed
`discrete` fine at 0.07); (2) **PARTIAL** — ICA's predicted asymptotic ordering has not emerged
even at n=10k, i.e. slow finite-sample convergence. No ICA-favorable region on this axis: this
panel maps the OLS-favorable corner (honest scope for the "OLS is the natural baseline" reviews);
criterion (b) "OLS breaks down when…" not yet evidenced. Enabling fix this round: hardened the
login-node against OOM — the local sanity run had been SIGKILLed (`failed(-9)`) by joblib
`n_jobs=-1` fan-out (32 workers on login1); `sweep_runner.py --mode local` now caps it
(`SWEEP_LOCAL_MAX_JOBS`), and the orchestrator should launch under `claude-guard` in tmux
(`docs/CLUSTER_BOOTSTRAP.md` §6). Findings: `autoresearch/rounds/findings_round02.md`.
Next: **r03_confounding_oracle** — oracle OFF × d ∈ {10, 50, 100} × n, at non-Gaussian η, to hunt
the ICA-favorable / OLS-breakdown region (criterion (b)).

## Round 01 (2026-07-02) — r01_smoke: validate sweep_runner end-to-end
Status: complete. Outcome: **confirmed (operational)** — 4/4 jobs DONE, 0 failed/held;
pipeline (grid → condor_submit_bid → .venv(care) → monte_carlo → .npy+DONE → aggregate)
works on real compute nodes. Enabling fix: restored `OMLParameterGrid.create_from_config`
(stale call site on master, removed by `a114722`; caught by the local sanity run before
condor submission). Env: `.venv` symlinked to `/is/cluster/fast/preizinger/nl-causal-representations/care`
(numpy 2.0.0 + torch 2.3.1 work together — the old numpy<2 pin was torch-2.2.2-specific).
Science: negligible (single discrete-η regime); ICA worst / OLS best at that one point,
consistent with discrete η being off-regime. Findings: `autoresearch/rounds/findings_round01.md`.
Next: **r02_regime_eta** — vary η via `--eta_noise_dist` (the axis that drives the ICA story),
n, all five estimators; first real WS1 regime-map panel.

## Round 00 (2026-07-02) — scaffolding
Status: complete (local). Scaffolding + program committed; campaign not yet started.
Next: bootstrap cluster session (docs/CLUSTER_BOOTSTRAP.md), then WS1 r01_smoke.
