# Research log — cluster autoresearch campaign

One entry per round, newest first. Format:
`## Round NN (date) — <one-line hypothesis>` then Status / Outcome / Link to findings doc.

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
