# Research log — cluster autoresearch campaign

One entry per round, newest first. Format:
`## Round NN (date) — <one-line hypothesis>` then Status / Outcome / Link to findings doc.

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
