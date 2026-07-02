# Research log — cluster autoresearch campaign

One entry per round, newest first. Format:
`## Round NN (date) — <one-line hypothesis>` then Status / Outcome / Link to findings doc.

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
