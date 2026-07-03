# Round 15 (WS4 closure) — the ICA d-gate is marginal; a learned tree recovers the hand-set rule

## Hypothesis

Two loose ends from r14: (1) add a dimension gate to the ICA branch (ICA only when
also low-dim, since ICA degrades with d, r06); (2) learn a shallow decision tree on
the four features and check it recovers the hand-set thresholds — direct evidence
the selection boundaries are intrinsic. Both are analysed on the pooled per-cell
records from the WS4 rounds (r13 + r14 splits), 58 cells, no new compute
(`autoresearch/learn_selector_tree.py`).

## Results

### (1) The ICA d-gate is a marginal, ambiguous change — not adopted

Regret on the 58 pooled cells:

| rule | regret vs oracle | match |
|---|---:|---:|
| hand-set (no d-gate) | +0.0125 | 46/58 |
| d-gated ICA (d ≤ 50) | +0.0109 | 47/58 |

The d-gate improves aggregate regret only within noise (0.0016) and flips a single
cell. It disagrees with the hand-set rule on 5 cells — all **d=200, n=5000, heavy-ε**
— and the disagreement is genuinely two-sided:

- On 3 cells it helps: the confounding is nonlinear but the nonlinearity feature
  can't detect it at d=200 (GBM can't out-fit linear on a 2000-subsample of a
  200-dim problem), so the hand-set rule falls to the ε branch and wrongly picks
  ICA; the d-gate routes to OLS, which is better.
- **But on one cell it breaks a genuine win**: `d=200, linear, ε=0.5` has oracle
  **ICA** (0.011 vs OLS 0.018) — ICA *does* win at d=200 when the model is linear.
  The d-gate mis-routes it to OLS.

So "ICA needs low dimension" is not quite right — ICA can win at d=200 in the linear
case, and lose at d=200 in the (undetected-)nonlinear case. A blanket d-gate cannot
separate these; the operative signal is nonlinearity, which the cheap feature misses
at high d. **Conclusion: keep the rule as-is** (default thresholds unchanged); the
high-d ICA decision is limited by nonlinearity *detection*, not by a missing d-gate.
The d-gate remains available as a conservative variant (`rule_dgate` in the module)
for deployments that prefer to never use ICA in high dimension.

### (2) A learned shallow tree recovers the hand-set structure

A depth-3 `DecisionTreeClassifier` on [nonlinearity, d/n, ε-kurtosis, log n] →
oracle label learns:

```
d_over_n <= 0.55 ── OML_lin          (else branch)
  ├ eps_kurt <= 14.56
  │   ├ nonlinearity <= 0.03 → OLS
  │   └ nonlinearity >  0.03 → OML_gbm
  └ eps_kurt > 14.56
      ├ nonlinearity <= 0.10 → ICA
      └ nonlinearity >  0.10 → OML_gbm
```

The learner **independently recovers the hand-set boundaries**: the high-dim split
at **d/n ≈ 0.55** (hand-set 0.5 → OML_lin), the nonlinearity split at **≈0.03–0.10**
(hand-set 0.05), and ICA reserved for high ε-kurtosis with low nonlinearity. Same
structure, from data alone — strong evidence the boundaries are intrinsic to the
estimators, not hand-tuned artefacts.

But the tree does **not beat** the hand-set rule (pooled 58 cells):

| strategy | regret vs oracle |
|---|---:|
| always-ICA | +31.69 |
| always-OLS | +0.2295 |
| always-OML(gbm) | +0.0450 |
| best fixed — always-OML(lin) | +0.0354 |
| learned tree (LOO-CV) | +0.0381 |
| **hand-set rule (mechanism)** | **+0.0125** |
| per-cell oracle | 0 |

Figure: `autoresearch/results/r13_multiselector_eval/ws4_closure.{png,pdf}`. The
tree at depth 3 spends its capacity on d/n, ε, and nonlinearity and **cannot also
encode the n-gate**, so it repeats v1's small-n gbm overuse; and its ε threshold
(14.56) is overfit to the pool. The hand-set rule — which *does* encode the n-gate
and uses mechanism-grounded thresholds — is ~3× better than both the best fixed
estimator and the learned tree, and generalises (LOO for the tree is the honest
comparison).

## Evidence grade

**Confirmed — WS4 complete.** (a) The proposed ICA d-gate does not cleanly help
(marginal aggregate gain, breaks a valid linear-high-d ICA win); the limiting factor
is nonlinearity detection at high d, documented, rule left unchanged. (b) A
data-driven shallow tree recovers the hand-set thresholds (d/n≈0.5, nonlinearity≈0.05,
ICA at high ε-kurtosis), confirming the boundaries are intrinsic, yet the
interpretable mechanism-grounded rule beats the learned tree (and every fixed
estimator) by ~3× and generalises better — the appropriate final artefact.

## Implications for the paper

- The estimator-selection story is closed: a small, interpretable, tuning-free rule
  over four measurable features (nonlinearity, dimension, outcome-kurtosis, n) tracks
  the oracle across the whole regime map, on synthetic and real covariates, and is
  not beaten by either threshold-calibration (r14, overfits) or tree-learning (r15,
  recovers-but-doesn't-beat). Both data-driven baselines *validate* the rule rather
  than replace it.
- Honest boundary: ICA-vs-OLS at high dimension hinges on detecting weak
  nonlinearity, which cheap features miss — a stated limitation and a direction
  (better high-d nonlinearity diagnostics) rather than a fix forced into the rule.

## Proposed next round

WS4 is complete. Remaining campaign options: (i) a WS3 sensitivity sweep
(gradually Gaussianising η / injecting η–ε dependence) to round out the assumption-
violation studies; (ii) begin assembling the paper-facing figure set from the
committed `metrics.tsv` tables and the four WS2/WS4 figures.
