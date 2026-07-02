---
name: housing-semisynthetic-deferred
description: "California Housing semi-synthetic experiment was run for the UAI 2026 rebuttal, found to be a method-misuse strawman, and deliberately deferred to camera-ready — no numbers committed"
metadata: 
  node_type: memory
  type: project
  originSessionId: bee41409-6da1-42d3-9e38-9ea3b877bbb5
---

The California Housing semi-synthetic experiment (the "housing" real-data experiment) WAS actually run during the UAI 2026 rebuttal, but the result was a negative one that was deliberately dropped. Feeding raw, highly correlated census features straight into FastICA violates ICA's source-independence assumption by construction, so ICA did poorly — but that reflects method *misuse*, not the method. The numbers were judged misleading and discarded.

Outcome: deferred to camera-ready, NOT in the rebuttal. Committed fix for a principled benchmark: (a) pre-disentangle X (PCA whitening or FastICA on X first) before running the TE estimator, OR (b) use a dataset with plausibly independent features. Camera-ready commitment is design (a) on California Housing or equivalent.

**Why:** No code, `.npy`, or git trace survives in either `ica_causal_effect` or `overlap-ica` — only the qualitative conclusion in prose. A git/filesystem sweep for "housing/california/semi-synthetic" comes up empty and looks like the experiment never happened.

**How to apply:** The only surviving record is `overlap-ica/reviews/uai2026_reviews_with_responses.md` §C (lines 301–305) and summary line 223. The actual RMSE numbers from that housing run are gone. The rebuttal experiments that DO have committed numbers are the Bernoulli/Rademacher ablation (§A), the OLS/per-coordinate-HOML/matching 900-config multi-treatment sweep (§B → `figures/multi_treatment/results_multi_treatment.npy`), and the binary-treatment study (`REBUTTAL_BINARY_TREATMENT.md`). See the binary treatment rebuttal results in `REBUTTAL_BINARY_TREATMENT.md`.
