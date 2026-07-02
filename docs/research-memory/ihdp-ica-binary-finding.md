---
name: ihdp-ica-binary-finding
description: "On real IHDP-100, ICA is poor (binary treatment); OLS/OML win — the 10-rep \"ICA best\" was an artifact"
metadata: 
  node_type: memory
  type: project
  originSessionId: b9ac3c8b-3119-4949-94c8-ffa98ebb1654
---

On the canonical IHDP-100 NPZ benchmark (672×25×100, from fredjo.com — the CEVAE CSV mirror only has 10 reps and silently fell back to a synthetic fixture), the honest result is: **OLS (RMSE 0.60) and first-order Ortho ML (0.56) win; ICA is poor (RMSE 3.99, bias +1.78, overshoots)**. GBM nuisance *hurts* OML (0.56→0.93) because n=672 is small and the effect is heterogeneous. HOML variants are high-variance (1.2–5.2).

**Why:** IHDP (and Jobs) have **binary treatment**, which violates ICA's continuous-non-Gaussian-η model. Consistent with the binary-treatment experiments and the Pareto analysis ([[venv-setup]] unrelated). The rebuttal message: ICA's domain is *continuous, heavy-tailed* treatment noise, not binary interventions — don't claim ICA shines on IHDP/Jobs.

**Why it matters:** an earlier 10-replication smoke run (CEVAE CSV, a different low-effect surface) reported "ICA best, RMSE 0.22". That does NOT reproduce on the real 100-rep benchmark and must not be cited. Corrected in `REBUTTAL_REALDATA.md` (commit on branch `rebuttal/tmlr-experiments`).
