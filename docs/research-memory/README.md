# Research memory snapshot

Snapshot of the local Claude Code auto-memory for this project, committed so
accumulated findings travel with the code to the cluster. The cluster
instance MUST read `MEMORY.md` (the index) at the start of every round and
the linked entries before designing experiments that touch their topics.

Source of truth: the local machine's
`~/.claude/projects/-Users-patrik-reizinger-Documents-GitHub-ica-causal-effect/memory/`.
Re-snapshot when local memory gains campaign-relevant entries. Cluster-side
findings do NOT go here — they go in `autoresearch/rounds/` findings docs.

Key settled results (do not re-derive):
- IHDP-100/Jobs (binary treatment): OLS 0.60 / OML 0.56 RMSE win, ICA 3.99 —
  binary T violates ICA's continuous non-Gaussian noise model. Scope, not bug.
- California Housing raw features into FastICA = method-misuse strawman;
  only the pre-disentangled design (whiten/ICA on X first) is admissible.
- Env: `pip install -r requirements.txt "numpy<2"` — numpy 2.x breaks torch.
