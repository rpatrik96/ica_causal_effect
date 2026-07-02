---
name: venv-setup
description: "How to set up/run the Python env for ica_causal_effect (empty .venv, numpy<2 pin for torch)"
metadata:
  node_type: memory
  type: project
  originSessionId: b9ac3c8b-3119-4949-94c8-ffa98ebb1654
---

The repo's `.venv` ships empty and there is no `[project]` table in `pyproject.toml`, so `uv run` does not resolve deps. Set up with `uv pip install` into the existing venv, then run everything via `.venv/bin/python` (the system python has no numpy).

**Critical:** torch 2.2.2 is built against NumPy 1.x. Installing `numpy>=2` makes every torch-importing module fail at runtime with `RuntimeError: Numpy is not available` (and silently fails some tests — `test_ica`, `test_mcc`, `test_multi_treatment_runner`). Pin `numpy<2` (1.26.4 works). Deps to install: numpy<2 scikit-learn scipy joblib pandas pytest pytest-xdist torch matplotlib seaborn tueplots.

Run tests with `.venv/bin/python -m pytest`. `test_baselines.py` is slow (~75s).
