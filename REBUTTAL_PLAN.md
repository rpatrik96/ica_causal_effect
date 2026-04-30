# UAI 2026 Rebuttal Experiment Package — Implementation Plan

**Source:** `/Users/patrik.reizinger/Documents/GitHub/overlap-ica/reviews/uai2026_reviews_with_responses.md`
**Codebase:** `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/`
**Cluster:** HTCondor (CPU-only)
**Audience:** downstream implementation agents — read this front-to-back, you should not need the rebuttal.

The five experiments below are non-negotiable rebuttal commitments:

1. OLS baseline (linear-SEM) — single + multi-treatment
2. Matching baseline (general regime)
3. Bernoulli treatment-noise experiment (Fig E.5 layout)
4. Semi-synthetic experiment (real X, PLR-imposed θ)
5. Fig 4 with OML and OLS baselines added

The codebase already gives us: ICA estimator (`ica.ica_treatment_effect_estimation`), Ortho ML / Robust Ortho ML / Higher-Order OML (`main_estimation.all_together_cross_fitting`), the `ablation_utils.run_parallel_experiments` Joblib dispatcher, and a five-method results layout
`[Ortho ML, OML/HOML, RobustOrthoEst, RobustOrthoSplit, ICA]` (`ablation_utils.py:25-29`). The plan respects this layout and appends new methods as additional indices to avoid breaking existing plotting code.

---

## Section 1 — File-by-file change map

### 1.1 New module: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/baselines.py`

Single home for the two new estimators. Keep dependencies tight (`numpy`, `scikit-learn`) so the cluster image needs no rebuild.

**Functions to define:**

```python
def ols_baseline(
    covariates: np.ndarray,           # (n, d)
    treatment: np.ndarray,            # (n,) or (n, m)
    outcome: np.ndarray,              # (n,)
    fit_intercept: bool = True,
) -> np.ndarray:
    """OLS of outcome on [treatment, covariates]. Returns theta of shape (m,)
    extracted from the first m coefficients. Internally uses
    sklearn.linear_model.LinearRegression. Handles both univariate (m=1) and
    multivariate (m>=2) treatment with no branching at the call site —
    treatment is reshaped to (n, m) on entry."""

def matching_baseline(
    covariates: np.ndarray,           # (n, d)
    treatment: np.ndarray,            # (n,) — continuous or binary
    outcome: np.ndarray,              # (n,)
    n_neighbors: int = 5,
    treatment_kind: str = "auto",    # "binary" | "continuous" | "auto"
) -> float:
    """k-NN matching on the propensity / GPS residual. For binary T:
    propensity-score matching using sklearn.linear_model.LogisticRegression
    + sklearn.neighbors.NearestNeighbors, ATE estimated by averaging
    matched-pair outcome differences. For continuous T: residualize T on
    X via Lasso (matches main_estimation.py's nuisance choice), then run
    1-NN-on-residual matching (Imbens-Rubin style GPS-matching), estimate
    theta from the slope of (Y_i - Y_match) on (T_i - T_match) via OLS.
    Returns scalar theta. NO causalml/EconML dependency — keep stdlib +
    sklearn so the cluster image is not invalidated."""
```

**Why a custom matching impl, not `causalml`/`EconML`:** the cluster venv at
`/is/cluster/fast/preizinger/nl-causal-representations/care` is shared
across projects (`run_experiment.sh:23`) — adding heavy deps (numba via
causalml, tf via econml) is a footgun. Stick to `numpy + sklearn`. The
matching estimator is a baseline; it does not need to be SOTA.

**Multi-treatment OLS (for §1.5):** `ols_baseline` already accepts `treatment`
of shape `(n, m)` and returns theta of shape `(m,)` — no second function
needed.

---

### 1.2 New module: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/semi_synthetic_data.py`

Loader + DGP for the semi-synthetic experiment.

```python
def load_real_covariates(
    name: str = "california_housing",   # "california_housing" | "ihdp"
    n_samples: int | None = None,
    standardize: bool = True,
    seed: int = 12143,
) -> np.ndarray:
    """Returns covariate matrix X. For california_housing: uses
    sklearn.datasets.fetch_california_housing (no download on cluster if
    sklearn data is cached; see Risk R3). For ihdp: loads from
    semi_synthetic/ihdp_npci_1.npz (must be vendored in the repo;
    fetch_california_housing is the safer default). Standardizes
    columns to zero mean / unit variance when standardize=True."""

def generate_semi_synthetic_pl(
    X: np.ndarray,
    treatment_effect: float = 1.0,
    eta_distribution: str = "discrete",   # uses oml_runner.setup_treatment_noise
    treatment_coef_scalar: float = 1.0,
    outcome_coef_scalar: float = 1.0,
    sigma_outcome: float = np.sqrt(3.0),
    nonlinearity: str = "identity",       # "identity" | "leaky_relu"
    support_size: int | None = None,      # None => use all columns
    seed: int = 12143,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply the PLR mechanism to real X:
        T = nonlinearity(X[:, support] @ treatment_coef) + eta
        Y = treatment_effect * T + nonlinearity(X[:, support] @ outcome_coef)
            + epsilon
    Returns (X, T, Y, ground_truth_metadata). Reuses
    oml_runner.setup_treatment_noise() so eta is identical to the
    fully-synthetic experiments — apples-to-apples comparison."""
```

`semi_synthetic_data.py` belongs at the top level (not in a subdir) to
match the flat-package layout of the rest of the codebase.

---

### 1.3 Modified: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/main_estimation.py`

**No new estimator goes here.** This file is the canonical home for OML
estimators only. Adding OLS/matching here would mix abstractions (LassoCV
nuisance machinery vs. trivial OLS).

The only edit is adding two lines to the `all_together_cross_fitting`
docstring noting that `baselines.ols_baseline` and `baselines.matching_baseline`
are the parallel "non-OML" baselines. No code change.

---

### 1.4 Modified: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/ablation_utils.py`

This is the central per-experiment dispatcher (`run_single_experiment` +
`run_parallel_experiments`, lines 423–511). Threading new estimators
through here gives every existing ablation (filtered_heatmap, variance,
coefficient, default) the OLS and matching baselines for free.

**Changes:**

1. Extend `METHOD_NAMES` and add new indices (line 25-32):
   ```
   ORTHO_ML_IDX = 0
   HOML_IDX = 1
   ROBUST_ORTHO_EST_IDX = 2
   ROBUST_ORTHO_SPLIT_IDX = 3
   ICA_IDX = 4
   OLS_IDX = 5            # NEW
   MATCHING_IDX = 6       # NEW
   METHOD_NAMES = ["Ortho ML", "OML", "Robust Ortho Est",
                   "Robust Ortho Split", "ICA", "OLS", "Matching"]
   ```

2. In `run_single_experiment` (the function called by `delayed(...)` at
   line 490), after the existing ICA call, add two extra calls:
   ```python
   ols_estimate = baselines.ols_baseline(covariates, treatment, outcome)
   matching_estimate = baselines.matching_baseline(covariates, treatment, outcome,
                                                   treatment_kind="auto")
   ```
   and append both to the returned tuple.

3. Update `extract_treatment_estimates` (line 519) to slice out the two
   new entries:
   ```python
   ortho_rec_tau = [
       [ortho_ml, robust_ortho_ml, robust_ortho_est_ml,
        robust_ortho_est_split_ml] + ica_estimate.tolist()
       + [ols_estimate, matching_estimate]   # NEW
       for ... in results
   ]
   ```

4. Add a `disable_baselines: bool = False` kwarg to
   `run_parallel_experiments` so the heaviest sweeps (sweep_all_experiments
   has 300+ jobs) can opt out if matching becomes a runtime bottleneck.

**Backwards compatibility:** all existing plotting code reads `biases[ICA_IDX]`,
`rmse[HOML_IDX]`, etc. — appending past index 4 leaves those untouched.

---

### 1.5 Modified: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/monte_carlo_single_instance.py`

Same conceptual change as `ablation_utils.py`, but in the dual code path
used by Fig 2 / Fig E.5 / `single_instance_seed`.

The `experiment()` function (lines 38-122) already returns a tuple
`(*all_together_cross_fitting(...), ica_treatment_effect_estimate, ica_mcc)`.
After ICA, append:
```python
ols_estimate = baselines.ols_baseline(covariates, treatment, outcome)
matching_estimate = baselines.matching_baseline(covariates, treatment, outcome)
return (*all_together_cross_fitting(...), ica_..., ica_mcc,
        ols_estimate, matching_estimate)
```

`run_experiments_for_configuration` (line 125) builds `ortho_rec_tau` at
line 310-314. Update the list comprehension to unpack and append the new
entries; the `result_dict` schema (line 335) does not need to change —
`ortho_rec_tau` is the per-experiment list and downstream plotting is
indexed by method-position.

CLI flag to add:
```
--disable_baselines       (default False) skip OLS/matching for legacy reproducibility
```

---

### 1.6 Modified: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/eta_ablation_experiments.py`

Once `ablation_utils.run_single_experiment` returns OLS/matching, every
ablation function in this file (`run_noise_ablation_experiments`,
`run_variance_ablation_experiments`, `run_coefficient_ablation_experiments`,
`run_sample_dimension_grid_experiments`) inherits them automatically via
`extract_treatment_estimates`.

Per-function edits are limited to the `method_names` list at line 358:
extend to 7 names. The Bernoulli experiment hooks in via the existing
distribution dispatch (`oml_runner.setup_treatment_noise`) — see §1.7.

---

### 1.7 Modified: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/oml_runner.py`

Add an asymmetric Bernoulli case alongside the existing rademacher case
(line 106-115). The rebuttal asks for "Bernoulli({0,1}) centred", i.e. a
Bernoulli(p) shifted to zero mean, distinct from the symmetric Rademacher
already in Fig E.5.

```python
elif distribution == "bernoulli":
    # Bernoulli(p) on {0, 1}, centred so E[eta]=0 → support {-p, 1-p}
    # Variance = p*(1-p), parameterized so Var = scale^2 by setting p
    # implicitly via scale; default p=0.3 (asymmetric, distinct from
    # rademacher's p=0.5).
    p = gennorm_beta if gennorm_beta is not None else 0.3
    centred_support = np.array([1.0 - p, -p])
    probs = np.array([p, 1.0 - p])
    discounts = centred_support  # reuses the discrete-style return
    def eta_sample(x):
        return np.random.choice(centred_support, size=x, p=probs)
    return discounts, eta_sample, 0.0, probs
```

Note we reuse `gennorm_beta` to carry `p` rather than adding a new kwarg
— the call sites (`eta_ablation_experiments.parse_distribution_spec`,
line 100-120) already accept `"bernoulli:0.3"`-style specs. Add `bernoulli`
to the `parse_distribution_spec` switch.

The rebuttal also asks to verify Rademacher behaves as advertised — no
change needed; it is already line 106-115.

---

### 1.8 New file: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/multi_treatment_runner.py`

The repo has `plot_utils.plot_multi_treatment` and
`regenerate_ica_heatmaps.regenerate_main_multi` that consume
`results_multi_treatment.npy`, but the producer is **not in the repo** —
it was either deleted or external. Fig 4 cannot be regenerated without it.

This file is the missing producer:

```python
def run_multi_treatment_experiment(
    sample_sizes: list[int]   = [500, 1000, 2000, 5000, 10000],
    n_treatments_grid: list[int] = [1, 2, 5],
    n_covariates_grid: list[int] = [10, 20, 50],
    n_experiments: int = 20,
    nonlinearity: str = "identity",      # linear PLR for the rebuttal
    eta_distribution: str = "discrete",
    output_path: str = "figures/multi_treatment/results_multi_treatment.npy",
    include_baselines: bool = True,      # NEW: OML + OLS baselines
    seed: int = 12143,
) -> dict:
    """Runs the Fig 4 grid: for each (n, m, d) triple, n_experiments
    Monte Carlo reps. Each rep:
      1. ica.generate_ica_data(n_covariates=d, n_treatments=m, batch_size=n)
      2. ica_treatment_effect_estimation → theta_hat (m,)
      3. If include_baselines:
         - ols_baseline → theta_hat_ols (m,)
         - For m=1: also call all_together_cross_fitting → HOML
         - For m>=2: vector-output Ortho ML — see note below
    Returns a dict with the same key schema regenerate_ica_heatmaps.py
    expects: 'sample_sizes', 'n_treatments', 'n_covariates',
    'true_params', 'treatment_effects', 'treatment_effects_iv', and adds
    new keys 'treatment_effects_ols', 'treatment_effects_oml'."""
```

**Note on vector-output Ortho ML for m ≥ 2:** the existing
`all_together_cross_fitting` is scalar-output (`main_estimation.py:142`).
For m ≥ 2 we need to either (a) run it m times, once per treatment
column, treating others as nuisance; or (b) call it column-wise on the
residualised treatment matrix. Option (a) is simpler, faithful to the
PLR scalar-treatment derivation, and what Mackey et al. (2018) do
implicitly. Default to (a) and document the choice. **Do not** introduce
a new vector-OML estimator — out of scope for the rebuttal.

CLI: this script gets its own `__main__` argparse block with
`--n_experiments`, `--sample_sizes`, `--n_treatments`, `--n_covariates`,
`--nonlinearity`, `--eta_distribution`, `--no_baselines`, `--output_dir`.

---

### 1.9 New entry: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/bernoulli_experiment.py`

The Bernoulli experiment is structurally identical to the Fig E.5 noise
ablation — a noise-distribution sweep over the Bernoulli (and Rademacher)
families, varying (a, b, θ, n) grids.

We deliberately do **not** add another script; instead, the Bernoulli
experiment is just `eta_noise_ablation.py --distributions bernoulli:0.3
bernoulli:0.5 rademacher --output_dir figures/bernoulli`, dispatched via
the new `bernoulli_experiment` cluster type (§3). One CLI argument added
to `eta_noise_ablation.py`:
```
--bernoulli_ps  nargs='+'  type=float  default=[0.3, 0.5, 0.7]
```
which, when set, expands to `bernoulli:p` distribution specs. This keeps
the Bernoulli case in the existing dispatcher and yields a Fig-E.5-shaped
output automatically (a, b, θ grids reuse `--randomize_coeffs`).

So: **no new file** for Bernoulli — extend `eta_noise_ablation.py` only.

---

### 1.10 New file: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/semi_synthetic_experiment.py`

Top-level driver for the semi-synthetic experiment. Imports
`semi_synthetic_data`, `baselines`, `main_estimation`, `ica`. Argparse
flags:
```
--dataset            "california_housing" (default) | "ihdp"
--n_samples          int (default = use all available)
--n_experiments      int (default 20)
--treatment_effect   float (default 1.0)
--eta_distribution   str  (default "discrete")
--nonlinearity       str  (default "identity")
--support_size       int  (default None → all dims)
--seed               int  (default 12143)
--output_dir         str  (default "figures/semi_synthetic")
--methods            comma-sep subset of {ica,homl,ols,matching}
                     (default "all")
```
Outputs:
* `<output_dir>/semi_synthetic_<dataset>_results.npy` — list of dicts
  (one per MC rep) with all five method estimates + ground truth.
* `<output_dir>/semi_synthetic_<dataset>_summary.{png,svg}` — bar plot of
  bias / std / RMSE per method.

---

### 1.11 Other touch-ups

* `requirements.txt`: no changes needed (sklearn already present).
* `pyproject.toml`: no changes.
* `README.md`: append a short "Rebuttal experiments" section pointing to
  the new entry points.

---

## Section 2 — Experiment-by-experiment specs

Conventions reused from existing code:
* Default `n_experiments = 20` (matches `eta_ablation_experiments.run_*`
  defaults at e.g. line 376).
* Default sample sizes from `eta_noise_ablation.py:148`: `[500, 1000,
  2000, 5000, 10000]`.
* Default sigma_outcome = √3.
* `seed = 12143`.
* Output paths follow the existing `figures/<experiment>/` structure
  observed in `figures/filtered_heatmap/`, `figures/variance_ablation/`,
  `figures/coefficient_ablation/`.

### Experiment 1 — OLS baseline (single + multi-treatment)

Not a standalone experiment. OLS is added as a method to **every**
existing experiment (Fig 2, Fig E.5, Fig 4, all eta-ablation runs).
Implementation is the §1.4 / §1.5 / §1.8 changes; nothing new to schedule.

Maps to: Fig 2, Fig E.5, Fig 4, Tab E.1, Tab E.2.

Runtime: zero added — OLS is O(n d²). Negligible vs. LassoCV.

### Experiment 2 — Matching baseline

Added as a method everywhere via §1.4. To validate it independently, run
a one-shot sanity experiment using `semi_synthetic_experiment.py
--methods matching,ols`.

Runtime: ~2× LassoCV per call (5-NN search dominates for n=5000). For
n=5000 and d=10, ~3s/rep on a single core. With n_experiments=20 and the
full eta-ablation grid (~150 configs), expect ~2.5 cpu-hours added per
ablation script — manageable on the existing 4-CPU jobs.

Maps to: Fig 2, Fig E.5, Tab E.1, Tab E.2 (the "general regime" tables).

### Experiment 3 — Bernoulli treatment-noise experiment

Entrypoint: `eta_noise_ablation.py` (extended per §1.9).

Exact CLI:
```
python eta_noise_ablation.py \
  --distributions bernoulli:0.3 bernoulli:0.5 bernoulli:0.7 rademacher \
  --randomize_coeffs --n_random_configs 20 \
  --treatment_effect_range 0.001 0.2 \
  --treatment_coef_range -10 10 \
  --outcome_coef_range -0.5 0.5 \
  --n_samples 5000 --n_experiments 20 \
  --output_dir figures/bernoulli
```

Parameter grid:
* (a, b, θ) — randomized via `--randomize_coeffs`, 20 configs (matches
  Tab E.2 protocol).
* Sample sizes — `[500, 1000, 2000, 5000, 10000]` (from Fig E.5 layout).
  Run as 5 separate jobs varying `--n_samples`.
* Bernoulli p ∈ {0.3, 0.5, 0.7}; Rademacher (≡ p=0.5 binary) for cross-check.

Output:
* `figures/bernoulli/noise_ablation_results_n20_*.npy` — raw results.
* `figures/bernoulli/noise_ablation_coeff_scatter.svg` — Fig E.5-style.
* `figures/bernoulli/noise_ablation_std_scatter.svg`
* `figures/bernoulli/diff_heatmap_*.svg`

Maps to: new Bernoulli figure mirroring Fig E.5; placed in §4.2 next to
the Rademacher panel.

Runtime per (n_samples, distribution, config) combo: ~10 min on 8 CPUs
for n=5000. Total: 5 sample sizes × 4 distributions × 20 configs × 1 job
= 400 configs × 20 reps. Wrap as 5 jobs (one per n_samples), each ~3.5
hours.

### Experiment 4 — Semi-synthetic experiment

Entrypoint: `semi_synthetic_experiment.py` (new, §1.10).

Exact CLI:
```
python semi_synthetic_experiment.py \
  --dataset california_housing \
  --n_samples 5000 --n_experiments 20 \
  --eta_distribution discrete \
  --nonlinearity identity \
  --treatment_effect 1.0 \
  --methods ica,homl,ols,matching \
  --output_dir figures/semi_synthetic
```

Parameter grid:
* dataset ∈ {california_housing}. (IHDP only if vendoring is approved —
  see Risk R3.)
* eta_distribution ∈ {discrete, laplace, rademacher, bernoulli:0.3} —
  4 jobs.
* nonlinearity ∈ {identity, leaky_relu} — 2 jobs each → 8 total.
* treatment_effect ∈ {0.5, 1.0, 2.0} — repeat for sensitivity.

Output:
* `figures/semi_synthetic/semi_synthetic_california_housing_results.npy`
* `figures/semi_synthetic/semi_synthetic_california_housing_summary.svg`
* `figures/semi_synthetic/semi_synthetic_california_housing_summary.md`
  (a markdown table mirroring `eta_ablation_experiments.print_*_summary`).

Maps to: new semi-synthetic figure / table in §4.

Runtime: California Housing has n≈20640, d=8. Drawing 5000-sample MC
reps from this fixed pool, n_experiments=20, total config count
4×2×3 = 24, each ≈ 5 min on 4 CPUs → ~2 cpu-hours total. One cluster job.

### Experiment 5 — Fig 4 with OML/OLS baselines

Entrypoint: `multi_treatment_runner.py` (new, §1.8).

Exact CLI:
```
python multi_treatment_runner.py \
  --n_experiments 20 \
  --sample_sizes 500 1000 2000 5000 10000 \
  --n_treatments 1 2 5 \
  --n_covariates 10 20 50 \
  --nonlinearity identity \
  --eta_distribution discrete \
  --include_baselines \
  --output_dir figures/multi_treatment
```

Parameter grid:
* sample_sizes = [500, 1000, 2000, 5000, 10000] (5)
* n_treatments = [1, 2, 5] (3) — matches Fig 4 / Fig E.15.
* n_covariates = [10, 20, 50] (3) — matches Fig 4 right.
* n_experiments = 20.
* Total configs = 5 × 3 × 3 = 45; total MC reps = 900.

Output:
* `figures/multi_treatment/results_multi_treatment.npy` — schema must
  match the existing `regenerate_ica_heatmaps.regenerate_main_multi`
  consumer (`regenerate_ica_heatmaps.py:188-210`): keys
  `sample_sizes`, `n_treatments`, `n_covariates`, `true_params`,
  `treatment_effects` (ICA), `treatment_effects_iv` (legacy OML);
  **adds** `treatment_effects_ols`, `treatment_effects_homl`.
* Re-run `regenerate_ica_heatmaps.py` to produce the heatmaps —
  modify `regenerate_main_multi` to plot one extra panel per baseline.

Maps to: Fig 4 (revised), Fig E.15 (revised).

Runtime: dominated by FastICA on (n=10000, d=50, m=5) which converges in
~30s per rep × 20 reps = 10 min. Smaller configs are seconds. Total
≈ 4 cpu-hours on a single 8-CPU job.

---

## Section 3 — Cluster integration

### 3.1 New experiment_type cases in `cluster/run_experiment.sh`

Append before the `*)` default case (line 96 of `run_experiment.sh`):

```bash
"bernoulli_experiment")
    python -u eta_noise_ablation.py \
      --distributions bernoulli:0.3 bernoulli:0.5 bernoulli:0.7 rademacher \
      --output_dir "${OUTPUT_DIR}/bernoulli" ${EXPERIMENT_ARGS}
    ;;

"semi_synthetic")
    python -u semi_synthetic_experiment.py \
      --output_dir "${OUTPUT_DIR}/semi_synthetic" ${EXPERIMENT_ARGS}
    ;;

"multi_treatment_with_baselines")
    python -u multi_treatment_runner.py \
      --output_dir "${OUTPUT_DIR}/multi_treatment" \
      --include_baselines ${EXPERIMENT_ARGS}
    ;;
```

Update the help text at line 9-12 and 99-105 to list the new types.

### 3.2 Submit-file strategy

| Experiment              | Where it goes                                   |
|-------------------------|--------------------------------------------------|
| OLS / Matching baselines | No new submit. Re-run `sweep_all_experiments.sub` once `ablation_utils.py` is updated; the existing queue clauses pick up baselines for free. |
| Bernoulli               | New section in `sweep_rebuttal.sub` (below).    |
| Semi-synthetic          | New section in `sweep_rebuttal.sub`.             |
| Fig 4 + baselines       | New section in `sweep_rebuttal.sub`.             |

### 3.3 New file: `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/cluster/sweep_rebuttal.sub`

Single submit covering experiments 3, 4, 5. Pattern follows
`sweep_all_experiments.sub` (which already has multi-section structure).

```
# HTCondor submit file for UAI 2026 rebuttal experiments
executable = run_experiment.sh
request_cpus = 8
request_memory = 32GB
request_disk = 20GB
+MaxRuntime = 86400
should_transfer_files = NO
getenv = True

n_experiments = 20

# ===== Section A: Bernoulli noise (5 jobs, one per n_samples) =====
exp_type = bernoulli_experiment
arguments = $(exp_type) --n_samples $(n_samples) --n_experiments $(n_experiments) \
            --randomize_coeffs --n_random_configs 20
queue n_samples in (500, 1000, 2000, 5000, 10000)

# ===== Section B: Semi-synthetic (one job per (eta, nonlinearity)) =====
exp_type = semi_synthetic
arguments = $(exp_type) --dataset california_housing \
            --n_samples 5000 --n_experiments $(n_experiments) \
            --eta_distribution $(eta_dist) --nonlinearity $(nonlin) \
            --treatment_effect $(theta)
queue eta_dist, nonlin, theta from (
    discrete,         identity,   1.0
    laplace,          identity,   1.0
    rademacher,       identity,   1.0
    bernoulli:0.3,    identity,   1.0
    discrete,         leaky_relu, 1.0
    laplace,          leaky_relu, 1.0
    discrete,         identity,   0.5
    discrete,         identity,   2.0
)

# ===== Section C: Multi-treatment with baselines (1 job, full grid) =====
exp_type = multi_treatment_with_baselines
arguments = $(exp_type) --n_experiments $(n_experiments) \
            --sample_sizes 500 1000 2000 5000 10000 \
            --n_treatments 1 2 5 \
            --n_covariates 10 20 50
queue 1
```

### 3.4 Resource requests — per experiment

| Experiment            | request_cpus | request_memory | MaxTime  | Justification                                                                                               |
|-----------------------|--------------|----------------|----------|-------------------------------------------------------------------------------------------------------------|
| Bernoulli             | 8            | 32 GB          | 14400 s (4h) | Joblib `n_jobs=-1` over 20 reps × 20 random configs = 400 calls at ≈30s each. 8 CPUs is the existing eta-ablation default (`sweep_eta_ablation.sub:7`). |
| Semi-synthetic        | 4            | 16 GB          | 7200 s (2h)  | 24 configs × 20 reps × ~5s/rep = ~40 min, +headroom. n=5000 fits in 16 GB easily.                          |
| Multi-treatment       | 8            | 32 GB          | 21600 s (6h) | FastICA at (n=10000, d=50, m=5) is the bottleneck; 45 configs × 20 reps with worst-case ~30s/rep = 7.5h serial / ~1h on 8 cores. |
| Re-run sweep_all      | 4            | 32 GB          | 259200 s (3d, unchanged) | Already provisioned in `sweep_all_experiments.sub:21-24`. Adding OLS+matching adds ~30% per-config cost; budget is fine. |

### 3.5 Re-running existing sweeps

After §1.4-1.7 land, re-submit `cluster/sweep_all_experiments.sub` and
`cluster/sweep_eta_ablation.sub` to regenerate Fig 2 / Fig E.5 with the
two new baselines. **Do not** delete existing `.npy` files — the
existence-check at e.g. `eta_noise_ablation.py:228, 262, 334, 460` will
short-circuit and skip re-running. Either delete those files explicitly
before the cluster submit, or add a `--force_rerun` flag to the
ablation scripts (recommended; one-line argparse + `if not opts.force_rerun and os.path.exists(...)`).

---

## Section 4 — Testing plan

All new tests live under `/Users/patrik.reizinger/Documents/GitHub/ica_causal_effect/tests/`,
using the existing pytest layout (`tests/conftest.py` already adds the
parent dir to `sys.path`).

### 4.1 New: `tests/test_baselines.py`

```python
class TestOLSBaseline:
    def test_ols_recovers_theta_in_linear_sem(self):
        # Deterministic toy SEM: T = X[:, 0] + eta, Y = 2*T + 0.5*X[:, 0] + eps
        # n=5000, eta ~ N(0, 1), eps ~ N(0, 0.1) — Gaussian noise so OLS is BLUE.
        # Assert |theta_hat - 2.0| < 0.05.

    def test_ols_returns_correct_shape_univariate(self):
        # treatment of shape (n,) → returns shape (1,) or scalar consistently.

    def test_ols_handles_multivariate_treatment(self):
        # treatment of shape (n, 3); assert returned theta has shape (3,)
        # and all three components recover their generating values within tol.

    def test_ols_no_intercept_option(self):
        # With fit_intercept=False on centred data, returned theta unchanged.

class TestMatchingBaseline:
    def test_matching_returns_scalar(self):
        # Smoke test: random data → finite scalar.

    def test_matching_bias_bound_continuous_T(self):
        # PLR with continuous T, theta=1.5, n=5000. Assert
        # |theta_hat - 1.5| < 0.5 (loose — matching is biased; we just
        # want "in the right neighbourhood").

    def test_matching_binary_T_propensity_path(self):
        # Bernoulli T, true ATE = 1.0. Assert |theta_hat - 1.0| < 0.3.

    def test_matching_treatment_kind_auto_detect(self):
        # Pass {0, 1} treatment with treatment_kind="auto" → goes through
        # binary path. Pass continuous → goes through GPS path.
```

### 4.2 New: `tests/test_semi_synthetic_data.py`

```python
class TestLoadRealCovariates:
    def test_california_housing_shape(self):
        X = load_real_covariates("california_housing")
        assert X.shape[0] > 10000 and X.shape[1] == 8

    def test_standardize_zero_mean(self):
        X = load_real_covariates("california_housing", standardize=True)
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-2)

    def test_n_samples_truncation(self):
        X = load_real_covariates("california_housing", n_samples=500)
        assert X.shape[0] == 500

class TestGenerateSemiSyntheticPLR:
    def test_outputs_have_correct_shapes(self):
        X = np.random.randn(1000, 8)
        X_out, T, Y, meta = generate_semi_synthetic_pl(X, treatment_effect=1.0)
        assert X_out.shape == (1000, 8)
        assert T.shape == (1000,)
        assert Y.shape == (1000,)

    def test_ground_truth_metadata(self):
        # meta must contain 'treatment_effect', 'treatment_coef',
        # 'outcome_coef', 'eta_distribution'.

    def test_eta_distribution_dispatch(self):
        # Pass eta_distribution="bernoulli" → empirical kurtosis matches
        # bernoulli's expected kurtosis within 10%.
```

### 4.3 New: `tests/test_bernoulli_dgp.py`

```python
def test_setup_treatment_noise_bernoulli_zero_mean():
    _, eta_sample, _, _ = setup_treatment_noise("bernoulli:0.3")
    samples = eta_sample(100000)
    assert abs(samples.mean()) < 0.01

def test_setup_treatment_noise_bernoulli_variance():
    # Var = p*(1-p) for p=0.3 ⇒ 0.21
    _, eta_sample, _, _ = setup_treatment_noise("bernoulli:0.3")
    samples = eta_sample(100000)
    assert abs(samples.var() - 0.21) < 0.01

def test_setup_treatment_noise_bernoulli_kurtosis_matches_theory():
    # Excess kurtosis of Bernoulli(p) = (1 - 6 p (1-p)) / (p (1-p))
    # For p=0.3: 1.05/0.21 ≈ 5.0
    samples = eta_sample(200000)
    assert abs(scipy_kurtosis(samples) - 5.0) < 0.5

def test_parse_distribution_spec_bernoulli():
    name, p = parse_distribution_spec("bernoulli:0.4")
    assert name == "bernoulli" and p == 0.4
```

### 4.4 New: `tests/test_multi_treatment_runner.py`

```python
def test_runner_produces_legacy_npy_schema():
    # Runs a tiny (sample_sizes=[200], n_treatments=[1, 2], n_cov=[5])
    # config and asserts the output dict has the keys
    # {'sample_sizes', 'n_treatments', 'n_covariates', 'true_params',
    #  'treatment_effects', 'treatment_effects_iv',
    #  'treatment_effects_ols', 'treatment_effects_homl'}.

def test_runner_baselines_finite():
    # All baseline outputs are finite (no NaN propagation).

def test_runner_consistent_across_m():
    # m=1 path returns scalars; m=2 returns shape (2,); m=5 returns (5,).
```

### 4.5 Modified: `tests/test_main_estimation.py` and `tests/test_ica.py`

No new test methods — but add a regression test guarding the assumption
that `run_single_experiment` returns 9 elements (was 7). Failure to
update existing call sites would break this test loudly.

### 4.6 Run all tests via:
```
pytest --tb=short -q tests/
```
Pre-existing tests must still pass.

---

## Section 5 — Agent dispatch plan

Reasoning: estimators (§1.1, §1.2) are leaves; everything else depends
on them. Tests are written first per the project convention (`testing.md`
rule). The cluster submit and experiment scripts are independent of each
other once estimators land.

### 5.1 Sequential phases

```
Phase 0 — Test scaffolding
└─ tdd-guide:   write tests/test_baselines.py + tests/test_semi_synthetic_data.py
                + tests/test_bernoulli_dgp.py + tests/test_multi_treatment_runner.py
                (FAILING tests, defining the contracts for §1.1-§1.10).

Phase 1 — Core estimators (parallel)
├─ executor:        baselines.py             (§1.1)
├─ executor:        semi_synthetic_data.py   (§1.2)
└─ executor:        oml_runner.py + parse_distribution_spec patch (§1.7)

[GATE] test-runner: pytest tests/test_baselines.py tests/test_semi_synthetic_data.py
                    tests/test_bernoulli_dgp.py — all green.

Phase 2 — Dispatcher integration (sequential, single agent)
└─ executor-high:   ablation_utils.py + monte_carlo_single_instance.py
                    + eta_ablation_experiments.py method-name updates (§1.4-§1.6).

[GATE] test-runner: pytest tests/test_main_estimation.py tests/test_ica.py
                    tests/test_baselines.py — all green; baselines now appear
                    in the result tuples.

Phase 3 — New experiments (parallel)
├─ executor-high:   multi_treatment_runner.py        (§1.8)
├─ executor:        semi_synthetic_experiment.py     (§1.10)
└─ executor:        eta_noise_ablation.py extension for --bernoulli_ps (§1.9)

[GATE] test-runner: pytest tests/test_multi_treatment_runner.py + smoke runs
                    of each new entrypoint with tiny grids.

Phase 4 — Cluster integration (sequential)
└─ executor:        run_experiment.sh + sweep_rebuttal.sub (§3).

[GATE] test-runner: dry-run condor_submit -dump on sweep_rebuttal.sub to
                    verify queue expansion is sane (no actual submission).

Phase 5 — Final verification
└─ test-runner:     full pytest suite + lint (black, isort, ruff/pylint).
```

### 5.2 Critical path

```
test scaffolding  →  baselines.py  →  ablation_utils integration  →  multi_treatment_runner  →  cluster wiring
        (5 min)        (45 min)            (60 min)                       (90 min)              (15 min)
```

Total wall-clock if Phase 1 / Phase 3 truly parallelize: ~3.5 hours of
agent time. Sequential worst-case: ~6 hours.

### 5.3 Parallelisation map

| Phase | Parallelisable items | Why                                   |
|-------|----------------------|---------------------------------------|
| 0     | All four test files  | Independent files.                    |
| 1     | 3 files              | No cross-dependencies.                |
| 2     | None                 | Single-purpose integration.           |
| 3     | 3 entrypoints        | Each consumes Phase 1 + Phase 2 only. |
| 4     | None                 | Cluster files reference each other.   |

---

## Section 6 — Risk register

| ID  | Risk                                                                                       | Likelihood | Impact | Mitigation                                                                                                                                                                                                                                                                                                          |
|-----|--------------------------------------------------------------------------------------------|------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R1  | **Matching dependency footprint.** `causalml` / `EconML` would invalidate the cluster venv. | Med        | High   | Custom k-NN matching in `baselines.matching_baseline` using only `numpy` + `sklearn`. Verified via `tests/test_baselines.py::TestMatchingBaseline`. Document the choice in the rebuttal: "we use a residual-based k-NN matching estimator (Imbens & Rubin 2015 Ch. 18) rather than a third-party library". |
| R2  | **Munkres permutation breaks for m > 2.** `ica.ica_treatment_effect_estimation` permutes the mixing matrix via `munkres_sort_idx` (`ica.py:243`); for m=5 with d=50 the permutation matrix has 56 components and Munkres can return ties or near-ties under noise. | Med        | High   | (a) `multi_treatment_runner.py` should also call `ica_treatment_effect_estimation_eps_row` (line 253) which does NOT depend on Munkres — store both estimates. (b) Failing-mode unit test in `test_multi_treatment_runner.py` for m=5: assert at least one of the two ICA variants returns finite values per rep. (c) Document fallback in the figure caption.       |
| R3  | **Semi-synthetic dataset download size on cluster.** `fetch_california_housing` requires internet on first call. The cluster job may be sandboxed. | High       | Med    | (a) Pre-fetch on a login node (one-time `python -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()"`); the file lands in `~/scikit_learn_data/`. (b) Vendor a copy under `data/california_housing.npz` + add a loader fallback that reads it. (c) `requirements.txt` already pins sklearn; no change. (d) AVOID IHDP unless someone vendors `ihdp_npci_1.npz` explicitly — do not rely on remote download.                                                                                                                              |
| R4  | **FastICA convergence on Bernoulli noise.** Discrete distributions have point masses → can stall FastICA's Newton iteration. Fig E.5 already includes Rademacher with no reported issue, but asymmetric Bernoulli is new. | Low        | Med    | Existing retry logic in `ica.ica_treatment_effect_estimation:213-239` increments `random_state` for up to 5 attempts. (a) Run a 1000-rep smoke test for `bernoulli:0.3` before launching the cluster sweep. (b) If failure rate >10%, fall back to `ica_treatment_effect_estimation_eps_row` and document. (c) Set `--check_convergence` in the cluster invocation so failed runs are filtered, not silently NaN-poisoning the aggregates.        |
| R5  | **`results_multi_treatment.npy` schema drift.** New baseline keys must NOT break the legacy consumer in `regenerate_ica_heatmaps.regenerate_main_multi`. | Low        | Med    | Add new keys `treatment_effects_ols`, `treatment_effects_homl` only — never rename or remove existing keys. The consumer ignores unknown keys (line 192-197 reads only the keys it cares about). Test in `test_multi_treatment_runner::test_runner_produces_legacy_npy_schema`.                                                                                |
| R6  | **Sweep re-run double-counts.** The existence checks (`eta_noise_ablation.py:228 etc.`) skip re-running, so the fresh OLS/matching columns will not appear in old `.npy` files. | High       | High   | Add `--force_rerun` argparse flag (one-liner) to `eta_noise_ablation.py` and `monte_carlo_single_instance.py`. In `sweep_rebuttal.sub`, pass `--force_rerun` for the OLS/matching update sweep. Operationally simpler alternative: `find figures -name "*.npy" -newer cluster/sweep_all_experiments.sub -print` to confirm what will rerun.                                                                                       |
| R7  | **OMC scoring still references 5-method index layout.** Plot scripts that index by integer (e.g. `eta_ablation_plotting.plot_*`) may not handle 7 methods. | Med        | Low    | Append-only ordering keeps OLS_IDX=5, MATCHING_IDX=6 — existing index lookups are unaffected. New plot panels for OLS/matching are scoped to the rebuttal (semi-synthetic figure, Fig 4 revision), not the entire pipeline.                                                                                                                              |
| R8  | **Vector-output Ortho ML for m ≥ 2 is not native.** `all_together_cross_fitting` is scalar. | High       | Med    | Use the column-by-column strategy described in §1.8; document explicitly that the multi-treatment OML baseline is the per-coordinate scalar-OML applied to each treatment. This matches the structure in Mackey et al. (2018) and is honest. **Do not** invent a new estimator.                                                                                                                                       |
| R9  | **Pre-commit hooks reformat new files mid-PR.** Repo uses black + isort + pylint pre-commit (CLAUDE.md "Pre-commit hooks automatically run before each commit"). | Med        | Low    | Run `pre-commit run --all-files` after each phase before the test gate. Don't bypass with `--no-verify`.                                                                                                                                                                                                |
| R10 | **Rebuttal text references c_ICA < 1.5 win-rate (96.3%).** Adding OLS to Fig 2 might shift this number when OLS wins in some cells. | Low        | High   | The win-rate is ICA-vs-HOML, not ICA-vs-anyone. OLS is added as a new column, not a comparator for the win-rate calculation. Verify in `eta_ablation_plotting.plot_ica_var_filtered_rmse_heatmap` that the "winner" computation is binary (ICA vs HOML only), and document in code that OLS is informational only.                                                                       |

---

## Appendix — Quick reference: which file produces which figure

| Figure / Table        | Producer script                                      | Add baselines via |
|-----------------------|-------------------------------------------------------|--------------------|
| Fig 2 (Right)         | `monte_carlo_single_instance.py`                      | §1.5               |
| Fig 4 (multi-treatment) | `multi_treatment_runner.py` (NEW, §1.8)              | built-in           |
| Fig E.5 (noise ablation) | `eta_noise_ablation.py` (default mode)             | §1.4 (via ablation_utils) |
| Fig E.13 (nonlinear PLR) | downstream of `monte_carlo_single_instance.py`     | §1.5               |
| Fig E.15 (multi-treatment heatmap) | `multi_treatment_runner.py`                | built-in           |
| Tab E.1               | `eta_ablation_experiments.run_coefficient_ablation_experiments` | §1.4 |
| Tab E.2               | `eta_ablation_experiments.run_noise_ablation_experiments`       | §1.4 |
| Bernoulli figure (NEW) | `eta_noise_ablation.py --distributions bernoulli:*` | §1.7 + §1.9        |
| Semi-synthetic table (NEW) | `semi_synthetic_experiment.py` (NEW, §1.10)     | built-in           |
