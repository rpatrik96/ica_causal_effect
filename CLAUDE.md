# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements methods from the paper "Independent Component Analysis for Treatment Effect Estimation" ([arXiv:2507.16467](https://arxiv.org/abs/2507.16467)). It compares second-order orthogonal methods with first-order orthogonal estimation methods and ICA-based approaches for the partially linear model.

## Core Architecture

### Estimation Pipeline

The codebase follows a modular pipeline structure:

1. **Data Generation** (`monte_carlo_single_instance.py`, `ica.py:generate_ica_data()`):
   - Generates synthetic data from partially linear model DGPs
   - Supports various distributions (Gaussian, uniform, generalized normal)
   - Configurable treatment/outcome relationships with sparsity patterns

2. **Estimation Methods** (`main_estimation.py`, `ica.py:ica_treatment_effect_estimation()`):
   - `all_together()`: Sample-split based estimation
   - `all_together_cross_fitting()`: Cross-fitted estimation with K-fold
   - Methods implemented:
     - **Ortho ML**: Standard first-order orthogonal method
     - **Robust Ortho ML**: Second-order orthogonal with known moments
     - **Robust Ortho ML (Est.)**: Second-order with estimated moments
     - **Robust Ortho ML (Split)**: Second-order with nested splitting
     - **ICA**: Treatment effect estimation via FastICA

3. **Disentanglement Metrics** (`mcc.py`):
   - Hungarian algorithm (Munkres) for optimal permutation matching
   - R² and MCC (Maximum Correlation Coefficient) calculations
   - Linear and permutation-based disentanglement scores

4. **Visualization** (`plot_utils.py`):
   - Standardized typography settings via `plot_typography()`
   - Method comparison plots, heatmaps, error bars
   - Asymptotic variance comparison plots

### Key Data Flow

```
DGP Parameters → Data Generation → Parallel Estimation → Results Storage (.npy) → Plotting
```

Results are saved as NumPy arrays containing dictionaries with keys:
- `ortho_rec_tau`: Treatment effect estimates from all methods
- `first_stage_mse`: First-stage regression errors
- `biases`, `sigmas`: Bias and standard deviation per method
- `eta_*_moment`, `ica_asymptotic_var`, `homl_asymptotic_var`: Theoretical quantities

## Commands

### Running Experiments

**Main estimation with configurable parameters:**
```bash
python monte_carlo_single_instance.py \
  --n_samples 500 \
  --n_experiments 20 \
  --sigma_outcome 1.732 \
  --covariate_pdf gennorm \
  --output_dir ./figures
```

**ICA functions:**
`ica.py` provides `generate_ica_data()`, `ica_treatment_effect_estimation()`, and `ica_treatment_effect_estimation_eps_row()` as library functions imported by experiment scripts.

**Eta noise ablation experiments (`eta_noise_ablation.py`):**
```bash
# Run filtered heatmap experiments (sample size vs dimension/beta)
python eta_noise_ablation.py --filtered_heatmap

# Constrain ICA variance coefficient to threshold (default 1.5)
python eta_noise_ablation.py --filtered_heatmap --constrain_ica_var

# Customize axis mode and parameters
python eta_noise_ablation.py --filtered_heatmap \
  --heatmap_axis_mode beta_vs_n \
  --ica_var_threshold 2.0 \
  --n_experiments 50

# Run coefficient ablation (varying treatment/outcome coefficients)
python eta_noise_ablation.py --coefficient_ablation

# Run variance ablation (varying noise variance across beta values)
python eta_noise_ablation.py --variance_ablation
```

**Common command-line flags:**
- `--n_samples`: Sample size (default: 500)
- `--n_experiments`: Number of Monte Carlo replications (default: 5)
- `--sigma_outcome`: Outcome noise standard deviation (default: √3)
- `--covariate_pdf`: Distribution for covariates - "gauss", "uniform", or "gennorm" (default: "gennorm")
- `--asymptotic_var`: Enable asymptotic variance analysis (default: False)
- `--check_convergence`: Verify ICA convergence (default: True)
- `--scalar_coeffs`: Use scalar coefficients only (default: True)

**Eta noise ablation flags (`eta_noise_ablation.py`):**
- `--filtered_heatmap`: Run filtered RMSE heatmap experiments
- `--constrain_ica_var`: Automatically compute treatment coefficient to achieve target ICA variance coefficient
- `--ica_var_threshold`: Target ICA variance coefficient threshold (default: 1.5)
- `--heatmap_axis_mode`: Axis mode - "d_vs_n" (dimension vs sample size) or "beta_vs_n" (beta vs sample size)
- `--heatmap_sample_sizes`: Sample sizes for heatmap grid (default: [500, 1000, 2000, 5000, 10000])
- `--heatmap_dimensions`: Covariate dimensions for d_vs_n mode (default: [5, 10, 20, 50])
- `--heatmap_betas`: Beta values for beta_vs_n mode (default: [0.5, 1.0, 2.0, 3.0, 4.0])
- `--coefficient_ablation`: Run coefficient ablation experiments
- `--variance_ablation`: Run variance ablation experiments

### Cluster Experiments

Large-scale experiments run via HTCondor. See `cluster/README.md` for setup and submission instructions.

### Loading and Plotting Existing Results

Results are cached in `.npy` files. If a results file exists, experiments are skipped and data is loaded:

```python
# In monte_carlo_single_instance.py:
if os.path.exists(results_file_path):
    structured_results = np.load(results_file_path, allow_pickle=True)
    all_results = structured_results
```

## Important Implementation Details

### ICA Convergence Handling

FastICA may fail to converge. The code uses a retry mechanism with increasing tolerances:
```python
# ica.py:ica_treatment_effect_estimation()
for attempt in range(5):
    ica = FastICA(n_components=X.shape[1], random_state=random_state+attempt, ...)
    # Returns np.nan if convergence fails and check_convergence=True
```

When `--check_convergence=True`, failed runs are filtered out from results.

### Treatment Effect Extraction from ICA Mixing Matrix

The ICA mixing matrix is permuted and scaled to extract treatment effects:
```python
# Resolve permutations using Munkres
permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
# Normalize to get 1 at epsilon -> Y
permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()
# Extract treatment effect (assumes outcome is last dimension)
treatment_effect_estimate = permuted_scaled_mixing[-1, n_covariates:-1]
```

### Asymptotic Variance Calculations

Two key asymptotic variance formulas are implemented:

**Higher-order OML (HOML):**
```python
homl_asymptotic_var = homl_asymptotic_var_num / eta_non_gauss_cond ** 2
```

**ICA:**
```python
ica_var_coeff = 1 + ||outcome_coef + treatment_coef * treatment_effect||²
ica_asymptotic_var = ica_var_coeff * eta_cubed_variance / eta_excess_kurtosis ** 2
```

### ICA Variance Coefficient Constraint

The `--constrain_ica_var` flag in `eta_noise_ablation.py` automatically computes coefficients to achieve a target ICA variance coefficient:

```python
# eta_noise_ablation.py:compute_ica_var_coeff()
ica_var_coeff = 1 + (outcome_coef_scalar + treatment_coef_scalar * treatment_effect)²

# eta_noise_ablation.py:compute_constrained_treatment_coef()
# Solves for treatment_coef_scalar given target ica_var_coeff:
treatment_coef_scalar = (sqrt(target - 1) - outcome_coef_scalar) / treatment_effect
```

When `--constrain_ica_var` is enabled:
1. If `outcome_coef_scalar` is zero, it's automatically set to 30% of the target coefficient sum
2. `treatment_coef_scalar` is computed to achieve `ica_var_coeff = ica_var_threshold`
3. Validation ensures both coefficients are non-zero

### Cross-Fitting Implementation

`all_together_cross_fitting()` uses 2-fold cross-fitting with nested splits for moment estimation:
- Outer loop: 2-fold split for nuisance function estimation
- Inner loop (for split estimator): Further 2-fold split of test data for moment estimation

This nested structure is critical for the "Robust Ortho ML (Split)" estimator.

## Output Organization

Results are saved in directories structured by experiment parameters:
```
figures/
  n_exp_{n_experiments}_sigma_outcome_{sigma}_pdf_{pdf}/
    error_bars/
    gennorm/
      treatment_effect_{value}/
    asymptotic_var_comparison/
```

ICA-specific plots are saved to:
```
figures/ica/
```

Eta noise ablation outputs:
```
figures/
  filtered_heatmap/
    filtered_heatmap_results_{axis_mode}.npy           # Raw results
    rmse_sample_size_vs_{dim|beta}_homl_filtered_*.svg # HOML RMSE heatmap
    rmse_sample_size_vs_{dim|beta}_ica_filtered_*.svg  # ICA RMSE heatmap
    rmse_sample_size_vs_{dim|beta}_diff_filtered_*.svg # Difference heatmap (ICA - HOML)
  coefficient_ablation/
    coefficient_ablation_results.npy
  variance_ablation/
    variance_ablation_results.npy
```

## Testing and Code Quality

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_main_estimation.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Formatting and Linting

The project uses pre-commit hooks to maintain code quality:

```bash
# Format code with black
black .

# Check linting with pylint
pylint *.py

# Sort imports with isort
isort .

# Run all pre-commit hooks manually
pre-commit run --all-files
```

Pre-commit hooks automatically run before each commit. See `TESTING.md` for detailed documentation.

### Test Coverage

Tests are located in `tests/` directory:
- `test_main_estimation.py` - Estimation methods (all_together, cross-fitting)
- `test_ica.py` - ICA data generation and treatment effect estimation
- `test_mcc.py` - Disentanglement metrics (R², MCC, Munkres)
- `test_plot_utils.py` - Data preparation and plotting utilities
