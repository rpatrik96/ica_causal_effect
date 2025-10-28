# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements methods from the paper "Independent Component Analysis for Treatment Effect Estimation" ([arXiv:2507.16467](https://arxiv.org/abs/2507.16467)). It compares second-order orthogonal methods with first-order orthogonal estimation methods and ICA-based approaches for the partially linear model.

## Core Architecture

### Estimation Pipeline

The codebase follows a modular pipeline structure:

1. **Data Generation** (`monte_carlo_single_instance.py`, `monte_carlo_single_instance_with_seed.py`, `ica.py:generate_ica_data()`):
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

**ICA-specific experiments:**
```bash
# Run sparsity ablation
python ica.py  # Runs main_sparsity() by default (see __main__ block)

# Other ICA experiments (uncomment in ica.py __main__ block):
# - main_multi(): Multiple treatment effects
# - main_nonlinear(): Nonlinear partially linear regression
# - main_gennorm(): Generalized normal distributions
# - main_gennorm_nonlinear(): Nonlinear with gennorm
# - main_nonlinear_theta(): Different theta distributions
# - main_nonlinear_noise_split(): Noise distribution ablations
```

**Common command-line flags:**
- `--n_samples`: Sample size (default: 500)
- `--n_experiments`: Number of Monte Carlo replications (default: 5)
- `--sigma_outcome`: Outcome noise standard deviation (default: √3)
- `--covariate_pdf`: Distribution for covariates - "gauss", "uniform", or "gennorm" (default: "gennorm")
- `--asymptotic_var`: Enable asymptotic variance analysis (default: False)
- `--check_convergence`: Verify ICA convergence (default: True)
- `--scalar_coeffs`: Use scalar coefficients only (default: True)

### Shell Scripts for Large-Scale Experiments

**Single instance parameter sweep:**
```bash
./single_instance_parameter_sweep.sh <output_directory>
```
Sweeps over sample sizes, dimensions, support sizes, and outcome noise levels.

**Multiple instances with fixed parameters:**
```bash
./multi_instance.sh <output_directory>
```
Runs 100 seeds for stability analysis.

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
    recovered_coefficients/
    model_errors/
    error_bars/
    gennorm/
      treatment_effect_{value}/
    asymptotic_var_comparison/
```

ICA-specific plots are saved to:
```
figures/ica/
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
