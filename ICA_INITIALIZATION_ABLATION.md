# ICA Initialization Ablation Study

This document describes the ablation study for comparing different initialization strategies across triangular, constrained, and regularized ICA methods.

## Overview

This ablation study investigates how different initialization strategies affect the performance of three ICA variants:

1. **Triangular ICA**: Enforces a triangular structure on the unmixing matrix
2. **Constrained ICA**: Supports orthogonality and non-negativity constraints
3. **Regularized ICA**: Adds L1/L2 regularization to the ICA objective

For each variant, we compare three initialization strategies:
- `random_triangular`: Random triangular matrix initialization
- `standard`: Standard random initialization
- `identity`: Identity matrix initialization

## Files Added

### Core Implementation

- **`ica_variants.py`**: Implementation of the three ICA variants
  - `TriangularICA`: ICA with triangular unmixing matrix constraint
  - `ConstrainedICA`: ICA with orthogonality/non-negativity constraints
  - `RegularizedICA`: ICA with L1/L2 regularization
  - `random_triangular_matrix()`: Utility for generating random triangular matrices
  - `whiten_data()`: ZCA whitening implementation
  - `ica_treatment_effect_estimation_variant()`: Wrapper function for treatment effect estimation

- **`ica_initialization_ablation.py`**: Main ablation study script
  - `run_initialization_ablation()`: Basic ablation across variants and initializations
  - `plot_ablation_results()`: Visualization of ablation results
  - `run_extended_ablation()`: Extended ablation across sparsity levels
  - `plot_sparsity_ablation()`: Sparsity-specific visualizations

### Testing

- **`tests/test_ica_variants.py`**: Comprehensive test suite for ICA variants
  - Tests for random triangular matrix generation
  - Tests for whitening functionality
  - Tests for each ICA variant
  - Integration tests with data generation
  - Comparison tests across methods

- **`validate_ica_variants.py`**: Quick validation script (no pytest required)

## Usage

### Basic Ablation Study

Run the basic ablation study with default parameters:

```bash
python ica_initialization_ablation.py
```

This will:
1. Generate synthetic data using the existing `generate_ica_data()` function
2. Test all combinations of variants × initializations across 20 random seeds
3. Save results to `figures/ica/initialization_ablation/`
4. Create visualizations:
   - Absolute error by initialization
   - Relative error by initialization
   - MCC scores by initialization
   - Convergence rate heatmap
   - Initialization comparison violin plots

### Extended Ablation Study

To run the extended ablation across different sparsity levels, uncomment the relevant section in `main()`:

```python
# In ica_initialization_ablation.py
def main():
    # ...

    # Uncomment this block:
    print("\n3. Running extended ablation across sparsity levels...")
    extended_results_df = run_extended_ablation(n_seeds=10)
```

### Custom Parameters

You can customize the ablation study parameters:

```python
from ica_initialization_ablation import run_initialization_ablation, plot_ablation_results

# Custom configuration
results_df = run_initialization_ablation(
    n_covariates=100,
    n_treatments=2,
    batch_size=10000,
    n_seeds=50,
    beta=2.0,
    sparse_prob=0.5,
    nonlinearity="relu"
)

plot_ablation_results(results_df)
```

## Results Structure

Results are saved in CSV format with the following columns:

- `seed`: Random seed for the experiment
- `variant`: ICA variant (`triangular`, `constrained`, or `regularized`)
- `initialization`: Initialization method (`random_triangular`, `standard`, or `identity`)
- `theta_true`: True treatment effect
- `theta_est`: Estimated treatment effect
- `abs_error`: Absolute estimation error
- `rel_error`: Relative estimation error
- `mcc`: Maximum Correlation Coefficient (disentanglement score)
- `converged`: Whether the ICA algorithm converged

### Output Files

**Basic ablation:**
- `figures/ica/initialization_ablation/initialization_ablation_results.csv`
- `figures/ica/initialization_ablation/summary_statistics.csv`
- Various PDF plots

**Extended ablation:**
- `figures/ica/initialization_ablation_extended/extended_ablation_results.csv`
- `figures/ica/initialization_ablation_extended/sparse_{prob}/` (per-sparsity results)

## Running Tests

### Quick Validation (No pytest required)

```bash
python validate_ica_variants.py
```

This runs basic sanity checks to ensure the implementation works correctly.

### Full Test Suite

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run all tests:

```bash
pytest tests/test_ica_variants.py -v
```

Run specific test classes:

```bash
# Test triangular matrix generation
pytest tests/test_ica_variants.py::TestRandomTriangularMatrix -v

# Test TriangularICA
pytest tests/test_ica_variants.py::TestTriangularICA -v

# Test integration with data generation
pytest tests/test_ica_variants.py::TestICAVariantIntegration -v
```

Run with coverage:

```bash
pytest tests/test_ica_variants.py --cov=. --cov-report=html
```

## Implementation Details

### Triangular ICA

The `TriangularICA` class enforces a lower (or upper) triangular structure on the unmixing matrix. This is useful for identifying causal orderings in the data.

**Key features:**
- Triangular constraint applied via masking after each gradient step
- Supports both lower and upper triangular matrices
- Random triangular initialization option

**Parameters:**
- `n_components`: Number of components to extract
- `lower`: Use lower (True) or upper (False) triangular
- `init`: Initialization method
- `max_iter`: Maximum iterations (default: 1000)
- `learning_rate`: Adam optimizer learning rate (default: 0.01)

### Constrained ICA

The `ConstrainedICA` class supports additional constraints:
- **Orthogonality**: Projects unmixing matrix onto orthogonal matrices via SVD
- **Non-negativity**: Enforces all elements ≥ 0

**Parameters:**
- `orthogonal`: Enable orthogonality constraint (default: False)
- `non_negative`: Enable non-negativity constraint (default: False)

### Regularized ICA

The `RegularizedICA` class adds regularization terms to the ICA objective:
- **L1 penalty**: Promotes sparsity in the unmixing matrix
- **L2 penalty**: Encourages smaller magnitudes (smoothness)

**Objective:**
```
L = -Σ H(s_i) + λ₁ ||W||₁ + λ₂ ||W||₂²
```

where H(s_i) is the negentropy of component i.

**Parameters:**
- `l1_penalty`: L1 regularization strength (default: 0.0)
- `l2_penalty`: L2 regularization strength (default: 0.0)

### Initialization Strategies

**Random Triangular (`random_triangular`):**
- Generates random Gaussian values
- Applies triangular mask (upper or lower)
- Ensures diagonal elements are non-zero
- Best for triangular ICA when enforcing causal structure

**Standard Random (`standard`):**
- Standard random Gaussian initialization
- No special structure
- Default for most ICA applications

**Identity (`identity`):**
- Initializes unmixing matrix to identity
- Can help with convergence when sources are nearly independent
- May get stuck in local optima

### Negentropy Approximation

All variants use the logcosh approximation for negentropy:

```python
H(s) ≈ E[log(cosh(s))]
```

This is a good compromise between robustness and computational efficiency.

## Expected Results

Based on the ICA literature and the treatment effect estimation setting, we expect:

1. **Triangular ICA**: Should perform well when the true mixing matrix has a triangular or near-triangular structure
2. **Random triangular initialization**: May help convergence for triangular ICA
3. **Convergence rates**: May vary across initializations, especially for constrained methods
4. **MCC scores**: Should be higher for methods that better match the data generation process

## Integration with Existing Code

This ablation study integrates seamlessly with the existing codebase:

- Uses `generate_ica_data()` from `ica.py` for data generation
- Uses `calc_disent_metrics()` from `mcc.py` for disentanglement evaluation
- Uses `calculate_mse()` from `ica_utils.py` for error computation
- Uses `plot_typography()` from `plot_utils.py` for consistent plotting style
- Follows the same structure as existing ablation studies in `ica.py`

## Future Extensions

Potential extensions to this ablation study:

1. **Additional initialization strategies:**
   - PCA-based initialization
   - Whitening matrix initialization
   - Data-driven initialization

2. **Additional ICA variants:**
   - FastICA (comparison baseline)
   - Sparse ICA with different sparsity penalties
   - Temporal ICA for time-series data

3. **Additional evaluation metrics:**
   - Amari distance for permutation and scaling
   - Component-wise correlation analysis
   - Computational efficiency metrics

4. **Parameter sweeps:**
   - Learning rate sensitivity
   - Regularization strength ablation
   - Constraint strength analysis

## References

- Original paper: "Independent Component Analysis for Treatment Effect Estimation" ([arXiv:2507.16467](https://arxiv.org/abs/2507.16467))
- FastICA: Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications
- Triangular ICA: Related to LiNGAM (Linear Non-Gaussian Acyclic Model) methods
