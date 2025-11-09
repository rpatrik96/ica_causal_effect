# ICA Diagonal Weights Fix - Complete Summary

## Executive Summary

**Problem**: The current ICA implementation produces a diagonal (or near-diagonal) mixing matrix, which defeats the purpose of Independent Component Analysis for treatment effect estimation.

**Root Cause**: In `ica.py:generate_ica_data()`, covariates are set equal to independent sources (line 72: `X[:, :n_covariates] = S[:, :n_covariates]`), creating an identity mapping that results in a diagonal mixing matrix.

**Solution**: Implemented new data generation functions in `ica_fixed.py` that create proper non-diagonal mixing by generating observed covariates as mixtures of latent sources while preserving the causal structure for treatment effect estimation.

## Files Created

### 1. `ICA_DIAGONAL_ISSUE_ANALYSIS.md`
Comprehensive analysis document explaining:
- Root cause of the diagonal mixing problem
- Why this breaks ICA
- Three proposed solutions with pros/cons
- Recommended approach
- Testing strategy

### 2. `ica_fixed.py`
New module with fixed data generation functions:

#### Main Functions:

**`generate_ica_data_with_mixing()`** - Full-featured fixed implementation
- Creates proper mixing for covariates
- Maintains causal structure (covariates → treatments → outcomes)
- Supports multiple mixing types: random, random_orthogonal, controlled
- Allows extra latent sources
- Returns mixing information for analysis

**`generate_ica_data_simple_mixing()`** - Simplified wrapper
- Easy drop-in replacement for original function
- Uses default random mixing
- Simpler parameter interface

**`generate_ica_data_full_mixing()`** - Pure ICA model
- All variables are linear mixtures
- For testing pure ICA recovery
- May not preserve exact causal interpretation

### 3. `diagnose_ica_diagonal.py`
Diagnostic script that:
- Runs the original data generation
- Fits ICA and examines mixing matrix
- Computes diagonality measures
- Identifies the specific problem (covariates = sources)
- Provides detailed analysis output

**Usage**:
```bash
python diagnose_ica_diagonal.py
```

### 4. `compare_ica_methods.py`
Comprehensive comparison script that:
- Compares original vs fixed methods side-by-side
- Tests three approaches: original, random mixing, orthogonal mixing
- Computes diagonality metrics for each
- Estimates and compares treatment effects
- Generates visualization heatmaps
- Provides statistical comparison

**Usage**:
```bash
python compare_ica_methods.py
```

**Output**:
- Console: Detailed statistics and comparison
- File: `figures/ica/mixing_matrix_comparison.png` (heatmap visualization)

### 5. `test_ica_fixed.py`
Comprehensive test suite with pytest:
- Tests mixing matrix properties (non-diagonal)
- Tests ICA recovery convergence
- Tests treatment effect estimation
- Tests all data generation variants
- Tests different configurations

**Usage**:
```bash
pytest test_ica_fixed.py -v
```

## Key Differences: Original vs Fixed

### Original Implementation (`ica.py`)

```python
# Sources
S = generate_independent_sources()

# Covariates = direct copy (PROBLEM!)
X[:, :n_covariates] = S[:, :n_covariates]  # Identity mapping!

# Treatments = source + small effect
X[:, treatments] = S[:, treatments] + f(covariates)

# Outcome = source + effects
X[:, outcome] = S[:, outcome] + g(treatments, covariates)
```

**Result**: Mixing matrix ≈ diagonal because each X[i] ≈ S[i]

### Fixed Implementation (`ica_fixed.py`)

```python
# Latent sources
S_latent = generate_independent_sources()

# Generate mixing matrix (non-diagonal!)
A = generate_mixing_matrix()  # random, orthogonal, or controlled

# Covariates = MIXTURE of latent sources
X_covariates = (A @ S_latent.T).T  # Proper mixing!

# Treatments = noise + f(mixed_covariates)
X_treatments = noise + f(X_covariates)

# Outcome = noise + g(treatments, mixed_covariates)
X_outcome = noise + g(X_treatments, X_covariates)
```

**Result**: Mixing matrix is non-diagonal with rich structure

## How to Use the Fixed Implementation

### Quick Start (Drop-in Replacement)

```python
from ica_fixed import generate_ica_data_simple_mixing

# Replace:
# from ica import generate_ica_data
# S, X, theta = generate_ica_data(...)

# With:
S, X, theta = generate_ica_data_simple_mixing(
    n_covariates=5,
    n_treatments=1,
    batch_size=1000,
    mixing_strength=1.0,  # Controls mixing intensity
)
```

### Advanced Usage (Full Control)

```python
from ica_fixed import generate_ica_data_with_mixing

S, X, theta, mixing_info = generate_ica_data_with_mixing(
    n_covariates=5,
    n_treatments=1,
    batch_size=1000,
    mixing_type="random_orthogonal",  # Well-conditioned mixing
    mixing_strength=1.0,
    n_extra_latent=2,  # Extra latent factors
    sparse_prob=0.3,
    beta=1.0,
    nonlinearity="leaky_relu",
    slope=0.2,
)

# Access mixing information
print("Covariate mixing matrix:", mixing_info["A_cov"])
print("Treatment effect:", mixing_info["theta"])
```

### Mixing Types

**`mixing_type="random"`**
- Random Gaussian matrix
- Most general, creates rich mixing structure
- Good for testing ICA robustness

**`mixing_type="random_orthogonal"`** (Recommended)
- Random orthogonal matrix
- Well-conditioned (stable numerics)
- Preserves variance
- Good balance of mixing and stability

**`mixing_type="controlled"`**
- Controlled singular values
- Predictable condition number
- Best for numerical stability

## Running the Diagnostics

### 1. Diagnose the Problem

```bash
python diagnose_ica_diagonal.py
```

Expected output showing:
- Original mixing matrix (near-diagonal)
- Diagonality measure ≈ 0.9-1.0 (bad)
- Identification that covariates = sources

### 2. Compare Methods

```bash
python compare_ica_methods.py
```

Expected output showing:
- Three mixing matrices side-by-side
- Original: diagonality ≈ 0.9-1.0
- Fixed methods: diagonality ≈ 0.4-0.7 (much better)
- Improved MCC scores
- Visualization saved to `figures/ica/mixing_matrix_comparison.png`

### 3. Run Tests

```bash
pytest test_ica_fixed.py -v
```

Expected output:
- All tests passing
- Verification of non-diagonal structure
- Confirmation of ICA convergence
- Treatment effect recoverability

## Integration with Existing Code

### Option 1: Update `ica.py` Directly

Replace the `generate_ica_data()` function in `ica.py` with the fixed version:

```python
# At top of ica.py, add:
from ica_fixed import generate_ica_data_with_mixing

# Then modify generate_ica_data to call the fixed version:
def generate_ica_data(
    n_covariates=1,
    n_treatments=1,
    batch_size=4096,
    slope=1.0,
    sparse_prob=0.3,
    beta=1.0,
    loc=0,
    scale=1,
    nonlinearity="leaky_relu",
    theta_choice="fixed",
    split_noise_dist=False,
):
    """Generate ICA data with proper mixing (fixed version)."""
    S, X, theta, _ = generate_ica_data_with_mixing(
        n_covariates=n_covariates,
        n_treatments=n_treatments,
        batch_size=batch_size,
        slope=slope,
        sparse_prob=sparse_prob,
        beta=beta,
        loc=loc,
        scale=scale,
        nonlinearity=nonlinearity,
        theta_choice=theta_choice,
        split_noise_dist=split_noise_dist,
        mixing_type="random_orthogonal",  # Recommended
        mixing_strength=1.0,
        n_extra_latent=0,
    )
    return S, X, theta
```

### Option 2: Create New Experiments

Keep original code intact and create new experiment functions:

```python
# In ica.py or new file
from ica_fixed import generate_ica_data_with_mixing

def main_sparsity_fixed():
    """Sparsity experiment with fixed ICA data generation."""
    # Same as main_sparsity() but using generate_ica_data_with_mixing
    ...
```

### Option 3: Add Flag for Backwards Compatibility

```python
def generate_ica_data(..., use_mixing=True):
    if use_mixing:
        return generate_ica_data_with_mixing(...)
    else:
        # Original implementation
        ...
```

## Expected Improvements

### Quantitative Metrics

**Diagonality Measure** (lower is better):
- Original: 0.90-0.98 (nearly diagonal)
- Fixed: 0.40-0.70 (proper mixing)

**Diagonal/Off-diagonal Ratio** (lower is better):
- Original: 10-100x (diagonal dominates)
- Fixed: 1-3x (balanced structure)

**MCC Scores** (higher is better):
- Depends on specific configuration
- Fixed version should improve or maintain scores
- Better disentanglement of sources

### Qualitative Improvements

1. **Proper ICA Model**: Data now follows standard ICA assumptions
2. **Realistic**: Observed features as mixtures of latent factors
3. **Non-trivial Recovery**: ICA actually has work to do
4. **Numerical Stability**: Especially with orthogonal mixing
5. **Interpretable**: Clear separation of mixing (covariates) vs causal (treatment → outcome)

## Theoretical Justification

### Why This Is Better

The fixed implementation creates a **two-stage model**:

**Stage 1: Latent → Observed (ICA Model)**
```
Latent sources S_latent → [Mixing A] → Observed covariates X_cov
```
This is the proper ICA model where A is non-diagonal.

**Stage 2: Causal Model**
```
X_cov → [Sparse connections] → Treatments
Treatments + X_cov → [Treatment effects] → Outcome
```
This preserves the causal interpretation from the paper.

### Alignment with Paper

From "Independent Component Analysis for Treatment Effect Estimation":

The partially linear model is:
```
Y = θ'T + g(X) + ε
```

Where:
- T: treatments
- X: covariates
- θ: treatment effects
- g(): nonlinear function

Our fixed implementation:
1. **Generates X as ICA mixtures** (realistic latent structure)
2. **Preserves causal graph**: X → T → Y
3. **Maintains identifiability**: Treatment effects still recoverable
4. **Adds non-triviality**: ICA must recover latent structure

## Verification Checklist

Before deploying the fix:

- [ ] Run `python diagnose_ica_diagonal.py` to confirm problem exists
- [ ] Run `python compare_ica_methods.py` to verify fix works
- [ ] Run `pytest test_ica_fixed.py -v` to ensure all tests pass
- [ ] Check that diagonality measure decreases significantly
- [ ] Verify treatment effect estimation still works
- [ ] Review visualization `figures/ica/mixing_matrix_comparison.png`
- [ ] Update existing experiments to use fixed version
- [ ] Re-run key experiments and compare results
- [ ] Update paper/documentation if results change significantly

## Troubleshooting

### Issue: ICA doesn't converge with fixed version

**Solution**: Try different mixing types
```python
# Try orthogonal mixing (more stable)
mixing_type="random_orthogonal"

# Or reduce mixing strength
mixing_strength=0.5
```

### Issue: Treatment effect estimates are worse

**Possible causes**:
1. Mixing strength too high → reduce it
2. Too many extra latent sources → set `n_extra_latent=0`
3. Need more samples → increase `batch_size`

**Diagnosis**:
```python
# Check condition number of mixing matrix
print("Condition number:", np.linalg.cond(mixing_info["A_cov"]))
# Should be < 100 for stability
```

### Issue: Results very different from original

**This is expected!** The original was incorrect (diagonal mixing). New results should be:
- Lower diagonality (good!)
- Different treatment effect estimates (check if more accurate)
- Different MCC scores (may improve or change)

**Action**: Re-baseline experiments with fixed version as new ground truth.

## Next Steps

1. **Immediate**: Run comparison script to verify fix
2. **Short-term**: Integrate into main codebase
3. **Medium-term**: Re-run all experiments with fixed version
4. **Long-term**: Update paper if results change significantly

## References

- Original implementation: `ica.py:18-92` (generate_ica_data)
- Fixed implementation: `ica_fixed.py` (this provides multiple options)
- Analysis: `ICA_DIAGONAL_ISSUE_ANALYSIS.md`
- Tests: `test_ica_fixed.py`

## Contact

For questions about this fix, refer to:
- Analysis document: `ICA_DIAGONAL_ISSUE_ANALYSIS.md`
- Comparison output: Run `compare_ica_methods.py`
- Test suite: `pytest test_ica_fixed.py -v`
