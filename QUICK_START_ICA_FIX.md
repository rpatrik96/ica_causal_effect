# Quick Start Guide: ICA Diagonal Weights Fix

## TL;DR

**Problem**: ICA mixing matrix is diagonal → ICA is useless
**Fix**: Use `ica_fixed.py` instead of original `generate_ica_data()`
**Verification**: Run `python compare_ica_methods.py`

## 5-Minute Quick Start

### Step 1: Verify the Problem Exists (Optional)

```bash
# Install dependencies if needed
pip install -r requirements.txt

# Run diagnostic
python diagnose_ica_diagonal.py
```

Look for output like:
```
⚠️  PROBLEM: Covariates are direct copies of sources!
Diagonality measure: 0.95 (1.0 = perfectly diagonal)
```

### Step 2: See the Fix in Action

```bash
# Run comparison script
python compare_ica_methods.py
```

Expected output:
```
SUMMARY COMPARISON
====================
Diagonality Measures (lower is better for ICA):
  Original:         0.95  ← BAD
  Fixed (random):   0.55  ← GOOD
  Fixed (orthog):   0.52  ← GOOD

Visualization saved to: figures/ica/mixing_matrix_comparison.png
```

### Step 3: Use the Fixed Version in Your Code

**Before** (in your existing code):
```python
from ica import generate_ica_data

S, X, theta = generate_ica_data(
    n_covariates=5,
    n_treatments=1,
    batch_size=1000,
)
```

**After** (recommended fix):
```python
from ica_fixed import generate_ica_data_with_mixing

S, X, theta, mixing_info = generate_ica_data_with_mixing(
    n_covariates=5,
    n_treatments=1,
    batch_size=1000,
    mixing_type="random_orthogonal",  # Well-conditioned mixing
    mixing_strength=1.0,
)
```

Or for simpler API:
```python
from ica_fixed import generate_ica_data_simple_mixing

S, X, theta = generate_ica_data_simple_mixing(
    n_covariates=5,
    n_treatments=1,
    batch_size=1000,
)
```

### Step 4: Run Tests

```bash
pytest test_ica_fixed.py -v
```

All tests should pass ✓

## What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `ica_fixed.py` | Fixed implementation | Import this in your code |
| `compare_ica_methods.py` | Side-by-side comparison | Run once to verify fix |
| `diagnose_ica_diagonal.py` | Problem diagnosis | Run to see original problem |
| `test_ica_fixed.py` | Test suite | Run with pytest |
| `ICA_FIX_SUMMARY.md` | Complete documentation | Read for details |
| `ICA_DIAGONAL_ISSUE_ANALYSIS.md` | Technical analysis | Read for theory |

## Choosing Mixing Type

**Use `random_orthogonal` (recommended for most cases)**:
```python
mixing_type="random_orthogonal"
```
- Well-conditioned (numerically stable)
- Preserves variance
- Good balance

**Use `random` (for maximum mixing)**:
```python
mixing_type="random"
```
- Maximum mixing complexity
- Tests ICA robustness
- May be less stable

**Use `controlled` (for specific condition number)**:
```python
mixing_type="controlled"
```
- Predictable conditioning
- Most numerically stable
- Less natural

## Common Integration Patterns

### Pattern 1: Replace Original Function

In `ica.py`, update `generate_ica_data()`:
```python
# Add at top of file
from ica_fixed import generate_ica_data_with_mixing as _generate_fixed

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
    """Generate ICA data with proper mixing (FIXED VERSION)."""
    S, X, theta, _ = _generate_fixed(
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
        mixing_type="random_orthogonal",
        mixing_strength=1.0,
        n_extra_latent=0,
    )
    return S, X, theta
```

### Pattern 2: Add Option Flag

For backwards compatibility:
```python
def generate_ica_data(..., use_proper_mixing=True):
    if use_proper_mixing:
        from ica_fixed import generate_ica_data_with_mixing
        S, X, theta, _ = generate_ica_data_with_mixing(...)
        return S, X, theta
    else:
        # Original implementation (with diagonal mixing)
        ...
```

### Pattern 3: New Experiments

Create new experiment functions:
```python
def main_sparsity_fixed():
    """Sparsity experiment with fixed ICA."""
    from ica_fixed import generate_ica_data_simple_mixing

    # Rest same as main_sparsity() but with generate_ica_data_simple_mixing
    ...
```

## Expected Results After Fix

### Visual Changes
Open `figures/ica/mixing_matrix_comparison.png` - you should see:
- **Left panel** (Original): Near-diagonal heatmap
- **Middle/Right panels** (Fixed): Non-diagonal structure with rich off-diagonal elements

### Numerical Changes
- **Diagonality**: Should drop from ~0.9-1.0 to ~0.4-0.7
- **Ratio**: Should drop from ~10-100 to ~1-3
- **MCC**: May change (this is expected)
- **Treatment effects**: May change (original was wrong!)

## Troubleshooting

### "ImportError: No module named ica_fixed"

Make sure `ica_fixed.py` is in the same directory as your script.

### "ICA doesn't converge"

Try:
```python
mixing_type="random_orthogonal"  # More stable
mixing_strength=0.5  # Reduce mixing intensity
```

### "Results are very different"

**This is expected!** The original had diagonal mixing (wrong). New results are correct.

Action items:
1. Verify fix is working: Run `compare_ica_methods.py`
2. Check diagonality decreased: Should be < 0.7
3. Re-baseline: New results are the correct baseline

### "Treatment effect estimates are worse"

Check if:
- Mixing strength too high → reduce to 0.5-1.0
- Need more samples → increase batch_size
- Mixing matrix ill-conditioned → use `mixing_type="random_orthogonal"`

Debug:
```python
S, X, theta, mixing_info = generate_ica_data_with_mixing(...)
print("Condition number:", np.linalg.cond(mixing_info["A_cov"]))
# Should be < 100
```

## One-Liner Summary

**Before**: Covariates = Sources (diagonal mixing, ICA useless)
**After**: Covariates = Mixtures of Sources (proper ICA model)

## Files to Read

1. **This file** - Quick start ← YOU ARE HERE
2. `ICA_FIX_SUMMARY.md` - Complete guide
3. `ICA_DIAGONAL_ISSUE_ANALYSIS.md` - Technical details

## Still Have Questions?

1. Run the comparison: `python compare_ica_methods.py`
2. Read the full summary: `ICA_FIX_SUMMARY.md`
3. Check the tests: `pytest test_ica_fixed.py -v`
4. Review the analysis: `ICA_DIAGONAL_ISSUE_ANALYSIS.md`
