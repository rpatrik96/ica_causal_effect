# ICA Diagonal Mixing Matrix Issue - Analysis and Solutions

## Problem Statement

The current ICA implementation in `ica.py` produces a diagonal (or near-diagonal) mixing matrix, which defeats the purpose of Independent Component Analysis. This happens because the data generation process in `generate_ica_data()` creates observed variables that are largely identical to the independent sources, rather than proper mixtures.

## Root Cause Analysis

### Current Data Generation (Problematic)

In `ica.py:generate_ica_data()` (lines 18-92):

```python
source_dim = n_covariates + n_treatments + 1  # +1 for outcome
S = torch.tensor(distribution.rvs(size=(batch_size, source_dim))).float()
X = S.clone()

# Covariates remain independent
X[:, :n_covariates] = S[:, :n_covariates]  # Identity mapping!

# Treatments
X[:, treatment_indices] = S[:, treatment_indices]  # Start with identity
X[:, treatment_indices] += covariate_effects  # Add small perturbation

# Outcome
X[:, -1] = S[:, -1]  # Start with identity
X[:, -1] += treatment_effects + covariate_effects  # Add perturbations
```

**Issue**: This creates a structural equation model (SEM) where:
- Covariates X[i] = S[i] (perfect identity, contributes to diagonal)
- Treatments X[j] = S[j] + small_effect (dominated by diagonal component)
- Outcome X[k] = S[k] + effects (dominated by diagonal component)

The resulting mixing matrix A in X ≈ A @ S.T is approximately diagonal because each observed variable is primarily its corresponding source plus small additive effects.

### Why This Breaks ICA

ICA assumes the mixing model:
```
X = A @ S.T + noise
```

Where:
- S contains independent latent sources
- A is a **full mixing matrix** (not diagonal)
- Each observed variable is a **mixture of multiple sources**

The current implementation violates this by making X[:, i] ≈ S[:, i] for covariates and X[:, i] ≈ S[:, i] + small_corrections for others.

## Proposed Solutions

### Solution 1: True Linear ICA Mixing (Recommended for Linear Case)

Generate data using a proper random mixing matrix:

```python
def generate_ica_data_with_mixing(
    n_covariates=1,
    n_treatments=1,
    batch_size=4096,
    beta=1.0,
    loc=0,
    scale=1,
    mixing_strength=1.0,
):
    """Generate ICA data with proper linear mixing."""

    # Generate independent sources
    source_dim = n_covariates + n_treatments + 1
    distribution = gennorm(beta, loc=loc, scale=scale)
    S = torch.tensor(distribution.rvs(size=(batch_size, source_dim))).float()

    # Create a random mixing matrix (non-diagonal)
    A = torch.randn(source_dim, source_dim) * mixing_strength

    # Ensure mixing matrix is well-conditioned
    # Option 1: Random orthogonal matrix (preserves variance)
    # A, _ = torch.linalg.qr(torch.randn(source_dim, source_dim))

    # Option 2: Random matrix with controlled condition number
    # U, _, Vt = torch.svd(torch.randn(source_dim, source_dim))
    # A = U @ Vt

    # Mix the sources to create observations
    X = (A @ S.T).T  # X = S @ A.T

    # Define treatment effect parameters (embedded in mixing matrix)
    # The treatment effect can be extracted from the mixing matrix structure
    theta = A[-1, n_covariates:n_covariates+n_treatments]

    return S, X, theta, A
```

**Pros**:
- Proper ICA model - all variables are mixtures
- ICA can recover the mixing matrix
- Mathematically sound

**Cons**:
- Changes the causal interpretation
- Treatment effects are now embedded in mixing matrix structure
- May not align with the partially linear model from the paper

### Solution 2: Modified SEM with Richer Mixing

Keep the causal structure but add proper mixing for covariates:

```python
def generate_ica_data_mixed_covariates(
    n_covariates=1,
    n_treatments=1,
    batch_size=4096,
    slope=1.0,
    sparse_prob=0.3,
    beta=1.0,
    mixing_strength=0.5,
):
    """Generate ICA data with mixed covariates but causal structure."""

    # Generate more sources than observed covariates
    n_latent = n_covariates + 2  # Extra latent sources
    source_dim = n_latent + n_treatments + 1

    distribution = gennorm(beta, loc=loc, scale=scale)
    S = torch.tensor(distribution.rvs(size=(batch_size, source_dim))).float()

    # Create mixing matrix for covariates only
    A_covariates = torch.randn(n_covariates, n_latent) * mixing_strength

    # Mix latent sources to create observed covariates
    X_covariates = (A_covariates @ S[:, :n_latent].T).T

    # Create treatment mixing (sparse connections)
    binary_mask = torch.bernoulli(torch.ones(n_treatments, n_covariates) * sparse_prob)
    random_coeffs = torch.randn(n_treatments, n_covariates)
    A_treatment = binary_mask * random_coeffs

    activation = F.leaky_relu  # or other

    # Treatments depend on mixed covariates
    treatment_noise = S[:, n_latent:n_latent+n_treatments]
    X_treatments = treatment_noise + activation(X_covariates) @ A_treatment.T

    # Outcome depends on treatments and covariates
    theta = torch.tensor([1.55, 0.65, -2.45, 1.75, -1.35])[:n_treatments]
    B = torch.randn(n_covariates)

    outcome_noise = S[:, -1]
    X_outcome = outcome_noise + (theta * X_treatments).sum(dim=1) + \
                (B * activation(X_covariates)).sum(dim=1)

    # Combine into full observation matrix
    X = torch.hstack([X_covariates, X_treatments, X_outcome.reshape(-1, 1)])

    # For ICA, we need to define what S should be
    # Option: Use the mixed covariates + treatment/outcome noise
    S_ica = torch.hstack([X_covariates, treatment_noise, outcome_noise.reshape(-1, 1)])

    return S_ica, X, theta
```

**Pros**:
- Maintains causal structure (treatments → outcomes)
- Covariates are now proper mixtures
- More realistic (observed covariates are often mixtures of latent factors)

**Cons**:
- More complex model
- Still not a pure linear ICA model
- Need to carefully define what "sources" means

### Solution 3: Post-Mixing Transformation (Simplest Fix)

Add a final mixing transformation to the current data:

```python
def generate_ica_data_with_postmixing(
    # ... all original parameters ...
    apply_mixing=True,
    mixing_strength=0.3,
):
    """Original data generation with optional final mixing."""

    # Generate data with original method
    # ... (all the original code) ...

    if apply_mixing:
        # Apply a random rotation/mixing to break diagonal structure
        mixing_matrix = torch.randn(X.shape[1], X.shape[1]) * mixing_strength
        # Make it close to identity but with off-diagonal elements
        mixing_matrix = torch.eye(X.shape[1]) + mixing_matrix
        X = (mixing_matrix @ X.T).T

        # Adjust S to be the pre-mixed version for ICA to recover
        S_original = S.clone()
        S = (torch.linalg.inv(mixing_matrix) @ X.T).T

    return S, X, theta
```

**Pros**:
- Minimal changes to existing code
- Easy to toggle on/off
- Preserves causal relationships in latent space

**Cons**:
- Somewhat artificial
- Mixing is applied after causal structure, which may not be theoretically clean
- May introduce numerical issues

## Recommendation

For the **partially linear model** application in your paper, I recommend **Solution 2** with modifications:

### Recommended Implementation

Create a new data generation function that:

1. **For covariates**: Generate from a proper ICA model with mixing
   - Latent sources → Mixed covariates
   - This is realistic (observed features are often mixtures of latent factors)

2. **For causal relationships**: Maintain the SEM structure
   - Mixed covariates → Treatments (with nonlinearity and sparsity)
   - Treatments + Covariates → Outcome

3. **For ICA recovery**: Define the recovery target appropriately
   - ICA should recover the latent covariate structure
   - Treatment effects can be identified from the partial mixing

This approach:
- Avoids diagonal mixing matrix (covariates are mixed)
- Preserves causal interpretation (treatments affect outcomes)
- Aligns with realistic data generation (features are latent mixtures)
- Makes ICA meaningful (recovers latent covariate structure)

## Testing Strategy

To verify the fix:

1. **Check mixing matrix diagonality**:
   ```python
   diag_norm = np.linalg.norm(np.diag(mixing))
   off_diag_norm = np.linalg.norm(mixing - np.diag(np.diag(mixing)))
   diagonality_ratio = diag_norm / off_diag_norm
   # Should be close to 1.0 for old method, >> 1.0 for fixed method
   ```

2. **Verify ICA recovery**:
   - Compute MCC between true and recovered sources
   - Should improve significantly with proper mixing

3. **Check treatment effect estimation**:
   - Compare estimated vs true treatment effects
   - Should improve or at least not degrade

4. **Visual inspection**:
   - Plot mixing matrix heatmaps
   - Check for off-diagonal structure
