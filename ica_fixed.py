"""
Fixed ICA data generation with proper mixing matrix (non-diagonal).

This module provides improved data generation functions that create
proper ICA mixing structures while maintaining causal relationships
for treatment effect estimation.
"""

import torch
import torch.nn.functional as F
from scipy.stats import gennorm


def generate_ica_data_with_mixing(
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
    mixing_type="random",  # "random", "random_orthogonal", or "controlled"
    mixing_strength=1.0,
    n_extra_latent=0,  # Extra latent sources for covariates
):
    """
    Generate ICA data with proper non-diagonal mixing matrix.

    This function creates data where observed covariates are mixtures of
    latent sources, avoiding the diagonal mixing matrix problem.

    Args:
        n_covariates: Number of observed covariates
        n_treatments: Number of treatment variables
        batch_size: Number of samples
        slope: Slope for leaky_relu nonlinearity
        sparse_prob: Probability of non-zero entries in treatment-covariate connections
        beta: Shape parameter for generalized normal distribution
        loc: Location parameter for distribution
        scale: Scale parameter for distribution
        nonlinearity: Nonlinearity type for covariate effects
        theta_choice: How to generate treatment effects ("fixed", "uniform", "gaussian")
        split_noise_dist: Whether to use different distributions for X and T,Y
        mixing_type: Type of mixing matrix ("random", "random_orthogonal", "controlled")
        mixing_strength: Scaling factor for mixing matrix elements
        n_extra_latent: Number of extra latent sources beyond n_covariates

    Returns:
        S: Independent latent sources (batch_size, source_dim)
        X: Observed mixed variables (batch_size, n_covariates + n_treatments + 1)
        theta: True treatment effect parameters
        mixing_info: Dictionary with mixing matrix and other info
    """

    # Number of latent sources for covariates
    n_latent_cov = n_covariates + n_extra_latent

    # Total source dimension: latent covariates + treatment noise + outcome noise
    source_dim = n_latent_cov + n_treatments + 1

    # Generate independent sources
    distribution = gennorm(beta, loc=loc, scale=scale)

    if split_noise_dist is False:
        S = torch.tensor(distribution.rvs(size=(batch_size, source_dim))).float()
    else:
        # Different distributions for covariates vs treatments/outcome
        S_cov = torch.tensor(
            gennorm(beta=2.0, loc=loc, scale=scale).rvs(size=(batch_size, n_latent_cov))
        ).float()
        S_rest = torch.tensor(
            distribution.rvs(size=(batch_size, n_treatments + 1))
        ).float()
        S = torch.hstack((S_cov, S_rest))

    # Generate mixing matrix for covariates
    if mixing_type == "random":
        # Random Gaussian mixing matrix
        A_cov = torch.randn(n_covariates, n_latent_cov) * mixing_strength
    elif mixing_type == "random_orthogonal":
        # Random orthogonal matrix (preserves variance, well-conditioned)
        if n_latent_cov >= n_covariates:
            A_temp = torch.randn(n_latent_cov, n_latent_cov)
            Q, _ = torch.linalg.qr(A_temp)
            A_cov = Q[:n_covariates, :] * mixing_strength
        else:
            A_temp = torch.randn(n_covariates, n_covariates)
            Q, _ = torch.linalg.qr(A_temp)
            A_cov = Q[:, :n_latent_cov] * mixing_strength
    elif mixing_type == "controlled":
        # Controlled condition number for stability
        A_temp = torch.randn(n_covariates, n_latent_cov)
        U, s, Vt = torch.svd(A_temp)
        # Set condition number to reasonable value (e.g., 10)
        s = torch.linspace(1.0, 0.1, min(n_covariates, n_latent_cov))
        if n_latent_cov >= n_covariates:
            A_cov = (U * s.unsqueeze(0)) @ Vt * mixing_strength
        else:
            A_cov = U[:, :n_latent_cov] @ (torch.diag(s) @ Vt) * mixing_strength
    else:
        raise ValueError(f"Unknown mixing_type: {mixing_type}")

    # Create mixed covariates
    S_latent_cov = S[:, :n_latent_cov]
    X_covariates = (A_cov @ S_latent_cov.T).T

    # Generate treatment effect parameters
    if theta_choice == "fixed":
        theta = torch.tensor([1.55, 0.65, -2.45, 1.75, -1.35])[:n_treatments]
    elif theta_choice == "uniform":
        theta = torch.rand(n_treatments)
    elif theta_choice == "gaussian":
        theta = torch.randn(n_treatments)
    else:
        raise ValueError(f"Unsupported theta_choice: {theta_choice}")

    # Create sparse treatment-covariate connection matrix
    binary_mask = torch.bernoulli(torch.ones(n_treatments, n_covariates) * sparse_prob)
    random_coeffs = torch.randn(n_treatments, n_covariates)
    A_treatment_cov = binary_mask * random_coeffs

    # Outcome-covariate connection
    B = torch.randn(n_covariates)

    # Define activation function
    activation_functions = {
        "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=slope),
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }

    if nonlinearity not in activation_functions:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    activation = activation_functions[nonlinearity]

    # Generate treatments with causal dependence on covariates
    treatment_noise = S[:, n_latent_cov : n_latent_cov + n_treatments]
    covariate_effects_on_treatment = activation(X_covariates) @ A_treatment_cov.T
    X_treatments = treatment_noise + covariate_effects_on_treatment

    # Generate outcome with causal dependence on treatments and covariates
    outcome_noise = S[:, -1]
    treatment_effects_on_outcome = (theta * X_treatments).sum(dim=1)
    covariate_effects_on_outcome = (B * activation(X_covariates)).sum(dim=1)
    X_outcome = outcome_noise + treatment_effects_on_outcome + covariate_effects_on_outcome

    # Combine all observed variables
    X = torch.hstack([X_covariates, X_treatments, X_outcome.reshape(-1, 1)])

    # Prepare mixing info for analysis
    mixing_info = {
        "A_cov": A_cov.numpy(),  # Covariate mixing matrix
        "A_treatment_cov": A_treatment_cov.numpy(),  # Treatment-covariate connections
        "B": B.numpy(),  # Outcome-covariate connections
        "theta": theta.numpy(),  # Treatment effects
        "n_latent_cov": n_latent_cov,
        "mixing_type": mixing_type,
    }

    return S, X, theta, mixing_info


def generate_ica_data_simple_mixing(
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
    mixing_strength=0.3,
):
    """
    Simplified version with basic random mixing for covariates only.

    This is a simpler alternative that just adds mixing to covariates
    while keeping the rest of the structure similar to the original.

    Args:
        Same as generate_ica_data_with_mixing, but with fewer options.

    Returns:
        S: Independent latent sources
        X: Observed mixed variables
        theta: True treatment effect parameters
    """

    # Use the full function with default settings
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
        mixing_type="random",
        mixing_strength=mixing_strength,
        n_extra_latent=0,
    )

    return S, X, theta


def generate_ica_data_full_mixing(
    n_covariates=1,
    n_treatments=1,
    batch_size=4096,
    beta=1.0,
    loc=0,
    scale=1,
    mixing_strength=1.0,
):
    """
    Generate data with full linear ICA mixing (all variables mixed).

    This creates a pure linear ICA model where all observed variables
    are linear mixtures of all independent sources. Suitable for
    testing ICA recovery but may not preserve causal structure.

    Args:
        n_covariates: Number of covariates
        n_treatments: Number of treatments
        batch_size: Number of samples
        beta: Shape parameter for generalized normal distribution
        loc: Location parameter
        scale: Scale parameter
        mixing_strength: Scaling for mixing matrix

    Returns:
        S: Independent sources
        X: Mixed observations
        theta: Treatment effects (embedded in mixing structure)
        A: Full mixing matrix
    """

    # Total dimension
    total_dim = n_covariates + n_treatments + 1

    # Generate independent sources
    distribution = gennorm(beta, loc=loc, scale=scale)
    S = torch.tensor(distribution.rvs(size=(batch_size, total_dim))).float()

    # Generate random mixing matrix
    A = torch.randn(total_dim, total_dim) * mixing_strength

    # Mix sources
    X = (A @ S.T).T

    # Extract treatment effect from mixing matrix structure
    # (coefficients from treatment sources to outcome)
    theta = A[-1, n_covariates : n_covariates + n_treatments]

    return S, X, theta, A
