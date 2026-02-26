"""
Utility classes and functions for ICA experiments.

This module provides reusable infrastructure for running ICA experiments,
including configuration management, results handling, and analysis utilities.
"""

from dataclasses import dataclass

import numpy as np


def calculate_mse(true_params, est_params, relative_error=True):
    """Calculate mean squared error between true and estimated parameters.

    Args:
        true_params: True parameter values
        est_params: Estimated parameter values
        relative_error: Whether to compute relative error

    Returns:
        Mean squared error or np.nan if estimation failed
    """
    if est_params is not None:
        if relative_error:
            errors = [
                np.linalg.norm((est - true) / (np.linalg.norm(true) + 1e-8))
                for est, true in zip(est_params, true_params)
            ]
        else:
            errors = [np.linalg.norm(est - true) for est, true in zip(est_params, true_params)]
        return np.mean(errors)
    return np.nan


@dataclass
class DataGenerationConfig:
    """Configuration for ICA data generation.

    Attributes:
        n_covariates: Number of covariates
        n_treatments: Number of treatment variables
        batch_size: Number of samples to generate
        slope: Slope parameter for leaky_relu activation
        sparse_prob: Probability of non-zero coefficients in sparse matrix
        beta: Shape parameter for generalized normal distribution
        loc: Location parameter for distribution
        scale: Scale parameter for distribution
        nonlinearity: Type of activation function ('leaky_relu', 'relu', 'sigmoid', 'tanh')
        theta_choice: Method for theta generation ('fixed', 'uniform', 'gaussian')
        split_noise_dist: Whether to use different distributions for covariates and treatments
    """

    n_covariates: int = 50
    n_treatments: int = 1
    batch_size: int = 5000
    slope: float = 1.0
    sparse_prob: float = 0.3
    beta: float = 1.0
    loc: float = 0
    scale: float = 1
    nonlinearity: str = "leaky_relu"
    theta_choice: str = "fixed"
    split_noise_dist: bool = False


@dataclass
class EstimationConfig:
    """Configuration for ICA treatment effect estimation.

    Attributes:
        whiten: Whitening strategy for ICA
        check_convergence: Whether to check for ICA convergence
        verbose: Whether to print verbose output
        fun: Functional form for ICA algorithm
        max_attempts: Maximum number of convergence attempts
    """

    whiten: str = "unit-variance"
    check_convergence: bool = True
    verbose: bool = False
    fun: str = "logcosh"
    max_attempts: int = 5


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution.

    Attributes:
        n_seeds: Number of random seeds to run
        results_file: Path to results file
        output_dir: Directory for output figures
    """

    n_seeds: int = 20
    results_file: str = "results.npy"
    output_dir: str = "figures/ica"
