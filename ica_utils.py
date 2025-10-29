"""
Utility classes and functions for ICA experiments.

This module provides reusable infrastructure for running ICA experiments,
including configuration management, results handling, and analysis utilities.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

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


class ExperimentResultsManager:
    """Manages loading, saving, and caching of experiment results.

    This class handles the common pattern of checking for existing results,
    loading them if available, or creating a new results dictionary.
    """

    def __init__(self, results_file: str):
        """Initialize the results manager.

        Args:
            results_file: Path to the results .npy file
        """
        self.results_file = results_file

    def exists(self) -> bool:
        """Check if the results file exists.

        Returns:
            True if results file exists, False otherwise
        """
        return os.path.exists(self.results_file)

    def load_or_create(self, keys: List[str]) -> dict:
        """Load existing results or create new dictionary.

        Args:
            keys: List of keys to initialize in new dictionary

        Returns:
            Either loaded results dict or new dict with specified keys
        """
        if self.exists():
            print(f"Results file '{self.results_file}' already exists. Loading data.")
            return np.load(self.results_file, allow_pickle=True).item()
        return {key: [] for key in keys}

    def save(self, results_dict: dict):
        """Save results dictionary to file.

        Args:
            results_dict: Dictionary containing experiment results
        """
        np.save(self.results_file, results_dict)
        print(f"Results saved to '{self.results_file}'")


class ResultsAnalyzer:
    """Analyzes experiment results and computes statistics.

    This class provides methods for computing MSE, filtering results,
    and preparing data for visualization.
    """

    @staticmethod
    def compute_method_statistics(
        results_dict: dict,
        indices: List[int],
        method_key: str = "treatment_effects",
        true_param_key: str = "true_params",
        relative_error: bool = True,
    ) -> Tuple[float, float]:
        """Compute mean and standard deviation of errors for a method.

        Args:
            results_dict: Dictionary containing experiment results
            indices: Indices of results to analyze
            method_key: Key for estimated parameters in results_dict
            true_param_key: Key for true parameters in results_dict
            relative_error: Whether to compute relative error

        Returns:
            Tuple of (mean_error, std_error)
        """
        est_params = [results_dict[method_key][i] for i in indices]
        true_params = [results_dict[true_param_key][i].numpy() for i in indices]

        errors = [calculate_mse(true, est, relative_error) for true, est in zip(true_params, est_params)]

        return np.nanmean(errors), np.nanstd(errors)

    @staticmethod
    def filter_by_parameter(results_dict: dict, parameter_key: str, parameter_value: any) -> List[int]:
        """Get indices of results matching a parameter value.

        Args:
            results_dict: Dictionary containing experiment results
            parameter_key: Key of parameter to filter by
            parameter_value: Value to match

        Returns:
            List of indices where parameter matches value
        """
        return [i for i, val in enumerate(results_dict[parameter_key]) if val == parameter_value]

    @staticmethod
    def prepare_error_bar_data(
        results_dict: dict, parameter_key: str, method_key: str = "treatment_effects", relative_error: bool = True
    ) -> Tuple[List, List[float], List[float]]:
        """Prepare data for error bar plotting.

        Args:
            results_dict: Dictionary containing experiment results
            parameter_key: Key of parameter to group by
            method_key: Key for estimated parameters
            relative_error: Whether to compute relative error

        Returns:
            Tuple of (parameter_values, means, stds)
        """
        parameter_values = sorted(set(results_dict[parameter_key]))
        means = []
        stds = []

        for param_val in parameter_values:
            indices = ResultsAnalyzer.filter_by_parameter(results_dict, parameter_key, param_val)

            est_params = [results_dict[method_key][i] for i in indices]
            true_params = [results_dict["true_params"][i].numpy() for i in indices]

            errors = [calculate_mse(true, est, relative_error) for true, est in zip(true_params, est_params)]

            means.append(np.nanmean(errors))
            stds.append(np.nanstd(errors))

        return parameter_values, means, stds


def initialize_results_dict(keys: List[str]) -> dict:
    """Initialize results dictionary with specified keys.

    Args:
        keys: List of keys to initialize

    Returns:
        Dictionary with keys initialized to empty lists
    """
    return {key: [] for key in keys}
