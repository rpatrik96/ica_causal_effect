"""
Experiment execution framework for OML experiments.

This module provides a unified interface for running Monte Carlo experiments
across different parameter configurations for Orthogonal Machine Learning.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from oml_utils import AsymptoticVarianceCalculator, OMLExperimentConfig, OMLMethodConfig, OMLParameterGrid


class OMLExperimentRunner:
    """Orchestrates OML experiment execution across parameter grids."""

    def __init__(
        self, experiment_config: OMLExperimentConfig, method_config: OMLMethodConfig, parameter_grid: OMLParameterGrid
    ):
        """Initialize the experiment runner.

        Args:
            experiment_config: Configuration for experiment execution
            method_config: Configuration for OML methods
            parameter_grid: Parameter grid for experiments
        """
        self.exp_config = experiment_config
        self.method_config = method_config
        self.param_grid = parameter_grid
        self.var_calculator = AsymptoticVarianceCalculator()

    def run_parameter_sweep(
        self,
        experiment_func: Callable,
        x_sample_func: Callable,
        eta_sample_func: Callable,
        epsilon_sample_func: Callable,
    ) -> List[Dict]:
        """Run experiments across the full parameter grid.

        Args:
            experiment_func: Function to run single experiment
            x_sample_func: Function to generate covariate samples
            eta_sample_func: Function to generate treatment noise samples
            epsilon_sample_func: Function to generate outcome noise samples

        Returns:
            List of result dictionaries
        """
        all_results = []

        # Create parameter combinations
        for n_samples in self.param_grid.data_samples:
            print(f"{n_samples=}")

            for treatment_coefficient in self.param_grid.treatment_coefs:
                for outcome_coefficient in self.param_grid.outcome_coefs:
                    for support_size in self.param_grid.support_sizes:
                        print(f"{support_size=}")

                        for beta in self.param_grid.beta_values:
                            print(f"{beta=}")

                            for treatment_effect in self.param_grid.treatment_effects:
                                print(f"{treatment_effect=}")

                                # Adjust sample size if tie_sample_dim is enabled
                                current_n_samples = n_samples
                                if self.exp_config.asymptotic_var and self.exp_config.tie_sample_dim:
                                    current_n_samples = support_size**4

                                # Run experiments for this configuration
                                result = self._run_single_configuration(
                                    n_samples=current_n_samples,
                                    support_size=support_size,
                                    beta=beta,
                                    treatment_effect=treatment_effect,
                                    treatment_coefficient=treatment_coefficient,
                                    outcome_coefficient=outcome_coefficient,
                                    experiment_func=experiment_func,
                                    x_sample_func=x_sample_func,
                                    eta_sample_func=eta_sample_func,
                                    epsilon_sample_func=epsilon_sample_func,
                                )

                                if result is not None:
                                    all_results.append(result)

        return all_results

    def _run_single_configuration(
        self,
        n_samples: int,
        support_size: int,
        beta: float,
        treatment_effect: float,
        treatment_coefficient: float,  # pylint: disable=unused-argument
        outcome_coefficient: float,  # pylint: disable=unused-argument
        experiment_func: Callable,
        x_sample_func: Callable,
        eta_sample_func: Callable,
        epsilon_sample_func: Callable,
    ) -> Optional[Dict]:
        """Run experiments for a single parameter configuration.

        Args:
            n_samples: Number of samples
            support_size: Support size
            beta: Beta parameter
            treatment_effect: Treatment effect value
            treatment_coefficient: Treatment coefficient value
            outcome_coefficient: Outcome coefficient value
            experiment_func: Function to run single experiment
            x_sample_func: Function to generate covariate samples
            eta_sample_func: Function to generate treatment noise samples
            epsilon_sample_func: Function to generate outcome noise samples

        Returns:
            Dictionary with results or None if no experiments converged
        """
        # Setup coefficients (this would typically be done by a helper function)
        # For now, we'll pass these through as-is
        # In the actual refactoring, this would call setup_treatment_outcome_coefs

        # Compute lambda regularization parameter
        cov_dim_max = self.param_grid.support_sizes[-1]
        lambda_reg = np.sqrt(np.log(cov_dim_max) / n_samples)

        # Run parallel experiments
        results = self._run_parallel_experiments(
            n_samples=n_samples,
            cov_dim_max=cov_dim_max,
            treatment_effect=treatment_effect,
            lambda_reg=lambda_reg,
            experiment_func=experiment_func,
            x_sample_func=x_sample_func,
            eta_sample_func=eta_sample_func,
            epsilon_sample_func=epsilon_sample_func,
        )

        if len(results) == 0:
            print("Configuration skipped as no runs converged")
            return None

        # Extract treatment effect estimates
        ortho_rec_tau = [
            [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml]
            + ica_treatment_effect_estimate.tolist()
            for (
                ortho_ml,
                robust_ortho_ml,
                robust_ortho_est_ml,
                robust_ortho_est_split_ml,
                _,
                _,
                ica_treatment_effect_estimate,
                _,
            ) in results
        ]

        # Extract first-stage metrics
        # Note: true_coef_treatment and true_coef_outcome need to be passed/computed
        # This is a placeholder that will be properly implemented during integration
        first_stage_mse = [
            [
                0.0,  # Placeholder for np.linalg.norm(true_coef_treatment - coef_treatment)
                0.0,  # Placeholder for np.linalg.norm(true_coef_outcome - coef_outcome)
                np.linalg.norm(ica_treatment_effect_estimate - treatment_effect),
                ica_mcc,
            ]
            for (
                _,
                _,
                _,
                _,
                coef_treatment,
                coef_outcome,
                ica_treatment_effect_estimate,
                ica_mcc,
            ) in results
        ]

        print(f"Experiments kept: {len(ortho_rec_tau)} out of {self.exp_config.n_experiments} seeds")

        # Create result dictionary
        result_dict = {
            "n_samples": n_samples,
            "support_size": support_size,
            "beta": beta,
            "treatment_effect": treatment_effect,
            "cov_dim_max": cov_dim_max,
            "sigma_outcome": self.exp_config.sigma_outcome,
            "ortho_rec_tau": ortho_rec_tau,
            "first_stage_mse": first_stage_mse,
        }

        return result_dict

    def _run_parallel_experiments(
        self,
        n_samples: int,
        cov_dim_max: int,
        treatment_effect: float,
        lambda_reg: float,
        experiment_func: Callable,
        x_sample_func: Callable,
        eta_sample_func: Callable,
        epsilon_sample_func: Callable,
    ) -> List[Tuple]:
        """Run experiments in parallel using joblib.

        Args:
            n_samples: Number of samples
            cov_dim_max: Maximum covariate dimension
            treatment_effect: Treatment effect value
            lambda_reg: Regularization parameter
            experiment_func: Function to run single experiment
            x_sample_func: Function to generate covariate samples
            eta_sample_func: Function to generate treatment noise samples
            epsilon_sample_func: Function to generate outcome noise samples

        Returns:
            List of experiment results
        """
        results = [
            r
            for r in Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_func)(
                    x_sample_func(n_samples, cov_dim_max),
                    eta_sample_func(n_samples),
                    epsilon_sample_func(n_samples),
                    treatment_effect,
                    # Additional parameters would be passed here
                    lambda_reg,
                    self.method_config.check_convergence,
                    self.method_config.verbose,
                )
                for _ in range(self.exp_config.n_experiments)
            )
            if (self.exp_config.check_convergence is False or r[-1] is not None)
        ]

        return results


def setup_treatment_noise(rademacher: bool = False) -> Tuple[np.ndarray, Callable, float, np.ndarray]:
    """Setup treatment noise distribution.

    Args:
        rademacher: Whether to use Rademacher distribution

    Returns:
        Tuple of (discounts, eta_sample, mean_discount, probs)
    """
    if not rademacher:
        discounts = np.array([0, -0.5, -2.0, -4.0])
        probs = np.array([0.65, 0.2, 0.1, 0.05])
    else:
        discounts = np.array([1, -1])
        probs = np.array([0.5, 0.5])

    mean_discount = np.dot(discounts, probs)

    def eta_sample(x):
        return np.array([discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)])

    return discounts, eta_sample, mean_discount, probs


def setup_covariate_pdf(config: OMLExperimentConfig, beta: float) -> Callable:
    """Setup covariate sampling function based on PDF choice.

    Args:
        config: Experiment configuration
        beta: Beta parameter for generalized normal distribution

    Returns:
        Function to sample covariates
    """
    if config.covariate_pdf == "gauss":
        return lambda n, d: np.random.normal(size=(n, d))
    if config.covariate_pdf == "uniform":
        return lambda n, d: np.random.uniform(-1, 1, size=(n, d))
    if config.covariate_pdf == "gennorm":
        from scipy.stats import gennorm

        return lambda n, d: gennorm.rvs(beta, size=(n, d))

    raise ValueError(f"Unknown covariate PDF: {config.covariate_pdf}")


def setup_treatment_outcome_coefs(
    cov_dim_max: int,
    config: OMLExperimentConfig,
    outcome_coef_array: Optional[np.ndarray],
    outcome_coef_list: np.ndarray,
    outcome_coefficient: Optional[float],
    support_size: int,
    treatment_coef_array: Optional[np.ndarray],
    treatment_coef_list: np.ndarray,
    treatment_coefficient: Optional[float],
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Setup treatment and outcome coefficients and supports.

    Args:
        cov_dim_max: Maximum covariate dimension
        config: Experiment configuration
        outcome_coef_array: Pre-generated outcome coefficient array (if not scalar)
        outcome_coef_list: List to store outcome coefficients
        outcome_coefficient: Scalar outcome coefficient (if scalar_coeffs=True)
        support_size: Support size
        treatment_coef_array: Pre-generated treatment coefficient array (if not scalar)
        treatment_coef_list: List to store treatment coefficients
        treatment_coefficient: Scalar treatment coefficient (if scalar_coeffs=True)

    Returns:
        Tuple of (outcome_coef, outcome_coefficient, outcome_support,
                 treatment_coef, treatment_support)
    """
    treatment_support = np.random.choice(cov_dim_max, size=support_size, replace=False)
    outcome_support = treatment_support

    if config.scalar_coeffs:
        treatment_coef_list[treatment_support] = treatment_coefficient
        outcome_coef_list[outcome_support] = outcome_coefficient
        treatment_coef = treatment_coef_list[treatment_support]
        outcome_coef = outcome_coef_list[outcome_support]
    else:
        treatment_coef = treatment_coef_array[treatment_support]
        outcome_coef = outcome_coef_array[outcome_support]

    return outcome_coef, outcome_coefficient, outcome_support, treatment_coef, treatment_support
