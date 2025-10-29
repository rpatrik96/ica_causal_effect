"""
Experiment execution framework for ICA experiments.

This module provides a unified interface for running parameter sweeps
across different experimental configurations.
"""

import itertools
from typing import Any, Callable, Dict, List

from ica import generate_ica_data, ica_treatment_effect_estimation
from ica_utils import DataGenerationConfig, EstimationConfig


class ExperimentRunner:
    """Orchestrates ICA experiment execution across parameter grids.

    This class eliminates duplicate experiment loop code by providing
    a unified interface for parameter sweeps.
    """

    def __init__(self, estimation_config: EstimationConfig = None):
        """Initialize the experiment runner.

        Args:
            estimation_config: Configuration for ICA estimation
        """
        self.estimation_config = estimation_config or EstimationConfig()

    def run_parameter_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        n_seeds: int,
        data_gen_config: DataGenerationConfig,
        results_dict: dict,
        result_callback: Callable = None,
    ) -> dict:
        """Run experiments across a parameter grid.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            n_seeds: Number of random seeds to run for each configuration
            data_gen_config: Base configuration for data generation
            results_dict: Dictionary to store results (modified in-place)
            result_callback: Optional callback function to store custom results
                           Signature: callback(results_dict, config, true_params, treatment_effects, mcc)

        Returns:
            Updated results_dict with experiment results
        """
        # Create all parameter combinations
        param_names = list(param_grid.keys())
        param_values_list = list(param_grid.values())

        for param_values in itertools.product(*param_values_list):
            # Create configuration for this parameter combination
            config_dict = dict(zip(param_names, param_values))

            # Update data generation config with current parameters
            current_config = self._update_config(data_gen_config, config_dict)

            # Run experiments for this configuration
            for seed in range(n_seeds):
                # Generate data
                S, X, true_params = generate_ica_data(
                    n_covariates=current_config.n_covariates,
                    n_treatments=current_config.n_treatments,
                    batch_size=current_config.batch_size,
                    slope=current_config.slope,
                    sparse_prob=current_config.sparse_prob,
                    beta=current_config.beta,
                    loc=current_config.loc,
                    scale=current_config.scale,
                    nonlinearity=current_config.nonlinearity,
                    theta_choice=current_config.theta_choice,
                    split_noise_dist=current_config.split_noise_dist,
                )

                # Run estimation
                treatment_effects, mcc = ica_treatment_effect_estimation(
                    X,
                    S,
                    random_state=seed,
                    whiten=self.estimation_config.whiten,
                    check_convergence=self.estimation_config.check_convergence,
                    n_treatments=current_config.n_treatments,
                    verbose=self.estimation_config.verbose,
                    fun=self.estimation_config.fun,
                )

                # Store results using callback or default storage
                if result_callback:
                    result_callback(results_dict, current_config, config_dict, true_params, treatment_effects, mcc)
                else:
                    self._default_store_results(
                        results_dict, current_config, config_dict, true_params, treatment_effects, mcc
                    )

        return results_dict

    def _update_config(self, base_config: DataGenerationConfig, param_dict: Dict[str, Any]) -> DataGenerationConfig:
        """Create a new config with updated parameters.

        Args:
            base_config: Base configuration to copy from
            param_dict: Dictionary of parameters to update

        Returns:
            New DataGenerationConfig with updated parameters
        """
        # Create a copy of the base config
        config_dict = vars(base_config).copy()

        # Update with new parameters
        config_dict.update(param_dict)

        # Return new config instance
        return DataGenerationConfig(**config_dict)

    def _default_store_results(
        self,
        results_dict: dict,
        config: DataGenerationConfig,
        param_dict: Dict[str, Any],
        true_params: Any,
        treatment_effects: Any,
        mcc: float,
    ):
        """Default method for storing experiment results.

        Args:
            results_dict: Dictionary to store results in
            config: Data generation configuration used
            param_dict: Parameter values for this run
            true_params: True parameter values
            treatment_effects: Estimated treatment effects
            mcc: Maximum correlation coefficient
        """
        # Store basic results
        if "sample_sizes" not in results_dict:
            results_dict["sample_sizes"] = []
        if "n_covariates" not in results_dict:
            results_dict["n_covariates"] = []
        if "n_treatments" not in results_dict:
            results_dict["n_treatments"] = []
        if "true_params" not in results_dict:
            results_dict["true_params"] = []
        if "treatment_effects" not in results_dict:
            results_dict["treatment_effects"] = []
        if "mccs" not in results_dict:
            results_dict["mccs"] = []

        results_dict["sample_sizes"].append(config.batch_size)
        results_dict["n_covariates"].append(config.n_covariates)
        results_dict["n_treatments"].append(config.n_treatments)
        results_dict["true_params"].append(true_params)
        results_dict["treatment_effects"].append(treatment_effects)
        results_dict["mccs"].append(mcc)

        # Store parameter-specific values
        for param_name, param_value in param_dict.items():
            if param_name not in results_dict:
                results_dict[param_name] = []
            results_dict[param_name].append(param_value)


def run_single_configuration(
    config: DataGenerationConfig, estimation_config: EstimationConfig, n_seeds: int
) -> List[tuple]:
    """Run experiments for a single configuration across multiple seeds.

    This is a simpler interface for running a single configuration.

    Args:
        config: Data generation configuration
        estimation_config: Estimation configuration
        n_seeds: Number of random seeds to run

    Returns:
        List of (true_params, treatment_effects, mcc) tuples
    """
    results = []

    for seed in range(n_seeds):
        # Generate data
        S, X, true_params = generate_ica_data(
            n_covariates=config.n_covariates,
            n_treatments=config.n_treatments,
            batch_size=config.batch_size,
            slope=config.slope,
            sparse_prob=config.sparse_prob,
            beta=config.beta,
            loc=config.loc,
            scale=config.scale,
            nonlinearity=config.nonlinearity,
            theta_choice=config.theta_choice,
            split_noise_dist=config.split_noise_dist,
        )

        # Run estimation
        treatment_effects, mcc = ica_treatment_effect_estimation(
            X,
            S,
            random_state=seed,
            whiten=estimation_config.whiten,
            check_convergence=estimation_config.check_convergence,
            n_treatments=config.n_treatments,
            verbose=estimation_config.verbose,
            fun=estimation_config.fun,
        )

        results.append((true_params, treatment_effects, mcc))

    return results
