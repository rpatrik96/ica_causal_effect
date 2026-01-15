"""
Utility classes and functions for Orthogonal Machine Learning experiments.

This module provides reusable infrastructure for running OML experiments,
including configuration management, results handling, and analysis utilities.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OMLExperimentConfig:
    """Configuration for OML experiment execution.

    Attributes:
        n_samples: Number of samples to generate
        n_experiments: Number of Monte Carlo replications
        seed: Random seed for reproducibility
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates ('gauss', 'uniform', 'gennorm')
        output_dir: Directory for output figures
        check_convergence: Whether to verify ICA convergence
        asymptotic_var: Flag to ablate asymptotic variance
        tie_sample_dim: Ties n=d**4 for dimensional analysis
        verbose: Enable verbose output
        small_data: Flag to use a small dataset for quick tests
        matched_coefficients: Whether treatment and outcome coefficients are matched
        scalar_coeffs: Whether only one coefficient is non-zero
        eta_noise_dist: Distribution for treatment noise eta. Options:
            'discrete' (default), 'laplace' (heavy-tailed), 'uniform' (bounded),
            'rademacher' (bounded discrete), 'gennorm_heavy', 'gennorm_light'
        treatment_coef_range: (min, max) range for random treatment coefficients
        outcome_coef_range: (min, max) range for random outcome coefficients
        oracle_support: If True, both OML and ICA receive x[:, support] (oracle knowledge).
            If False, both methods receive full x matrix.
    """

    n_samples: int = 500
    n_experiments: int = 5
    seed: int = 12143
    sigma_outcome: float = np.sqrt(3.0)
    covariate_pdf: str = "gennorm"
    output_dir: str = "./figures"
    check_convergence: bool = True
    asymptotic_var: bool = False
    tie_sample_dim: bool = False
    verbose: bool = False
    small_data: bool = False
    matched_coefficients: bool = False
    scalar_coeffs: bool = True
    eta_noise_dist: str = "discrete"
    treatment_coef_range: Tuple[float, float] = (-5.0, 5.0)
    outcome_coef_range: Tuple[float, float] = (-5.0, 5.0)
    oracle_support: bool = True


@dataclass
class OMLParameterGrid:
    """Parameter grid configuration for OML experiments.

    Attributes:
        support_sizes: List of support sizes to test
        data_samples: List of sample sizes to test
        beta_values: List of beta values for generalized normal distribution
        treatment_effects: List of treatment effect values
        treatment_coefs: List of treatment coefficient values
        outcome_coefs: List of outcome coefficient values
        beta_filter: Default beta value for filtering results
        support_filter: Default support size for filtering results
    """

    support_sizes: List[int] = field(default_factory=lambda: [2, 5, 10, 20, 50])
    data_samples: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000, 5000])
    beta_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0])
    treatment_effects: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.5, 1.0, 3.0, 10])
    treatment_coefs: List[float] = field(default_factory=lambda: [-0.002, 0.05, -0.43, 1.56])
    outcome_coefs: List[float] = field(default_factory=lambda: [0.003, -0.02, 0.63, -1.45])
    beta_filter: float = 4.0
    support_filter: int = 10

    @classmethod
    def create_from_config(cls, config: OMLExperimentConfig) -> "OMLParameterGrid":
        """Create parameter grid based on experiment configuration.

        Args:
            config: Experiment configuration

        Returns:
            Configured parameter grid
        """
        grid = cls()

        # Adjust for small_data mode
        if config.small_data:
            grid.support_sizes = [2, 5, 10]
            grid.data_samples = [20, 50, 100]
            grid.support_filter = 5
        else:
            # Adjust for asymptotic_var or scalar_coeffs modes
            if config.asymptotic_var or config.scalar_coeffs:
                grid.support_sizes = [10]
                grid.treatment_effects = [1]

            if config.asymptotic_var:
                grid.data_samples = [10**4]
                # Asymptotic variance specific coefficients
                grid.treatment_coefs = [-0.002, -0.33, 1.26]
                grid.outcome_coefs = [-0.05, 0.7, 1.9]

        # Adjust beta values
        if config.covariate_pdf != "gennorm" or config.asymptotic_var:
            grid.beta_values = [1.0]

        # Adjust treatment effects
        if config.matched_coefficients:
            grid.treatment_effects = [1.0]

        return grid


@dataclass
class OMLMethodConfig:
    """Configuration for OML estimation methods.

    Attributes:
        lambda_reg: Regularization parameter for Lasso
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
    """

    lambda_reg: float = 0.01
    check_convergence: bool = True
    verbose: bool = False


@dataclass
class OMLResultsContainer:
    """Container for OML experiment results.

    Attributes:
        n_samples: Number of samples used
        support_size: Support size used
        beta: Beta parameter value
        treatment_effect: True treatment effect
        cov_dim_max: Maximum covariate dimension
        sigma_outcome: Outcome noise standard deviation
        ortho_rec_tau: Recovered treatment effects from all methods
        first_stage_mse: First-stage regression MSEs
        biases: Bias for each method
        sigmas: Standard deviation for each method
        eta_second_moment: Second moment of treatment noise
        eta_third_moment: Third moment of treatment noise
        eta_non_gauss_cond: Non-Gaussianity condition value
        eta_cubed_variance: Variance of cubed treatment noise
        eta_fourth_moment: Fourth moment of treatment noise
        eta_skewness_squared: Squared skewness of treatment noise
        eta_excess_kurtosis: Excess kurtosis of treatment noise
        ica_var_coeff: ICA variance coefficient
        ica_asymptotic_var: ICA asymptotic variance
        ica_asymptotic_var_hyvarinen: Hyvarinen's ICA asymptotic variance
        ica_asymptotic_var_num: Numerator of ICA asymptotic variance
        homl_asymptotic_var: HOML asymptotic variance
        homl_asymptotic_var_num: Numerator of HOML asymptotic variance
    """

    n_samples: int
    support_size: int
    beta: float
    treatment_effect: float
    cov_dim_max: int
    sigma_outcome: float
    ortho_rec_tau: List
    first_stage_mse: List
    biases: Optional[np.ndarray] = None
    sigmas: Optional[np.ndarray] = None
    eta_second_moment: Optional[float] = None
    eta_third_moment: Optional[float] = None
    eta_non_gauss_cond: Optional[float] = None
    eta_cubed_variance: Optional[float] = None
    eta_fourth_moment: Optional[float] = None
    eta_skewness_squared: Optional[float] = None
    eta_excess_kurtosis: Optional[float] = None
    ica_var_coeff: Optional[float] = None
    ica_asymptotic_var: Optional[float] = None
    ica_asymptotic_var_hyvarinen: Optional[float] = None
    ica_asymptotic_var_num: Optional[float] = None
    homl_asymptotic_var: Optional[float] = None
    homl_asymptotic_var_num: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert container to dictionary for saving.

        Returns:
            Dictionary representation of results
        """
        return {
            "n_samples": self.n_samples,
            "support_size": self.support_size,
            "beta": self.beta,
            "treatment_effect": self.treatment_effect,
            "cov_dim_max": self.cov_dim_max,
            "sigma_outcome": self.sigma_outcome,
            "ortho_rec_tau": self.ortho_rec_tau,
            "first_stage_mse": self.first_stage_mse,
            "biases": self.biases,
            "sigmas": self.sigmas,
            "eta_second_moment": self.eta_second_moment,
            "eta_third_moment": self.eta_third_moment,
            "eta_non_gauss_cond": self.eta_non_gauss_cond,
            "eta_cubed_variance": self.eta_cubed_variance,
            "eta_fourth_moment": self.eta_fourth_moment,
            "eta_skewness_squared": self.eta_skewness_squared,
            "eta_excess_kurtosis": self.eta_excess_kurtosis,
            "ica_var_coeff": self.ica_var_coeff,
            "ica_asymptotic_var": self.ica_asymptotic_var,
            "ica_asymptotic_var_hyvarinen": self.ica_asymptotic_var_hyvarinen,
            "ica_asymptotic_var_num": self.ica_asymptotic_var_num,
            "homl_asymptotic_var": self.homl_asymptotic_var,
            "homl_asymptotic_var_num": self.homl_asymptotic_var_num,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OMLResultsContainer":
        """Create container from dictionary.

        Args:
            data: Dictionary with result data

        Returns:
            OMLResultsContainer instance
        """
        return cls(**data)


def compute_distribution_moments(
    distribution: str, params: np.ndarray = None, probs: np.ndarray = None, scale: float = 1.0, beta: float = None
) -> Dict:
    """Compute analytical moments for different noise distributions.

    Args:
        distribution: Type of noise distribution (discrete, laplace, uniform, rademacher,
                     gennorm_heavy, gennorm_light, gennorm)
        params: Distribution parameters (discounts for discrete, or [beta, loc, scale] for gennorm)
        probs: Probabilities for discrete distributions
        scale: Scale parameter (used for standardization)
        beta: Shape parameter for generalized normal distribution. If None and distribution
              is 'gennorm', will be extracted from params[0].

    Returns:
        Dictionary with moments: second_moment, third_moment, fourth_moment,
        cubed_variance, excess_kurtosis, skewness
    """
    from scipy.special import gamma

    if distribution == "discrete":
        # Discrete distribution with specified discounts and probabilities
        if params is None or probs is None:
            raise ValueError("params (discounts) and probs required for discrete distribution")

        discounts = params
        mean_discount = np.dot(discounts, probs)
        centered_discounts = discounts - mean_discount

        second_moment = np.dot(centered_discounts**2, probs)
        third_moment = np.dot(centered_discounts**3, probs)
        fourth_moment = np.dot(centered_discounts**4, probs)

    elif distribution == "laplace":
        # Laplace distribution: symmetric, heavy-tailed
        # For Laplace(0, b): E[X^2] = 2b^2, E[X^3] = 0, E[X^4] = 24b^4
        # We use b = scale/sqrt(2) so variance = scale^2
        b = scale / np.sqrt(2)
        second_moment = 2 * b**2  # = scale^2
        third_moment = 0.0  # Symmetric distribution
        fourth_moment = 24 * b**4  # = 6 * scale^4

    elif distribution == "uniform":
        # Uniform distribution on [-a, a]: symmetric, bounded
        # For Uniform(-a, a): E[X^2] = a^2/3, E[X^3] = 0, E[X^4] = a^4/5
        # We use a = sqrt(3)*scale so variance = scale^2
        a = np.sqrt(3) * scale
        second_moment = a**2 / 3  # = scale^2
        third_moment = 0.0  # Symmetric distribution
        fourth_moment = a**4 / 5  # = 9/5 * scale^4 = 1.8 * scale^4

    elif distribution == "rademacher":
        # Rademacher distribution: {-scale, +scale} with prob 0.5 each
        # E[X^2] = scale^2, E[X^3] = 0, E[X^4] = scale^4
        second_moment = scale**2
        third_moment = 0.0  # Symmetric distribution
        fourth_moment = scale**4

    elif distribution == "gennorm_heavy":
        # Generalized normal with beta=1 (equivalent to Laplace)
        gn_beta = 1.0
        # Raw moments of standard gennorm(beta): E[X^n] = Gamma((n+1)/beta) / Gamma(1/beta)
        # For centered (mean=0) symmetric distribution with unit variance
        gn_var = gamma(3 / gn_beta) / gamma(1 / gn_beta)
        gn_fourth = gamma(5 / gn_beta) / gamma(1 / gn_beta)
        # Scale to have variance = scale^2
        second_moment = scale**2
        third_moment = 0.0  # Symmetric
        fourth_moment = (gn_fourth / gn_var**2) * scale**4

    elif distribution == "gennorm_light":
        # Generalized normal with beta=4 (lighter tails than Gaussian)
        gn_beta = 4.0
        gn_var = gamma(3 / gn_beta) / gamma(1 / gn_beta)
        gn_fourth = gamma(5 / gn_beta) / gamma(1 / gn_beta)
        # Scale to have variance = scale^2
        second_moment = scale**2
        third_moment = 0.0  # Symmetric
        fourth_moment = (gn_fourth / gn_var**2) * scale**4

    elif distribution == "gennorm":
        # Generalized normal with arbitrary beta
        # Extract beta from params[0] if not provided directly
        if beta is not None:
            gn_beta = beta
        elif params is not None and len(params) >= 1:
            gn_beta = params[0]
        else:
            raise ValueError("beta parameter required for gennorm distribution (via beta arg or params[0])")

        gn_var = gamma(3 / gn_beta) / gamma(1 / gn_beta)
        gn_fourth = gamma(5 / gn_beta) / gamma(1 / gn_beta)
        # Scale to have variance = scale^2
        second_moment = scale**2
        third_moment = 0.0  # Symmetric
        fourth_moment = (gn_fourth / gn_var**2) * scale**4

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Compute derived quantities
    cubed_variance = fourth_moment * second_moment - third_moment**2
    excess_kurtosis = fourth_moment / (second_moment**2) - 3
    skewness = third_moment / (second_moment ** (3 / 2)) if second_moment > 0 else 0.0

    return {
        "second_moment": second_moment,
        "third_moment": third_moment,
        "fourth_moment": fourth_moment,
        "cubed_variance": cubed_variance,
        "excess_kurtosis": excess_kurtosis,
        "skewness": skewness,
        "skewness_squared": skewness**2,
    }


class AsymptoticVarianceCalculator:
    """Calculates asymptotic variance metrics for OML and ICA methods."""

    @staticmethod
    def calc_homl_asymptotic_var_from_distribution(
        distribution: str, params: np.ndarray = None, probs: np.ndarray = None, scale: float = 1.0, beta: float = None
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Calculate HOML asymptotic variance for any supported distribution.

        Args:
            distribution: Type of noise distribution
            params: Distribution parameters
            probs: Probabilities for discrete distributions
            scale: Scale parameter
            beta: Shape parameter for generalized normal distribution

        Returns:
            Tuple of (eta_cubed_variance, eta_fourth_moment, eta_non_gauss_cond,
                     eta_second_moment, eta_third_moment, homl_asymptotic_var,
                     homl_asymptotic_var_num)
        """
        moments = compute_distribution_moments(distribution, params, probs, scale, beta)

        eta_second_moment = moments["second_moment"]
        eta_third_moment = moments["third_moment"]
        eta_fourth_moment = moments["fourth_moment"]
        eta_cubed_variance = moments["cubed_variance"]

        # Non-Gaussianity condition (requires non-zero third moment for HOML)
        # For symmetric distributions, this will be 0 which makes HOML inapplicable
        if abs(eta_third_moment) > 1e-10:
            eta_non_gauss_cond = eta_third_moment / np.sqrt(eta_cubed_variance)
            homl_asymptotic_var_num = eta_cubed_variance
            homl_asymptotic_var = homl_asymptotic_var_num / (eta_non_gauss_cond**2)
        else:
            # For symmetric distributions, HOML is not applicable
            eta_non_gauss_cond = 0.0
            homl_asymptotic_var_num = eta_cubed_variance
            homl_asymptotic_var = np.inf  # HOML not applicable

        return (
            eta_cubed_variance,
            eta_fourth_moment,
            eta_non_gauss_cond,
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            homl_asymptotic_var_num,
        )

    @staticmethod
    def calc_homl_asymptotic_var(
        discounts: np.ndarray, mean_discount: float, probs: np.ndarray
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Calculate HOML asymptotic variance components.

        Args:
            discounts: Discount values for treatment noise
            mean_discount: Mean of discounts
            probs: Probabilities for each discount

        Returns:
            Tuple of (eta_cubed_variance, eta_fourth_moment, eta_non_gauss_cond,
                     eta_second_moment, eta_third_moment, homl_asymptotic_var,
                     homl_asymptotic_var_num)
        """
        # Center discounts
        centered_discounts = discounts - mean_discount

        # Compute moments
        eta_second_moment = np.dot(centered_discounts**2, probs)
        eta_third_moment = np.dot(centered_discounts**3, probs)
        eta_fourth_moment = np.dot(centered_discounts**4, probs)

        # Compute variance of cubed noise
        eta_cubed_variance = eta_fourth_moment * eta_second_moment - eta_third_moment**2

        # Non-Gaussianity condition
        eta_non_gauss_cond = eta_third_moment / np.sqrt(eta_cubed_variance)

        # HOML asymptotic variance
        homl_asymptotic_var_num = eta_cubed_variance
        homl_asymptotic_var = homl_asymptotic_var_num / (eta_non_gauss_cond**2)

        return (
            eta_cubed_variance,
            eta_fourth_moment,
            eta_non_gauss_cond,
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            homl_asymptotic_var_num,
        )

    @staticmethod
    def calc_ica_asymptotic_var(
        treatment_coef: np.ndarray,
        outcome_coef: np.ndarray,
        treatment_effect: float,
        discounts: np.ndarray,
        mean_discount: float,
        probs: np.ndarray,
        eta_cubed_variance: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate ICA asymptotic variance components.

        Args:
            treatment_coef: Treatment coefficients
            outcome_coef: Outcome coefficients
            treatment_effect: True treatment effect
            discounts: Discount values for treatment noise
            mean_discount: Mean of discounts
            probs: Probabilities for each discount
            eta_cubed_variance: Variance of cubed treatment noise

        Returns:
            Tuple of (eta_excess_kurtosis, eta_skewness_squared, ica_asymptotic_var,
                     ica_asymptotic_var_hyvarinen, ica_asymptotic_var_num, ica_var_coeff)
        """
        # Center discounts
        centered_discounts = discounts - mean_discount

        # Compute moments
        eta_second_moment = np.dot(centered_discounts**2, probs)
        eta_third_moment = np.dot(centered_discounts**3, probs)
        eta_fourth_moment = np.dot(centered_discounts**4, probs)

        # Skewness and kurtosis
        eta_skewness_squared = (eta_third_moment / eta_second_moment ** (3 / 2)) ** 2
        eta_excess_kurtosis = eta_fourth_moment / (eta_second_moment**2) - 3

        # ICA variance coefficient
        ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect) ** 2

        # ICA asymptotic variances
        ica_asymptotic_var_num = ica_var_coeff * eta_cubed_variance
        ica_asymptotic_var = ica_asymptotic_var_num / (eta_excess_kurtosis**2)
        ica_asymptotic_var_hyvarinen = ica_asymptotic_var_num / (12 * eta_skewness_squared)

        return (
            eta_excess_kurtosis,
            eta_skewness_squared,
            ica_asymptotic_var,
            ica_asymptotic_var_hyvarinen,
            ica_asymptotic_var_num,
            ica_var_coeff,
        )

    @staticmethod
    def calc_ica_asymptotic_var_from_distribution(
        treatment_coef: np.ndarray,
        outcome_coef: np.ndarray,
        treatment_effect: float,
        distribution: str,
        params: np.ndarray = None,
        probs: np.ndarray = None,
        scale: float = 1.0,
        beta: float = None,
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate ICA asymptotic variance for any supported distribution.

        Args:
            treatment_coef: Treatment coefficients
            outcome_coef: Outcome coefficients
            treatment_effect: True treatment effect
            distribution: Type of noise distribution
            params: Distribution parameters
            probs: Probabilities for discrete distributions
            scale: Scale parameter
            beta: Shape parameter for generalized normal distribution

        Returns:
            Tuple of (eta_excess_kurtosis, eta_skewness_squared, ica_asymptotic_var,
                     ica_asymptotic_var_hyvarinen, ica_asymptotic_var_num, ica_var_coeff)
        """
        moments = compute_distribution_moments(distribution, params, probs, scale, beta)

        eta_excess_kurtosis = moments["excess_kurtosis"]
        eta_skewness_squared = moments["skewness_squared"]
        eta_cubed_variance = moments["cubed_variance"]

        # ICA variance coefficient
        ica_var_coeff = 1 + np.linalg.norm(outcome_coef + treatment_coef * treatment_effect) ** 2

        # ICA asymptotic variances
        ica_asymptotic_var_num = ica_var_coeff * eta_cubed_variance

        # Handle cases where kurtosis or skewness is zero
        if abs(eta_excess_kurtosis) > 1e-10:
            ica_asymptotic_var = ica_asymptotic_var_num / (eta_excess_kurtosis**2)
        else:
            ica_asymptotic_var = np.inf  # ICA based on kurtosis not applicable

        if abs(eta_skewness_squared) > 1e-10:
            ica_asymptotic_var_hyvarinen = ica_asymptotic_var_num / (12 * eta_skewness_squared)
        else:
            ica_asymptotic_var_hyvarinen = np.inf  # ICA based on skewness not applicable

        return (
            eta_excess_kurtosis,
            eta_skewness_squared,
            ica_asymptotic_var,
            ica_asymptotic_var_hyvarinen,
            ica_asymptotic_var_num,
            ica_var_coeff,
        )


class OMLResultsManager:
    """Manages loading, saving, and caching of OML experiment results."""

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

    def load(self) -> List[Dict]:
        """Load existing results.

        Returns:
            List of result dictionaries
        """
        if self.exists():
            print(f"Results file '{self.results_file}' already exists. Loading data.")
            return list(np.load(self.results_file, allow_pickle=True))
        return []

    def save(self, results: List[Dict]):
        """Save results to file.

        Args:
            results: List of result dictionaries to save
        """
        np.save(self.results_file, np.array(results))
        print(f"Results saved to '{self.results_file}'")


def setup_output_dir(config: OMLExperimentConfig) -> str:
    """Setup output directory based on configuration.

    Args:
        config: Experiment configuration

    Returns:
        Full output directory path
    """
    dir_parts = [
        config.output_dir,
        f"n_exp_{config.n_experiments}_sigma_outcome_{config.sigma_outcome}_pdf_{config.covariate_pdf}",
    ]

    if config.check_convergence:
        dir_parts.append("convergence")

    if config.small_data:
        dir_parts.append("small_data")

    if config.asymptotic_var:
        dir_parts.append("asymptotic_var")

    if config.matched_coefficients:
        dir_parts.append("matched_coefficients")

    if config.tie_sample_dim:
        dir_parts.append("tie_sample_dim")

    if config.scalar_coeffs:
        dir_parts.append("scalar_coeffs")

    # Add eta noise distribution to output path if not default
    eta_noise_dist = getattr(config, "eta_noise_dist", "discrete")
    if eta_noise_dist != "discrete":
        dir_parts.append(f"eta_{eta_noise_dist}")

    output_dir = os.path.join(*dir_parts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_results_filename(config: OMLExperimentConfig) -> str:
    """Setup results filename based on configuration.

    Args:
        config: Experiment configuration

    Returns:
        Results filename
    """
    filename_parts = []

    if config.asymptotic_var:
        filename_parts.append("all_results_asymptotic_var")
    else:
        filename_parts.append("all_results")

    filename_parts.append(f"n_exp_{config.n_experiments}")

    if config.covariate_pdf == "gennorm":
        filename_parts.append("gennorm")

    filename_parts.append(f"sigma_outcome_{config.sigma_outcome}")

    if config.check_convergence:
        filename_parts.append("check_convergence")

    if config.small_data:
        filename_parts.append("small_data")

    if config.tie_sample_dim:
        filename_parts.append("tie_sample_dim")

    if config.scalar_coeffs:
        filename_parts.append("scalar_coeffs")

    if config.matched_coefficients:
        filename_parts.append("matched_coefficients")

    # Add eta noise distribution to filename if not default
    eta_noise_dist = getattr(config, "eta_noise_dist", "discrete")
    if eta_noise_dist != "discrete":
        filename_parts.append(f"eta_{eta_noise_dist}")

    # Add no_oracle to filename when oracle_support is disabled
    oracle_support = getattr(config, "oracle_support", True)
    if not oracle_support:
        filename_parts.append("no_oracle")

    return "_".join(filename_parts) + ".npy"
