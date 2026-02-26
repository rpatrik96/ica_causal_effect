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
    single_config: bool = False
    beta: Optional[float] = None
    support_size: Optional[int] = None


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
    cov_dim_max: int = 50


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


def compute_distribution_moments(
    distribution: str, params: np.ndarray = None, probs: np.ndarray = None, scale: float = 1.0, beta: float = None
) -> Dict:
    """Compute analytical moments for supported treatment noise distributions.

    Parameters
    ----------
    distribution : str
        Noise distribution type: "discrete", "laplace", "uniform", "rademacher",
        "gennorm_heavy", "gennorm_light", or "gennorm".
    params : np.ndarray, optional
        For "discrete": discount values. For "gennorm": [beta, loc, scale].
    probs : np.ndarray, optional
        Probability vector for discrete distributions.
    scale : float
        Scale parameter used to normalize continuous distributions to variance
        ``scale²``.
    beta : float, optional
        Shape parameter for "gennorm". If None, extracted from ``params[0]``.

    Returns
    -------
    dict
        Keys: "second_moment", "third_moment", "fourth_moment",
        "cubed_variance", "excess_kurtosis", "skewness", "skewness_squared".

    Raises
    ------
    ValueError
        If required parameters are missing or the distribution is unknown.
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

        Parameters
        ----------
        distribution : str
            Noise distribution type (see ``compute_distribution_moments``).
        params : np.ndarray, optional
            Distribution parameters.
        probs : np.ndarray, optional
            Probability vector for discrete distributions.
        scale : float
            Scale parameter.
        beta : float, optional
            Shape parameter for generalized normal distribution.

        Returns
        -------
        tuple
            ``(eta_cubed_variance, eta_fourth_moment, eta_non_gauss_cond,
            eta_second_moment, eta_third_moment, homl_asymptotic_var,
            homl_asymptotic_var_num)``
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
        """Calculate HOML asymptotic variance components for discrete treatment noise.

        The HOML asymptotic variance is Var[eta³] / (E[eta³] / sqrt(Var[eta³]))²,
        which simplifies to Var[eta³] / (E[eta³])² * Var[eta³] = Var[eta³]².
        Returns inf when E[eta³] = 0 (symmetric distribution).

        Parameters
        ----------
        discounts : np.ndarray
            Discrete support values of the treatment noise distribution.
        mean_discount : float
            Mean of the discrete distribution.
        probs : np.ndarray
            Probability vector corresponding to each discount value.

        Returns
        -------
        tuple
            ``(eta_cubed_variance, eta_fourth_moment, eta_non_gauss_cond,
            eta_second_moment, eta_third_moment, homl_asymptotic_var,
            homl_asymptotic_var_num)``
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
        """Calculate ICA asymptotic variance components for discrete treatment noise.

        The ICA variance coefficient is c_ICA = 1 + ||b + a*theta||², where b is
        the outcome coefficient vector and a is the treatment coefficient vector.
        The asymptotic variance is c_ICA * Var[eta³] / kurtosis(eta)².

        Parameters
        ----------
        treatment_coef : np.ndarray
            Treatment coefficient vector (a in the PLM).
        outcome_coef : np.ndarray
            Outcome coefficient vector (b in the PLM).
        treatment_effect : float
            True treatment effect theta.
        discounts : np.ndarray
            Discrete support values of the treatment noise distribution.
        mean_discount : float
            Mean of the discrete distribution.
        probs : np.ndarray
            Probability vector corresponding to each discount value.
        eta_cubed_variance : float
            Pre-computed Var[eta³] = E[eta⁴]*E[eta²] - (E[eta³])².

        Returns
        -------
        tuple
            ``(eta_excess_kurtosis, eta_skewness_squared, ica_asymptotic_var,
            ica_asymptotic_var_hyvarinen, ica_asymptotic_var_num, ica_var_coeff)``
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

        Parameters
        ----------
        treatment_coef : np.ndarray
            Treatment coefficient vector (a in the PLM).
        outcome_coef : np.ndarray
            Outcome coefficient vector (b in the PLM).
        treatment_effect : float
            True treatment effect theta.
        distribution : str
            Noise distribution type (see ``compute_distribution_moments``).
        params : np.ndarray, optional
            Distribution parameters.
        probs : np.ndarray, optional
            Probability vector for discrete distributions.
        scale : float
            Scale parameter.
        beta : float, optional
            Shape parameter for generalized normal distribution.

        Returns
        -------
        tuple
            ``(eta_excess_kurtosis, eta_skewness_squared, ica_asymptotic_var,
            ica_asymptotic_var_hyvarinen, ica_asymptotic_var_num, ica_var_coeff)``
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

        Parameters
        ----------
        results_file : str
            Path to the results ``.npy`` file.
        """
        self.results_file = results_file

    def exists(self) -> bool:
        """Check if the results file exists.

        Returns
        -------
        bool
            True if the results file exists on disk, False otherwise.
        """
        return os.path.exists(self.results_file)

    def load(self) -> List[Dict]:
        """Load existing results from the ``.npy`` file.

        Returns
        -------
        List[Dict]
            List of result dictionaries, or an empty list if the file does not
            exist.
        """
        if self.exists():
            print(f"Results file '{self.results_file}' already exists. Loading data.")
            return list(np.load(self.results_file, allow_pickle=True))
        return []

    def save(self, results: List[Dict]) -> None:
        """Save results to the ``.npy`` file.

        Parameters
        ----------
        results : List[Dict]
            List of result dictionaries to serialize with ``np.save``.
        """
        np.save(self.results_file, np.array(results))
        print(f"Results saved to '{self.results_file}'")


def setup_output_dir(config: OMLExperimentConfig) -> str:
    """Construct and create the output directory for an experiment run.

    The path encodes key configuration flags (n_experiments, sigma_outcome,
    covariate_pdf, and optional suffixes for convergence, small_data, etc.)
    so that results from different configurations are stored separately.

    Parameters
    ----------
    config : OMLExperimentConfig
        Experiment configuration.

    Returns
    -------
    str
        Absolute path to the created output directory.
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
    """Construct a results filename that encodes the experiment configuration.

    Filename components are joined with underscores and reflect n_experiments,
    covariate_pdf, sigma_outcome, convergence/scalar flags, n_samples, beta,
    support_size, and oracle_support as applicable.

    Parameters
    ----------
    config : OMLExperimentConfig
        Experiment configuration.

    Returns
    -------
    str
        Results filename (e.g., ``all_results_n_exp_20_gennorm_sigma_outcome_1.732_..._n500.npy``).
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

    # Add sample size to filename (each split job runs one n_samples)
    filename_parts.append(f"n{config.n_samples}")

    # Add beta to filename when explicitly specified (per-beta job splitting)
    beta = getattr(config, "beta", None)
    if beta is not None:
        filename_parts.append(f"beta{beta}")

    # Add support_size to filename when explicitly specified (per-support-size job splitting)
    support_size = getattr(config, "support_size", None)
    if support_size is not None:
        filename_parts.append(f"d{support_size}")

    # Add no_oracle to filename when oracle_support is disabled
    oracle_support = getattr(config, "oracle_support", True)
    if not oracle_support:
        filename_parts.append("no_oracle")

    return "_".join(filename_parts) + ".npy"
