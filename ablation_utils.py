"""
Shared utilities for ablation studies.

This module provides common functionality used across noise distribution
and coefficient ablation experiments.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso

from ica import ica_treatment_effect_estimation
from main_estimation import all_together_cross_fitting
from oml_utils import AsymptoticVarianceCalculator

# =============================================================================
# Constants
# =============================================================================

# Method indices in results arrays
# Order: OrthoML, RobustOrthoML(HOML), RobustOrthoEst, RobustOrthoSplit, ICA
ORTHO_ML_IDX = 0
HOML_IDX = 1  # Robust Ortho ML (Higher-Order ML)
ROBUST_ORTHO_EST_IDX = 2
ROBUST_ORTHO_SPLIT_IDX = 3
ICA_IDX = 4

# Method names for display
METHOD_NAMES = ["Ortho ML", "OML", "Robust Ortho Est", "Robust Ortho Split", "ICA"]
METHOD_NAMES_SHORT = ["OML", "OML", "ROE", "ROS", "ICA"]

# Colors for OML and ICA (primary comparison methods)
OML_COLOR = "#1f77b4"  # Blue
ICA_COLOR = "#ff7f0e"  # Orange

# Colors for comparison plots (colorblind-friendly)
ICA_BETTER_COLOR = "blue"  # matplotlib default blue
OML_BETTER_COLOR = "red"  # matplotlib default red

# Markers for comparison plots (colorblind-friendly)
ICA_BETTER_MARKER = "s"  # Square
OML_BETTER_MARKER = "o"  # Circle

# Distribution label mapping
DISTRIBUTION_LABELS = {
    "discrete": "Discrete",
    "laplace": "Laplace",
    "uniform": "Uniform",
    "rademacher": "Rademacher",
    "gennorm_heavy": r"GenNorm ($\beta$=1)",
    "gennorm_light": r"GenNorm ($\beta$=4)",
}


def get_distribution_label(dist_key: str) -> str:
    """Generate a display label for a distribution key.

    Parameters
    ----------
    dist_key : str
        Distribution specification, e.g. "discrete", "laplace", or
        "gennorm:1.5" for a generalized normal with beta=1.5.

    Returns
    -------
    str
        Human-readable label suitable for plot legends.
    """
    if dist_key in DISTRIBUTION_LABELS:
        return DISTRIBUTION_LABELS[dist_key]
    if dist_key.startswith("gennorm:"):
        beta_val = dist_key.split(":")[1]
        return rf"GenNorm ($\beta$={beta_val})"
    return dist_key


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class AblationExperimentConfig:
    """Configuration for ablation experiments.

    Attributes:
        n_samples: Number of samples per experiment
        n_experiments: Number of Monte Carlo replications
        support_size: Size of support for coefficients
        beta: Beta parameter for gennorm covariate distribution
        sigma_outcome: Standard deviation of outcome noise
        covariate_pdf: Distribution for covariates ('gennorm', 'gauss', 'uniform')
        check_convergence: Whether to check ICA convergence
        verbose: Enable verbose output
        seed: Random seed for reproducibility
    """

    n_samples: int = 5000
    n_experiments: int = 20
    support_size: int = 10
    beta: float = 1.0
    sigma_outcome: float = np.sqrt(3.0)
    covariate_pdf: str = "gennorm"
    check_convergence: bool = False
    verbose: bool = False
    seed: int = 12143


# =============================================================================
# Sampling Functions
# =============================================================================


def create_covariate_sampler(covariate_pdf: str, beta: float = 1.0) -> Callable[[int, int], np.ndarray]:
    """Create a covariate sampling function for the specified distribution.

    Parameters
    ----------
    covariate_pdf : str
        Distribution type: "gennorm", "gauss", or "uniform".
    beta : float
        Shape parameter for the generalized normal distribution (ignored for
        "gauss" and "uniform").

    Returns
    -------
    Callable[[int, int], np.ndarray]
        Function ``f(n_samples, n_dimensions) -> np.ndarray`` of shape
        ``(n_samples, n_dimensions)``.

    Raises
    ------
    ValueError
        If an unknown ``covariate_pdf`` is specified.
    """
    if covariate_pdf == "gennorm":
        from scipy.stats import gennorm

        return lambda n, d: gennorm.rvs(beta, size=(n, d))

    if covariate_pdf == "gauss":
        return lambda n, d: np.random.normal(size=(n, d))

    if covariate_pdf == "uniform":
        return lambda n, d: np.random.uniform(-1, 1, size=(n, d))

    raise ValueError(f"Unknown covariate PDF: {covariate_pdf}")


def create_outcome_noise_sampler(sigma_outcome: float) -> Callable[[int], np.ndarray]:
    """Create an outcome noise sampling function (uniform on [-sigma, sigma]).

    Parameters
    ----------
    sigma_outcome : float
        Half-width of the uniform distribution; Var[eps] = sigma_outcome² / 3.

    Returns
    -------
    Callable[[int], np.ndarray]
        Function ``f(n_samples) -> np.ndarray`` of shape ``(n_samples,)``.
    """
    return lambda n: np.random.uniform(-sigma_outcome, sigma_outcome, size=n)


# =============================================================================
# Moment Calculation
# =============================================================================


def calculate_homl_moments(
    noise_dist: str,
    params_or_discounts: np.ndarray,
    mean_discount: float,
    probs: Optional[np.ndarray],
) -> Dict[str, float]:
    """Calculate HOML-related moments for a treatment noise distribution.

    Parameters
    ----------
    noise_dist : str
        Distribution type (see ``oml_utils.compute_distribution_moments``).
    params_or_discounts : np.ndarray
        Discount values for "discrete", or distribution parameter array for
        continuous distributions.
    mean_discount : float
        Mean of the distribution (used for centering in the discrete case).
    probs : np.ndarray or None
        Probability vector for "discrete"; None for continuous distributions.

    Returns
    -------
    Dict[str, float]
        Keys: "eta_cubed_variance", "eta_fourth_moment", "eta_second_moment",
        "eta_third_moment", "homl_asymptotic_var".
    """
    var_calculator = AsymptoticVarianceCalculator()

    if noise_dist == "discrete":
        (
            eta_cubed_variance,
            eta_fourth_moment,
            _,  # eta_non_gauss_cond
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            _,  # homl_asymptotic_var_num
        ) = var_calculator.calc_homl_asymptotic_var(params_or_discounts, mean_discount, probs)
    else:
        (
            eta_cubed_variance,
            eta_fourth_moment,
            _,  # eta_non_gauss_cond
            eta_second_moment,
            eta_third_moment,
            homl_asymptotic_var,
            _,  # homl_asymptotic_var_num
        ) = var_calculator.calc_homl_asymptotic_var_from_distribution(noise_dist, params_or_discounts, probs)

    return {
        "eta_cubed_variance": eta_cubed_variance,
        "eta_fourth_moment": eta_fourth_moment,
        "eta_second_moment": eta_second_moment,
        "eta_third_moment": eta_third_moment,
        "homl_asymptotic_var": homl_asymptotic_var,
    }


def calculate_ica_moments(
    noise_dist: str,
    treatment_coef: np.ndarray,
    outcome_coef: np.ndarray,
    treatment_effect: float,
    params_or_discounts: np.ndarray,
    mean_discount: float,
    probs: Optional[np.ndarray],
    eta_cubed_variance: float,
) -> Dict[str, float]:
    """Calculate ICA asymptotic variance components for a noise distribution.

    Parameters
    ----------
    noise_dist : str
        Distribution type (see ``oml_utils.compute_distribution_moments``).
    treatment_coef : np.ndarray
        Treatment coefficient vector (a in the PLM).
    outcome_coef : np.ndarray
        Outcome coefficient vector (b in the PLM).
    treatment_effect : float
        True treatment effect theta.
    params_or_discounts : np.ndarray
        Discount values for "discrete", or distribution parameter array for
        continuous distributions.
    mean_discount : float
        Mean of the distribution (used for centering in the discrete case).
    probs : np.ndarray or None
        Probability vector for "discrete"; None for continuous distributions.
    eta_cubed_variance : float
        Pre-computed Var[eta³] = E[eta⁴]*E[eta²] - (E[eta³])².

    Returns
    -------
    Dict[str, float]
        Keys: "eta_excess_kurtosis", "eta_skewness_squared", "ica_asymptotic_var".
    """
    var_calculator = AsymptoticVarianceCalculator()

    if noise_dist == "discrete":
        (
            eta_excess_kurtosis,
            eta_skewness_squared,
            ica_asymptotic_var,
            _,  # ica_asymptotic_var_hyvarinen
            _,  # ica_asymptotic_var_num
            _,
        ) = var_calculator.calc_ica_asymptotic_var(
            treatment_coef,
            outcome_coef,
            treatment_effect,
            params_or_discounts,
            mean_discount,
            probs,
            eta_cubed_variance,
        )
    else:
        (
            eta_excess_kurtosis,
            eta_skewness_squared,
            ica_asymptotic_var,
            _,  # ica_asymptotic_var_hyvarinen
            _,  # ica_asymptotic_var_num
            _,
        ) = var_calculator.calc_ica_asymptotic_var_from_distribution(
            treatment_coef, outcome_coef, treatment_effect, noise_dist, params_or_discounts, probs
        )

    return {
        "eta_excess_kurtosis": eta_excess_kurtosis,
        "eta_skewness_squared": eta_skewness_squared,
        "ica_asymptotic_var": ica_asymptotic_var,
    }


# =============================================================================
# Experiment Execution
# =============================================================================


def run_single_experiment(
    x: np.ndarray,
    eta: np.ndarray,
    epsilon: np.ndarray,
    treatment_effect: float,
    treatment_support: np.ndarray,
    treatment_coef: np.ndarray,
    outcome_support: np.ndarray,
    outcome_coef: np.ndarray,
    eta_second_moment: float,
    eta_third_moment: float,
    lambda_reg: float,
    check_convergence: bool = False,
    verbose: bool = False,
    oracle_support: bool = True,
) -> Tuple:
    """Run a single OML experiment with specified noise samples.

    Constructs the PLM D = X*a + eta, Y = theta*D + X*b + eps, fits Lasso
    nuisance models, runs cross-fitted OML and ICA estimation, and returns
    all method estimates.

    Parameters
    ----------
    x : np.ndarray
        Covariate matrix of shape (n_samples, cov_dim_max).
    eta : np.ndarray
        Treatment noise samples of shape (n_samples,).
    epsilon : np.ndarray
        Outcome noise samples of shape (n_samples,).
    treatment_effect : float
        True treatment effect theta.
    treatment_support : np.ndarray
        Integer indices of active covariates for the treatment model.
    treatment_coef : np.ndarray
        Treatment coefficient vector of shape (support_size,).
    outcome_support : np.ndarray
        Integer indices of active covariates for the outcome model (must equal
        ``treatment_support``).
    outcome_coef : np.ndarray
        Outcome coefficient vector of shape (support_size,).
    eta_second_moment : float
        E[eta²], used by the HOML estimator.
    eta_third_moment : float
        E[eta³], used by the HOML estimator.
    lambda_reg : float
        Lasso regularization parameter for nuisance models.
    check_convergence : bool
        If True, filter out runs where ICA did not converge.
    verbose : bool
        Print per-run diagnostic information.
    oracle_support : bool
        If True, both OML and ICA receive ``x[:, support]`` (oracle support).
        If False, both methods receive the full covariate matrix.

    Returns
    -------
    tuple
        ``(ortho_ml, robust_ortho_ml, robust_ortho_est_ml,
        robust_ortho_est_split_ml, first_stage_mse_treatment,
        first_stage_mse_outcome, ica_estimate, ica_mcc)``
    """
    # Generate treatment as a function of covariates
    treatment = np.dot(x[:, treatment_support], treatment_coef) + eta

    # Generate outcome as a function of treatment and covariates
    outcome = treatment_effect * treatment + np.dot(x[:, outcome_support], outcome_coef) + epsilon

    model_treatment = Lasso(alpha=lambda_reg)
    model_outcome = Lasso(alpha=lambda_reg)

    assert (treatment_support == outcome_support).all()

    # Select covariates based on oracle_support flag
    # When oracle_support=True, both OML and ICA receive x[:, support] (oracle knowledge)
    # When oracle_support=False, both methods receive full x
    if oracle_support:
        covariates = x[:, treatment_support]
    else:
        covariates = x

    try:
        ica_treatment_effect_estimate, ica_mcc = ica_treatment_effect_estimation(
            np.hstack((covariates, treatment.reshape(-1, 1), outcome.reshape(-1, 1))),
            np.hstack((covariates, eta.reshape(-1, 1), epsilon.reshape(-1, 1))),
            check_convergence=check_convergence,
            verbose=verbose,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An error occurred during ICA treatment effect estimation: {e}")
        ica_mcc = None
        ica_treatment_effect_estimate = None

    if verbose:
        print(f"Estimated vs true treatment effect: {ica_treatment_effect_estimate}, {treatment_effect}")

    return (
        *all_together_cross_fitting(
            covariates,
            treatment,
            outcome,
            eta_second_moment,
            eta_third_moment,
            model_treatment=model_treatment,
            model_outcome=model_outcome,
        ),
        ica_treatment_effect_estimate,
        ica_mcc,
    )


def run_parallel_experiments(
    n_experiments: int,
    x_sample: Callable,
    eta_sample: Callable,
    epsilon_sample: Callable,
    n_samples: int,
    cov_dim_max: int,
    treatment_effect: float,
    treatment_support: np.ndarray,
    treatment_coef: np.ndarray,
    outcome_support: np.ndarray,
    outcome_coef: np.ndarray,
    eta_second_moment: float,
    eta_third_moment: float,
    lambda_reg: float,
    check_convergence: bool = False,
    verbose: bool = False,
    oracle_support: bool = True,
) -> List[Tuple]:
    """Run Monte Carlo experiments in parallel using joblib.

    Parameters
    ----------
    n_experiments : int
        Number of Monte Carlo replications.
    x_sample : Callable
        Function ``f(n_samples, cov_dim_max) -> np.ndarray`` for covariates.
    eta_sample : Callable
        Function ``f(n_samples) -> np.ndarray`` for treatment noise.
    epsilon_sample : Callable
        Function ``f(n_samples) -> np.ndarray`` for outcome noise.
    n_samples : int
        Number of samples per experiment.
    cov_dim_max : int
        Total covariate dimension passed to ``x_sample``.
    treatment_effect : float
        True treatment effect theta.
    treatment_support : np.ndarray
        Integer indices of active covariates for treatment.
    treatment_coef : np.ndarray
        Treatment coefficient vector of shape (support_size,).
    outcome_support : np.ndarray
        Integer indices of active covariates for outcome.
    outcome_coef : np.ndarray
        Outcome coefficient vector of shape (support_size,).
    eta_second_moment : float
        E[eta²], used by the HOML estimator.
    eta_third_moment : float
        E[eta³], used by the HOML estimator.
    lambda_reg : float
        Lasso regularization parameter.
    check_convergence : bool
        If True, filter out runs where ICA did not converge.
    verbose : bool
        Pass through to ``run_single_experiment``.
    oracle_support : bool
        If True, pass oracle support to both OML and ICA.

    Returns
    -------
    List[Tuple]
        Experiment result tuples; filtered for ICA convergence when
        ``check_convergence=True``.
    """
    results = [
        r
        for r in Parallel(n_jobs=-1, verbose=0)(
            delayed(run_single_experiment)(
                x_sample(n_samples, cov_dim_max),
                eta_sample(n_samples),
                epsilon_sample(n_samples),
                treatment_effect,
                treatment_support,
                treatment_coef,
                outcome_support,
                outcome_coef,
                eta_second_moment,
                eta_third_moment,
                lambda_reg,
                check_convergence,
                verbose,
                oracle_support,
            )
            for _ in range(n_experiments)
        )
        if (check_convergence is False or r[-1] is not None)
    ]

    return results


# =============================================================================
# Result Processing
# =============================================================================


def extract_treatment_estimates(results: List[Tuple]) -> List[List[float]]:
    """Extract treatment effect estimates from experiment result tuples.

    Parameters
    ----------
    results : List[Tuple]
        Output of ``run_parallel_experiments``; each tuple has the form
        ``(ortho_ml, robust_ortho_ml, robust_ortho_est_ml,
        robust_ortho_est_split_ml, ..., ica_estimate, ica_mcc)``.

    Returns
    -------
    List[List[float]]
        One list per experiment: ``[ortho_ml, robust_ortho_ml,
        robust_ortho_est, robust_ortho_split, ica_0, ...]``.
    """
    ortho_rec_tau = [
        [ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] + ica_estimate.tolist()
        for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_estimate, _ in results
    ]
    return ortho_rec_tau


def compute_estimation_statistics(
    ortho_rec_tau: List[List[float]], treatment_effect: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bias, standard deviation, and RMSE across Monte Carlo estimates.

    Parameters
    ----------
    ortho_rec_tau : List[List[float]]
        Estimates from ``extract_treatment_estimates``; shape
        (n_experiments, n_methods).
    treatment_effect : float
        True treatment effect theta (constant across experiments).

    Returns
    -------
    biases : np.ndarray
        Mean error per method, shape (n_methods,).
    sigmas : np.ndarray
        Standard deviation per method, shape (n_methods,).
    rmse : np.ndarray
        Root mean squared error per method, shape (n_methods,).
    """
    ortho_rec_tau_array = np.array(ortho_rec_tau)
    biases = np.mean(ortho_rec_tau_array - treatment_effect, axis=0)
    sigmas = np.std(ortho_rec_tau_array, axis=0)
    rmse = np.sqrt(biases**2 + sigmas**2)
    return biases, sigmas, rmse


def compute_estimation_statistics_varying_te(
    ortho_rec_tau: List[List[float]], treatment_effects: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bias, standard deviation, and RMSE when theta varies per experiment.

    Use this variant instead of ``compute_estimation_statistics`` when the
    true treatment effect differs across Monte Carlo replications.

    Parameters
    ----------
    ortho_rec_tau : List[List[float]]
        Estimates from ``extract_treatment_estimates``; shape
        (n_experiments, n_methods).
    treatment_effects : np.ndarray
        True treatment effect per experiment, shape (n_experiments,).

    Returns
    -------
    biases : np.ndarray
        Mean error per method, shape (n_methods,).
    sigmas : np.ndarray
        Standard deviation of estimates per method, shape (n_methods,).
    rmse : np.ndarray
        Root mean squared error per method, shape (n_methods,).
    """
    ortho_rec_tau_array = np.array(ortho_rec_tau)
    # Compute bias relative to each experiment's true treatment effect
    biases = np.mean(ortho_rec_tau_array - treatment_effects[:, None], axis=0)
    sigmas = np.std(ortho_rec_tau_array, axis=0)
    rmse = np.sqrt(np.mean((ortho_rec_tau_array - treatment_effects[:, None]) ** 2, axis=0))
    return biases, sigmas, rmse
