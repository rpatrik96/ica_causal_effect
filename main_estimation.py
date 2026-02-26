"""Estimation methods for the partially linear model.

Implements four estimators compared in the paper:

- **Ortho ML**: Standard first-order orthogonal (Double ML) estimator.
- **Robust Ortho ML (Known)**: Second-order orthogonal estimator using known
  treatment noise moments (eta_second_moment, eta_third_moment).
- **Robust Ortho ML (Est.)**: Second-order orthogonal estimator where moments
  are estimated from the training fold residuals.
- **Robust Ortho ML (Split)**: Second-order orthogonal estimator where moments
  are estimated on a held-out sub-split of the test fold.

Both a simple sample-split variant (``all_together``) and a 2-fold
cross-fitted variant (``all_together_cross_fitting``) are provided.
"""

from typing import Tuple

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

# Regularisation grid used by default for nuisance function estimation.
DEFAULT_LASSO_ALPHAS = [0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]


def all_together(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    eta_second_moment: float,
    eta_third_moment: float,
    model_treatment: LassoCV = LassoCV(alphas=DEFAULT_LASSO_ALPHAS),
    model_outcome: LassoCV = LassoCV(alphas=DEFAULT_LASSO_ALPHAS),
) -> Tuple[float, float, float, float]:
    """Estimate the treatment effect via sample splitting.

    The dataset is split 50/50 into a training fold and a test fold.
    Nuisance functions (E[treatment|covariates] and E[outcome|covariates])
    are fitted on the training fold via LassoCV.  Treatment effects are
    estimated from the test-fold residuals.

    Implements the partially linear model::

        treatment = g(covariates) + eta
        outcome   = theta * treatment + f(covariates) + epsilon

    where the goal is to recover ``theta``.

    Args:
        covariates: Covariate matrix of shape ``(n_samples, n_features)``.
        treatment: Treatment vector of shape ``(n_samples,)``.
        outcome: Outcome vector of shape ``(n_samples,)``.
        eta_second_moment: Known second central moment of the treatment noise
            eta (i.e. ``E[eta^2]``).
        eta_third_moment: Known third cumulant of the treatment noise eta
            (i.e. ``E[eta^3] - 3*E[eta]*E[eta^2]``).
        model_treatment: Fitted or unfitted sklearn estimator for the
            treatment nuisance function.  Defaults to
            ``LassoCV(alphas=DEFAULT_LASSO_ALPHAS)``.
        model_outcome: Fitted or unfitted sklearn estimator for the
            outcome nuisance function.  Defaults to
            ``LassoCV(alphas=DEFAULT_LASSO_ALPHAS)``.

    Returns:
        Tuple of four scalar treatment-effect estimates:
        ``(ortho_ml, robust_ortho_ml, robust_ortho_est_ml,
        robust_ortho_est_split_ml)``.
    """
    # Split the data in half, train and test
    split_size = covariates.shape[0] // 2
    covariates_train = covariates[:split_size]
    treatment_train = treatment[:split_size]
    outcome_train = outcome[:split_size]
    covariates_test = covariates[split_size:]
    treatment_test = treatment[split_size:]
    outcome_test = outcome[split_size:]

    # Fit with LassoCV the treatment as a function of covariates and the
    # outcome as a function of covariates, using only the train fold.
    # treatment = g0(covariates)
    # outcome   = f0(covariates) + theta * g0(covariates)
    model_treatment.fit(covariates_train, treatment_train)
    model_outcome.fit(covariates_train, outcome_train)

    # Compute residuals on the test fold
    residual_treatment = (treatment_test - model_treatment.predict(covariates_test)).flatten()
    residual_outcome = (outcome_test - model_outcome.predict(covariates_test)).flatten()

    """ ORTHO ML """
    # Compute coefficient by OLS on residuals
    ortho_ml = np.sum(np.multiply(residual_treatment, residual_outcome)) / np.sum(
        np.multiply(residual_treatment, residual_treatment)
    )

    """ ROBUST ORTHO ML with KNOWN MOMENTS """
    # Compute for each sample the quantity:
    #
    #          (Z_i-f(X_i))^3 - 3*(sigma^2)*(Z_i-f(X_i)) - cube_p
    #
    # The coefficient is a simple division:
    #
    #       E_n{ (Y-m(X)) * ((Z-f(X))^3-3(sigma^2)(Z-f(X))) }
    #   -----------------------------------------------------------------
    #   E_n{ (Z-f(x)) * ((Z-f(x))^3 - 3 * (sigma^2) * (Z-f(x)) - cube_p)}
    #
    mult_known = residual_treatment**3 - 3 * eta_second_moment * residual_treatment - eta_third_moment
    robust_ortho_ml = np.mean(residual_outcome * mult_known) / np.mean(residual_treatment * mult_known)

    """ ROBUST ORTHO ML with ESTIMATED MOMENTS """
    # Estimate the moments from the training-fold residuals
    eta_residual_train = treatment_train - model_treatment.predict(covariates_train)
    eta_second_moment_est = np.mean(eta_residual_train**2)
    eta_third_moment_est = np.mean(eta_residual_train**3) - 3 * np.mean(eta_residual_train) * np.mean(
        eta_residual_train**2
    )
    # Estimate the treatment effect from the test fold
    mult_est = residual_treatment**3 - 3 * eta_second_moment_est * residual_treatment - eta_third_moment_est
    robust_ortho_est_ml = np.mean(residual_outcome * mult_est) / np.mean(residual_treatment * mult_est)

    """ ROBUST ORTHO ML with ESTIMATED MOMENTS on THIRD SPLIT """
    # Further split the test fold: estimate moments on one half, evaluate on
    # the other half.
    test_split = covariates_test.shape[0] // 2
    eta_residual_split = residual_treatment[:test_split]
    eta_second_moment_est = np.mean(eta_residual_split**2)
    eta_third_moment_est = np.mean(eta_residual_split**3) - 3 * np.mean(eta_residual_split) * np.mean(
        eta_residual_split**2
    )
    residual_treatment_second = residual_treatment[test_split:]
    residual_outcome_second = residual_outcome[test_split:]
    mult_est_split = (
        residual_treatment_second**3 - 3 * eta_second_moment_est * residual_treatment_second - eta_third_moment_est
    )
    robust_ortho_est_split_ml = np.mean(residual_outcome_second * mult_est_split) / np.mean(
        residual_treatment_second * mult_est_split
    )

    return ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml


def all_together_cross_fitting(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    treatment_second_moment: float,
    treatment_third_moment: float,
    model_treatment: LassoCV = LassoCV(alphas=DEFAULT_LASSO_ALPHAS),
    model_outcome: LassoCV = LassoCV(alphas=DEFAULT_LASSO_ALPHAS),
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Estimate the treatment effect via 2-fold cross-fitting.

    Uses ``sklearn.model_selection.KFold`` with ``n_splits=2``.  For each
    fold, nuisance functions are fitted on the complementary fold, and
    residuals are accumulated for all observations.  Final estimates are
    computed from the full-sample residual arrays.

    For the split-moment estimator (``robust_ortho_est_split_ml``) a nested
    2-fold split is applied inside each test fold so that the moment
    estimates and the score evaluation use disjoint subsets.

    Implements the partially linear model::

        treatment = g(covariates) + eta
        outcome   = theta * treatment + f(covariates) + epsilon

    Args:
        covariates: Covariate matrix of shape ``(n_samples, n_features)``.
        treatment: Treatment vector of shape ``(n_samples,)``.
        outcome: Outcome vector of shape ``(n_samples,)``.
        treatment_second_moment: Known second central moment of the treatment
            noise eta (i.e. ``E[eta^2]``).
        treatment_third_moment: Known third cumulant of the treatment noise
            eta (i.e. ``E[eta^3] - 3*E[eta]*E[eta^2]``).
        model_treatment: Fitted or unfitted sklearn estimator for the
            treatment nuisance function.  Defaults to
            ``LassoCV(alphas=DEFAULT_LASSO_ALPHAS)``.
        model_outcome: Fitted or unfitted sklearn estimator for the
            outcome nuisance function.  Defaults to
            ``LassoCV(alphas=DEFAULT_LASSO_ALPHAS)``.

    Returns:
        Tuple of six values:
        ``(ortho_ml, robust_ortho_ml, robust_ortho_est_ml,
        robust_ortho_est_split_ml, treatment_coef, outcome_coef)``
        where the last two are the final LassoCV coefficient vectors of
        shape ``(n_features,)``.
    """
    residual_treatment = np.zeros(covariates.shape[0])
    residual_outcome = np.zeros(covariates.shape[0])
    mult_known = np.zeros(covariates.shape[0])
    mult_est = np.zeros(covariates.shape[0])
    mult_est_split = np.zeros(covariates.shape[0])

    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(covariates):
        # Split the data, train and test
        covariates_train = covariates[train_index]
        treatment_train = treatment[train_index]
        outcome_train = outcome[train_index]
        covariates_test = covariates[test_index]
        treatment_test = treatment[test_index]
        outcome_test = outcome[test_index]

        # Fit nuisance functions on the training fold
        model_treatment.fit(covariates_train, treatment_train)
        model_outcome.fit(covariates_train, outcome_train)

        # Accumulate test-fold residuals
        residual_treatment[test_index] = (treatment_test - model_treatment.predict(covariates_test)).flatten()
        residual_outcome[test_index] = (outcome_test - model_outcome.predict(covariates_test)).flatten()

        # Estimate multipliers for robust orthogonal methods

        # 1. Multiplier with known moments
        mult_known[test_index] = (
            residual_treatment[test_index] ** 3
            - 3 * treatment_second_moment * residual_treatment[test_index]
            - treatment_third_moment
        )

        # 2. Multiplier with moments estimated on the training fold residuals
        residual_treatment_train = treatment_train - model_treatment.predict(covariates_train)
        second_moment_est = np.mean(residual_treatment_train**2)
        third_moment_est = np.mean(residual_treatment_train**3) - 3 * np.mean(residual_treatment_train) * np.mean(
            residual_treatment_train**2
        )
        mult_est[test_index] = (
            residual_treatment[test_index] ** 3
            - 3 * second_moment_est * residual_treatment[test_index]
            - third_moment_est
        )

        # 3. Multiplier with moments estimated on a nested split of the test fold
        nested_kf = KFold(n_splits=2)
        for nested_train_index, nested_test_index in nested_kf.split(test_index):
            residual_treatment_nested_train = residual_treatment[test_index[nested_train_index]]
            second_moment_est = np.mean(residual_treatment_nested_train**2)
            third_moment_est = np.mean(residual_treatment_nested_train**3) - 3 * np.mean(
                residual_treatment_nested_train
            ) * np.mean(residual_treatment_nested_train**2)
            residual_treatment_nested_test = residual_treatment[test_index[nested_test_index]]
            mult_est_split[test_index[nested_test_index]] = (
                residual_treatment_nested_test**3
                - 3 * second_moment_est * residual_treatment_nested_test
                - third_moment_est
            )

    """ ORTHO ML """
    # Compute coefficient by OLS on residuals
    ortho_ml = np.mean(residual_outcome * residual_treatment) / np.mean(residual_treatment * residual_treatment)

    """ ROBUST ORTHO ML with KNOWN MOMENTS """
    # Compute for each sample the quantity:
    #
    #          (Z_i-f(X_i))^3 - 3*(sigma^2)*(Z_i-f(X_i)) - cube_p
    #
    # The coefficient is a simple division:
    #
    #       E_n{ (Y-m(X)) * ((Z-f(X))^3-3(sigma^2)(Z-f(X))) }
    #   -----------------------------------------------------------------
    #   E_n{ (Z-f(x)) * ((Z-f(x))^3 - 3 * (sigma^2) * (Z-f(x)) - cube_p)}
    #
    robust_ortho_ml = np.mean(residual_outcome * mult_known) / np.mean(residual_treatment * mult_known)

    """ ROBUST ORTHO ML with ESTIMATED MOMENTS """
    robust_ortho_est_ml = np.mean(residual_outcome * mult_est) / np.mean(residual_treatment * mult_est)

    """ ROBUST ORTHO ML with ESTIMATED MOMENTS on THIRD SPLIT """
    robust_ortho_est_split_ml = np.mean(residual_outcome * mult_est_split) / np.mean(
        residual_treatment * mult_est_split
    )

    return (
        ortho_ml,
        robust_ortho_ml,
        robust_ortho_est_ml,
        robust_ortho_est_split_ml,
        model_treatment.coef_,
        model_outcome.coef_,
    )
