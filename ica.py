"""
ICA-based treatment effect estimation for the partially linear model.

Implements data generation and estimation routines using FastICA to recover
treatment effects theta in Y = D*theta + X*b + eps, where D, X, eps are
mutually independent non-Gaussian sources.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gennorm
from sklearn.decomposition import FastICA

from mcc import calc_disent_metrics

# FastICA hyperparameters
ICA_MAX_ITER = 1000
ICA_BASE_TOL = 1e-4
ICA_MAX_RETRIES = 5

# Fixed treatment effect vector used when theta_choice="fixed"
DEFAULT_TREATMENT_EFFECTS = [1.55, 0.65, -2.45, 1.75, -1.35]


def generate_ica_data(
    n_covariates: int = 1,
    n_treatments: int = 1,
    batch_size: int = 4096,
    slope: float = 1.0,
    sparse_prob: float = 0.3,
    beta: float = 1.0,
    loc: float = 0,
    scale: float = 1,
    nonlinearity: str = "leaky_relu",
    theta_choice: str = "fixed",
    split_noise_dist: bool = False,
    split_eta_eps: bool = False,
    beta_eta: float = None,
) -> tuple:
    """Generate synthetic data from a partially linear model for ICA-based estimation.

    Sources S are drawn i.i.d. from a generalized normal distribution. Covariates
    pass through unchanged; treatments mix covariates via a sparse nonlinear map;
    the outcome follows Y = theta*D + b*f(X) + eps.

    Parameters
    ----------
    n_covariates : int
        Number of covariate dimensions.
    n_treatments : int
        Number of treatment variables.
    batch_size : int
        Number of samples to generate.
    slope : float
        Negative slope for leaky_relu nonlinearity.
    sparse_prob : float
        Probability of non-zero entries in the covariate-to-treatment coefficient
        matrix A.
    beta : float
        Shape parameter of the generalized normal distribution (beta=1 is Laplace,
        beta=2 is Gaussian).
    loc : float
        Location parameter of the generalized normal distribution.
    scale : float
        Scale parameter of the generalized normal distribution.
    nonlinearity : str
        Activation function applied to covariates before mixing. One of
        "leaky_relu", "relu", "sigmoid", "tanh".
    theta_choice : str
        How to generate treatment effects theta. One of "fixed" (deterministic
        vector), "uniform", or "gaussian".
    split_noise_dist : bool
        If True, covariates are drawn from N(0,1) while treatment and outcome
        noise share gennorm(beta).
    split_eta_eps : bool
        If True, decouple eta and eps: covariates ~ N(0,1), eta ~ gennorm(beta_eta),
        eps ~ gennorm(beta).
    beta_eta : float, optional
        Shape parameter for the treatment noise distribution when split_eta_eps
        is True. Defaults to 2.0 (Gaussian).

    Returns
    -------
    S : torch.Tensor
        Source matrix of shape (batch_size, n_covariates + n_treatments + 1).
    X : torch.Tensor
        Observed data matrix of shape (batch_size, n_covariates + n_treatments + 1),
        ordered as [covariates, treatments, outcome].
    theta : torch.Tensor
        True treatment effect vector of shape (n_treatments,).
    """
    # Create sparse matrix of shape (n_treatments x n_covariates)
    binary_mask = torch.bernoulli(torch.ones(n_treatments, n_covariates) * sparse_prob)
    random_coeffs = torch.randn(n_treatments, n_covariates)
    A_covariates = binary_mask * random_coeffs

    if theta_choice == "fixed":
        theta = torch.tensor(DEFAULT_TREATMENT_EFFECTS)[:n_treatments]  # Fixed vector of thetas
    elif theta_choice == "uniform":
        theta = torch.rand(n_treatments)  # Draw theta from a uniform distribution
    elif theta_choice == "gaussian":
        theta = torch.randn(n_treatments)  # Draw theta from a Gaussian distribution
    else:
        raise ValueError(f"Unsupported theta_choice for theta generation: {theta_choice}")

    B = torch.randn(n_covariates)  # Base effects on outcome per covariate

    distribution = gennorm(beta, loc=loc, scale=scale)

    source_dim = n_covariates + n_treatments + 1  # +1 for outcome
    if split_eta_eps:
        # Decouple eta and eps distributions: eta gets beta_eta, eps gets beta
        _beta_eta = beta_eta if beta_eta is not None else 2.0
        S_X = torch.tensor(gennorm(beta=2.0, loc=loc, scale=scale).rvs(size=(batch_size, n_covariates))).float()
        S_eta = torch.tensor(gennorm(beta=_beta_eta, loc=loc, scale=scale).rvs(size=(batch_size, n_treatments))).float()
        S_eps = torch.tensor(distribution.rvs(size=(batch_size, 1))).float()
        S = torch.hstack((S_X, S_eta, S_eps))
    elif split_noise_dist is False:
        S = torch.tensor(distribution.rvs(size=(batch_size, source_dim))).float()
    else:
        S_X = torch.tensor(gennorm(beta=2.0, loc=loc, scale=scale).rvs(size=(batch_size, n_covariates))).float()
        S_TY = torch.tensor(distribution.rvs(size=(batch_size, n_treatments + 1))).float()
        S = torch.hstack((S_X, S_TY))
    X = S.clone()

    # Define activation function based on the nonlinearity parameter
    activation_functions = {
        "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=slope),
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }

    if nonlinearity not in activation_functions:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    activation = activation_functions[nonlinearity]

    # Covariates remain independent
    X[:, :n_covariates] = S[:, :n_covariates]

    # Treatments depend on all covariates
    treatment_indices = torch.arange(n_covariates, n_covariates + n_treatments)
    X[:, treatment_indices] = S[:, treatment_indices]
    # Modified to use the sparse matrix multiplication
    covariate_effects = activation(S[:, :n_covariates]) @ A_covariates.t()
    X[:, treatment_indices] += covariate_effects

    # Outcome depends on all covariates and treatments
    X[:, -1] = S[:, -1]

    # Add treatment effects
    X[:, -1] += (theta * X[:, treatment_indices]).sum(dim=1)

    # Add covariate effects

    # Use the sum of all sparse connections for the outcome
    X[:, -1] += (B * activation(S[:, :n_covariates])).sum(dim=1)

    return S, X, theta


def ica_treatment_effect_estimation(
    X: np.ndarray,
    S: np.ndarray,
    random_state: int = 0,
    whiten: str = "unit-variance",
    check_convergence: bool = True,
    n_treatments: int = 1,
    verbose: bool = True,
    fun: str = "logcosh",
) -> tuple:
    """Estimate treatment effects via FastICA with Munkres permutation matching.

    Fits FastICA to X, resolves the permutation ambiguity by matching recovered
    sources to ground-truth S via the Hungarian algorithm (MCC), then extracts
    theta from the normalized mixing matrix column corresponding to the outcome.

    Parameters
    ----------
    X : np.ndarray
        Observed data matrix of shape (n_samples, n_variables), ordered as
        [covariates, treatments, outcome].
    S : np.ndarray
        Ground-truth source matrix of shape (n_samples, n_variables), used for
        permutation resolution via MCC.
    random_state : int
        Base random seed (incremented on each convergence retry).
    whiten : str
        Whitening strategy passed to FastICA.
    check_convergence : bool
        If True, return NaN when FastICA fails to converge within max iterations.
    n_treatments : int
        Number of treatment variables.
    verbose : bool
        Print convergence attempt information.
    fun : str
        Contrast function for FastICA ("logcosh", "exp", or "cube").

    Returns
    -------
    treatment_effect_estimate : np.ndarray
        Estimated theta of shape (n_treatments,), or NaN array on convergence
        failure.
    mcc : float or None
        Permutation disentanglement score (MCC), or None on failure.
    """
    from warnings import catch_warnings  # pylint: disable=import-outside-toplevel

    tol = ICA_BASE_TOL  # Initial tolerance

    for attempt in range(ICA_MAX_RETRIES):
        with catch_warnings(record=True) as w:

            ica = FastICA(
                n_components=X.shape[1],
                random_state=random_state + attempt,
                max_iter=ICA_MAX_ITER,
                whiten=whiten,
                tol=tol,
                fun=fun,
            )
            S_hat = ica.fit_transform(X)

            if len(w) > 0 and check_convergence is True:
                if verbose:
                    print(f"warning at {attempt=}")
                print("Max tolerance reached without convergence")
                return (
                    np.nan
                    * np.ones(
                        n_treatments,
                    ),
                    None,
                )
            if verbose:
                print(f"success at {attempt=}")
            break

    results = calc_disent_metrics(S, S_hat)
    # resolve the permutations
    permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
    # normalize to get 1 at epsilon -> Y
    permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

    n_covariates = X.shape[1] - 1 - n_treatments  # Assuming  and 1 outcome
    treatment_effect_estimate = permuted_scaled_mixing[-1, n_covariates:-1]

    return treatment_effect_estimate, results["permutation_disentanglement_score"]


def ica_treatment_effect_estimation_eps_row(
    X: np.ndarray,
    S: np.ndarray = None,
    random_state: int = 0,
    whiten: str = "unit-variance",
    check_convergence: bool = True,
    n_treatments: int = 1,
    verbose: bool = True,
    fun: str = "logcosh",
) -> tuple:
    """Estimate treatment effects by identifying the eps component via non-Gaussianity.

    Unlike ``ica_treatment_effect_estimation``, this does not rely on Munkres
    matching. It identifies the outcome noise eps as the component with the
    highest absolute excess kurtosis, then reads theta from the corresponding
    row of the unmixing matrix W. This is valid even when eta is Gaussian,
    since theta is encoded in the eps row of W.

    Parameters
    ----------
    X : np.ndarray
        Observed data matrix of shape (n_samples, n_variables), ordered as
        [covariates, treatments, outcome].
    S : np.ndarray, optional
        Ground-truth source matrix of shape (n_samples, n_variables). Used
        only to compute MCC; pass None to skip.
    random_state : int
        Base random seed (incremented on convergence retries).
    whiten : str
        Whitening strategy for FastICA.
    check_convergence : bool
        If True, return NaN when FastICA fails to converge.
    n_treatments : int
        Number of treatment variables.
    verbose : bool
        Print convergence and component identification information.
    fun : str
        Contrast function for FastICA ("logcosh", "exp", or "cube").

    Returns
    -------
    treatment_effect_estimate : np.ndarray
        Estimated theta of shape (n_treatments,), or NaN array on failure.
    mcc_score : float or None
        Permutation disentanglement score, or None if S is not provided or
        convergence fails.
    """
    from warnings import catch_warnings  # pylint: disable=import-outside-toplevel

    from scipy.stats import kurtosis  # pylint: disable=import-outside-toplevel

    tol = ICA_BASE_TOL

    for attempt in range(ICA_MAX_RETRIES):
        with catch_warnings(record=True) as w:
            ica = FastICA(
                n_components=X.shape[1],
                random_state=random_state + attempt,
                max_iter=ICA_MAX_ITER,
                whiten=whiten,
                tol=tol,
                fun=fun,
            )
            S_hat = ica.fit_transform(X)

            if len(w) > 0 and check_convergence is True:
                if verbose:
                    print(f"warning at {attempt=}")
                print("Max tolerance reached without convergence")
                return (
                    np.nan * np.ones(n_treatments),
                    None,
                )
            if verbose:
                print(f"success at {attempt=}")
            break

    # Identify the eps component: the most non-Gaussian component that loads
    # heavily on Y (the last observed variable).
    # The unmixing matrix W satisfies S_hat = W @ X, so W = ica.components_
    # The mixing matrix A = ica.mixing_ satisfies X = A @ S_hat
    # The eps row of W is the row whose projection extracts eps from observations.
    # We identify it by: (1) restrict to components with nonzero loading on Y,
    # (2) pick the one with highest absolute excess kurtosis.
    n_components = S_hat.shape[1]
    abs_kurtosis = np.array([np.abs(kurtosis(S_hat[:, j])) for j in range(n_components)])

    # Among components, find the one with max |kurtosis| — this is eps
    eps_idx = int(np.argmax(abs_kurtosis))

    if verbose:
        print(f"Identified eps component: {eps_idx} (|kurtosis|={abs_kurtosis[eps_idx]:.3f})")

    # The eps row of the unmixing matrix W
    # W = ica.components_ has shape (n_components, n_features)
    w_eps = ica.components_[eps_idx, :]

    # Normalize so that the Y-entry (last column) is 1 (since eps enters Y with coeff 1)
    w_eps_normalized = w_eps / w_eps[-1]

    # theta is in the treatment columns: columns n_covariates to n_covariates+n_treatments
    n_covariates = X.shape[1] - 1 - n_treatments
    # The unmixing row has the NEGATIVE of theta (since W = A^{-1} and the eps row
    # of W is [-b', -theta, 1]), so we negate to get theta
    treatment_effect_estimate = -w_eps_normalized[n_covariates : n_covariates + n_treatments]

    # Compute MCC if ground truth sources are provided
    mcc_score = None
    if S is not None:
        results = calc_disent_metrics(S, S_hat)
        mcc_score = results["permutation_disentanglement_score"]

    return treatment_effect_estimate, mcc_score
