"""
ICA Variants: Triangular, Constrained, and Regularized ICA

This module implements variants of Independent Component Analysis (ICA) with support
for different initialization strategies including random triangular matrices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import FastICA


def random_triangular_matrix(n_components, lower=True, random_state=None):
    """
    Generate a random triangular matrix for ICA initialization.

    Parameters
    ----------
    n_components : int
        Dimension of the square matrix
    lower : bool, default=True
        If True, generate lower triangular. If False, generate upper triangular.
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Random triangular matrix of shape (n_components, n_components)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random matrix
    matrix = np.random.randn(n_components, n_components)

    # Apply triangular constraint
    if lower:
        matrix = np.tril(matrix)
    else:
        matrix = np.triu(matrix)

    # Ensure diagonal elements are non-zero
    diag_indices = np.diag_indices(n_components)
    if np.any(np.abs(matrix[diag_indices]) < 1e-6):
        matrix[diag_indices] = np.where(
            np.abs(matrix[diag_indices]) < 1e-6,
            np.random.randn(n_components),
            matrix[diag_indices]
        )

    return matrix


def whiten_data(X):
    """
    Whiten the data using ZCA whitening.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)

    Returns
    -------
    X_white : np.ndarray
        Whitened data
    K : np.ndarray
        Whitening matrix
    """
    # Center the data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # Compute covariance matrix
    cov = np.cov(X_centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Add small constant for numerical stability
    eigenvalues = eigenvalues + 1e-10

    # Whitening matrix
    K = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    # Whiten the data
    X_white = X_centered @ K.T

    return X_white, K


class TriangularICA:
    """
    Independent Component Analysis with triangular unmixing matrix constraint.

    This variant enforces a lower triangular structure on the unmixing matrix,
    which can be useful for identifying causal orderings.

    Parameters
    ----------
    n_components : int
        Number of components to extract
    max_iter : int, default=1000
        Maximum number of optimization iterations
    tol : float, default=1e-4
        Convergence tolerance
    learning_rate : float, default=0.01
        Learning rate for optimization
    random_state : int or None
        Random seed
    init : str, default='random_triangular'
        Initialization method: 'random_triangular', 'standard', or 'identity'
    lower : bool, default=True
        Use lower triangular (True) or upper triangular (False)
    whiten : bool, default=True
        Whether to whiten the data before ICA
    """

    def __init__(
        self,
        n_components,
        max_iter=1000,
        tol=1e-4,
        learning_rate=0.01,
        random_state=None,
        init="random_triangular",
        lower=True,
        whiten=True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.init = init
        self.lower = lower
        self.whiten = whiten
        self.unmixing_ = None
        self.mixing_ = None
        self.whitening_ = None
        self.converged_ = False

    def _initialize_unmixing(self):
        """Initialize the unmixing matrix."""
        if self.init == "random_triangular":
            W = random_triangular_matrix(
                self.n_components, lower=self.lower, random_state=self.random_state
            )
        elif self.init == "identity":
            W = np.eye(self.n_components)
            if self.lower:
                W = np.tril(W)
            else:
                W = np.triu(W)
        else:  # standard random initialization
            if self.random_state is not None:
                np.random.seed(self.random_state)
            W = np.random.randn(self.n_components, self.n_components)
            if self.lower:
                W = np.tril(W)
            else:
                W = np.triu(W)

        return torch.tensor(W, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def _negentropy_logcosh(s):
        """Compute negative entropy approximation using logcosh."""
        return torch.mean(torch.log(torch.cosh(s)))

    def _apply_triangular_constraint(self, W):
        """Apply triangular constraint to the unmixing matrix."""
        if self.lower:
            return torch.tril(W)
        else:
            return torch.triu(W)

    def fit_transform(self, X):
        """
        Fit the model and return the transformed data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)

        Returns
        -------
        S : np.ndarray
            Estimated independent components
        """
        # Whiten the data if requested
        if self.whiten:
            X_white, K = whiten_data(X)
            self.whitening_ = K
        else:
            X_white = X - X.mean(axis=0)
            self.whitening_ = None

        X_tensor = torch.tensor(X_white, dtype=torch.float32)

        # Initialize unmixing matrix
        W = self._initialize_unmixing()

        optimizer = optim.Adam([W], lr=self.learning_rate)

        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Apply triangular constraint
            W_constrained = self._apply_triangular_constraint(W)

            # Compute sources
            S = X_tensor @ W_constrained.T

            # Compute negative entropy for each component
            loss = 0
            for i in range(self.n_components):
                loss -= self._negentropy_logcosh(S[:, i])

            # Add small regularization to encourage non-degenerate solutions
            loss += 1e-4 * torch.norm(W_constrained)

            loss.backward()
            optimizer.step()

            # Check convergence
            if abs(prev_loss - loss.item()) < self.tol:
                self.converged_ = True
                break
            prev_loss = loss.item()

        # Store final unmixing matrix
        W_final = self._apply_triangular_constraint(W.detach())
        self.unmixing_ = W_final.numpy()

        # Compute mixing matrix (pseudo-inverse for triangular matrices)
        try:
            self.mixing_ = np.linalg.inv(self.unmixing_)
        except np.linalg.LinAlgError:
            self.mixing_ = np.linalg.pinv(self.unmixing_)

        # Compute final sources
        S_final = X_white @ self.unmixing_.T

        return S_final


class ConstrainedICA:
    """
    Independent Component Analysis with additional constraints.

    Supports orthogonality and non-negativity constraints on the unmixing matrix.

    Parameters
    ----------
    n_components : int
        Number of components to extract
    max_iter : int, default=1000
        Maximum number of optimization iterations
    tol : float, default=1e-4
        Convergence tolerance
    learning_rate : float, default=0.01
        Learning rate for optimization
    random_state : int or None
        Random seed
    init : str, default='random_triangular'
        Initialization method
    orthogonal : bool, default=False
        Enforce orthogonality constraint
    non_negative : bool, default=False
        Enforce non-negativity constraint
    whiten : bool, default=True
        Whether to whiten the data
    """

    def __init__(
        self,
        n_components,
        max_iter=1000,
        tol=1e-4,
        learning_rate=0.01,
        random_state=None,
        init="random_triangular",
        orthogonal=False,
        non_negative=False,
        whiten=True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.init = init
        self.orthogonal = orthogonal
        self.non_negative = non_negative
        self.whiten = whiten
        self.unmixing_ = None
        self.mixing_ = None
        self.whitening_ = None
        self.converged_ = False

    def _initialize_unmixing(self):
        """Initialize the unmixing matrix."""
        if self.init == "random_triangular":
            W = random_triangular_matrix(
                self.n_components, lower=True, random_state=self.random_state
            )
        elif self.init == "identity":
            W = np.eye(self.n_components)
        else:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            W = np.random.randn(self.n_components, self.n_components)

        # Apply non-negativity if needed
        if self.non_negative:
            W = np.abs(W)

        return torch.tensor(W, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def _negentropy_logcosh(s):
        """Compute negative entropy approximation using logcosh."""
        return torch.mean(torch.log(torch.cosh(s)))

    def _apply_constraints(self, W):
        """Apply constraints to the unmixing matrix."""
        W_constrained = W

        # Apply non-negativity
        if self.non_negative:
            W_constrained = torch.abs(W_constrained)

        # Apply orthogonality via Gram-Schmidt-like projection
        if self.orthogonal:
            # Use SVD to project onto orthogonal matrices
            U, _, Vt = torch.linalg.svd(W_constrained, full_matrices=False)
            W_constrained = U @ Vt

        return W_constrained

    def fit_transform(self, X):
        """
        Fit the model and return the transformed data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)

        Returns
        -------
        S : np.ndarray
            Estimated independent components
        """
        # Whiten the data if requested
        if self.whiten:
            X_white, K = whiten_data(X)
            self.whitening_ = K
        else:
            X_white = X - X.mean(axis=0)
            self.whitening_ = None

        X_tensor = torch.tensor(X_white, dtype=torch.float32)

        # Initialize unmixing matrix
        W = self._initialize_unmixing()

        optimizer = optim.Adam([W], lr=self.learning_rate)

        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Apply constraints
            W_constrained = self._apply_constraints(W)

            # Compute sources
            S = X_tensor @ W_constrained.T

            # Compute negative entropy for each component
            loss = 0
            for i in range(self.n_components):
                loss -= self._negentropy_logcosh(S[:, i])

            # Add regularization
            loss += 1e-4 * torch.norm(W_constrained)

            loss.backward()
            optimizer.step()

            # Check convergence
            if abs(prev_loss - loss.item()) < self.tol:
                self.converged_ = True
                break
            prev_loss = loss.item()

        # Store final unmixing matrix
        W_final = self._apply_constraints(W.detach())
        self.unmixing_ = W_final.numpy()

        # Compute mixing matrix
        try:
            self.mixing_ = np.linalg.inv(self.unmixing_)
        except np.linalg.LinAlgError:
            self.mixing_ = np.linalg.pinv(self.unmixing_)

        # Compute final sources
        S_final = X_white @ self.unmixing_.T

        return S_final


class RegularizedICA:
    """
    Independent Component Analysis with L1/L2 regularization.

    Parameters
    ----------
    n_components : int
        Number of components to extract
    max_iter : int, default=1000
        Maximum number of optimization iterations
    tol : float, default=1e-4
        Convergence tolerance
    learning_rate : float, default=0.01
        Learning rate for optimization
    random_state : int or None
        Random seed
    init : str, default='random_triangular'
        Initialization method
    l1_penalty : float, default=0.0
        L1 regularization strength (promotes sparsity)
    l2_penalty : float, default=0.0
        L2 regularization strength (promotes smoothness)
    whiten : bool, default=True
        Whether to whiten the data
    """

    def __init__(
        self,
        n_components,
        max_iter=1000,
        tol=1e-4,
        learning_rate=0.01,
        random_state=None,
        init="random_triangular",
        l1_penalty=0.0,
        l2_penalty=0.0,
        whiten=True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.init = init
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.whiten = whiten
        self.unmixing_ = None
        self.mixing_ = None
        self.whitening_ = None
        self.converged_ = False

    def _initialize_unmixing(self):
        """Initialize the unmixing matrix."""
        if self.init == "random_triangular":
            W = random_triangular_matrix(
                self.n_components, lower=True, random_state=self.random_state
            )
        elif self.init == "identity":
            W = np.eye(self.n_components)
        else:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            W = np.random.randn(self.n_components, self.n_components)

        return torch.tensor(W, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def _negentropy_logcosh(s):
        """Compute negative entropy approximation using logcosh."""
        return torch.mean(torch.log(torch.cosh(s)))

    def fit_transform(self, X):
        """
        Fit the model and return the transformed data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)

        Returns
        -------
        S : np.ndarray
            Estimated independent components
        """
        # Whiten the data if requested
        if self.whiten:
            X_white, K = whiten_data(X)
            self.whitening_ = K
        else:
            X_white = X - X.mean(axis=0)
            self.whitening_ = None

        X_tensor = torch.tensor(X_white, dtype=torch.float32)

        # Initialize unmixing matrix
        W = self._initialize_unmixing()

        optimizer = optim.Adam([W], lr=self.learning_rate)

        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Compute sources
            S = X_tensor @ W.T

            # Compute negative entropy for each component
            loss = 0
            for i in range(self.n_components):
                loss -= self._negentropy_logcosh(S[:, i])

            # Add L1 regularization (sparsity)
            if self.l1_penalty > 0:
                loss += self.l1_penalty * torch.norm(W, p=1)

            # Add L2 regularization (smoothness)
            if self.l2_penalty > 0:
                loss += self.l2_penalty * torch.norm(W, p=2)

            loss.backward()
            optimizer.step()

            # Check convergence
            if abs(prev_loss - loss.item()) < self.tol:
                self.converged_ = True
                break
            prev_loss = loss.item()

        # Store final unmixing matrix
        self.unmixing_ = W.detach().numpy()

        # Compute mixing matrix
        try:
            self.mixing_ = np.linalg.inv(self.unmixing_)
        except np.linalg.LinAlgError:
            self.mixing_ = np.linalg.pinv(self.unmixing_)

        # Compute final sources
        S_final = X_white @ self.unmixing_.T

        return S_final


def ica_treatment_effect_estimation_variant(
    X,
    S,
    variant="triangular",
    random_state=0,
    n_treatments=1,
    verbose=True,
    init="random_triangular",
    **kwargs
):
    """
    Estimate treatment effects using ICA variants.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Observed data matrix
    S : np.ndarray or torch.Tensor
        True sources (for evaluation)
    variant : str
        ICA variant: 'triangular', 'constrained', or 'regularized'
    random_state : int
        Random seed
    n_treatments : int
        Number of treatment variables
    verbose : bool
        Print diagnostic information
    init : str
        Initialization method: 'random_triangular', 'standard', or 'identity'
    **kwargs
        Additional arguments passed to the ICA variant

    Returns
    -------
    treatment_effect_estimate : np.ndarray
        Estimated treatment effects
    disentanglement_score : float
        Disentanglement metric (MCC)
    """
    from mcc import calc_disent_metrics  # pylint: disable=import-outside-toplevel

    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(S, torch.Tensor):
        S = S.numpy()

    # Select ICA variant
    if variant == "triangular":
        ica = TriangularICA(
            n_components=X.shape[1],
            random_state=random_state,
            init=init,
            **kwargs
        )
    elif variant == "constrained":
        ica = ConstrainedICA(
            n_components=X.shape[1],
            random_state=random_state,
            init=init,
            **kwargs
        )
    elif variant == "regularized":
        ica = RegularizedICA(
            n_components=X.shape[1],
            random_state=random_state,
            init=init,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    try:
        S_hat = ica.fit_transform(X)

        if not ica.converged_ and verbose:
            print(f"Warning: ICA did not converge (variant={variant}, init={init})")

        # Compute disentanglement metrics
        results = calc_disent_metrics(S, S_hat)

        # Resolve permutations
        permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]

        # Normalize to get 1 at epsilon -> Y
        permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

        # Extract treatment effect
        n_covariates = X.shape[1] - 1 - n_treatments
        treatment_effect_estimate = permuted_scaled_mixing[-1, n_covariates:-1]

        return treatment_effect_estimate, results["permutation_disentanglement_score"]

    except Exception as e:
        if verbose:
            print(f"Error in ICA variant {variant} with init {init}: {e}")
        return np.nan * np.ones(n_treatments), None
