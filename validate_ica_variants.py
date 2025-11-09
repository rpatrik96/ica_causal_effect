"""
Quick validation script for ICA variants.
Run this to verify basic functionality without pytest.
"""

import numpy as np
import torch
from ica import generate_ica_data
from ica_variants import (
    TriangularICA,
    ConstrainedICA,
    RegularizedICA,
    random_triangular_matrix,
    whiten_data,
    ica_treatment_effect_estimation_variant,
)


def test_random_triangular_matrix():
    """Test random triangular matrix generation."""
    print("Testing random_triangular_matrix...")

    # Lower triangular
    matrix_lower = random_triangular_matrix(5, lower=True, random_state=42)
    assert matrix_lower.shape == (5, 5)
    assert np.allclose(np.triu(matrix_lower, k=1), 0.0), "Lower triangular constraint failed"

    # Upper triangular
    matrix_upper = random_triangular_matrix(5, lower=False, random_state=42)
    assert np.allclose(np.tril(matrix_upper, k=-1), 0.0), "Upper triangular constraint failed"

    print("  ✓ Random triangular matrix generation works")


def test_whitening():
    """Test whitening function."""
    print("Testing whitening...")

    X = np.random.randn(1000, 10)
    X_white, K = whiten_data(X)

    assert X_white.shape == X.shape, "Whitening changed shape"
    assert np.allclose(X_white.mean(axis=0), 0.0, atol=1e-10), "Whitened data not centered"

    cov = np.cov(X_white.T)
    assert np.allclose(cov, np.eye(10), atol=0.1), "Covariance not identity"

    print("  ✓ Whitening works correctly")


def test_triangular_ica():
    """Test TriangularICA."""
    print("Testing TriangularICA...")

    X = np.random.randn(200, 5)
    ica = TriangularICA(n_components=5, max_iter=100, random_state=42, init="random_triangular")

    S = ica.fit_transform(X)

    assert S.shape == X.shape, "Output shape mismatch"
    assert ica.unmixing_ is not None, "Unmixing matrix not computed"
    assert np.allclose(np.triu(ica.unmixing_, k=1), 0.0, atol=1e-5), "Triangular constraint not enforced"

    print("  ✓ TriangularICA works correctly")


def test_constrained_ica():
    """Test ConstrainedICA."""
    print("Testing ConstrainedICA...")

    X = np.random.randn(200, 5)
    ica = ConstrainedICA(
        n_components=5,
        max_iter=100,
        random_state=42,
        init="random_triangular",
        non_negative=True
    )

    S = ica.fit_transform(X)

    assert S.shape == X.shape, "Output shape mismatch"
    assert ica.unmixing_ is not None, "Unmixing matrix not computed"
    assert np.all(ica.unmixing_ >= -1e-6), "Non-negativity constraint not enforced"

    print("  ✓ ConstrainedICA works correctly")


def test_regularized_ica():
    """Test RegularizedICA."""
    print("Testing RegularizedICA...")

    X = np.random.randn(200, 5)
    ica = RegularizedICA(
        n_components=5,
        max_iter=100,
        random_state=42,
        init="random_triangular",
        l1_penalty=0.01,
        l2_penalty=0.01
    )

    S = ica.fit_transform(X)

    assert S.shape == X.shape, "Output shape mismatch"
    assert ica.unmixing_ is not None, "Unmixing matrix not computed"

    print("  ✓ RegularizedICA works correctly")


def test_integration_with_data_generation():
    """Test ICA variants with generated data."""
    print("Testing integration with generated data...")

    S, X, theta_true = generate_ica_data(
        n_covariates=10,
        n_treatments=1,
        batch_size=500,
        beta=1.0,
        sparse_prob=0.3,
    )

    X_np = X.numpy()
    S_np = S.numpy()

    # Test all variants
    for variant in ["triangular", "constrained", "regularized"]:
        variant_kwargs = {}
        if variant == "regularized":
            variant_kwargs = {"l1_penalty": 0.01, "l2_penalty": 0.01}

        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X_np,
            S_np,
            variant=variant,
            random_state=42,
            n_treatments=1,
            init="random_triangular",
            max_iter=200,
            **variant_kwargs,
        )

        assert theta_est is not None, f"{variant} failed to estimate treatment effect"
        assert theta_est.shape == (1,), f"{variant} returned wrong shape"
        print(f"  ✓ {variant} ICA works with generated data")


def test_all_initializations():
    """Test all initialization methods."""
    print("Testing all initialization methods...")

    S, X, theta_true = generate_ica_data(
        n_covariates=10,
        n_treatments=1,
        batch_size=500,
        beta=1.0,
    )

    X_np = X.numpy()
    S_np = S.numpy()

    for init in ["random_triangular", "standard", "identity"]:
        theta_est, mcc = ica_treatment_effect_estimation_variant(
            X_np,
            S_np,
            variant="triangular",
            random_state=42,
            n_treatments=1,
            init=init,
            max_iter=200,
        )

        assert theta_est is not None, f"{init} initialization failed"
        print(f"  ✓ {init} initialization works")


def main():
    """Run all validation tests."""
    print("="*60)
    print("Validating ICA Variants Implementation")
    print("="*60)
    print()

    try:
        test_random_triangular_matrix()
        test_whitening()
        test_triangular_ica()
        test_constrained_ica()
        test_regularized_ica()
        test_integration_with_data_generation()
        test_all_initializations()

        print()
        print("="*60)
        print("✓ All validation tests passed!")
        print("="*60)
        print()
        print("Note: For comprehensive testing, run:")
        print("  pip install -r requirements-dev.txt")
        print("  pytest tests/test_ica_variants.py -v")

    except AssertionError as e:
        print()
        print("="*60)
        print("✗ Validation failed!")
        print(f"Error: {e}")
        print("="*60)
        return 1

    except Exception as e:
        print()
        print("="*60)
        print("✗ Unexpected error!")
        print(f"Error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
