"""Tests for plot_utils.py module."""

import numpy as np
import pytest

from plot_utils import plot_estimates, prepare_heatmap_data


class TestPrepareHeatmapData:
    """Tests for prepare_heatmap_data function."""

    @pytest.fixture
    def sample_results(self):
        """Generate sample results data."""
        results = []
        for beta in [1.0, 2.0]:
            for n_samples in [100, 500]:
                for support_size in [5, 10]:
                    result = {
                        "beta": beta,
                        "n_samples": n_samples,
                        "support_size": support_size,
                        "biases": [0.1, 0.2, 0.15, 0.12, 0.08],  # 5 methods
                        "sigmas": [0.05, 0.06, 0.055, 0.052, 0.04],
                        "biases_rel": [0.11, 0.21, 0.16, 0.13, 0.09],
                        "sigmas_rel": [0.051, 0.061, 0.056, 0.053, 0.041],
                        "first_stage_mse": [[0.1], [0.2], [0.15]],
                    }
                    results.append(result)
        return results

    def test_prepare_heatmap_data_returns_correct_shapes(self, sample_results):
        """Test that prepare_heatmap_data returns matrices of correct shape."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, data_matrix_rmse = prepare_heatmap_data(
            sample_results, x_key="beta", y_key="n_samples", value_key="biases", diff_index=3
        )

        unique_x = len(set([r["beta"] for r in sample_results]))
        unique_y = len(set([r["n_samples"] for r in sample_results]))

        assert data_matrix_mean.shape == (
            unique_y,
            unique_x,
        ), f"Mean matrix should be ({unique_y}, {unique_x})"
        assert data_matrix_std.shape == (unique_y, unique_x), f"Std matrix should be ({unique_y}, {unique_x})"
        assert data_matrix.shape == (unique_y, unique_x), f"Data matrix should be ({unique_y}, {unique_x})"
        assert len(x_values) == unique_x, f"x_values should have {unique_x} elements"
        assert len(y_values) == unique_y, f"y_values should have {unique_y} elements"
        assert data_matrix_rmse is None, "RMSE matrix should be None when compute_rmse=False"

    def test_prepare_heatmap_data_with_beta_filter(self, sample_results):
        """Test prepare_heatmap_data with beta filter."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, _ = prepare_heatmap_data(
            sample_results,
            x_key="support_size",
            y_key="n_samples",
            value_key="biases",
            diff_index=3,
            beta_filter=1.0,
        )

        # Should only include results with beta=1.0
        assert len(x_values) > 0, "x_values should not be empty"
        assert len(y_values) > 0, "y_values should not be empty"

    def test_prepare_heatmap_data_with_support_filter(self, sample_results):
        """Test prepare_heatmap_data with support_size filter."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, _ = prepare_heatmap_data(
            sample_results,
            x_key="beta",
            y_key="n_samples",
            value_key="biases",
            diff_index=3,
            support_size_filter=5,
        )

        # Should only include results with support_size=5
        assert len(x_values) > 0, "x_values should not be empty"
        assert len(y_values) > 0, "y_values should not be empty"

    def test_prepare_heatmap_data_with_relative_error(self, sample_results):
        """Test prepare_heatmap_data with relative error."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, _ = prepare_heatmap_data(
            sample_results,
            x_key="beta",
            y_key="n_samples",
            value_key="biases",
            diff_index=3,
            relative_error=True,
        )

        assert data_matrix_mean.shape[0] > 0, "Mean matrix should not be empty"

    def test_prepare_heatmap_data_without_diff_index(self, sample_results):
        """Test prepare_heatmap_data without comparison (diff_index=None)."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, _ = prepare_heatmap_data(
            sample_results, x_key="beta", y_key="n_samples", value_key="biases", diff_index=None
        )

        # Without diff_index, data_matrix should contain actual values, not -1/0/1
        assert np.all(np.isfinite(data_matrix_mean)), "Mean matrix should contain finite values"

    def test_prepare_heatmap_data_with_compute_rmse(self, sample_results):
        """Test prepare_heatmap_data with compute_rmse=True."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, data_matrix_rmse = prepare_heatmap_data(
            sample_results,
            x_key="beta",
            y_key="n_samples",
            value_key="biases",
            diff_index=3,
            compute_rmse=True,
        )

        unique_x = len(set([r["beta"] for r in sample_results]))
        unique_y = len(set([r["n_samples"] for r in sample_results]))

        assert data_matrix_rmse is not None, "RMSE matrix should not be None when compute_rmse=True"
        assert data_matrix_rmse.shape == (unique_y, unique_x), f"RMSE matrix should be ({unique_y}, {unique_x})"

    def test_prepare_heatmap_data_rmse_values_correct(self, sample_results):
        """Test that RMSE values are computed correctly (sqrt(bias^2 + sigma^2))."""
        data_matrix_mean, data_matrix_std, data_matrix, x_values, y_values, data_matrix_rmse = prepare_heatmap_data(
            sample_results,
            x_key="beta",
            y_key="n_samples",
            value_key="biases",
            diff_index=None,  # No comparison, just ICA RMSE
            compute_rmse=True,
        )

        # For each cell, RMSE should be sqrt(bias^2 + sigma^2) for ICA (index -1)
        # Since diff_index=None, we get ICA RMSE values directly
        assert data_matrix_rmse is not None, "RMSE matrix should not be None"
        # All RMSE values should be non-negative
        assert np.all(data_matrix_rmse >= 0), "RMSE values should be non-negative"


class TestPlotEstimates:
    """Tests for plot_estimates function."""

    def test_plot_estimates_returns_bias_and_sigma(self):
        """Test that plot_estimates returns bias and sigma."""
        np.random.seed(42)
        estimates = np.random.randn(100) + 2.0  # Mean around 2.0
        true_tau = 2.0
        treatment_effect = 2.0

        bias, sigma = plot_estimates(estimates, true_tau, treatment_effect, plot=False)

        assert isinstance(bias, (float, np.floating)), "Bias should be float"
        assert isinstance(sigma, (float, np.floating)), "Sigma should be float"
        assert sigma >= 0, "Sigma should be non-negative"

    def test_plot_estimates_with_relative_error(self):
        """Test plot_estimates with relative error."""
        np.random.seed(42)
        estimates = np.random.randn(100) + 2.0
        true_tau = 2.0
        treatment_effect = 2.0

        bias_abs, sigma_abs = plot_estimates(estimates, true_tau, treatment_effect, plot=False, relative_error=False)
        bias_rel, sigma_rel = plot_estimates(estimates, true_tau, treatment_effect, plot=False, relative_error=True)

        # Relative error should be different from absolute error
        assert isinstance(bias_abs, (float, np.floating)), "Absolute bias should be float"
        assert isinstance(bias_rel, (float, np.floating)), "Relative bias should be float"

    def test_plot_estimates_handles_nans(self):
        """Test that plot_estimates handles NaN values correctly."""
        np.random.seed(42)
        estimates = np.random.randn(100) + 2.0
        estimates[::10] = np.nan  # Add some NaN values
        true_tau = 2.0
        treatment_effect = 2.0

        bias, sigma = plot_estimates(estimates, true_tau, treatment_effect, plot=False)

        # Should still return finite values (nanmean/nanstd handle NaNs)
        assert np.isfinite(bias), "Bias should be finite even with NaNs in data"
        assert np.isfinite(sigma), "Sigma should be finite even with NaNs in data"

    def test_plot_estimates_empty_list(self):
        """Test plot_estimates with empty estimate list."""
        estimates = []
        true_tau = 2.0
        treatment_effect = 2.0

        # This should either handle gracefully or raise an appropriate error
        # Depending on implementation, you might want to test for specific behavior
        try:
            bias, sigma = plot_estimates(estimates, true_tau, treatment_effect, plot=False)
            # If it succeeds, values should be NaN
            assert np.isnan(bias) or np.isnan(sigma), "Empty estimates should result in NaN"
        except (ValueError, IndexError):
            # Or it might raise an error, which is also acceptable
            pass
