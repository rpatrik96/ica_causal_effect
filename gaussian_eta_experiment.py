"""Experiment: Gaussian eta (treatment noise) with non-Gaussian eps (outcome noise).

Tests Option B from Proposition cor:ica_gauss: theta is identifiable via ICA when
eps is non-Gaussian, even if eta is Gaussian. This is because theta resides in the
eps row of the unmixing matrix W, and eps is identifiable as the sole non-Gaussian
source after quotienting the Gaussian covariate block.

Compares two theta extraction methods:
  1. Munkres-based (standard): expected to FAIL with Gaussian eta
  2. Eps-row-based (new): expected to SUCCEED with Gaussian eta
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tueplots import bundles

from ica import generate_ica_data, ica_treatment_effect_estimation, ica_treatment_effect_estimation_eps_row
from plot_utils import plot_typography


def run_single_config(
    n_covariates,
    n_treatments,
    batch_size,
    beta_eps,
    beta_eta,
    sparse_prob,
    n_seeds,
    verbose=False,
):
    """Run both extraction methods for a single configuration across seeds."""
    munkres_errors = []
    eps_row_errors = []
    munkres_thetas = []
    eps_row_thetas = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        S, X, theta_true = generate_ica_data(
            n_covariates=n_covariates,
            n_treatments=n_treatments,
            batch_size=batch_size,
            beta=beta_eps,
            split_eta_eps=True,
            beta_eta=beta_eta,
            sparse_prob=sparse_prob,
        )

        # Method 1: Munkres-based (standard)
        theta_munkres, _ = ica_treatment_effect_estimation(
            X, S, random_state=seed, n_treatments=n_treatments, verbose=verbose
        )

        # Method 2: Eps-row-based (new)
        theta_eps_row, _ = ica_treatment_effect_estimation_eps_row(
            X, S, random_state=seed, n_treatments=n_treatments, verbose=verbose
        )

        theta_np = theta_true.numpy()
        munkres_err = np.mean((np.array(theta_munkres) - theta_np) ** 2)
        eps_row_err = np.mean((np.array(theta_eps_row) - theta_np) ** 2)

        munkres_errors.append(munkres_err)
        eps_row_errors.append(eps_row_err)
        munkres_thetas.append(np.array(theta_munkres).tolist())
        eps_row_thetas.append(np.array(theta_eps_row).tolist())

        if verbose:
            print(
                f"  seed={seed}: theta_true={theta_np}, "
                f"munkres={np.array(theta_munkres):.4f} (MSE={munkres_err:.4f}), "
                f"eps_row={np.array(theta_eps_row):.4f} (MSE={eps_row_err:.4f})"
            )

    return {
        "munkres_mse_mean": float(np.nanmean(munkres_errors)),
        "munkres_mse_std": float(np.nanstd(munkres_errors)),
        "eps_row_mse_mean": float(np.nanmean(eps_row_errors)),
        "eps_row_mse_std": float(np.nanstd(eps_row_errors)),
        "munkres_nan_frac": float(np.mean(np.isnan(munkres_errors))),
        "eps_row_nan_frac": float(np.mean(np.isnan(eps_row_errors))),
        "munkres_thetas": munkres_thetas,
        "eps_row_thetas": eps_row_thetas,
    }


def run_experiment(args):
    """Run the full Gaussian eta experiment across a parameter grid."""
    sample_sizes = [500, 1000, 2000, 5000]
    beta_eps_values = [0.5, 1.0, 4.0]  # Laplace=1, sub-Gaussian=0.5, super-Gaussian=4
    beta_eta_values = [2.0]  # Gaussian eta (the key test)
    n_covariate_values = [5, 10]

    # Also run a control: non-Gaussian eta for comparison
    beta_eta_control = [1.0]

    all_results = {}

    for n_cov in n_covariate_values:
        for n_samples in sample_sizes:
            for beta_eps in beta_eps_values:
                for beta_eta in beta_eta_values + beta_eta_control:
                    config_key = f"n={n_samples}_d={n_cov}_beta_eps={beta_eps}_beta_eta={beta_eta}"
                    print(f"\n{'='*60}")
                    print(f"Config: {config_key}")
                    print(f"{'='*60}")

                    result = run_single_config(
                        n_covariates=n_cov,
                        n_treatments=1,
                        batch_size=n_samples,
                        beta_eps=beta_eps,
                        beta_eta=beta_eta,
                        sparse_prob=0.3,
                        n_seeds=args.n_experiments,
                        verbose=args.verbose,
                    )

                    result["config"] = {
                        "n_samples": n_samples,
                        "n_covariates": n_cov,
                        "beta_eps": beta_eps,
                        "beta_eta": beta_eta,
                        "n_treatments": 1,
                        "sparse_prob": 0.3,
                    }

                    all_results[config_key] = result

                    print(
                        f"  Munkres MSE: {result['munkres_mse_mean']:.4f} +/- {result['munkres_mse_std']:.4f} "
                        f"(NaN frac: {result['munkres_nan_frac']:.2f})"
                    )
                    print(
                        f"  Eps-row MSE: {result['eps_row_mse_mean']:.4f} +/- {result['eps_row_mse_std']:.4f} "
                        f"(NaN frac: {result['eps_row_nan_frac']:.2f})"
                    )

    return all_results


def plot_results(all_results, output_dir):
    """Generate comparison plots."""
    plot_typography()

    with plt.rc_context({**bundles.icml2022()}):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))

        for ax_idx, n_cov in enumerate([5, 10]):
            ax = axes[ax_idx]

            # Extract results for Gaussian eta (beta_eta=2.0) and control (beta_eta=1.0)
            for beta_eta, ls, label_suffix in [(2.0, "-", "Gaussian $\\eta$"), (1.0, "--", "Laplace $\\eta$")]:
                for beta_eps, color in [(0.5, "C0"), (1.0, "C1"), (4.0, "C2")]:
                    sample_sizes = []
                    munkres_means = []
                    eps_row_means = []

                    for n_samples in [500, 1000, 2000, 5000]:
                        key = f"n={n_samples}_d={n_cov}_beta_eps={beta_eps}_beta_eta={beta_eta}"
                        if key in all_results:
                            r = all_results[key]
                            sample_sizes.append(n_samples)
                            munkres_means.append(r["munkres_mse_mean"])
                            eps_row_means.append(r["eps_row_mse_mean"])

                    if sample_sizes:
                        ax.plot(
                            sample_sizes,
                            eps_row_means,
                            marker="o",
                            ls=ls,
                            color=color,
                            markersize=4,
                            label=f"$\\beta_\\varepsilon$={beta_eps}, {label_suffix}" if ax_idx == 0 else None,
                        )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Sample size $n$")
            ax.set_ylabel("MSE($\\hat\\theta$)")
            ax.set_title(f"$d = {n_cov}$ covariates")

        axes[0].legend(fontsize=6, ncol=1)
        fig.suptitle("Eps-row extraction: Gaussian vs. non-Gaussian $\\eta$")
        fig.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "gaussian_eta_comparison.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "gaussian_eta_comparison.pdf"), bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Gaussian eta experiment for Option B (cor:ica_gauss)")
    parser.add_argument("--n_experiments", type=int, default=20, help="Number of random seeds")
    parser.add_argument("--output_dir", type=str, default="figures/gaussian_eta", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed details")
    args = parser.parse_args()

    results = run_experiment(args)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "gaussian_eta_results.json")

    # Convert results to JSON-serializable format
    serializable = {}
    for key, val in results.items():
        serializable[key] = dict(val.items())

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
