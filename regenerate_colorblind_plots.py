#!/usr/bin/env python3
"""Regenerate all plots affected by the colorblind-friendly color scheme update.

This script regenerates all difference plots that use the red/blue color scheme
with different markers (circles for OML better, squares for ICA better).

Run this after updating the color scheme in ablation_utils.py to regenerate
all affected figures.
"""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np


def regenerate_n20_all_plots():
    """Regenerate all n20 plots (heatmaps + scatter) via eta_ablation_plotting."""
    print("=" * 60)
    print("Regenerating n20 plots (heatmaps + scatter)...")
    print("=" * 60)

    from eta_ablation_plotting import regenerate_n20_plots  # pylint: disable=import-outside-toplevel

    regenerate_n20_plots(scatter=True)


def regenerate_noise_ablation_plots():
    """Regenerate noise ablation plots from eta_ablation_plotting.py."""
    print("\n" + "=" * 60)
    print("Regenerating noise ablation plots...")
    print("=" * 60)

    # Import only the functions we actually use
    # pylint: disable=import-outside-toplevel
    from eta_ablation_plotting import (
        plot_diff_heatmaps,
        plot_noise_ablation_coeff_scatter,
        plot_noise_ablation_std_scatter,
    )

    noise_ablation_dir = "figures/noise_ablation"

    # Find all results files
    results_files = glob.glob(os.path.join(noise_ablation_dir, "noise_ablation_results*.npy"))

    for results_file in results_files:
        filename = os.path.basename(results_file)
        print(f"\nProcessing {filename}...")

        try:
            results = np.load(results_file, allow_pickle=True).item()
        except (OSError, ValueError) as e:
            print(f"  Error loading {filename}: {e}")
            continue

        # Skip if not a dictionary with distribution keys
        if not isinstance(results, dict):
            print(f"  Skipping {filename}: not a valid results dictionary")
            continue

        # Check if this has the expected structure
        has_distributions = any(
            isinstance(v, dict) and ("rmse" in v or "biases" in v or "sigmas" in v) for v in results.values()
        )

        if not has_distributions:
            print(f"  Skipping {filename}: no distribution results found")
            continue

        # Filter out metadata keys
        dist_results = {
            k: v
            for k, v in results.items()
            if k not in ["treatment_coef_range", "outcome_coef_range", "treatment_effect_range"] and isinstance(v, dict)
        }

        if len(dist_results) == 0:
            print(f"  Skipping {filename}: no distribution results")
            continue

        # Create output directory based on filename
        output_dir = noise_ablation_dir

        # Extract parameters from filename for scatter plots
        tc_match = re.search(r"tc(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        oc_match = re.search(r"oc(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        te_match = re.search(r"te(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        coef_match = re.search(r"coef(-?\d+\.?\d*)to(-?\d+\.?\d*)", filename)
        n_match = re.search(r"_n(\d+)_", filename)

        if tc_match and oc_match:
            treatment_coef_range = (float(tc_match.group(1)), float(tc_match.group(2)))
            outcome_coef_range = (float(oc_match.group(1)), float(oc_match.group(2)))
        elif coef_match:
            treatment_coef_range = (float(coef_match.group(1)), float(coef_match.group(2)))
            outcome_coef_range = treatment_coef_range
        else:
            treatment_coef_range = (-2.0, 2.0)
            outcome_coef_range = (-2.0, 2.0)

        if te_match:
            treatment_effect_range = (float(te_match.group(1)), float(te_match.group(2)))
        else:
            treatment_effect_range = (0.1, 5.0)

        n_configs = int(n_match.group(1)) if n_match else 0

        print(f"  Distributions: {list(dist_results.keys())}")

        # Generate coefficient scatter plots (RMSE)
        try:
            if any("config_results" in v for v in dist_results.values() if isinstance(v, dict)):
                print("  Generating RMSE scatter plots...")
                plot_noise_ablation_coeff_scatter(
                    dist_results,
                    output_dir=output_dir,
                    n_configs=n_configs,
                    treatment_coef_range=treatment_coef_range,
                    outcome_coef_range=outcome_coef_range,
                    treatment_effect_range=treatment_effect_range,
                )
        except (KeyError, ValueError) as e:
            print(f"    Error generating RMSE scatter plots: {e}")

        # Generate coefficient scatter plots (Std)
        try:
            if any("config_results" in v for v in dist_results.values() if isinstance(v, dict)):
                print("  Generating Std scatter plots...")
                plot_noise_ablation_std_scatter(
                    dist_results,
                    output_dir=output_dir,
                    n_configs=n_configs,
                    treatment_coef_range=treatment_coef_range,
                    outcome_coef_range=outcome_coef_range,
                    treatment_effect_range=treatment_effect_range,
                )
        except (KeyError, ValueError) as e:
            print(f"    Error generating Std scatter plots: {e}")

        # Generate heatmaps
        try:
            if any("config_results" in v for v in dist_results.values() if isinstance(v, dict)):
                print("  Generating heatmaps...")
                plot_diff_heatmaps(
                    dist_results,
                    output_dir=output_dir,
                    n_configs=n_configs,
                    treatment_coef_range=treatment_coef_range,
                    outcome_coef_range=outcome_coef_range,
                    treatment_effect_range=treatment_effect_range,
                )
        except (KeyError, ValueError) as e:
            print(f"    Error generating heatmaps: {e}")


def regenerate_coefficient_ablation_plots():
    """Regenerate coefficient ablation plots."""
    print("\n" + "=" * 60)
    print("Regenerating coefficient ablation plots...")
    print("=" * 60)

    # pylint: disable=import-outside-toplevel
    from eta_ablation_plotting import plot_coefficient_ablation_results

    coeff_ablation_dir = "figures/coefficient_ablation"

    if not os.path.exists(coeff_ablation_dir):
        print(f"  Directory {coeff_ablation_dir} does not exist, skipping")
        return

    results_files = glob.glob(os.path.join(coeff_ablation_dir, "*.npy"))

    for results_file in results_files:
        filename = os.path.basename(results_file)
        print(f"\nProcessing {filename}...")

        try:
            results = np.load(results_file, allow_pickle=True)
            # Convert to list if it's an array of dicts
            if hasattr(results, "tolist"):
                results = results.tolist()
            print("  Generating coefficient ablation plots...")
            plot_coefficient_ablation_results(results, coeff_ablation_dir)
        except (OSError, ValueError, TypeError) as e:
            print(f"  Error processing {filename}: {e}")


def main():
    """Main entry point."""
    print("Regenerating all plots with colorblind-friendly color scheme")
    print("Color scheme: Blue (ICA better) / Red (OML better)")
    print("Markers: Squares (ICA better) / Circles (OML better)")
    print()

    # Regenerate n20 plots (heatmaps + scatter, consolidated)
    try:
        regenerate_n20_all_plots()
    except ImportError as e:
        print(f"Error regenerating n20 plots: {e}")

    # Regenerate noise ablation plots
    try:
        regenerate_noise_ablation_plots()
    except ImportError as e:
        print(f"Error regenerating noise ablation plots: {e}")

    # Regenerate coefficient ablation plots
    try:
        regenerate_coefficient_ablation_plots()
    except ImportError as e:
        print(f"Error regenerating coefficient ablation plots: {e}")

    print("\n" + "=" * 60)
    print("Done! All affected plots have been regenerated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
