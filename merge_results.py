"""Merge partial OML experiment results from split HTCondor jobs.

After running split jobs (one per n_samples × beta combination), this script:
1. Loads all partial .npy result files matching a glob pattern
2. Concatenates them into a single results list
3. Saves the merged file
4. Optionally generates combined plots via generate_all_oml_plots

Usage:
    # Merge all partial results in the output directory
    python merge_results.py /path/to/figures/n_exp_20_sigma_outcome_1.732_pdf_gennorm/scalar_coeffs/

    # Merge with custom pattern and generate plots
    python merge_results.py /path/to/output/ --pattern "all_results_*_n*_beta*.npy" --plot

    # Specify experiment config for plotting
    python merge_results.py /path/to/output/ --plot --n_experiments 20 --covariate_pdf gennorm
"""

import argparse
import glob
import os
import sys

import numpy as np


def find_partial_results(results_dir, pattern="all_results_*_n[0-9]*_beta*.npy"):
    """Find all partial result files matching the pattern.

    Args:
        results_dir: Directory containing partial result files
        pattern: Glob pattern for matching files

    Returns:
        Sorted list of matching file paths
    """
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    return files


def merge_results(files):
    """Load and merge partial result files.

    Args:
        files: List of .npy file paths to merge

    Returns:
        Combined list of result dictionaries
    """
    all_results = []
    for f in files:
        partial = list(np.load(f, allow_pickle=True))
        print(f"  {os.path.basename(f)}: {len(partial)} configurations")
        all_results.extend(partial)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Merge partial OML experiment results")
    parser.add_argument("results_dir", help="Directory containing partial .npy result files")
    parser.add_argument(
        "--pattern",
        default="all_results_*_n[0-9]*_beta*.npy",
        help="Glob pattern for partial result files (default: all_results_*_n[0-9]*_beta*.npy)",
    )
    parser.add_argument(
        "--output",
        default="all_results_merged.npy",
        help="Output filename for merged results (default: all_results_merged.npy)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate combined plots after merging",
    )
    parser.add_argument("--n_experiments", type=int, default=20)
    parser.add_argument("--sigma_outcome", type=float, default=np.sqrt(3.0))
    parser.add_argument("--covariate_pdf", type=str, default="gennorm")
    parser.add_argument("--no_oracle_support", action="store_true")

    opts = parser.parse_args()

    # Find partial results
    files = find_partial_results(opts.results_dir, opts.pattern)
    if not files:
        print(f"No files matching '{opts.pattern}' in {opts.results_dir}")
        return 1

    print(f"Found {len(files)} partial result files:")
    all_results = merge_results(files)
    print(f"\nTotal configurations after merge: {len(all_results)}")

    # Save merged results
    output_path = os.path.join(opts.results_dir, opts.output)
    np.save(output_path, np.array(all_results, dtype=object))
    print(f"Merged results saved to {output_path}")

    # Generate plots if requested
    if opts.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tueplots import bundles

        from oml_plotting import generate_all_oml_plots
        from oml_utils import OMLExperimentConfig, OMLParameterGrid
        from plot_utils import plot_typography

        plt.rcParams.update(bundles.icml2022(usetex=True))
        plot_typography()

        config = OMLExperimentConfig(
            n_experiments=opts.n_experiments,
            sigma_outcome=opts.sigma_outcome,
            covariate_pdf=opts.covariate_pdf,
            output_dir=opts.results_dir,
            oracle_support=not opts.no_oracle_support,
        )

        # Build full parameter grid (not restricted) for plotting
        param_grid = OMLParameterGrid()
        if config.covariate_pdf != "gennorm":
            param_grid.beta_values = [1.0]

        print("\nGenerating combined plots...")
        generate_all_oml_plots(all_results, config, param_grid, param_grid.treatment_effects)
        print("Plots generated successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
