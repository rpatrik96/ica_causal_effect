"""
Noise distribution and coefficient ablation study for treatment effect estimation.

This script runs experiments comparing:
1. Different noise distributions for eta (treatment noise)
2. Different coefficient configurations (treatment_coef, outcome_coef, treatment_effect)

Noise distributions:
- Discrete (default asymmetric)
- Heavy-tailed: Laplace, gennorm (beta<2)
- Bounded: Uniform, Rademacher
- Light-tailed: gennorm (beta>2)

Supports flexible gennorm distributions with configurable beta parameter:
- gennorm:0.5  - Very heavy tails
- gennorm:1.0  - Equivalent to Laplace (gennorm_heavy)
- gennorm:2.0  - Equivalent to Gaussian
- gennorm:4.0  - Lighter tails (gennorm_light)
- gennorm:8.0  - Approaching uniform

Coefficient ablation varies the ICA variance coefficient:
  ica_var_coeff = 1 + ||outcome_coef + treatment_coef * treatment_effect||^2

This allows studying how the relationship between coefficients affects estimation error.
"""

import argparse
import os
import sys

import numpy as np

from eta_ablation_experiments import (
    compute_constrained_treatment_coef,
    compute_ica_var_coeff,
    print_coefficient_ablation_summary,
    print_noise_ablation_summary,
    print_variance_ablation_summary,
    run_coefficient_ablation_experiments,
    run_noise_ablation_experiments,
    run_sample_dimension_grid_experiments,
    run_variance_ablation_experiments,
)
from eta_ablation_plotting import (
    plot_coefficient_ablation_results,
    plot_diff_heatmaps,
    plot_ica_var_filtered_bias_heatmaps,
    plot_ica_var_filtered_rmse_heatmap,
    plot_noise_ablation_coeff_scatter,
    plot_noise_ablation_std_scatter,
    plot_variance_ablation_heatmaps,
)


def main(args=None):
    """Main function for noise distribution and coefficient ablation study."""
    parser = argparse.ArgumentParser(
        description="Noise distribution and coefficient ablation study for treatment effect estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default noise distribution ablation
  python eta_noise_ablation.py

  # Run noise ablation with specific gennorm beta values
  python eta_noise_ablation.py --distributions gennorm:0.5 gennorm:1.0 gennorm:2.0

  # Run coefficient ablation
  python eta_noise_ablation.py --coefficient_ablation

  # Run variance ablation (beta vs variance heatmaps)
  python eta_noise_ablation.py --variance_ablation

  # Run variance ablation with custom grid
  python eta_noise_ablation.py --variance_ablation \
    --variance_beta_values 0.5 1.0 1.5 2.5 3.0 --variance_values 0.5 1.0 2.0 4.0
        """,
    )

    # Common arguments
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples per experiment")
    parser.add_argument("--n_experiments", type=int, default=20, help="Number of Monte Carlo replications")
    parser.add_argument("--support_size", type=int, default=10, help="Support size for coefficients")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for gennorm covariates")
    parser.add_argument("--sigma_outcome", type=float, default=np.sqrt(3.0), help="Outcome noise std")
    parser.add_argument("--covariate_pdf", type=str, default="gennorm", help="Covariate distribution")
    parser.add_argument("--check_convergence", action="store_true", help="Check ICA convergence")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=12143, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="figures/noise_ablation", help="Output directory")

    # Noise ablation arguments
    parser.add_argument("--treatment_effect", type=float, default=1.0, help="True treatment effect")
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["discrete", "laplace", "uniform", "rademacher", "gennorm_heavy", "gennorm_light"],
        help="Noise distributions to test",
    )
    parser.add_argument("--gennorm_betas", nargs="+", type=float, default=None, help="Gennorm beta values to add")
    parser.add_argument("--randomize_coeffs", action="store_true", help="Randomize coefficients")
    parser.add_argument("--n_random_configs", type=int, default=20, help="Number of random configs")
    parser.add_argument(
        "--treatment_effect_range", nargs=2, type=float, default=[0.001, 0.2], help="Treatment effect range"
    )
    parser.add_argument(
        "--treatment_coef_range", nargs=2, type=float, default=[-10.0, 10.0], help="Treatment coef range"
    )
    parser.add_argument("--outcome_coef_range", nargs=2, type=float, default=[-0.5, 0.5], help="Outcome coef range")

    # Coefficient ablation arguments
    parser.add_argument("--coefficient_ablation", action="store_true", help="Run coefficient ablation")
    parser.add_argument(
        "--noise_distribution", type=str, default="discrete", help="Noise distribution for coef ablation"
    )

    # Variance ablation arguments
    parser.add_argument("--variance_ablation", action="store_true", help="Run variance ablation")
    parser.add_argument(
        "--variance_beta_values",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.5, 3.0, 4.0],
        help="Gennorm beta values for variance ablation",
    )
    parser.add_argument(
        "--variance_values",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Variance values (scale^2) for variance ablation",
    )

    # Filtered heatmap arguments
    parser.add_argument("--filtered_heatmap", action="store_true", help="Run filtered RMSE heatmap experiments")
    parser.add_argument(
        "--heatmap_axis_mode",
        type=str,
        default="d_vs_n",
        choices=["d_vs_n", "beta_vs_n"],
        help="Axis mode: 'd_vs_n' (dimension vs sample size) or 'beta_vs_n' (beta vs sample size)",
    )
    parser.add_argument(
        "--heatmap_sample_sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 5000, 10000],
        help="Sample sizes for heatmap",
    )
    parser.add_argument(
        "--heatmap_dimensions",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50],
        help="Covariate dimensions for d_vs_n mode",
    )
    parser.add_argument(
        "--heatmap_betas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0, 4.0],
        help="Beta values for beta_vs_n mode",
    )
    parser.add_argument(
        "--ica_var_threshold",
        type=float,
        default=1.5,
        help="ICA variance coefficient threshold for filtering",
    )
    parser.add_argument(
        "--fixed_beta",
        type=float,
        default=1.0,
        help="Fixed beta for d_vs_n mode",
    )
    parser.add_argument(
        "--fixed_dimension",
        type=int,
        default=10,
        help="Fixed covariate dimension for beta_vs_n mode",
    )
    parser.add_argument(
        "--heatmap_treatment_coef",
        type=float,
        default=0.5,
        help="Treatment coefficient scalar for filtered heatmap experiments (default 0.5 gives ica_var_coeff=1.25)",
    )
    parser.add_argument(
        "--heatmap_outcome_coef",
        type=float,
        default=0.0,
        help="Outcome coefficient scalar for filtered heatmap experiments",
    )
    parser.add_argument(
        "--constrain_ica_var",
        action="store_true",
        help="Automatically set treatment coefficient to achieve ica_var_coeff = ica_var_threshold. "
        "When enabled, --heatmap_treatment_coef is ignored and computed from --ica_var_threshold.",
    )

    # Oracle support arguments
    parser.add_argument(
        "--oracle_support",
        dest="oracle_support",
        action="store_true",
        default=True,
        help="If True, both OML and ICA receive x[:, support] (oracle knowledge). Default: True.",
    )
    parser.add_argument(
        "--no_oracle_support",
        dest="oracle_support",
        action="store_false",
        help="Disable oracle support (both methods receive full x).",
    )

    if args is None:
        args = sys.argv[1:]
    opts = parser.parse_args(args)

    # Compute oracle suffix for output directories
    oracle_suffix = "" if opts.oracle_support else "_no_oracle"

    if opts.variance_ablation:
        # Run variance ablation
        var_output_dir = os.path.join(opts.output_dir, f"variance_ablation{oracle_suffix}")
        results_file = os.path.join(var_output_dir, f"variance_ablation_results{oracle_suffix}.npy")

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            var_results = np.load(results_file, allow_pickle=True).item()
        else:
            var_results = run_variance_ablation_experiments(
                beta_values=opts.variance_beta_values,
                variance_values=opts.variance_values,
                n_samples=opts.n_samples,
                n_experiments=opts.n_experiments,
                support_size=opts.support_size,
                treatment_effect=opts.treatment_effect,
                covariate_beta=opts.beta,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
                oracle_support=opts.oracle_support,
            )

            os.makedirs(var_output_dir, exist_ok=True)
            np.save(results_file, var_results)
            print(f"Results saved to {results_file}")

        plot_variance_ablation_heatmaps(var_results, var_output_dir)

        # Print summary
        print_variance_ablation_summary(var_results, opts)

    elif opts.coefficient_ablation:
        # Run coefficient ablation
        coef_output_dir = os.path.join(opts.output_dir, f"coefficient_ablation{oracle_suffix}")
        results_file = os.path.join(coef_output_dir, f"coefficient_ablation_results{oracle_suffix}.npy")

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            coef_results = np.load(results_file, allow_pickle=True).tolist()
        else:
            coef_results = run_coefficient_ablation_experiments(
                noise_distribution=opts.noise_distribution,
                n_samples=opts.n_samples,
                n_experiments=opts.n_experiments,
                support_size=opts.support_size,
                beta=opts.beta,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
                oracle_support=opts.oracle_support,
            )

            os.makedirs(coef_output_dir, exist_ok=True)
            np.save(results_file, coef_results)
            print(f"Results saved to {results_file}")

        plot_coefficient_ablation_results(coef_results, coef_output_dir)

        # Print summary
        print_coefficient_ablation_summary(coef_results, opts)

    elif opts.filtered_heatmap:
        # Run filtered heatmap experiments
        heatmap_output_dir = os.path.join(opts.output_dir, f"filtered_heatmap{oracle_suffix}")

        # Compute treatment coefficient if constrain_ica_var is enabled
        if opts.constrain_ica_var:
            # Ensure outcome_coef is non-zero (default to 0.2 if zero)
            outcome_coef_scalar = opts.heatmap_outcome_coef
            if outcome_coef_scalar == 0.0:
                # Set a default non-zero outcome coefficient
                # Use 30% of the target coefficient sum to ensure treatment_coef is also non-zero
                target_coef_sum = np.sqrt(opts.ica_var_threshold - 1)
                outcome_coef_scalar = 0.3 * target_coef_sum
                print(f"Setting outcome_coef_scalar to {outcome_coef_scalar:.6f} (30% of target sum)")

            treatment_coef_scalar = compute_constrained_treatment_coef(
                target_ica_var_coeff=opts.ica_var_threshold,
                treatment_effect=opts.treatment_effect,
                outcome_coef_scalar=outcome_coef_scalar,
            )

            # Validate both coefficients are non-zero
            if treatment_coef_scalar == 0.0:
                raise ValueError(
                    f"Computed treatment_coef_scalar is zero. "
                    f"Adjust outcome_coef_scalar ({outcome_coef_scalar})"
                    f" or ica_var_threshold ({opts.ica_var_threshold})."
                )
            if outcome_coef_scalar == 0.0:
                raise ValueError("outcome_coef_scalar must be non-zero when --constrain_ica_var is enabled.")

            computed_ica_var = compute_ica_var_coeff(treatment_coef_scalar, outcome_coef_scalar, opts.treatment_effect)
            print(f"Constraining ICA variance coefficient to {opts.ica_var_threshold}")
            print(f"  Computed treatment_coef_scalar: {treatment_coef_scalar:.6f}")
            print(f"  Outcome_coef_scalar: {outcome_coef_scalar:.6f}")
            print(f"  Resulting ica_var_coeff: {computed_ica_var:.6f}")
        else:
            treatment_coef_scalar = opts.heatmap_treatment_coef
            outcome_coef_scalar = opts.heatmap_outcome_coef

        results_file = os.path.join(
            heatmap_output_dir,
            f"filtered_heatmap_results_{opts.heatmap_axis_mode}{oracle_suffix}.npy",
        )

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            heatmap_results = np.load(results_file, allow_pickle=True).tolist()
        else:
            heatmap_results = run_sample_dimension_grid_experiments(
                sample_sizes=opts.heatmap_sample_sizes,
                dimension_values=opts.heatmap_dimensions if opts.heatmap_axis_mode == "d_vs_n" else None,
                beta_values=opts.heatmap_betas if opts.heatmap_axis_mode == "beta_vs_n" else None,
                axis_mode=opts.heatmap_axis_mode,
                fixed_beta=opts.fixed_beta,
                fixed_dimension=opts.fixed_dimension,
                noise_distribution=opts.noise_distribution,
                n_experiments=opts.n_experiments,
                treatment_effect=opts.treatment_effect,
                treatment_coef_scalar=treatment_coef_scalar,
                outcome_coef_scalar=outcome_coef_scalar,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
                oracle_support=opts.oracle_support,
            )

            os.makedirs(heatmap_output_dir, exist_ok=True)
            np.save(results_file, heatmap_results)
            print(f"Results saved to {results_file}")

        # Validate that all results are below the threshold when constrain_ica_var is enabled
        if opts.constrain_ica_var:
            ica_var_coeffs = [r["ica_var_coeff"] for r in heatmap_results]
            max_ica_var = max(ica_var_coeffs)
            min_ica_var = min(ica_var_coeffs)
            results_above_threshold = [c for c in ica_var_coeffs if c > opts.ica_var_threshold]

            if results_above_threshold:
                print(
                    f"\nWARNING: {len(results_above_threshold)}/{len(ica_var_coeffs)} results have "
                    f"ica_var_coeff > {opts.ica_var_threshold}"
                )
                print(f"  Max ica_var_coeff: {max_ica_var:.6f}")
                print(f"  Min ica_var_coeff: {min_ica_var:.6f}")
                raise ValueError(
                    f"ICA variance constraint violated: {len(results_above_threshold)} results "
                    f"exceed threshold {opts.ica_var_threshold}. Max: {max_ica_var:.6f}"
                )
            print(
                f"\nValidation passed: All {len(ica_var_coeffs)} results have "
                f"ica_var_coeff <= {opts.ica_var_threshold}"
            )
            print(f"  Range: [{min_ica_var:.6f}, {max_ica_var:.6f}]")

        # Plot filtered RMSE heatmap
        plot_ica_var_filtered_rmse_heatmap(
            heatmap_results,
            heatmap_output_dir,
            axis_mode=opts.heatmap_axis_mode,
            ica_var_threshold=opts.ica_var_threshold,
            filter_below=True,
        )

        # Plot filtered bias heatmaps
        plot_ica_var_filtered_bias_heatmaps(
            heatmap_results,
            heatmap_output_dir,
            axis_mode=opts.heatmap_axis_mode,
            ica_var_threshold=opts.ica_var_threshold,
            filter_below=True,
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print("SUMMARY: Filtered Heatmap Experiments")
        print(f"{'=' * 80}")
        print(f"Axis mode: {opts.heatmap_axis_mode}")
        print(f"Sample sizes: {opts.heatmap_sample_sizes}")
        if opts.heatmap_axis_mode == "d_vs_n":
            print(f"Dimensions: {opts.heatmap_dimensions}")
            print(f"Fixed beta: {opts.fixed_beta}")
        else:
            print(f"Beta values: {opts.heatmap_betas}")
            print(f"Fixed dimension: {opts.fixed_dimension}")
        print(f"Covariate distribution: {opts.covariate_pdf}")
        print(f"Noise distribution: {opts.noise_distribution}")
        print(f"ICA var threshold: {opts.ica_var_threshold}")
        print(f"Constrain ICA var: {opts.constrain_ica_var}")
        print(f"Treatment coef: {treatment_coef_scalar:.6f}" + (" (computed)" if opts.constrain_ica_var else ""))
        outcome_coef_computed = opts.constrain_ica_var and opts.heatmap_outcome_coef == 0.0
        print(f"Outcome coef: {outcome_coef_scalar:.6f}" + (" (computed)" if outcome_coef_computed else ""))
        print(f"Treatment effect: {opts.treatment_effect}")
        actual_ica_var = compute_ica_var_coeff(treatment_coef_scalar, outcome_coef_scalar, opts.treatment_effect)
        print(f"Resulting ICA var coeff: {actual_ica_var:.6f}")
        print(f"Total configurations: {len(heatmap_results)}")

        # Count filtered results
        filtered_count = sum(1 for r in heatmap_results if r["ica_var_coeff"] <= opts.ica_var_threshold)
        print(f"Results with ica_var_coeff <= {opts.ica_var_threshold}: {filtered_count}/{len(heatmap_results)}")
        print(f"Output directory: {heatmap_output_dir}")

    else:
        # Run noise distribution ablation
        distributions = list(opts.distributions)
        if opts.gennorm_betas is not None:
            for beta_val in opts.gennorm_betas:
                gennorm_spec = f"gennorm:{beta_val}"
                if gennorm_spec not in distributions:
                    distributions.append(gennorm_spec)
            print(f"Final distribution list: {distributions}")

        # Setup results file path
        # Use subdirectory with oracle suffix for noise ablation too
        noise_output_dir = opts.output_dir + oracle_suffix if oracle_suffix else opts.output_dir
        if opts.randomize_coeffs:
            tc_range = tuple(opts.treatment_coef_range)
            oc_range = tuple(opts.outcome_coef_range)
            te_range = tuple(opts.treatment_effect_range)
            results_file = os.path.join(
                noise_output_dir,
                f"noise_ablation_results_n{opts.n_random_configs}"
                f"_tc{tc_range[0]:.1f}to{tc_range[1]:.1f}"
                f"_oc{oc_range[0]:.1f}to{oc_range[1]:.1f}"
                f"_te{te_range[0]:.1f}to{te_range[1]:.1f}{oracle_suffix}.npy",
            )
        else:
            results_file = os.path.join(noise_output_dir, f"noise_ablation_results{oracle_suffix}.npy")

        if os.path.exists(results_file):
            print(f"Loading existing results from {results_file}")
            results = np.load(results_file, allow_pickle=True).item()
        else:
            results = run_noise_ablation_experiments(
                noise_distributions=distributions,
                n_samples=opts.n_samples,
                n_experiments=opts.n_experiments,
                support_size=opts.support_size,
                treatment_effect=opts.treatment_effect,
                beta=opts.beta,
                sigma_outcome=opts.sigma_outcome,
                covariate_pdf=opts.covariate_pdf,
                check_convergence=opts.check_convergence,
                verbose=opts.verbose,
                seed=opts.seed,
                randomize_coeffs=opts.randomize_coeffs,
                n_random_configs=opts.n_random_configs,
                treatment_effect_range=tuple(opts.treatment_effect_range),
                treatment_coef_range=tuple(opts.treatment_coef_range),
                outcome_coef_range=tuple(opts.outcome_coef_range),
                oracle_support=opts.oracle_support,
            )

            os.makedirs(noise_output_dir, exist_ok=True)
            np.save(results_file, results)
            print(f"Results saved to {results_file}")

        if opts.randomize_coeffs:
            plot_noise_ablation_coeff_scatter(
                results,
                noise_output_dir,
                n_configs=opts.n_random_configs,
                treatment_coef_range=tc_range,
                outcome_coef_range=oc_range,
                treatment_effect_range=te_range,
            )
            plot_noise_ablation_std_scatter(
                results,
                noise_output_dir,
                n_configs=opts.n_random_configs,
                treatment_coef_range=tc_range,
                outcome_coef_range=oc_range,
                treatment_effect_range=te_range,
            )
            plot_diff_heatmaps(
                results,
                noise_output_dir,
                n_configs=opts.n_random_configs,
                treatment_coef_range=tc_range,
                outcome_coef_range=oc_range,
                treatment_effect_range=te_range,
            )

        # Print summary
        print_noise_ablation_summary(results, opts)


if __name__ == "__main__":
    main()
