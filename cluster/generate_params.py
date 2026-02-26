#!/usr/bin/env python3
"""Generate parameter files for HTCondor sweep experiments.

This script generates parameter combinations for cluster experiments,
replicating parameter grids for cluster submission.
"""

import argparse
import itertools
from pathlib import Path


def generate_single_instance_params(output_file: str = "single_instance_params.txt"):
    """Generate single-instance parameter grid for cluster sweep."""
    n_samples_list = [2000, 5000, 10000]
    n_experiments_list = [2000]
    n_dim_list = [1000, 2000, 5000]
    support_size_list = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    sigma_q_list = [3, 10, 20]

    params = []
    for n_samples, n_experiments, n_dim, support_size, sigma_q in itertools.product(
        n_samples_list, n_experiments_list, n_dim_list, support_size_list, sigma_q_list
    ):
        # Skip invalid combinations where support_size > n_dim
        if support_size > n_dim:
            continue
        params.append(f"{n_samples} {n_experiments} {n_dim} {support_size} {sigma_q}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(params))

    print(f"Generated {len(params)} parameter combinations to {output_file}")
    return params


def generate_multi_instance_params(output_file: str = "multi_instance_params.txt", n_seeds: int = 100):
    """Generate multi-instance (multi-seed) parameter grid for cluster sweep."""
    n_samples = 5000
    n_experiments = 2000
    n_dim = 1000
    support_size_list = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

    params = []
    for support_size, seed in itertools.product(support_size_list, range(1, n_seeds + 1)):
        params.append(f"{n_samples} {n_experiments} {n_dim} {support_size} {seed}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(params))

    print(f"Generated {len(params)} parameter combinations to {output_file}")
    return params


def generate_heatmap_params(output_file: str = "heatmap_params.txt"):
    """Generate parameters for eta ablation heatmap experiments."""
    # Default parameters from eta_noise_ablation.py
    sample_sizes = [500, 1000, 2000, 5000, 10000]
    dimensions = [5, 10, 20, 50]
    betas = [0.5, 1.0, 2.0, 3.0, 4.0]

    params = []

    # d_vs_n mode
    for n_samples, n_dim in itertools.product(sample_sizes, dimensions):
        params.append(f"d_vs_n {n_samples} {n_dim} -")

    # beta_vs_n mode
    for n_samples, beta in itertools.product(sample_sizes, betas):
        params.append(f"beta_vs_n {n_samples} - {beta}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(params))

    print(f"Generated {len(params)} parameter combinations to {output_file}")
    return params


def main():
    parser = argparse.ArgumentParser(description="Generate parameter files for HTCondor experiments")
    parser.add_argument(
        "--type",
        choices=["single_instance", "multi_instance", "heatmap", "all"],
        default="all",
        help="Type of parameter file to generate",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for parameter files")
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of seeds for multi_instance")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ["single_instance", "all"]:
        generate_single_instance_params(str(output_dir / "single_instance_params.txt"))

    if args.type in ["multi_instance", "all"]:
        generate_multi_instance_params(str(output_dir / "multi_instance_params.txt"), args.n_seeds)

    if args.type in ["heatmap", "all"]:
        generate_heatmap_params(str(output_dir / "heatmap_params.txt"))


if __name__ == "__main__":
    main()
