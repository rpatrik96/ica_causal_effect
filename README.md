# Introduction

[![CI](https://github.com/rpatrik96/ica_causal_effect/actions/workflows/ci.yml/badge.svg)](https://github.com/rpatrik96/ica_causal_effect/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/rpatrik96/ica_causal_effect/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/rpatrik96/ica_causal_effect/actions/workflows/pre-commit.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains code related to the paper: [Independent Component Analysis for Treatment Effect Estimation](https://arxiv.org/abs/2507.16467).

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For development (testing, linting, formatting):

```bash
pip install -r requirements-dev.txt
```

### Pre-commit Hooks

Install pre-commit hooks for automated code quality checks:

```bash
pre-commit install
```

## Repository Structure

### Core Modules

* `main_estimation.py`: Implements second-order orthogonal methods and benchmark first-order orthogonal estimation methods for the partially linear model.

* `ica.py`: Contains functions for generating Independent Component Analysis (ICA) data, estimating treatment effects using ICA, and various main functions for running different ICA-related experiments.

* `mcc.py`: Implements the Munkres algorithm (also known as the Hungarian algorithm) for solving the assignment problem and includes functions for calculating disentanglement metrics such as R2 and MCC.

### Experiment Runners

* `monte_carlo_single_instance.py` and `monte_carlo_single_instance_with_seed.py`: Generate data from the partially linear model DGP based on input parameters, execute the proposed second-order orthogonal method and benchmarks, and save results. The former is for single instance plots and parameter sweeps, while the latter is used for generating plots with multiple instances.

* `experiment_runner.py`: Unified experiment runner for OML experiments with parallel execution support.

* `oml_runner.py`: Runner for orthogonal machine learning experiments.

* `eta_noise_ablation_refactored.py`: Ablation studies for noise distributions, variance parameters, and coefficient configurations. Supports:
  - **Filtered heatmap experiments**: Compare HOML vs ICA RMSE across sample sizes and dimensions/beta values
  - **ICA variance coefficient constraint**: Automatically compute coefficients to achieve a target ICA variance coefficient
  - **Coefficient ablation**: Vary treatment/outcome coefficients to study their effect on estimation error
  - **Variance ablation**: Study how noise variance affects estimation across different distributions

### Utilities

* `plot_utils.py`: Provides utility functions for plotting, including typography settings, estimate histograms, method comparisons, and multi-treatment plots.

* `ica_utils.py`: Utility functions for ICA experiments.

* `ica_plotting.py`: Plotting functions specific to ICA experiments.

* `oml_utils.py`: Utility functions for OML experiments.

* `oml_plotting.py`: Plotting functions for OML experiments.

* `ablation_utils.py`: Shared utilities for ablation studies.

### Plotting Scripts

* `plot_dumps_single_instance.py`: Generate figures from single instance results.

* `plot_multi_instance.py`: Generate figures from multi-instance results.

* `regenerate_all_n20_plots.py`: Regenerate all plots for n=20 experiments.

* `regenerate_n20_heatmaps.py`: Regenerate heatmap plots for n=20 experiments.

* `regenerate_ica_heatmaps.py`: Regenerate ICA-specific heatmap plots.

### Shell Scripts

* `single_instance_parameter_sweep.sh`: A shell script that runs a single instance as the DGP parameters vary, creating all related plots.

* `multi_instance.sh`: A shell script that runs multiple instances of the DGP for a fixed set of parameters, generating related plots.

## Testing

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run with coverage:

```bash
pytest --cov=. --cov-report=html
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Code Quality

Format code:

```bash
black .
```

Sort imports:

```bash
isort .
```

Run linting:

```bash
pylint *.py
flake8 .
```

Run all pre-commit hooks:

```bash
pre-commit run --all-files
```

## CI/CD

The repository uses GitHub Actions for continuous integration:

- **CI Workflow**: Runs tests on Python 3.8-3.11, code quality checks, and coverage reporting
- **Pre-commit Workflow**: Validates code formatting and linting
- **Quick Test Workflow**: Fast feedback on any branch

See [CI_CD.md](CI_CD.md) for detailed CI/CD documentation.

## Running Ablation Experiments

### Eta Noise Ablation

Run filtered heatmap experiments comparing HOML vs ICA across sample sizes and dimensions:

```bash
# Basic filtered heatmap (dimension vs sample size)
python eta_noise_ablation_refactored.py --filtered_heatmap

# With ICA variance coefficient constraint (automatically computes coefficients)
python eta_noise_ablation_refactored.py --filtered_heatmap --constrain_ica_var

# Beta vs sample size mode with custom parameters
python eta_noise_ablation_refactored.py --filtered_heatmap \
  --heatmap_axis_mode beta_vs_n \
  --ica_var_threshold 2.0 \
  --n_experiments 50
```

Run coefficient and variance ablation studies:

```bash
# Coefficient ablation (varying treatment/outcome coefficients)
python eta_noise_ablation_refactored.py --coefficient_ablation

# Variance ablation (noise variance across beta values)
python eta_noise_ablation_refactored.py --variance_ablation
```

Key flags:
- `--constrain_ica_var`: Automatically compute treatment coefficient to achieve `ica_var_coeff = ica_var_threshold`
- `--ica_var_threshold`: Target ICA variance coefficient (default: 1.5)
- `--heatmap_axis_mode`: Choose "d_vs_n" (dimension vs sample size) or "beta_vs_n" (beta vs sample size)

## Re-creating the Figures in the Paper

To recreate the figures in the paper, execute the following scripts:
