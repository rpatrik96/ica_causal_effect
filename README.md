# Introduction

This repository contains code related to the paper: [Independent Component Analysis for Treatment Effect Estimation](https://arxiv.org/abs/2507.16467).

# Repository Structure

* `main_estimation.py`: Implements second-order orthogonal methods and benchmark first-order orthogonal estimation methods for the partially linear model.

* `monte_carlo_single_instance_with_seed.py` and `monte_carlo_single_instance.py`: Generate data from the partially linear model DGP based on input parameters, execute the proposed second-order orthogonal method and benchmarks, and save results in joblib dumps. The former is used for generating plots with multiple instances, while the latter is for single instance plots and parameter sweeps.

* `plot_dumps_multi_instance.py` and `plot_dumps_single_instance.py`: Generate figures from the corresponding dumps created by the above files.

* `single_instance_parameter_sweep.sh`: A shell script that runs a single instance as the DGP parameters vary, creating all related plots.

* `multi_instance.sh`: A shell script that runs multiple instances of the DGP for a fixed set of parameters, generating related plots.

* `ica.py`: Contains functions for generating Independent Component Analysis (ICA) data, estimating treatment effects using ICA, and various main functions for running different ICA-related experiments.

* `mcc.py`: Implements the Munkres algorithm (also known as the Hungarian algorithm) for solving the assignment problem and includes functions for calculating disentanglement metrics such as R2 and MCC.

* `plot_utils.py`: Provides utility functions for plotting, including typography settings, estimate histograms, method comparisons, and multi-treatment plots.

# Re-creating the Figures in the Paper

To recreate the figures in the paper, execute the following scripts: