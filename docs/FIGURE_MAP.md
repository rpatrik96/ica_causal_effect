# ICA companion-paper figure map

Run commands from the repository root with the full historical grids. The
experiment drivers write data caches in the repository root and figures under
`figures/ica/`. For cached drivers, the plotting command reloads the `.npy`
file and skips data generation. `main_fun` is the exception: its combined
driver always regenerates its data before plotting.

| Paper output | Data artifact | Generate data | Plot | Status |
|---|---|---|---|---|
| `heatmap_dimension_vs_nonlinearity.svg` | `results_main_nonlinear.npy` | `./.venv/bin/python ica_experiments.py nonlinear` | `./.venv/bin/python ica_experiments.py nonlinear` | Covered |
| `heatmap_dimension_vs_nonlinearity_std.svg` | `results_main_nonlinear.npy` | `./.venv/bin/python ica_experiments.py nonlinear` | `./.venv/bin/python ica_experiments.py nonlinear` | Covered |
| `heatmap_dimension_vs_slope_leaky_relu.svg` | `results_main_nonlinear.npy` | `./.venv/bin/python ica_experiments.py nonlinear` | `./.venv/bin/python ica_experiments.py nonlinear` | Covered |
| `heatmap_ica_multi_dimensions_vs_samples_rel.svg` | `results_multi_treatment.npy` | `./.venv/bin/python ica_experiments.py multi` | `./.venv/bin/python ica_experiments.py multi` | Covered |
| `heatmap_ica_multi_dimensions_vs_samples_rel_m2.svg` | `results_multi_treatment.npy` | `./.venv/bin/python ica_experiments.py multi` | `./.venv/bin/python ica_experiments.py multi` | Covered |
| `ica_mse_fun.svg` | `results_main_fun.npy` | `./.venv/bin/python ica_experiments.py fun` | `./.venv/bin/python ica_experiments.py fun` | Covered; combined driver regenerates data |
| `ica_mse_vs_beta_nonlinear_n5000.svg` | `results_main_gennorm_nonlinear.npy` | `./.venv/bin/python ica_experiments.py gennorm_nonlinear` | `./.venv/bin/python ica_experiments.py gennorm_nonlinear` | Covered |
| `ica_mse_heatmap_loc_scale_n5000.svg` | `results_main_loc_scale.npy` | `./.venv/bin/python ica_experiments.py loc_scale` | `./.venv/bin/python ica_experiments.py loc_scale` | Covered |
| `ica_mse_vs_noise_split_nonlinear_n5000.svg` | `results_main_gennorm_nonlinear_noise_split.npy` | `./.venv/bin/python ica_experiments.py nonlinear_noise_split` | `./.venv/bin/python ica_experiments.py nonlinear_noise_split` | Covered |
| `ica_mse_vs_theta_choice_nonlinear_n5000.svg` | `results_main_gennorm_nonlinear_theta.npy` | `./.venv/bin/python ica_experiments.py nonlinear_theta` | `./.venv/bin/python ica_experiments.py nonlinear_theta` | Covered |
| `ica_mse_vs_dim_sparsity.svg` | `results_main_sparsity.npy` | `./.venv/bin/python ica_experiments.py sparsity` | `./.venv/bin/python ica_experiments.py sparsity` | Covered |
| `ica_iv_mse_vs_dim_n5000.*` | — | — | — | **MISSING:** generator last present at `44e2e9e`; not used by the paper |
| `rmse_diff_boxplot_by_te.svg` | coefficient-ablation results | `./.venv/bin/python eta_noise_ablation.py --coefficient_ablation` | same command (`eta_ablation_plotting.plot_coefficient_ablation_results`) | Covered; plot restored from `11b225d^:eta_noise_ablation.py` |

## DirectLiNGAM comparisons

These commands each produce the error figure, runtime figure, and Markdown
summary shown below. They require the `lingam` dependency declared in
`requirements.txt`.

| Outputs | Data artifact | Generate data | Plot | Status |
|---|---|---|---|---|
| `ica_vs_directlingam_sparsity_n5000.svg`, `ica_vs_directlingam_sparsity_runtime_n5000.svg`, `ica_vs_directlingam_sparsity_n5000_summary.md` | `results_main_sparsity_comparison.npy` | `./.venv/bin/python ica_experiments.py sparsity_comparison` | `./.venv/bin/python ica_experiments.py sparsity_comparison` | Covered |
| `ica_vs_directlingam_gennorm_n200.svg`, `ica_vs_directlingam_gennorm_runtime_n200.svg`, `ica_vs_directlingam_gennorm_n200_summary.md` | `results_main_gennorm_comparison_200.npy` | `./.venv/bin/python ica_experiments.py gennorm_comparison` | `./.venv/bin/python ica_experiments.py gennorm_comparison` | Covered |
| `ica_vs_directlingam_sample_size.svg`, `ica_vs_directlingam_sample_size_runtime.svg`, `ica_vs_directlingam_sample_size_summary.md` | `results_main_sample_size_comparison.npy` | `./.venv/bin/python ica_experiments.py sample_size_comparison` | `./.venv/bin/python ica_experiments.py sample_size_comparison` | Covered |
| `ica_vs_directlingam_n_covariates_n1000.svg`, `ica_vs_directlingam_n_covariates_runtime_n1000.svg`, `ica_vs_directlingam_n_covariates_n1000_summary.md` | `results_main_n_covariates_comparison_n1000.npy` | `./.venv/bin/python ica_experiments.py n_covariates_comparison` | `./.venv/bin/python ica_experiments.py n_covariates_comparison` | Covered |
| `ica_vs_directlingam_n_treatments_n1000.svg`, `ica_vs_directlingam_n_treatments_runtime_n1000.svg`, `ica_vs_directlingam_n_treatments_n1000_summary.md` | `results_main_n_treatments_comparison_n1000.npy` | `./.venv/bin/python ica_experiments.py n_treatments_comparison` | `./.venv/bin/python ica_experiments.py n_treatments_comparison` | Covered |

Use `--n_seeds N` to override the historical default of 20. Add `--quick` only
for smoke tests; it deliberately changes grids, sample sizes, and numeric output
suffixes and therefore does not reproduce the paper figures.
