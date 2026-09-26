# E2: theory versus simulation

The paper-demand defaults are standardized generalized-normal covariates with shape beta=4, standardized discrete treatment noise with support inherited from `DEFAULT_TREATMENT_DISCOUNTS/PROBS`, and unit-variance uniform outcome noise. Accordingly, `paper_default` epsilon and the separately requested `uniform` epsilon are aliases; they remain separate seeded cells in the grid.

For d=3, every covariate is independent with the named law and the scalar a and b are broadcast to all three coordinates (L=I). The report distinguishes gross finite-sample mismatches (observed/predicted outside [0.5, 2]) from cells where the predicted AV lies outside the percentile bootstrap 95% interval for observed n Var. Failed fits are excluded from variance and MSE calculations and reported separately.

## Formula agreement

| Estimator | Median obs./pred. | Range | Gross mismatches | 95% exclusions | Fit failures |
|---|---:|---:|---:|---:|---:|
| A_paper_cube_population | 1 | 0.866-1.31 | 0/88 | 7/88 | 0 |
| B_logcosh_eta_column | 1.02 | 0.775-21.5 | 4/88 | 5/88 | 0 |
| B_cube_eta_column | 1.01 | 0.83-28.1 | 13/88 | 5/88 | 0 |
| C_logcosh_eps_row | 1.02 | 0.768-21.2 | 4/88 | 6/88 | 0 |
| C_cube_eps_row | 1 | 0.738-27.9 | 9/88 | 13/88 | 0 |
| D_OLS | 1.01 | 0.85-1.17 | 0/88 | 1/88 | 0 |
| E_HOML_known | 0.998 | 0.747-1.23 | 0/88 | 5/88 | 0 |

## Estimator A comparison

Estimator A's observed variance divided by the paper formula has median 1 (range 0.866-1.31). Dividing the same observations by formula B with the cube contrast gives median 2.26 (range 0.0441-30.4).

## ICA win/loss pattern

A win means lower AV (predicted comparison) or lower n MSE (observed comparison) within the same cell.

| ICA estimator | Pred. vs HOML W/L | Obs. vs HOML W/L | Pred. vs OLS W/L | Obs. vs OLS W/L |
|---|---:|---:|---:|---:|
| B_logcosh_eta_column | 48/40 | 45/43 | 31/57 | 34/54 |
| B_cube_eta_column | 35/53 | 30/58 | 12/76 | 10/78 |
| C_logcosh_eps_row | 78/10 | 73/15 | 51/37 | 51/37 |
| C_cube_eps_row | 56/32 | 50/38 | 21/67 | 19/69 |

## Gross finite-sample mismatches

These are the cells with observed/predicted outside [0.5, 2]. All such cells below are symmetric FastICA readouts that formally converged; the pairwise AV describes the regular local mode but does not capture their finite-sample tail events.

| Estimator | Cell | n | d | a | b | eta | eps | covariate | Obs./pred. |
|---|---|---:|---:|---:|---:|---|---|---|---:|
| B_cube_eta_column | c005 | 5000 | 1 | 0 | 0 | laplace | laplace | paper_gennorm_beta4 | 4.67 |
| C_cube_eps_row | c005 | 5000 | 1 | 0 | 0 | laplace | laplace | paper_gennorm_beta4 | 4.67 |
| B_logcosh_eta_column | c021 | 5000 | 1 | 2 | 0 | laplace | laplace | paper_gennorm_beta4 | 21.5 |
| C_logcosh_eps_row | c021 | 5000 | 1 | 2 | 0 | laplace | laplace | paper_gennorm_beta4 | 21.2 |
| B_cube_eta_column | c026 | 5000 | 1 | 2 | 0 | uniform | gaussian | paper_gennorm_beta4 | 3.09 |
| C_cube_eps_row | c026 | 5000 | 1 | 2 | 0 | uniform | gaussian | paper_gennorm_beta4 | 2.9 |
| B_cube_eta_column | c033 | 5000 | 1 | 0 | 2 | discrete | laplace | paper_gennorm_beta4 | 4.15 |
| C_cube_eps_row | c033 | 5000 | 1 | 0 | 2 | discrete | laplace | paper_gennorm_beta4 | 14.8 |
| C_cube_eps_row | c038 | 5000 | 1 | 0 | 2 | laplace | gaussian | paper_gennorm_beta4 | 2.56 |
| B_cube_eta_column | c040 | 5000 | 1 | 0 | 2 | uniform | paper_default | paper_gennorm_beta4 | 4.19 |
| C_cube_eps_row | c040 | 5000 | 1 | 0 | 2 | uniform | paper_default | paper_gennorm_beta4 | 26.2 |
| B_cube_eta_column | c042 | 5000 | 1 | 0 | 2 | uniform | gaussian | paper_gennorm_beta4 | 6.07 |
| B_logcosh_eta_column | c046 | 5000 | 1 | 0 | 2 | rademacher | gaussian | paper_gennorm_beta4 | 3.91 |
| B_cube_eta_column | c046 | 5000 | 1 | 0 | 2 | rademacher | gaussian | paper_gennorm_beta4 | 19.6 |
| C_logcosh_eps_row | c046 | 5000 | 1 | 0 | 2 | rademacher | gaussian | paper_gennorm_beta4 | 10.1 |
| B_cube_eta_column | c047 | 5000 | 1 | 0 | 2 | rademacher | uniform | paper_gennorm_beta4 | 15.5 |
| B_logcosh_eta_column | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 3.98 |
| C_logcosh_eps_row | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 14.8 |
| B_cube_eta_column | c057 | 5000 | 1 | 2 | -2 | uniform | laplace | paper_gennorm_beta4 | 9.17 |
| C_cube_eps_row | c058 | 5000 | 1 | 2 | -2 | uniform | gaussian | paper_gennorm_beta4 | 2.95 |
| B_cube_eta_column | c061 | 5000 | 1 | 2 | -2 | rademacher | laplace | paper_gennorm_beta4 | 2.07 |
| B_logcosh_eta_column | c066 | 5000 | 1 | 0.5 | 0.05 | discrete | gaussian | paper_gennorm_beta4 | 7.22 |
| C_logcosh_eps_row | c066 | 5000 | 1 | 0.5 | 0.05 | discrete | gaussian | paper_gennorm_beta4 | 7.47 |
| B_cube_eta_column | c069 | 5000 | 1 | 0.5 | 0.05 | laplace | laplace | paper_gennorm_beta4 | 4.35 |
| C_cube_eps_row | c069 | 5000 | 1 | 0.5 | 0.05 | laplace | laplace | paper_gennorm_beta4 | 4.25 |
| B_cube_eta_column | c076 | 5000 | 1 | 0.5 | 0.05 | rademacher | paper_default | paper_gennorm_beta4 | 28.1 |
| C_cube_eps_row | c076 | 5000 | 1 | 0.5 | 0.05 | rademacher | paper_default | paper_gennorm_beta4 | 27.9 |
| B_cube_eta_column | c079 | 5000 | 1 | 0.5 | 0.05 | rademacher | uniform | paper_gennorm_beta4 | 26.6 |
| C_cube_eps_row | c079 | 5000 | 1 | 0.5 | 0.05 | rademacher | uniform | paper_gennorm_beta4 | 26.6 |
| B_cube_eta_column | c085 | 20000 | 1 | 2 | -2 | rademacher | uniform | paper_gennorm_beta4 | 14.2 |

## Bootstrap-interval exclusions

These are the cells where the predicted AV lies outside the percentile bootstrap 95% interval for observed n Var. With 88 cells per estimator, some exclusions are expected from Monte Carlo error.

| Estimator | Cell | n | d | a | b | eta | eps | covariate | Obs./pred. | 95% CI for n Var | Pred. AV |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| B_cube_eta_column | c004 | 5000 | 1 | 0 | 0 | laplace | paper_default | paper_gennorm_beta4 | 0.853 | [2.69, 3.48] | 3.61 |
| C_cube_eps_row | c004 | 5000 | 1 | 0 | 0 | laplace | paper_default | paper_gennorm_beta4 | 0.853 | [2.68, 3.48] | 3.61 |
| B_cube_eta_column | c005 | 5000 | 1 | 0 | 0 | laplace | laplace | paper_gennorm_beta4 | 4.67 | [3.32, 38] | 3.25 |
| C_cube_eps_row | c005 | 5000 | 1 | 0 | 0 | laplace | laplace | paper_gennorm_beta4 | 4.67 | [3.3, 38.4] | 3.25 |
| B_logcosh_eta_column | c011 | 5000 | 1 | 0 | 0 | uniform | uniform | paper_gennorm_beta4 | 0.869 | [0.444, 0.59] | 0.595 |
| C_logcosh_eps_row | c011 | 5000 | 1 | 0 | 0 | uniform | uniform | paper_gennorm_beta4 | 0.868 | [0.443, 0.591] | 0.595 |
| C_cube_eps_row | c011 | 5000 | 1 | 0 | 0 | uniform | uniform | paper_gennorm_beta4 | 0.879 | [0.35, 0.463] | 0.464 |
| A_paper_cube_population | c016 | 5000 | 1 | 2 | 0 | discrete | paper_default | paper_gennorm_beta4 | 1.18 | [17.5, 23.2] | 17.1 |
| A_paper_cube_population | c017 | 5000 | 1 | 2 | 0 | discrete | laplace | paper_gennorm_beta4 | 1.2 | [17.8, 23.1] | 17.1 |
| B_logcosh_eta_column | c019 | 5000 | 1 | 2 | 0 | discrete | uniform | paper_gennorm_beta4 | 0.881 | [0.572, 0.736] | 0.744 |
| C_logcosh_eps_row | c019 | 5000 | 1 | 2 | 0 | discrete | uniform | paper_gennorm_beta4 | 0.871 | [0.565, 0.735] | 0.744 |
| C_cube_eps_row | c019 | 5000 | 1 | 2 | 0 | discrete | uniform | paper_gennorm_beta4 | 0.874 | [0.946, 1.23] | 1.24 |
| B_logcosh_eta_column | c021 | 5000 | 1 | 2 | 0 | laplace | laplace | paper_gennorm_beta4 | 21.5 | [1.35, 69.6] | 1.26 |
| C_logcosh_eps_row | c021 | 5000 | 1 | 2 | 0 | laplace | laplace | paper_gennorm_beta4 | 21.2 | [1.32, 64.2] | 1.26 |
| B_cube_eta_column | c025 | 5000 | 1 | 2 | 0 | uniform | laplace | paper_gennorm_beta4 | 0.83 | [2.28, 2.98] | 3.18 |
| C_cube_eps_row | c025 | 5000 | 1 | 2 | 0 | uniform | laplace | paper_gennorm_beta4 | 0.835 | [2.32, 3.01] | 3.18 |
| D_OLS | c029 | 5000 | 1 | 2 | 0 | rademacher | laplace | paper_gennorm_beta4 | 1.17 | [1.01, 1.33] | 1 |
| E_HOML_known | c029 | 5000 | 1 | 2 | 0 | rademacher | laplace | paper_gennorm_beta4 | 1.17 | [1.02, 1.33] | 1 |
| A_paper_cube_population | c032 | 5000 | 1 | 0 | 2 | discrete | paper_default | paper_gennorm_beta4 | 1.18 | [17.4, 23] | 17.1 |
| C_cube_eps_row | c033 | 5000 | 1 | 0 | 2 | discrete | laplace | paper_gennorm_beta4 | 14.8 | [1.58, 57.2] | 1.57 |
| B_cube_eta_column | c034 | 5000 | 1 | 0 | 2 | discrete | gaussian | paper_gennorm_beta4 | 0.873 | [5.95, 7.89] | 7.91 |
| C_cube_eps_row | c045 | 5000 | 1 | 0 | 2 | rademacher | laplace | paper_gennorm_beta4 | 0.862 | [1.75, 2.25] | 2.32 |
| A_paper_cube_population | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 1.31 | [3.87, 5.09] | 3.42 |
| B_logcosh_eta_column | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 3.98 | [5.06, 48.1] | 4.96 |
| B_cube_eta_column | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 1.2 | [7.73, 9.91] | 7.37 |
| C_cube_eps_row | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 1.2 | [1.59, 2.19] | 1.57 |
| E_HOML_known | c049 | 5000 | 1 | 2 | -2 | discrete | laplace | paper_gennorm_beta4 | 1.23 | [2.02, 2.63] | 1.88 |
| C_cube_eps_row | c052 | 5000 | 1 | 2 | -2 | laplace | paper_default | paper_gennorm_beta4 | 0.845 | [2.64, 3.48] | 3.61 |
| E_HOML_known | c052 | 5000 | 1 | 2 | -2 | laplace | paper_default | paper_gennorm_beta4 | 0.863 | [5.15, 6.9] | 7 |
| C_logcosh_eps_row | c053 | 5000 | 1 | 2 | -2 | laplace | laplace | paper_gennorm_beta4 | 1.2 | [1.28, 1.72] | 1.26 |
| C_cube_eps_row | c057 | 5000 | 1 | 2 | -2 | uniform | laplace | paper_gennorm_beta4 | 0.86 | [2.32, 3.15] | 3.18 |
| C_logcosh_eps_row | c060 | 5000 | 1 | 2 | -2 | rademacher | paper_default | paper_gennorm_beta4 | 0.866 | [0.443, 0.576] | 0.584 |
| C_cube_eps_row | c060 | 5000 | 1 | 2 | -2 | rademacher | paper_default | paper_gennorm_beta4 | 0.872 | [0.343, 0.446] | 0.451 |
| E_HOML_known | c060 | 5000 | 1 | 2 | -2 | rademacher | paper_default | paper_gennorm_beta4 | 0.886 | [0.775, 0.998] | 1 |
| A_paper_cube_population | c064 | 5000 | 1 | 0.5 | 0.05 | discrete | paper_default | paper_gennorm_beta4 | 1.19 | [4.62, 5.99] | 4.46 |
| A_paper_cube_population | c068 | 5000 | 1 | 0.5 | 0.05 | laplace | paper_default | paper_gennorm_beta4 | 0.866 | [9.8, 12.8] | 13 |
| A_paper_cube_population | c080 | 5000 | 1 | 0.5 | 0.05 | discrete | paper_default | laplace_covariate | 1.18 | [4.53, 6.01] | 4.46 |
| C_cube_eps_row | c083 | 20000 | 1 | 2 | 0 | laplace | gaussian | paper_gennorm_beta4 | 1.21 | [7.75, 10.8] | 7.67 |
| B_logcosh_eta_column | c086 | 20000 | 1 | 0.5 | 0.05 | discrete | paper_default | laplace_covariate | 0.775 | [0.473, 0.692] | 0.746 |
| C_logcosh_eps_row | c086 | 20000 | 1 | 0.5 | 0.05 | discrete | paper_default | laplace_covariate | 0.768 | [0.46, 0.683] | 0.744 |
| C_cube_eps_row | c086 | 20000 | 1 | 0.5 | 0.05 | discrete | paper_default | laplace_covariate | 0.738 | [0.748, 1.1] | 1.24 |
| E_HOML_known | c086 | 20000 | 1 | 0.5 | 0.05 | discrete | paper_default | laplace_covariate | 0.747 | [1.14, 1.7] | 1.88 |
