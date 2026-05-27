"""Re-run the production paths with check_convergence=False so we see the
actual theta the runner would consume."""

# ruff: noqa: E402, E501
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from binary_treatment_dgp import BinaryTreatmentDGPConfig, generate_binary_treatment_data
from ica import ica_treatment_effect_estimation, ica_treatment_effect_estimation_eps_row

warnings.filterwarnings("ignore")

cfg = BinaryTreatmentDGPConfig(
    n_samples=2000,
    n_covariates=10,
    support_size=5,
    treatment_effect=1.5,
    propensity_strength=0.7,
    outcome_coef_scale=0.5,
    sigma_outcome=0.5,
    seed=2024,
)
X_cov, T, Y, propensity, eta, alpha, beta = generate_binary_treatment_data(cfg)
obs = np.column_stack([X_cov, T, Y])
eps = Y - cfg.treatment_effect * T - X_cov @ beta
sources = np.column_stack([X_cov, eta, eps])

print("Production calls with check_convergence=False (what the runner consumes):")
print(f"{'seed':>4s} {'fun':>8s} | {'theta_eps':>12s} {'theta_munkres':>14s} {'MCC_munk':>9s}")
for fun in ("logcosh", "exp", "cube"):
    for seed in range(5):
        t_eps, _ = ica_treatment_effect_estimation_eps_row(
            obs,
            S=sources,
            random_state=seed,
            n_treatments=1,
            verbose=False,
            fun=fun,
            check_convergence=False,
        )
        t_mun, mcc_mun = ica_treatment_effect_estimation(
            obs,
            S=sources,
            random_state=seed,
            n_treatments=1,
            verbose=False,
            fun=fun,
            check_convergence=False,
        )
        mcc_str = f"{mcc_mun:.3f}" if mcc_mun is not None else "  nan"
        print(f"{seed:>4d} {fun:>8s} | {float(t_eps[0]):>12.4f} " f"{float(t_mun[0]):>14.4f} {mcc_str:>9s}")
