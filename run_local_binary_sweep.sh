#!/usr/bin/env bash
# Local execution of the 13-job binary-treatment sweep that mirrors
# cluster/sweep_binary_treatment.sub. Uses Python 3.10 because that is
# the interpreter on this machine that has both numpy and torch.
set -euo pipefail

PYTHON="/usr/local/bin/python3.10"
OUT="figures/binary_treatment"
N_EXP=30
mkdir -p "${OUT}/logs"

run_job () {
    local label="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> ${label}"
    "${PYTHON}" binary_treatment_runner.py "$@" --output_dir "${OUT}" \
        --results_file "${label}.npy" \
        > "${OUT}/logs/${label}.log" 2>&1
    echo "[$(date +%H:%M:%S)] <<< ${label} done"
}

# Section A: sample-size sweep ----------------------------------------------
for N in 500 1000 2000 5000 10000; do
    run_job "A_n${N}" \
        --n_samples "${N}" --n_experiments "${N_EXP}" \
        --n_covariates 10 --support_size 5 \
        --treatment_effect 1.5 --propensity_strength 0.7 \
        --outcome_coef_scale 0.5 --sigma_outcome 0.5 \
        --base_seed 12143
done

# Section B: propensity-strength sweep --------------------------------------
for P in 0.3 0.7 1.5 3.0; do
    run_job "B_p${P}" \
        --n_samples 2000 --n_experiments "${N_EXP}" \
        --n_covariates 10 --support_size 5 \
        --treatment_effect 1.5 --propensity_strength "${P}" \
        --outcome_coef_scale 0.5 --sigma_outcome 0.5 \
        --base_seed 12143
done

# Section C: covariate-dimension sweep --------------------------------------
for D in 10 20 50 100; do
    run_job "C_d${D}" \
        --n_samples 2000 --n_experiments "${N_EXP}" \
        --n_covariates "${D}" --support_size 5 \
        --treatment_effect 1.5 --propensity_strength 0.7 \
        --outcome_coef_scale 0.5 --sigma_outcome 0.5 \
        --base_seed 12143
done

echo "[$(date +%H:%M:%S)] All 13 jobs done."
