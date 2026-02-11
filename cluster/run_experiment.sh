#!/bin/bash
# Wrapper script for running double_orthogonal_ml experiments on HTCondor cluster
set -e

# Parse arguments
EXPERIMENT_TYPE=${1:-"single_instance"}
shift
EXPERIMENT_ARGS="$@"

# Setup paths - adjust these to your cluster configuration
CLUSTER_HOME="/home/preizinger"
PROJECT_DIR="${CLUSTER_HOME}/double_orthogonal_ml"
VENV_DIR="${PROJECT_DIR}/.venv"
# Output to project figures directory (will be synced back)
OUTPUT_DIR="${PROJECT_DIR}/figures"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to project directory (so relative imports work)
cd "${PROJECT_DIR}"

# Activate virtual environment
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "Virtual environment not found at ${VENV_DIR}"
    echo "Please run: python -m venv ${VENV_DIR} && pip install -r requirements.txt"
    exit 1
fi

# Print environment info for debugging
echo "============================================"
echo "Experiment Type: ${EXPERIMENT_TYPE}"
echo "Arguments: ${EXPERIMENT_ARGS}"
echo "Python: $(which python)"
echo "Working Directory: $(pwd)"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Hostname: $(hostname)"
echo "Start Time: $(date)"
echo "============================================"

# Run the appropriate experiment
case "${EXPERIMENT_TYPE}" in
    "single_instance")
        # Single instance Monte Carlo experiment
        # Example args: --n_samples 5000 --n_experiments 20 [--no_oracle_support]
        python monte_carlo_single_instance.py --output_dir "${OUTPUT_DIR}" ${EXPERIMENT_ARGS}
        ;;

    "single_instance_seed")
        # Single instance with specific seed (reuses monte_carlo_single_instance.py which accepts --seed)
        # Example args: --n_samples 500 --n_experiments 20 --seed 42
        python monte_carlo_single_instance.py --output_dir "${OUTPUT_DIR}" ${EXPERIMENT_ARGS}
        ;;

    "eta_filtered_heatmap")
        # Eta noise ablation - filtered heatmap experiments
        # Example args: --n_experiments 20 [--no_oracle_support]
        python eta_noise_ablation_refactored.py --filtered_heatmap --output_dir "${OUTPUT_DIR}" ${EXPERIMENT_ARGS}
        ;;

    "eta_variance_ablation")
        # Eta noise ablation - variance ablation experiments
        # Example args: --n_experiments 20 [--no_oracle_support]
        python eta_noise_ablation_refactored.py --variance_ablation --output_dir "${OUTPUT_DIR}" ${EXPERIMENT_ARGS}
        ;;

    "eta_coefficient_ablation")
        # Eta noise ablation - coefficient ablation experiments
        # Example args: --n_experiments 20 [--no_oracle_support]
        python eta_noise_ablation_refactored.py --coefficient_ablation --output_dir "${OUTPUT_DIR}" ${EXPERIMENT_ARGS}
        ;;

    "eta_default")
        # Eta noise ablation - default mode (no specific ablation flag)
        # Example args: --n_experiments 20 [--no_oracle_support]
        python eta_noise_ablation_refactored.py --output_dir "${OUTPUT_DIR}" ${EXPERIMENT_ARGS}
        ;;

    "ica")
        # ICA-specific experiments
        python ica.py ${EXPERIMENT_ARGS}
        ;;

    *)
        echo "Unknown experiment type: ${EXPERIMENT_TYPE}"
        echo "Available types:"
        echo "  single_instance         - Monte Carlo OML experiments"
        echo "  single_instance_seed    - Seeded Monte Carlo experiments"
        echo "  eta_filtered_heatmap    - Eta ablation: filtered heatmap"
        echo "  eta_variance_ablation   - Eta ablation: variance ablation"
        echo "  eta_coefficient_ablation - Eta ablation: coefficient ablation"
        echo "  eta_default             - Eta ablation: default mode"
        echo "  ica                     - ICA experiments"
        exit 1
        ;;
esac

echo "============================================"
echo "End Time: $(date)"
echo "Experiment completed successfully!"
echo "============================================"
