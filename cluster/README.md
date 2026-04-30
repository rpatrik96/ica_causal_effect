# HTCondor Cluster Setup for double_orthogonal_ml

This directory contains configuration files for running experiments on an HTCondor cluster.

## Setup (One-time)

### 1. Clone the repository on the cluster

```bash
cd ~
git clone https://github.com/rpatrik96/double_orthogonal_ml.git
cd double_orthogonal_ml
```

### 2. Create and configure the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create jobs output directory

```bash
mkdir -p ~/jobs
```

### 4. Make the run script executable

```bash
chmod +x cluster/run_experiment.sh
```

### 5. Adjust paths in configuration files

Edit `cluster/run_experiment.sh` and update these paths to match your cluster setup:
- `CLUSTER_HOME`: Your home directory on the cluster
- `PROJECT_DIR`: Where you cloned the repository
- `VENV_DIR`: Path to your virtual environment

Results are automatically saved to `${PROJECT_DIR}/figures/`.

## Experiment Configurations

The cluster setup covers all 10 experiment configurations:

| Experiment Type | Oracle Support | Command |
|-----------------|----------------|---------|
| `single_instance` | Yes (default) | `monte_carlo_single_instance.py` |
| `single_instance` | No | `monte_carlo_single_instance.py --no_oracle_support` |
| `eta_filtered_heatmap` | Yes | `eta_noise_ablation.py --filtered_heatmap` |
| `eta_filtered_heatmap` | No | `eta_noise_ablation.py --filtered_heatmap --no_oracle_support` |
| `eta_variance_ablation` | Yes | `eta_noise_ablation.py --variance_ablation` |
| `eta_variance_ablation` | No | `eta_noise_ablation.py --variance_ablation --no_oracle_support` |
| `eta_coefficient_ablation` | Yes | `eta_noise_ablation.py --coefficient_ablation` |
| `eta_coefficient_ablation` | No | `eta_noise_ablation.py --coefficient_ablation --no_oracle_support` |
| `eta_default` | Yes | `eta_noise_ablation.py` |
| `eta_default` | No | `eta_noise_ablation.py --no_oracle_support` |

## Submit Files

| File | Description |
|------|-------------|
| `sweep_all_experiments.sub` | **All 10 configurations** - single instance + all eta ablations |
| `sweep_eta_ablation.sub` | All 8 eta ablation configurations |
| `sweep_rebuttal.sub` | **UAI 2026 rebuttal** - Bernoulli + semi-synthetic + multi-treatment with OLS/HOML baselines |
| `cluster.sub` | Basic submit file for single custom experiments |
| `sweep_single_instance.sub` | Parameter sweep for single instance experiments |
| `sweep_multi_instance.sub` | Multi-seed experiments for variance estimation |
| `sweep_full.sub` | Full parameter sweep from file |

## Running Experiments

### Run all 10 configurations (recommended)

```bash
cd ~/double_orthogonal_ml/cluster
condor_submit sweep_all_experiments.sub
```

This submits 10 jobs covering all experiment types with both oracle configurations.

### Run only eta ablation experiments (8 jobs)

```bash
condor_submit sweep_eta_ablation.sub
```

### Run UAI 2026 rebuttal experiments (14 jobs)

```bash
condor_submit sweep_rebuttal.sub
```

This submits the three new rebuttal experiment families:
- **Section A** — Bernoulli noise ablation (5 jobs, one per `n_samples`).
- **Section B** — Semi-synthetic experiment on California Housing (8 jobs, one per `(eta_dist, nonlinearity, theta)`).
- **Section C** — Multi-treatment Fig 4 with OLS+HOML baselines (1 job, full grid).

OLS and matching baselines are now part of the main dispatcher, so re-submitting `sweep_all_experiments.sub` (or `sweep_eta_ablation.sub`) automatically appends them to every existing ablation. Delete or rename existing `figures/**/*.npy` files before re-submission so the existence-check in `eta_noise_ablation.py` does not short-circuit the re-run.

### Run a single custom experiment

```bash
# With oracle support (default)
condor_submit cluster.sub \
    experiment_type=single_instance \
    experiment_args="--n_samples 5000 --n_experiments 20"

# Without oracle support
condor_submit cluster.sub \
    experiment_type=eta_filtered_heatmap \
    experiment_args="--n_experiments 20 --no_oracle_support"
```

## Experiment Types for run_experiment.sh

| Type | Script | Description |
|------|--------|-------------|
| `single_instance` | `monte_carlo_single_instance.py` | Main OML Monte Carlo experiments |
| `single_instance_seed` | `monte_carlo_single_instance.py --seed` | Seeded experiments for reproducibility |
| `eta_filtered_heatmap` | `eta_noise_ablation.py --filtered_heatmap` | Filtered RMSE heatmaps |
| `eta_variance_ablation` | `eta_noise_ablation.py --variance_ablation` | Variance ablation studies |
| `eta_coefficient_ablation` | `eta_noise_ablation.py --coefficient_ablation` | Coefficient ablation studies |
| `eta_default` | `eta_noise_ablation.py` | Default eta ablation mode |
| `ica` | `ica.py` | ICA-specific experiments |

## Output Location

All results are saved to the **project directory**: `~/double_orthogonal_ml/figures/`

This ensures results are automatically available when you pull the repo locally:

```bash
# On local machine - sync results from cluster
rsync -avz cluster:~/double_orthogonal_ml/figures/ ./figures/

# Or use git if figures are tracked
ssh cluster "cd ~/double_orthogonal_ml && git add figures/ && git commit -m 'Add cluster results' && git push"
git pull
```

## Monitoring Jobs

```bash
# Check job status
condor_q

# Check job status for specific user
condor_q -submitter preizinger

# View detailed job info
condor_q -long <job_id>

# View job logs
tail -f ~/jobs/doml_<cluster_id>_<exp_type>_oracle<flag>.out

# Remove a job
condor_rm <job_id>

# Remove all your jobs
condor_rm -all
```

## Resource Configuration

Default resource requests (adjust in submit files based on experiment size):

| Resource | Default | Notes |
|----------|---------|-------|
| CPUs | 4-8 | Eta ablations use 8 CPUs |
| Memory | 32GB | For large matrix operations |
| Disk | 20GB | For result files |
| Max Time | 3 days | Set via `MaxTime` in submit file; must match `+MaxRuntime` |

These experiments are CPU-only (no GPU required).

## Troubleshooting

### Job held or removed

Check the error log:
```bash
cat ~/jobs/doml_<cluster_id>_<exp_type>_oracle<flag>.err
```

Common issues:
- **Memory exceeded**: Increase `request_memory`
- **Time limit exceeded**: Increase `MaxTime` **and** `+MaxRuntime` — both must match, as `periodic_remove` uses `MaxTime` independently of `+MaxRuntime`
- **Missing dependencies**: Ensure venv is properly set up

### Virtual environment issues

```bash
# Recreate the venv if needed
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Module import errors

Ensure you're running from the project directory. The `run_experiment.sh` script does `cd "${PROJECT_DIR}"` automatically.
