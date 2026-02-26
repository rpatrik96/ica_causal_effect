#!/usr/bin/env bash
# Usage: single_instance_parameter_sweep.sh <output_dir>
# Sweeps over sample sizes, dimensions, support sizes, and sigma_q values.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir>" >&2
    exit 1
fi

output_dir="${1}"
mkdir -p "${output_dir}"
echo "${output_dir}"

for n_samples in 2000 5000 10000;
do
    for n_experiments in 2000;
    do
        for n_dim in 1000 2000 5000;
        do
            for support_size in 1 20 40 60 80 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800;
            do
                for sigma_q in 3 10 20;
                do
                    python3 monte_carlo_single_instance.py --n_samples "${n_samples}" --n_experiments "${n_experiments}" --n_dim "${n_dim}" --support_size "${support_size}" --output_dir "${output_dir}" --sigma_q "${sigma_q}"
                done
            done
        done
    done
done

mkdir -p figures
mkdir -p figures/single_instance
python3 plot_dumps_single_instance.py --input_dir "${output_dir}" --output_dir figures/single_instance

echo "Saved figures in figures/single_instance!"
