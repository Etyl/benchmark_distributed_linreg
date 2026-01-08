#!/bin/bash
#
#SBATCH --job-name=benchopt-linreg
#
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --partition=parietal,normal
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --error error_%A_%a.out
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate benchopt-linreg

cd ~/benchmarks/benchmark_distributed_linreg
python -m benchopt run . --config config.yml