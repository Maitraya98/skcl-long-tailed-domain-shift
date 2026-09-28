#!/bin/bash -l
#SBATCH --job-name=concutmix-cifar10
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
mkdir -p logs

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module add python
conda activate skcl

echo "=== Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

cd ~/skcl-project/ConCutMix/run_files
bash cifar10_imb100.sh

echo "=== Training complete ==="
