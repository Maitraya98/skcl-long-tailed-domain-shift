#!/bin/bash -l
#SBATCH --job-name=skcl-cifar10
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

python3 train_skcl_cifar.py \
  --dataset cifar10 \
  --data_root "$WORK/data" \
  --graph graph_cifar10_real.json \
  --imb_factor 0.01 \
  --epochs 200 \
  --batch_size 256 \
  --out_dir results/skcl_cifar10_lt100

echo "=== Training complete ==="
