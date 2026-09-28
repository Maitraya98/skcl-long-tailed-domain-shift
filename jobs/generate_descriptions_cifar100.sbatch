#!/bin/bash -l
#SBATCH --job-name=gen-desc-cifar100
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
mkdir -p logs

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module add python
conda activate skcl

python3 src/llm_descriptions/generate_descriptions.py \
  --dataset cifar100 \
  --data_root $WORK/data \
  --backend local \
  --local_model Qwen/Qwen2.5-3B-Instruct \
  --out descriptions_cifar100_real.json

echo "=== Description generation complete ==="
