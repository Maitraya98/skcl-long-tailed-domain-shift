#!/bin/bash -l
#SBATCH --job-name=skcl-cifar-r2
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

python3 train_skcl_cifar.py \
  --dataset cifar100 --data_root "$WORK/data" \
  --graph graph_cifar100_real.json --imb_factor 0.01 \
  --epochs 200 --batch_size 256 --seed 123 \
  --out_dir results/skcl_cifar100_lt100_run2
