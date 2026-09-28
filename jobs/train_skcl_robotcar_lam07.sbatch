#!/bin/bash -l
#SBATCH --job-name=skcl-rc-l07
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

python3 train_skcl_robotcar.py --manifest robotcar_extracted/manifest.csv \
  --graph graph_robotcar_real.json --epochs 100 --batch_size 128 --lam 0.7 --seed 42 \
  --out_dir results/skcl_robotcar_lam07
