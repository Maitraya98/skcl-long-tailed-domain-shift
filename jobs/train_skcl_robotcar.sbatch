#!/bin/bash -l
#SBATCH --job-name=skcl-robotcar
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=02:00:00
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
python3 train_skcl_robotcar.py --manifest robotcar_extracted/manifest.csv \
  --graph graph_robotcar_real.json --epochs 100 --batch_size 128 --out_dir results/skcl_robotcar
echo "=== Training complete ==="
