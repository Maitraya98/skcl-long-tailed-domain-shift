#!/bin/bash -l
#SBATCH --job-name=concutmix-r2
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

cd ~/skcl-project/ConCutMix/run_files
sed 's|--root_log "../log"|--root_log "../log_run2"|' cifar100_imb100.sh > cifar100_imb100_run2.sh
bash cifar100_imb100_run2.sh
