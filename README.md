[README.md](https://github.com/user-attachments/files/32736708/README.md)
# Long-Tailed Recognition with SKCL

**Reproduction on CIFAR-100-LT and CIFAR-10-LT, and a day-to-night domain-shift experiment on Oxford driving data**

Project Representation Learning · Image Data Exploration and Analysis (IDEA) Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg

Author: Maitraya Prasad Goswami · Supervisors: Prof. Dr. Bernhard Kainz, Mischa Dombrowski, Luca Hagen

---

## Overview

Semantic Knowledge-driven Contrastive Learning (SKCL, Liu et al., ICCV 2025) helps rare ("tail") classes by generating a text description of every class with a language model, linking classes whose descriptions are similar in a semantic graph, and pulling each image toward its own class and its semantic neighbours during contrastive training. The paper has no public code.

This project has two parts:

1. **Reproduction.** SKCL implemented from the paper, and compared with two image-only baselines, **BCL** (Zhu et al., CVPR 2022) and **ConCutMix** (Pan et al., IEEE TIP 2024), on CIFAR-100-LT and CIFAR-10-LT (β = 100, ResNet-32).
2. **Own experiment.** A new day/night dataset built from ROAD annotations on Oxford RobotCar video. BCL and SKCL are trained on daytime images only and tested on unseen daytime and nighttime drives, to see whether SKCL's text-based knowledge makes it more robust to lighting changes.

## Main results

**Part 1: reproduction (Top-1 %, β = 100)**

| Method    | CIFAR-100-LT paper | This project (mean ± sd, 3 runs) | vs. paper | CIFAR-10-LT paper | This project (1 run) | vs. paper |
|-----------|-------------------:|---------------------------------:|----------:|------------------:|---------------------:|----------:|
| BCL       | 52.01 | 47.2 ± 0.9 | −9.3%  | 87.26 | 78.04 | −10.6% |
| ConCutMix | 53.16 | 52.8 ± 0.3 | −0.7%  | 88.00 | 85.16 | −3.2%  |
| SKCL      | 54.02 | 43.7 ± 0.8 | −19.1% | 88.16 | 75.19 | −14.7% |

Only ConCutMix reproduces within the ±5% criterion. SKCL falls furthest on the rare (Few) classes.

**Part 2: day → night (5 runs per method, trained on day data only)**

| Metric | BCL | SKCL |
|---|---:|---:|
| Day − night gap, overall Top-1 (points) | 1.74 ± 0.72 | 2.10 ± 1.01 |
| Cyclist Top-1 at night (%) | 47.8 ± 4.4 | 51.7 ± 6.1 |

SKCL is ahead on night cyclists in 4 of 5 runs, but the difference is not statistically significant (p = 0.29). Overall robustness is the same for both methods. The report traces the small effect to the semantic graph: the cyclist is its least connected class, and its shared coarse node (`vulnerable_road_user`) is semantically closer to pedestrians than to cyclists.

The full report is in [`report/`](report/).

## Repository structure

```
skcl-long-tailed-domain-shift/
├── ConCutMix/                 public ConCutMix repository, adapted (5 compatibility fixes)
├── jobs/                      SLURM batch scripts for TinyGPU (incl. repeat runs: *_run2, *_seed7,
│                              and parameter tests: *_beta025, *_beta05, *_lam03, *_lam07)
├── loss/                      skcl_loss.py, contrastive.py (BCL), logitadjust.py
├── models/                    resnet_cifar.py (CIFAR ResNet-32), skcl_model.py
├── scripts/                   extract_road_dataset.py, analyze_box_sizes.py, inspect_cifar_lt.py
├── src/
│   ├── datasets/              long-tailed CIFAR, multi-view loader, CIFAR class hierarchy
│   ├── llm_descriptions/      generate_descriptions.py (Qwen2.5-3B-Instruct)
│   ├── robotcar/              ROAD annotation parsing, RobotCar hierarchy and loaders
│   └── semantic_graph/        build_graph.py (Sentence-BERT + Top-K graph)
├── tests/                     unit tests (long-tailed sampler, BCL and SKCL training steps)
├── results/                   per-run log.txt files and figures (no checkpoints)
├── descriptions_*.json        generated class descriptions (CIFAR-10, CIFAR-100, RobotCar)
├── graph_*.json               semantic graphs used for training
├── train_bcl_cifar.py         BCL training, CIFAR
├── train_skcl_cifar.py        SKCL training, CIFAR
├── train_bcl_robotcar.py      BCL training, day-to-night driving data
├── train_skcl_robotcar.py     SKCL training, day-to-night driving data
├── demo_predict.py            prediction demo on Few-class examples
├── environment.yml            conda environment
└── report/                    final project report (PDF)
```

Datasets, extracted image crops and model checkpoints are **not** included (see *Data* below).

## Setup

```bash
git clone https://github.com/Maitraya98/skcl-long-tailed-domain-shift.git
cd skcl-long-tailed-domain-shift
conda env create -f environment.yml
conda activate skcl
```

On the FAU TinyGPU cluster:

```bash
module add python
conda activate skcl
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

## Running the pipeline

**1. Check that everything works (no data or GPU needed)**

```bash
python3 models/resnet_cifar.py
python3 models/skcl_model.py
python3 tests/test_cifar_lt.py
python3 tests/test_train_bcl_cifar.py
python3 tests/test_train_skcl_cifar.py
```

**2. Generate class descriptions and build the semantic graph (SKCL only)**

```bash
python3 src/llm_descriptions/generate_descriptions.py \
  --dataset cifar100 --data_root $WORK/data --backend local \
  --local_model Qwen/Qwen2.5-3B-Instruct \
  --out descriptions_cifar100_real.json

python3 src/semantic_graph/build_graph.py \
  --descriptions descriptions_cifar100_real.json \
  --dataset cifar100 --data_root $WORK/data \
  --top_k 2 --out graph_cifar100_real.json
```

Use `--dataset cifar10` (Top-2) or `--dataset robotcar` (Top-1) for the other graphs. The description step needs a GPU.

**3. Train (Part 1, CIFAR)**

```bash
sbatch.tinygpu jobs/train_bcl_cifar.sbatch
sbatch.tinygpu jobs/train_concutmix_cifar.sbatch
sbatch.tinygpu jobs/train_skcl_cifar.sbatch
sbatch.tinygpu jobs/train_bcl_cifar10.sbatch
```

**4. Train (Part 2, day → night)**

```bash
sbatch.tinygpu jobs/train_bcl_robotcar.sbatch
sbatch.tinygpu jobs/train_skcl_robotcar.sbatch
```

Check jobs with `squeue.tinygpu -u $USER`. Logs are written to `logs/`, and each run's summary to `results/<run>/log.txt`.

## Data

- **CIFAR-10 / CIFAR-100** are downloaded automatically by torchvision into `$WORK/data`. The long-tailed versions (β = 100: 10,847 and 12,406 training images) are created on the fly.
- **Oxford RobotCar** and **ROAD** are released for academic use only and require registration: <https://robotcar-dataset.robots.ox.ac.uk> and the ROAD dataset page. The extracted crops are therefore not redistributed here. The extraction keeps every 12th frame of each annotated object track, drops boxes smaller than 20 px, resizes crops to 64 × 64, and splits by whole drive (16 day-train, 5 day-test, 2 night-test drives).

## Key settings

- Backbone: CIFAR ResNet-32 for all methods; 200 epochs on CIFAR, 100 epochs on the driving data.
- SKCL loss: L = CE(fine) + CE(coarse) + λ · L_SKCL with λ = 0.5 (not specified in the paper).
- Descriptions: Qwen2.5-3B-Instruct with the paper's prompt, ≤ 300 words; embeddings: Sentence-BERT.
- Hardware: NVIDIA RTX 3080 and A100 GPUs, PyTorch 2.7.1, SLURM on TinyGPU.

## References

- Liu, Yang & Wang (2025). *Long-Tailed Classification with Multi-Granularity Semantics.* ICCV.
- Zhu et al. (2022). *Balanced Contrastive Learning for Long-Tailed Visual Recognition.* CVPR.
- Pan et al. (2024). *Enhanced Long-Tailed Recognition with Contrastive CutMix Augmentation.* IEEE TIP.
- Maddern et al. (2017). *1 Year, 1000 km: The Oxford RobotCar Dataset.* IJRR.
- Singh et al. (2023). *ROAD: The Road Event Awareness Dataset for Autonomous Driving.* IEEE TPAMI.

## License

Code written for this project: MIT (see `LICENSE`). The `ConCutMix/` folder keeps the license of the original ConCutMix repository.
