#!/bin/bash -l
#SBATCH --job-name=skcl-smoke
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
mkdir -p logs

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module add python
conda activate skcl

echo "=== Node / GPU info ==="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python3 -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Smoke test: cached data + one training step ==="
python3 - <<'PY'
import torch, torchvision, os
from torchvision import transforms

data_root = os.environ["WORK"] + "/data"
ds = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=transforms.ToTensor())
print(f"CIFAR-100 loaded from cache: {len(ds)} images")

loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
x, y = next(iter(loader))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.resnet18(num_classes=100).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x, y = x.to(device), y.to(device)
loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward(); opt.step()
print(f"Smoke test OK. Loss: {loss.item():.4f}  Device: {device}  GPU: {torch.cuda.get_device_name(0)}")
PY

echo "=== Smoke test complete ==="
