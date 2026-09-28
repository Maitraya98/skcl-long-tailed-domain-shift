import os, sys, torch, numpy as np, torchvision, torchvision.transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "models")
from skcl_model import SKCLModelCIFAR
from resnet_cifar import BCLModelCIFAR

WORK = os.environ["WORK"]
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
test = torchvision.datasets.CIFAR100(f"{WORK}/data", train=False, download=False,
        transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
names = test.classes

def load(kind, path):
    sd = torch.load(path, map_location="cpu")
    if kind == "SKCL":
        m = SKCLModelCIFAR(num_fine=100, num_coarse=20, feat_dim=sd["contrast_head.3.weight"].shape[0])
    else:
        m = BCLModelCIFAR(num_classes=100, feat_dim=sd["head.3.weight"].shape[0])
    m.load_state_dict(sd)
    return m.eval()

models = {"BCL":  load("BCL",  "results/bcl_cifar100_lt100/best.pt"),
          "SKCL": load("SKCL", "results/skcl_cifar100_lt100/best.pt")}

counts = np.array([int(500 * 0.01 ** (i / 99)) for i in range(100)])
group = np.where(counts > 100, "Many", np.where(counts >= 20, "Medium", "Few"))

loader = torch.utils.data.DataLoader(test, batch_size=500)
for name, m in models.items():
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(m(x)[1].argmax(1)); labels.append(y)
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    ok = p == y
    parts = "  ".join(f"{g}: {100*ok[group[y]==g].mean():.1f}" for g in ["Many", "Medium", "Few"])
    print(f"{name:5s} Top-1: {100*ok.mean():.2f}%   |  {parts}")

few_classes = np.where(group == "Few")[0]
idx = [i for i in np.random.RandomState(3).permutation(len(test)) if test.targets[i] in few_classes][:8]
fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
for ax, i in zip(axes.flat, idx):
    x, y = test[i]
    with torch.no_grad():
        pb = models["BCL"](x[None])[1].argmax(1).item()
        ps = models["SKCL"](x[None])[1].argmax(1).item()
    ax.imshow(test.data[i]); ax.axis("off")
    ax.set_title(f"true: {names[y]}\nBCL: {names[pb]}\nSKCL: {names[ps]}", fontsize=10,
                 color="green" if ps == y else "black")
plt.suptitle("Rare (Few) classes: fewer than 20 training images each", fontsize=12)
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/demo_few_predictions.png", dpi=150)
print("Saved image: results/figures/demo_few_predictions.png")
