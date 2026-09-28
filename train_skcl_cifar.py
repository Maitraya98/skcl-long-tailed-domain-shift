"""
train_skcl_cifar.py -- full SKCL training on CIFAR-100/10-LT (Eq. 6).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "datasets"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "loss"))

from cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100, classify_many_medium_few
from cifar_lt_multiview import build_cifar_bcl_transforms
from cifar_hierarchy import get_hierarchy
from skcl_model import SKCLModelCIFAR
from skcl_loss import SKCLLoss


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_top1(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item() * 100.0


def shot_accuracy(preds, targets, cls_num_list, many_thresh=100, few_thresh=20):
    many_idx, medium_idx, few_idx = classify_many_medium_few(cls_num_list, many_thresh, few_thresh)
    many_idx, medium_idx, few_idx = set(many_idx), set(medium_idx), set(few_idx)
    counts = {"many": [0, 0], "medium": [0, 0], "few": [0, 0]}
    for p, t in zip(preds, targets):
        t = int(t)
        bucket = "many" if t in many_idx else ("few" if t in few_idx else "medium")
        counts[bucket][1] += 1
        counts[bucket][0] += int(p == t)
    out = {}
    for k, (correct, total) in counts.items():
        out[k] = 100.0 * correct / total if total else 0.0
    return out["many"], out["medium"], out["few"]


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    else:
        lr *= 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


class MultiViewCIFARWithCoarse(torch.utils.data.Dataset):
    def __init__(self, base_dataset, view_transforms, fine_to_coarse):
        self.base_dataset = base_dataset
        self.view_transforms = view_transforms
        self.fine_to_coarse = fine_to_coarse

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        from PIL import Image
        img, fine_target = self.base_dataset.data[idx], self.base_dataset.targets[idx]
        img = Image.fromarray(img)
        views = tuple(t(img) for t in self.view_transforms)
        coarse_target = self.fine_to_coarse[fine_target]
        return views, fine_target, coarse_target


def train_one_epoch(loader, model, criterion_ce_fine, criterion_ce_coarse, criterion_skcl,
                     optimizer, epoch, neighbor_lists, args):
    model.train()
    ce_meter, skcl_meter, acc_meter, time_meter = (AverageMeter() for _ in range(4))
    end = time.time()

    for i, (views, fine_targets, coarse_targets) in enumerate(loader):
        x_ce = views[0].cuda(non_blocking=True)
        x_contrast = views[1].cuda(non_blocking=True)
        fine_targets = fine_targets.cuda(non_blocking=True)
        coarse_targets = coarse_targets.cuda(non_blocking=True)

        z_bar_ce, logits_fine, logits_coarse, prototypes = model(x_ce)
        z_bar_contrast, _, _, _ = model(x_contrast)

        ce_loss_fine = criterion_ce_fine(logits_fine, fine_targets)
        ce_loss_coarse = criterion_ce_coarse(logits_coarse, coarse_targets)
        ce_loss = args.w_fine * ce_loss_fine + args.w_coarse * ce_loss_coarse

        skcl_loss = criterion_skcl(z_bar_contrast, fine_targets, prototypes, neighbor_lists)

        loss = ce_loss + args.lam * skcl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = fine_targets.shape[0]
        ce_meter.update(ce_loss.item(), bs)
        skcl_meter.update(skcl_loss.item(), bs)
        acc_meter.update(accuracy_top1(logits_fine, fine_targets), bs)
        time_meter.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f"Epoch [{epoch}][{i}/{len(loader)}]  Time {time_meter.val:.3f} ({time_meter.avg:.3f})  "
                  f"CE {ce_meter.val:.4f} ({ce_meter.avg:.4f})  SKCL {skcl_meter.val:.4f} ({skcl_meter.avg:.4f})  "
                  f"Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})")

    return ce_meter.avg, skcl_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(loader, model, cls_num_list, args):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        _, logits_fine, _, _ = model(x)
        preds = logits_fine.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
    top1 = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    many, medium, few = shot_accuracy(all_preds, all_targets, cls_num_list)
    return top1, many, medium, few


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--tau_prime", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--w_fine", type=float, default=1.0)
    parser.add_argument("--w_coarse", type=float, default=1.0)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/skcl_cifar")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.graph) as f:
        graph = json.load(f)
    n_fine, n_coarse = graph["n_fine"], graph["n_coarse"]
    neighbor_lists = graph["neighbors"][:n_fine]
    print(f"Loaded graph: {n_fine} fine + {n_coarse} coarse nodes, Top-{graph['top_k']} neighbors")

    fine_names, coarse_names, fine_to_coarse = get_hierarchy(args.dataset, args.data_root)
    assert len(fine_names) == n_fine and len(coarse_names) == n_coarse

    ds_cls = IMBALANCECIFAR100 if args.dataset == "cifar100" else IMBALANCECIFAR10
    train_base = ds_cls(root=args.data_root, imb_type="exp", imb_factor=args.imb_factor,
                         rand_number=args.seed, train=True, download=False, transform=None)
    cls_num_list = train_base.get_cls_num_list()
    print(f"{args.dataset}-LT (beta={round(1/args.imb_factor)}): "
          f"{sum(cls_num_list)} training images, head={cls_num_list[0]}, tail={cls_num_list[-1]}")

    view_transforms = build_cifar_bcl_transforms(args.dataset)
    train_dataset = MultiViewCIFARWithCoarse(train_base, view_transforms, fine_to_coarse)

    from torchvision import transforms as T
    from cifar_lt import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if args.dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_ds_cls = __import__("torchvision").datasets.CIFAR100 if args.dataset == "cifar100" \
        else __import__("torchvision").datasets.CIFAR10
    test_dataset = test_ds_cls(root=args.data_root, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = SKCLModelCIFAR(num_fine=n_fine, num_coarse=n_coarse, feat_dim=args.feat_dim).cuda()

    from logitadjust import LogitAdjust
    coarse_cls_num_list = [0] * n_coarse
    for fine_idx, coarse_idx in enumerate(fine_to_coarse):
        coarse_cls_num_list[coarse_idx] += cls_num_list[fine_idx]
    criterion_ce_fine = LogitAdjust(cls_num_list).cuda()
    criterion_ce_coarse = LogitAdjust(coarse_cls_num_list).cuda()
    criterion_skcl = SKCLLoss(tau=args.tau, tau_prime=args.tau_prime).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    best_top1 = 0.0
    log_path = os.path.join(args.out_dir, "log.txt")

    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        ce_avg, skcl_avg, train_acc = train_one_epoch(
            train_loader, model, criterion_ce_fine, criterion_ce_coarse, criterion_skcl,
            optimizer, epoch, neighbor_lists, args
        )
        line = f"Epoch {epoch}  lr={lr:.5f}  CE={ce_avg:.4f}  SKCL={skcl_avg:.4f}  TrainAcc={train_acc:.2f}"

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            top1, many, medium, few = evaluate(test_loader, model, cls_num_list, args)
            line += f"  || Test Top1={top1:.2f}  Many={many:.2f}  Medium={medium:.2f}  Few={few:.2f}"
            if top1 > best_top1:
                best_top1 = top1
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    print(f"\nBest Top-1: {best_top1:.2f}")
    with open(log_path, "a") as f:
        f.write(f"\nBest Top-1: {best_top1:.2f}\n")


if __name__ == "__main__":
    main()
