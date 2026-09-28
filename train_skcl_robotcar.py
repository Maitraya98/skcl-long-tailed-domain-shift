"""
train_skcl_robotcar.py -- SKCL training on RobotCar, day_train only,
evaluated on both day_test and night_test.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "robotcar"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "loss"))

from robotcar_lt import RobotCarLT, CLASS_NAMES, get_robotcar_transforms
from robotcar_multiview import MultiViewRobotCarWithCoarse, build_robotcar_bcl_transforms
from robotcar_hierarchy import get_robotcar_hierarchy
from skcl_model import SKCLModelCIFAR
from skcl_loss import SKCLLoss
from logitadjust import LogitAdjust


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


def per_class_accuracy(preds, targets, num_classes):
    correct = [0] * num_classes
    total = [0] * num_classes
    for p, t in zip(preds, targets):
        total[t] += 1
        correct[t] += int(p == t)
    return [100.0 * c / t if t else 0.0 for c, t in zip(correct, total)]


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    else:
        lr *= 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


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
def evaluate(loader, model, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        _, logits_fine, _, _ = model(x)
        preds = logits_fine.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
    top1 = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    per_class = per_class_accuracy(all_preds, all_targets, num_classes)
    return top1, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
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
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/skcl_robotcar")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.graph) as f:
        graph = json.load(f)
    n_fine, n_coarse = graph["n_fine"], graph["n_coarse"]
    neighbor_lists = graph["neighbors"][:n_fine]
    print(f"Loaded graph: {n_fine} fine + {n_coarse} coarse nodes, Top-{graph['top_k']} neighbors")

    fine_names, coarse_names, fine_to_coarse = get_robotcar_hierarchy()
    assert fine_names == CLASS_NAMES, f"Hierarchy fine classes {fine_names} must match RobotCarLT's {CLASS_NAMES}"
    assert len(fine_names) == n_fine and len(coarse_names) == n_coarse

    train_base = RobotCarLT(args.manifest, split="day_train", transform=None)
    cls_num_list = train_base.get_cls_num_list()
    print(f"day_train: {sum(cls_num_list)} images, per-class {dict(zip(CLASS_NAMES, cls_num_list))}")

    view_transforms = build_robotcar_bcl_transforms(args.crop_size)
    train_dataset = MultiViewRobotCarWithCoarse(train_base, view_transforms[:2], fine_to_coarse)

    _, test_transform = get_robotcar_transforms(args.crop_size)
    day_test_dataset = RobotCarLT(args.manifest, split="day_test", transform=test_transform)
    night_test_dataset = RobotCarLT(args.manifest, split="night_test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    day_test_loader = DataLoader(day_test_dataset, batch_size=256, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    night_test_loader = DataLoader(night_test_dataset, batch_size=256, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

    model = SKCLModelCIFAR(num_fine=n_fine, num_coarse=n_coarse, feat_dim=args.feat_dim).cuda()

    coarse_cls_num_list = [0] * n_coarse
    for fine_idx, coarse_idx in enumerate(fine_to_coarse):
        coarse_cls_num_list[coarse_idx] += cls_num_list[fine_idx]
    criterion_ce_fine = LogitAdjust(cls_num_list).cuda()
    criterion_ce_coarse = LogitAdjust(coarse_cls_num_list).cuda()
    criterion_skcl = SKCLLoss(tau=args.tau, tau_prime=args.tau_prime).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    best_day_top1 = 0.0
    log_path = os.path.join(args.out_dir, "log.txt")

    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        ce_avg, skcl_avg, train_acc = train_one_epoch(
            train_loader, model, criterion_ce_fine, criterion_ce_coarse, criterion_skcl,
            optimizer, epoch, neighbor_lists, args
        )
        line = f"Epoch {epoch}  lr={lr:.5f}  CE={ce_avg:.4f}  SKCL={skcl_avg:.4f}  TrainAcc={train_acc:.2f}"

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            day_top1, day_per_class = evaluate(day_test_loader, model, n_fine)
            night_top1, night_per_class = evaluate(night_test_loader, model, n_fine)
            domain_gap = day_top1 - night_top1
            line += (f"  || Day Top1={day_top1:.2f} {dict(zip(CLASS_NAMES, [round(x,1) for x in day_per_class]))}"
                      f"  || Night Top1={night_top1:.2f} {dict(zip(CLASS_NAMES, [round(x,1) for x in night_per_class]))}"
                      f"  || DomainGap={domain_gap:.2f}")
            if day_top1 > best_day_top1:
                best_day_top1 = day_top1
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    print(f"\nBest Day Top-1: {best_day_top1:.2f}")
    with open(log_path, "a") as f:
        f.write(f"\nBest Day Top-1: {best_day_top1:.2f}\n")


if __name__ == "__main__":
    main()
