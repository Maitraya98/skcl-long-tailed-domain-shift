"""
train_bcl_robotcar.py -- BCL training on RobotCar, day_train only,
evaluated on both day_test and night_test.
"""
from __future__ import annotations
import argparse
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
from robotcar_multiview import MultiViewRobotCar, build_robotcar_bcl_transforms
from resnet_cifar import BCLModelCIFAR
from contrastive import BalSCL
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


def train_one_epoch(loader, model, criterion_ce, criterion_scl, optimizer, epoch, args):
    model.train()
    ce_meter, scl_meter, acc_meter, time_meter = (AverageMeter() for _ in range(4))
    end = time.time()

    for i, (views, targets) in enumerate(loader):
        inputs = torch.cat([views[0], views[1], views[2]], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = targets.shape[0]

        feat_mlp, logits, centers = model(inputs)
        centers = centers[: args.num_classes]
        _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
        features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        logits_ce, _, _ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)

        scl_loss = criterion_scl(centers, features, targets)
        ce_loss = criterion_ce(logits_ce, targets)
        loss = args.alpha * ce_loss + args.beta * scl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ce_meter.update(ce_loss.item(), batch_size)
        scl_meter.update(scl_loss.item(), batch_size)
        acc_meter.update(accuracy_top1(logits_ce, targets), batch_size)
        time_meter.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f"Epoch [{epoch}][{i}/{len(loader)}]  Time {time_meter.val:.3f} ({time_meter.avg:.3f})  "
                  f"CE {ce_meter.val:.4f} ({ce_meter.avg:.4f})  SCL {scl_meter.val:.4f} ({scl_meter.avg:.4f})  "
                  f"Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})")

    return ce_meter.avg, scl_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(loader, model, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        _, logits, _ = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
    top1 = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    per_class = per_class_accuracy(all_preds, all_targets, num_classes)
    return top1, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/bcl_robotcar")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)
    args.num_classes = len(CLASS_NAMES)

    train_base = RobotCarLT(args.manifest, split="day_train", transform=None)
    cls_num_list = train_base.get_cls_num_list()
    print(f"day_train: {sum(cls_num_list)} images, per-class {dict(zip(CLASS_NAMES, cls_num_list))}")

    view_transforms = build_robotcar_bcl_transforms(args.crop_size)
    train_dataset = MultiViewRobotCar(train_base, view_transforms)

    _, test_transform = get_robotcar_transforms(args.crop_size)
    day_test_dataset = RobotCarLT(args.manifest, split="day_test", transform=test_transform)
    night_test_dataset = RobotCarLT(args.manifest, split="night_test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    day_test_loader = DataLoader(day_test_dataset, batch_size=256, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    night_test_loader = DataLoader(night_test_dataset, batch_size=256, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

    model = BCLModelCIFAR(num_classes=args.num_classes, feat_dim=args.feat_dim).cuda()
    criterion_ce = LogitAdjust(cls_num_list).cuda()
    criterion_scl = BalSCL(cls_num_list, args.temp).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    best_day_top1 = 0.0
    log_path = os.path.join(args.out_dir, "log.txt")

    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        ce_avg, scl_avg, train_acc = train_one_epoch(
            train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args
        )
        line = f"Epoch {epoch}  lr={lr:.5f}  CE={ce_avg:.4f}  SCL={scl_avg:.4f}  TrainAcc={train_acc:.2f}"

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            day_top1, day_per_class = evaluate(day_test_loader, model, args.num_classes)
            night_top1, night_per_class = evaluate(night_test_loader, model, args.num_classes)
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
