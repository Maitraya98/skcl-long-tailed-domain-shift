"""
test_train_bcl_cifar.py

Integration test for the BCL CIFAR pipeline: model forward, both losses,
one optimizer step -- runs on CPU too (this GPU node will run it on GPU),
validates the wiring before spending real training hours on it.
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "loss"))

from cifar_lt_multiview import MultiViewCIFAR, build_cifar_bcl_transforms
from resnet_cifar import BCLModelCIFAR
from contrastive import BalSCL
from logitadjust import LogitAdjust


class _FakeBaseDataset:
    def __init__(self, num_classes=5, per_class=8):
        n = num_classes * per_class
        self.data = np.random.randint(0, 255, size=(n, 32, 32, 3), dtype=np.uint8)
        self.targets = []
        for c in range(num_classes):
            self.targets.extend([c] * per_class)
        self._cls_num_list = [per_class] * num_classes

    def get_cls_num_list(self):
        return self._cls_num_list


def test_multiview_wrapper_shapes():
    base = _FakeBaseDataset(num_classes=5, per_class=8)
    view_transforms = build_cifar_bcl_transforms("cifar100")
    ds = MultiViewCIFAR(base, view_transforms)
    views, target = ds[0]
    assert len(views) == 3, f"expected 3 views, got {len(views)}"
    assert views[0].shape == (3, 32, 32), f"unexpected view shape: {views[0].shape}"
    assert isinstance(target, (int, np.integer))
    print(f"[OK] MultiViewCIFAR: 3 views of shape {tuple(views[0].shape)}, target={target}")


def test_full_training_step():
    num_classes = 5
    batch_size = 4
    cls_num_list = [50, 40, 30, 20, 10]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BCLModelCIFAR(num_classes=num_classes, feat_dim=64).to(device)
    criterion_ce = LogitAdjust(cls_num_list).to(device)
    criterion_scl = BalSCL(cls_num_list, temperature=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    view1 = torch.randn(batch_size, 3, 32, 32, device=device)
    view2 = torch.randn(batch_size, 3, 32, 32, device=device)
    view3 = torch.randn(batch_size, 3, 32, 32, device=device)
    inputs = torch.cat([view1, view2, view3], dim=0)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    feat_mlp, logits, centers = model(inputs)
    centers = centers[:num_classes]
    _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
    features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
    logits_ce, _, _ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)

    scl_loss = criterion_scl(centers, features, targets)
    ce_loss = criterion_ce(logits_ce, targets)
    loss = 1.0 * ce_loss + 0.35 * scl_loss

    assert not torch.isnan(loss), "loss is NaN"
    assert loss.item() > 0, "loss should be positive"

    optimizer.zero_grad()
    loss.backward()

    encoder_grad_norm = sum(
        p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None
    )
    assert encoder_grad_norm > 0, "no gradient reached the encoder"

    optimizer.step()
    print(f"[OK] full training step ({device}): CE={ce_loss.item():.4f}  SCL={scl_loss.item():.4f}  "
          f"total={loss.item():.4f}  encoder_grad_norm={encoder_grad_norm:.4f}")


def test_logitadjust_portable():
    cls_num_list = [100, 50, 10]
    crit = LogitAdjust(cls_num_list)
    x = torch.randn(4, 3)
    y = torch.randint(0, 3, (4,))
    loss = crit(x, y)
    assert not torch.isnan(loss)
    assert crit.m_list.device == x.device
    print(f"[OK] LogitAdjust portable: loss={loss.item():.4f}")


if __name__ == "__main__":
    tests = [test_multiview_wrapper_shapes, test_full_training_step, test_logitadjust_portable]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
