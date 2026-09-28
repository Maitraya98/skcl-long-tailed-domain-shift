import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "loss"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))

from skcl_model import SKCLModelCIFAR
from skcl_loss import SKCLLoss
from logitadjust import LogitAdjust
from cifar_hierarchy import get_cifar10_hierarchy


def test_cifar10_hierarchy_consistency():
    fine_names, coarse_names, fine_to_coarse = get_cifar10_hierarchy()
    assert len(fine_names) == 10
    assert len(coarse_names) == 2
    assert len(fine_to_coarse) == 10
    assert all(0 <= c < 2 for c in fine_to_coarse)
    print(f"[OK] CIFAR-10 hierarchy: {len(fine_names)} fine -> {len(coarse_names)} coarse, mapping={fine_to_coarse}")


def test_full_skcl_training_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fine, n_coarse = 10, 2
    batch_size = 8
    feat_dim = 32

    model = SKCLModelCIFAR(num_fine=n_fine, num_coarse=n_coarse, feat_dim=feat_dim).to(device)

    fine_cls_num_list = [500, 400, 300, 200, 150, 100, 80, 50, 30, 10]
    coarse_cls_num_list = [1000, 1220]

    criterion_ce_fine = LogitAdjust(fine_cls_num_list).to(device)
    criterion_ce_coarse = LogitAdjust(coarse_cls_num_list).to(device)
    criterion_skcl = SKCLLoss(tau=0.1, tau_prime=0.2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    x_ce = torch.randn(batch_size, 3, 32, 32, device=device)
    x_contrast = torch.randn(batch_size, 3, 32, 32, device=device)
    fine_targets = torch.randint(0, n_fine, (batch_size,), device=device)
    fine_to_coarse = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    coarse_targets = torch.tensor([fine_to_coarse[t] for t in fine_targets.cpu().tolist()], device=device)

    neighbor_lists = [[(i + 1) % 12, (i + 5) % 12] for i in range(n_fine)]

    z_bar_ce, logits_fine, logits_coarse, prototypes = model(x_ce)
    z_bar_contrast, _, _, _ = model(x_contrast)

    assert logits_fine.shape == (batch_size, n_fine)
    assert logits_coarse.shape == (batch_size, n_coarse)
    assert prototypes.shape == (n_fine + n_coarse, feat_dim)

    ce_loss_fine = criterion_ce_fine(logits_fine, fine_targets)
    ce_loss_coarse = criterion_ce_coarse(logits_coarse, coarse_targets)
    ce_loss = 1.0 * ce_loss_fine + 1.0 * ce_loss_coarse

    skcl_loss = criterion_skcl(z_bar_contrast, fine_targets, prototypes, neighbor_lists)
    loss = ce_loss + 0.5 * skcl_loss

    assert not torch.isnan(loss)
    assert loss.item() > 0

    optimizer.zero_grad()
    loss.backward()

    encoder_grad_norm = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
    fc_fine_grad_norm = sum(p.grad.norm().item() for p in model.fc_fine.parameters() if p.grad is not None)
    fc_coarse_grad_norm = sum(p.grad.norm().item() for p in model.fc_coarse.parameters() if p.grad is not None)
    proto_fine_grad_norm = sum(p.grad.norm().item() for p in model.proto_head_fine.parameters() if p.grad is not None)

    assert encoder_grad_norm > 0
    assert fc_fine_grad_norm > 0
    assert fc_coarse_grad_norm > 0
    assert proto_fine_grad_norm > 0

    optimizer.step()
    print(f"[OK] full SKCL step ({device}): CE_fine={ce_loss_fine.item():.4f}  CE_coarse={ce_loss_coarse.item():.4f}  "
          f"SKCL={skcl_loss.item():.4f}  total={loss.item():.4f}")
    print(f"     gradient norms: encoder={encoder_grad_norm:.4f}  fc_fine={fc_fine_grad_norm:.4f}  "
          f"fc_coarse={fc_coarse_grad_norm:.4f}  proto_fine={proto_fine_grad_norm:.4f}")


def test_coarse_label_aggregation():
    cls_num_list = [500, 400, 300, 200, 150, 100, 80, 50, 30, 10]
    fine_to_coarse = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    n_coarse = 2
    coarse_cls_num_list = [0] * n_coarse
    for fine_idx, coarse_idx in enumerate(fine_to_coarse):
        coarse_cls_num_list[coarse_idx] += cls_num_list[fine_idx]
    assert coarse_cls_num_list[0] == 500 + 400 + 300 + 200
    assert coarse_cls_num_list[1] == 150 + 100 + 80 + 50 + 30 + 10
    assert sum(coarse_cls_num_list) == sum(cls_num_list)
    print(f"[OK] coarse aggregation: {coarse_cls_num_list}, sums match total ({sum(cls_num_list)})")


if __name__ == "__main__":
    tests = [test_cifar10_hierarchy_consistency, test_full_skcl_training_step, test_coarse_label_aggregation]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
