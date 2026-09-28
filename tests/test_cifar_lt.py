"""
test_cifar_lt.py — synthetic tests, no network required.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))
from cifar_lt import _LongTailedCIFARMixin, classify_many_medium_few


class _FakeBalancedDataset:
    def __init__(self, num_classes: int, per_class: int):
        n = num_classes * per_class
        self.data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
        self.targets = []
        for c in range(num_classes):
            self.targets.extend([c] * per_class)
        perm = np.random.permutation(n)
        self.data = self.data[perm]
        self.targets = list(np.array(self.targets)[perm])


class FakeIMBALANCECIFAR(_LongTailedCIFARMixin, _FakeBalancedDataset):
    def __init__(self, num_classes, per_class, **kwargs):
        self.cls_num = num_classes
        super().__init__(num_classes, per_class, **kwargs)


def test_exp_imbalance_endpoints():
    ds = FakeIMBALANCECIFAR(num_classes=100, per_class=500, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert counts[0] == 500
    assert 4 <= counts[-1] <= 5
    print(f"[OK] exp imbalance endpoints: head={counts[0]}, tail={counts[-1]}")


def test_monotonically_decreasing():
    ds = FakeIMBALANCECIFAR(num_classes=100, per_class=500, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    print("[OK] monotonically decreasing across all 100 classes")


def test_total_sample_count_matches_data_shape():
    ds = FakeIMBALANCECIFAR(num_classes=100, per_class=500, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert len(ds.data) == sum(counts)
    assert len(ds.targets) == sum(counts)
    print(f"[OK] total sample count matches: {sum(counts)} images")


def test_cifar10_beta100():
    ds = FakeIMBALANCECIFAR(num_classes=10, per_class=5000, imb_type="exp", imb_factor=0.01)
    counts = ds.get_cls_num_list()
    assert counts[0] == 5000
    assert 49 <= counts[-1] <= 50
    print(f"[OK] CIFAR-10-LT beta=100: head={counts[0]}, tail={counts[-1]}, total={sum(counts)}")


def test_many_medium_few_split():
    cls_num_list = [200] * 20 + [50] * 30 + [10] * 50
    many, medium, few = classify_many_medium_few(cls_num_list)
    assert len(many) == 20 and len(medium) == 30 and len(few) == 50
    print(f"[OK] many/medium/few split: {len(many)}/{len(medium)}/{len(few)}")


def test_step_imbalance():
    ds = FakeIMBALANCECIFAR(num_classes=10, per_class=1000, imb_type="step", imb_factor=0.1)
    counts = ds.get_cls_num_list()
    assert counts[:5] == [1000] * 5
    assert counts[5:] == [100] * 5
    print(f"[OK] step imbalance: {counts}")


def test_seed_reproducibility():
    ds1 = FakeIMBALANCECIFAR(num_classes=10, per_class=500, imb_type="exp", imb_factor=0.01, rand_number=42)
    ds2 = FakeIMBALANCECIFAR(num_classes=10, per_class=500, imb_type="exp", imb_factor=0.01, rand_number=42)
    assert ds1.get_cls_num_list() == ds2.get_cls_num_list()
    print("[OK] seeded runs produce identical per-class counts")


if __name__ == "__main__":
    tests = [test_exp_imbalance_endpoints, test_monotonically_decreasing,
             test_total_sample_count_matches_data_shape, test_cifar10_beta100,
             test_many_medium_few_split, test_step_imbalance, test_seed_reproducibility]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
