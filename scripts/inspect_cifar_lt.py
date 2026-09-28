import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))
from cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100, classify_many_medium_few


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    cls = IMBALANCECIFAR100 if args.dataset == "cifar100" else IMBALANCECIFAR10
    ds = cls(root=args.root, imb_type="exp", imb_factor=args.imb_factor, train=True, download=False)
    cls_num_list = ds.get_cls_num_list()
    many, medium, few = classify_many_medium_few(cls_num_list)

    beta = round(1 / args.imb_factor)
    print(f"=== {args.dataset.upper()}-LT (beta={beta}) ===")
    print(f"Total training images: {sum(cls_num_list)}")
    print(f"Head class count:      {cls_num_list[0]}")
    print(f"Tail class count:      {cls_num_list[-1]}")
    print(f"Many-shot classes:     {len(many)}")
    print(f"Medium-shot classes:   {len(medium)}")
    print(f"Few-shot classes:      {len(few)}")

    ratio = cls_num_list[0] / cls_num_list[-1]
    print(f"Head/tail ratio:       {ratio:.1f} (target: {beta})")
    if abs(ratio - beta) > beta * 0.1:
        print("WARNING: head/tail ratio deviates >10% from target beta.")

    os.makedirs(args.out_dir, exist_ok=True)
    _plot_distribution(cls_num_list, args.dataset, beta, args.out_dir)


def _plot_distribution(cls_num_list, dataset_name, beta, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(cls_num_list)), cls_num_list, color="#2E5E4E", width=1.0)
    ax.set_xlabel("Class index (sorted head to tail)")
    ax.set_ylabel("Training samples")
    ax.set_title(f"{dataset_name.upper()}-LT class distribution (beta={beta})")
    ax.set_yscale("log")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset_name}_lt_beta{beta}_distribution.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved distribution plot to: {out_path}")


if __name__ == "__main__":
    main()
