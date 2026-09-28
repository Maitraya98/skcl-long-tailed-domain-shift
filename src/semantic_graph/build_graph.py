"""
build_graph.py -- Sections 3.3-3.4: SBERT embeddings, cosine similarity,
Top-K neighbor selection.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datasets"))
from cifar_hierarchy import get_hierarchy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "robotcar"))


def get_dataset_hierarchy(dataset, data_root):
    if dataset == "robotcar":
        from robotcar_hierarchy import get_robotcar_hierarchy
        return get_robotcar_hierarchy()
    return get_hierarchy(dataset, data_root)


def build_similarity_matrix(descriptions: dict, node_order: list, model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [descriptions[name] for name in node_order]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / norms
    S = embeddings_normed @ embeddings_normed.T
    return embeddings_normed, S


def top_k_neighbors(S: np.ndarray, k: int) -> list:
    n = S.shape[0]
    neighbors = []
    for i in range(n):
        row = S[i].copy()
        row[i] = -np.inf
        top_k_idx = np.argsort(-row)[:k]
        neighbors.append(top_k_idx.tolist())
    return neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "robotcar"], required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    with open(args.descriptions) as f:
        descriptions = json.load(f)

    fine_names, coarse_names, fine_to_coarse = get_dataset_hierarchy(args.dataset, args.data_root)
    node_order = list(fine_names) + list(coarse_names)
    n_fine = len(fine_names)

    missing = [c for c in node_order if c not in descriptions]
    if missing:
        raise ValueError(f"Missing descriptions for: {missing}")

    embeddings, S = build_similarity_matrix(descriptions, node_order, args.sbert_model)
    neighbors = top_k_neighbors(S, args.top_k)

    print(f"Built {S.shape[0]}x{S.shape[0]} similarity matrix ({n_fine} fine + {len(coarse_names)} coarse)")
    print(f"Example: {node_order[0]!r} top-{args.top_k}: {[node_order[j] for j in neighbors[0]]} "
          f"(sims: {[round(float(S[0, j]), 3) for j in neighbors[0]]})")

    out = {
        "node_order": node_order, "n_fine": n_fine, "n_coarse": len(coarse_names),
        "fine_to_coarse": fine_to_coarse, "similarity_matrix": S.tolist(),
        "top_k": args.top_k, "neighbors": neighbors,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Saved graph to {args.out}")


if __name__ == "__main__":
    main()
