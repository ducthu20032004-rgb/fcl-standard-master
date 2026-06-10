from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except ImportError as exc:
    raise ImportError(
        "This script requires torch-geometric. Please install torch-geometric first."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Download Cora through PyG Planetoid.")
    parser.add_argument("--output-dir", type=str, default="data/cora")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = Planetoid(
        root=str(out_root),
        name="Cora",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0]

    stats = {
        "dataset": "Cora",
        "root": str(out_root),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "num_train_mask": int(data.train_mask.sum().item()),
        "num_val_mask": int(data.val_mask.sum().item()),
        "num_test_mask": int(data.test_mask.sum().item()),
    }

    with (out_root / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[Cora] Download and preprocessing completed.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()