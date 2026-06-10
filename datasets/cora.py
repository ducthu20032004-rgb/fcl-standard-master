from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import DatasetBundle
from .registry import register_dataset


try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except ImportError as exc:
    raise ImportError(
        "Cora support requires torch-geometric. "
        "Please install torch-geometric first."
    ) from exc


@dataclass
class CoraNodeBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    target_nodes: torch.Tensor

    def to(self, device, non_blocking: bool = False):
        return CoraNodeBatch(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            target_nodes=self.target_nodes.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self):
        return CoraNodeBatch(
            x=self.x.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            target_nodes=self.target_nodes.pin_memory(),
        )


class CoraNodeIndexDataset(Dataset):
    def __init__(self, node_ids: np.ndarray, labels: np.ndarray) -> None:
        self.node_ids = node_ids.astype(np.int64)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.node_ids)

    def __getitem__(self, idx: int):
        return int(self.node_ids[idx]), int(self.labels[idx])


@register_dataset("cora")
def build_cora(args) -> DatasetBundle:
    root = Path(args.data_root) / "cora"
    dataset = Planetoid(
        root=str(root),
        name="Cora",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0]

    if args.cora_train_pool == "train":
        train_mask = data.train_mask
    elif args.cora_train_pool == "trainval":
        train_mask = data.train_mask | data.val_mask
    elif args.cora_train_pool == "nontest":
        train_mask = ~data.test_mask
    else:
        raise ValueError(f"Unknown cora_train_pool: {args.cora_train_pool}")

    test_mask = data.test_mask

    train_node_ids = torch.where(train_mask)[0].cpu().numpy()
    test_node_ids = torch.where(test_mask)[0].cpu().numpy()

    train_labels = data.y[train_mask].cpu().numpy()
    test_labels = data.y[test_mask].cpu().numpy()

    full_x = data.x.detach().cpu().contiguous()
    edge_index = data.edge_index.detach().cpu().contiguous()

    train_dataset = CoraNodeIndexDataset(train_node_ids, train_labels)
    test_dataset = CoraNodeIndexDataset(test_node_ids, test_labels)

    def collate_fn(batch: List[Tuple[int, int]]):
        node_ids, labels = zip(*batch)
        node_ids_t = torch.tensor(node_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        batch_obj = CoraNodeBatch(
            x=full_x,
            edge_index=edge_index,
            target_nodes=node_ids_t,
        )
        return batch_obj, labels_t

    args.graph_input_dim = int(dataset.num_features)

    return DatasetBundle(
        name="cora",
        modality="graph",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=train_labels.astype(np.int64),
        test_targets=test_labels.astype(np.int64),
        num_classes=int(dataset.num_classes),
        class_names=[str(i) for i in range(int(dataset.num_classes))],
        collate_train_fn=collate_fn,
        collate_test_fn=collate_fn,
        default_backbone="cora_gcn",
        metadata={
            "graph_input_dim": int(dataset.num_features),
            "num_nodes_total": int(data.num_nodes),
            "num_edges_total": int(data.num_edges),
            "source_root": str(root),
            "train_pool": args.cora_train_pool,
        },
        train_task_ids=None,
        test_task_ids=None,
        task_names=None,
        default_scenario="class-il",
    )