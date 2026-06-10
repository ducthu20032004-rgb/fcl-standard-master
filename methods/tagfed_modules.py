from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


class GroupLogitHead(nn.Module):
    """
    Lightweight server-side group model.

    Paper gốc distill trên uploaded feature maps + logits theo group task.
    Trong dense adaptation này, server group model là 1 linear head nhận
    feature vector từ client backbone và học CE + KD theo đúng tinh thần Eq.(1).
    """
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def maybe_subsample_indices(indices: Sequence[int], ratio: float, seed: int) -> List[int]:
    indices = list(indices)
    if len(indices) == 0:
        return []
    ratio = float(max(0.0, min(1.0, ratio)))
    if ratio >= 1.0:
        return indices
    rng = np.random.RandomState(seed)
    count = max(1, int(round(len(indices) * ratio)))
    chosen = rng.choice(np.asarray(indices), size=count, replace=False)
    return chosen.astype(np.int64).tolist()


def build_message_loader(dataset_bundle, indices, batch_size: int, num_workers: int):
    subset = Subset(dataset_bundle.train_dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_bundle.collate_train_fn,
        pin_memory=torch.cuda.is_available(),
    )


def build_tagfed_message(
    model: torch.nn.Module,
    dataset_bundle,
    train_indices,
    ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """
    Client -> server message in TagFed-style adaptation:
    features + logits + labels from a subset of current-task local data.
    """
    sampled_indices = maybe_subsample_indices(train_indices, ratio=ratio, seed=seed)
    if len(sampled_indices) == 0:
        return None

    loader = build_message_loader(
        dataset_bundle=dataset_bundle,
        indices=sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model.eval()
    features_list = []
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            features = model.extract_features(x)
            logits = model.forward_from_features(features)

            features_list.append(features.detach().cpu())
            logits_list.append(logits.detach().cpu())
            labels_list.append(y.detach().cpu())

    if len(features_list) == 0:
        return None

    return {
        "features": torch.cat(features_list, dim=0).contiguous(),
        "logits": torch.cat(logits_list, dim=0).contiguous(),
        "labels": torch.cat(labels_list, dim=0).long().contiguous(),
    }


def build_group_train_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    batch_size: int,
):
    # save memory by moving tensors to CPU and making them contiguous before creating dataset/loader
    dataset = TensorDataset(
        features.detach().cpu().contiguous(),
        labels.detach().cpu().long().contiguous(),
        logits.detach().cpu().contiguous(),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )