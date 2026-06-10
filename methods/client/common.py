from __future__ import annotations

import copy
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.misc import state_dict_to_cpu


def maybe_subsample_indices(indices: Sequence[int], ratio: float, seed: int) -> List[int]:
    if len(indices) == 0:
        return []
    if ratio >= 1.0:
        return list(indices)
    ratio = max(0.0, min(1.0, float(ratio)))
    count = max(1, int(round(len(indices) * ratio)))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(np.asarray(indices), size=count, replace=False)
    return chosen.tolist()


def build_loader(dataset_bundle, indices, batch_size, num_workers, shuffle=True):
    return DataLoader(
        Subset(dataset_bundle.train_dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset_bundle.collate_train_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last = True, # Drop last batch to avoid BatchNorm issues with small batch sizes
    )


def build_loss_context(args, task_labels, num_classes: int, task_id: int, device: torch.device):
    if args.loss_mode == "partial":
        current_classes = list(task_labels[task_id])
        class_map = torch.full((num_classes,), -1, dtype=torch.long, device=device)
        for new_idx, cls in enumerate(current_classes):
            class_map[int(cls)] = int(new_idx)
        return current_classes, class_map
    return None, None


def compute_task_loss(logits, y, args, current_classes, class_map):
    if args.loss_mode == "partial":
        assert current_classes is not None and class_map is not None
        return F.cross_entropy(logits[:, current_classes], class_map[y])
    return F.cross_entropy(logits, y)


def train_standard_local_model(local_model, args, dataset_bundle, task_labels, device, task_id, train_indices):
    optimizer = optim.SGD(
        local_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    loader = build_loader(
        dataset_bundle=dataset_bundle,
        indices=train_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    current_classes, class_map = build_loss_context(
        args=args,
        task_labels=task_labels,
        num_classes=dataset_bundle.num_classes,
        task_id=task_id,
        device=device,
    )

    local_model.train()
    for _ in range(args.local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = local_model(x)
            loss = compute_task_loss(logits, y, args, current_classes, class_map)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return local_model


def clone_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return copy.deepcopy(model).to(device)


def state_from_model(model: torch.nn.Module):
    return state_dict_to_cpu(model.state_dict())