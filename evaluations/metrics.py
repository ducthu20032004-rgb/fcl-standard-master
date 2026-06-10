from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.base import DatasetBundle
from datasets.partitioners import PartitionBundle


def _build_loader(dataset, indices, batch_size, num_workers, collate_fn):
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def _predict_with_task_mask(logits: torch.Tensor, allowed_classes: Sequence[int]) -> torch.Tensor:
    class_ids = torch.tensor(list(allowed_classes), device=logits.device, dtype=torch.long)
    local_pred = logits[:, class_ids].argmax(dim=1)
    return class_ids[local_pred]


@torch.no_grad()
def _eval_task_accuracy(
    model: torch.nn.Module,
    dataset_bundle: DatasetBundle,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    allowed_classes: Sequence[int] | None = None,
) -> Tuple[float, int, int]:
    if len(indices) == 0:
        return 0.0, 0, 0

    model.eval()
    loader = _build_loader(
        dataset=dataset_bundle.test_dataset,
        indices=indices,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset_bundle.collate_test_fn,
    )

    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        if allowed_classes is None:
            pred = logits.argmax(dim=1)
        else:
            pred = _predict_with_task_mask(logits, allowed_classes)

        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return float(correct / max(total, 1)), correct, total


@torch.no_grad()
def eval_taskwise_accuracy(
    model: torch.nn.Module,
    dataset_bundle: DatasetBundle,
    partition: PartitionBundle,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_acc = []
    task_correct = []
    task_total = []

    for task_id in range(partition.num_tasks):
        if partition.scenario == "task-il":
            allowed_classes = partition.task_labels[task_id]
        else:
            allowed_classes = None

        acc, correct, total = _eval_task_accuracy(
            model=model,
            dataset_bundle=dataset_bundle,
            indices=partition.test_task_indices[task_id],
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            allowed_classes=allowed_classes,
        )
        task_acc.append(acc)
        task_correct.append(correct)
        task_total.append(total)

    return (
        np.asarray(task_acc, dtype=np.float32),
        np.asarray(task_correct, dtype=np.int64),
        np.asarray(task_total, dtype=np.int64),
    )


def compute_client_first_avg_acc(
    current_task_acc: np.ndarray,
    client_seen_tasks: Sequence[Sequence[int]],
) -> float:
    per_client = []
    for seen in client_seen_tasks:
        if len(seen) == 0:
            per_client.append(0.0)
        else:
            per_client.append(float(np.mean(current_task_acc[list(seen)])))
    return float(np.mean(per_client))


def compute_client_first_forgetting(
    task_acc_history: np.ndarray,
    client_seen_tasks: Sequence[Sequence[int]],
) -> float:
    current_task_acc = task_acc_history[-1]
    max_up_to_now = task_acc_history.max(axis=0)
    forgetting_per_task = max_up_to_now - current_task_acc

    per_client = []
    for seen in client_seen_tasks:
        if len(seen) <= 1:
            per_client.append(0.0)
        else:
            per_client.append(float(np.mean(forgetting_per_task[list(seen[:-1])])))
    return float(np.mean(per_client))


@torch.no_grad()
def eval_accuracy_over_tasks(
    model: torch.nn.Module,
    dataset_bundle: DatasetBundle,
    partition: PartitionBundle,
    task_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    if len(task_ids) == 0:
        return 0.0

    weighted_correct = 0
    weighted_total = 0
    for task_id in task_ids:
        allowed_classes = partition.task_labels[task_id] if partition.scenario == "task-il" else None
        _, correct, total = _eval_task_accuracy(
            model=model,
            dataset_bundle=dataset_bundle,
            indices=partition.test_task_indices[task_id],
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            allowed_classes=allowed_classes,
        )
        weighted_correct += int(correct)
        weighted_total += int(total)

    return float(weighted_correct / max(weighted_total, 1))


def compute_local_global_gap(
    global_model: torch.nn.Module,
    local_model_states: Dict[int, Dict[str, torch.Tensor]],
    model_builder,
    dataset_bundle: DatasetBundle,
    partition: PartitionBundle,
    client_seen_tasks: Sequence[Sequence[int]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    gaps = []
    for client_id, seen_tasks in enumerate(client_seen_tasks):
        if len(seen_tasks) == 0:
            gaps.append(0.0)
            continue

        global_acc = eval_accuracy_over_tasks(
            model=global_model,
            dataset_bundle=dataset_bundle,
            partition=partition,
            task_ids=seen_tasks,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        local_model = model_builder().to(device)
        local_model.load_state_dict(local_model_states[client_id], strict=True)
        local_acc = eval_accuracy_over_tasks(
            model=local_model,
            dataset_bundle=dataset_bundle,
            partition=partition,
            task_ids=seen_tasks,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        gaps.append(local_acc - global_acc)

    return float(np.mean(gaps))