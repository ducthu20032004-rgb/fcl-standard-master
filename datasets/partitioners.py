from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import DatasetBundle


@dataclass
class PartitionBundle:
    scenario: str
    task_labels: List[List[int]]
    task_names: List[str]
    class_to_task: Optional[np.ndarray]
    test_task_indices: List[List[int]]
    client_train_indices: List[List[int]]
    client_test_indices: List[List[int]]
    client_test_task_indices: List[List[List[int]]]
    client_task_orders: List[List[int]]       # base unique order
    client_stream_orders: List[List[int]]     # actual training stream
    client_task_indices: List[List[List[int]]]
    client_seen_tasks_by_pos: List[List[List[int]]]
    client_eval_indices_by_pos: List[List[List[int]]]
    stream_length: int

    @property
    def num_tasks(self) -> int:
        return len(self.task_labels)

    @property
    def num_base_tasks(self) -> int:
        return len(self.task_labels)


def build_task_labels(
    num_classes: int,
    classes_per_task: int,
    seed: int,
    order_mode: str,
) -> List[List[int]]:
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive.")
    if num_classes % classes_per_task != 0:
        raise ValueError(
            f"num_classes={num_classes} must be divisible by classes_per_task={classes_per_task}"
        )

    if order_mode == "sequential":
        class_order = list(range(num_classes))
    elif order_mode == "random":
        rng = np.random.RandomState(seed)
        class_order = rng.permutation(np.arange(num_classes)).tolist()
    else:
        raise ValueError(f"Unsupported task label order mode: {order_mode}")

    num_tasks = num_classes // classes_per_task
    return [
        class_order[i * classes_per_task : (i + 1) * classes_per_task]
        for i in range(num_tasks)
    ]


def dirichlet_split_indices(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    allocation_mode: str,
) -> List[List[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0.0:
        raise ValueError("dirichlet alpha must be > 0.")

    rng = np.random.RandomState(seed)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for _, idxs in idx_by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        weights = rng.dirichlet(alpha * np.ones(num_clients))

        if allocation_mode == "multinomial":
            counts = rng.multinomial(len(idxs), weights)
        elif allocation_mode == "floor_remainder":
            raw = weights * len(idxs)
            counts = np.floor(raw).astype(int)
            remainder = int(len(idxs) - counts.sum())
            if remainder > 0:
                frac = raw - counts
                top = np.argsort(-frac, kind="mergesort")[:remainder]
                counts[top] += 1
        else:
            raise ValueError(f"Unsupported allocation_mode: {allocation_mode}")

        start = 0
        for client_id in range(num_clients):
            take = int(counts[client_id])
            if take > 0:
                client_indices[client_id].extend(idxs[start : start + take].tolist())
            start += take

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


def adjacent_swap_permutation(
    num_tasks: int,
    psi: float,
    seed: int,
    swap_mode: str,
) -> List[int]:
    if not 0.0 <= psi <= 1.0:
        raise ValueError("order_psi must be in [0, 1].")

    rng = np.random.RandomState(seed)
    if psi <= 0.0:
        return list(range(num_tasks))

    if swap_mode == "disjoint":
        perm = list(range(num_tasks))
        idx = 0
        while idx < num_tasks - 1:
            if rng.rand() < psi:
                perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
                idx += 2
            else:
                idx += 1
        return perm

    if swap_mode == "scan":
        if psi >= 1.0 - 1e-12:
            return rng.permutation(np.arange(num_tasks)).tolist()
        perm = list(range(num_tasks))
        for task_id in range(num_tasks - 1):
            if rng.rand() < psi:
                perm[task_id], perm[task_id + 1] = perm[task_id + 1], perm[task_id]
        return perm

    raise ValueError(f"Unsupported swap_mode: {swap_mode}")


def _build_seen_task_cache(
    client_stream_orders: List[List[int]],
    client_task_indices: List[List[List[int]]],
    test_task_indices: List[List[int]],
    eval_filter_empty_tasks: bool,
):
    num_clients = len(client_stream_orders)
    stream_length = len(client_stream_orders[0]) if client_stream_orders else 0

    client_seen_tasks_by_pos = [[[] for _ in range(stream_length)] for _ in range(num_clients)]
    client_eval_indices_by_pos = [[[] for _ in range(stream_length)] for _ in range(num_clients)]

    for client_id in range(num_clients):
        seen = []
        for pos in range(stream_length):
            task_id = int(client_stream_orders[client_id][pos])
            has_train_data = len(client_task_indices[client_id][task_id]) > 0
            if (not eval_filter_empty_tasks) or has_train_data:
                if task_id not in seen:
                    seen.append(task_id)

            client_seen_tasks_by_pos[client_id][pos] = list(seen)

            eval_indices: List[int] = []
            for seen_task_id in seen:
                eval_indices.extend(test_task_indices[seen_task_id])
            client_eval_indices_by_pos[client_id][pos] = sorted(set(eval_indices))

    return client_seen_tasks_by_pos, client_eval_indices_by_pos


def _extend_stream_orders(
    base_orders: List[List[int]],
    stream_length: int,
    repeat_mode: str,
    seed: int,
) -> List[List[int]]:
    if len(base_orders) == 0:
        return []

    num_base_tasks = len(base_orders[0])
    if stream_length <= num_base_tasks:
        return [order[:stream_length] for order in base_orders]

    streams = []
    for client_id, base_order in enumerate(base_orders):
        base_order = list(base_order)
        stream = list(base_order)
        rng = np.random.RandomState(seed + 1000 + client_id)

        while len(stream) < stream_length:
            if repeat_mode == "cycle":
                stream.extend(base_order)
            elif repeat_mode == "random":
                stream.append(int(rng.choice(np.asarray(base_order))))
            else:
                raise ValueError(f"Unsupported domain_repeat_mode: {repeat_mode}")
        streams.append(stream[:stream_length])

    return streams


def _partition_domain_tasks(
    train_targets: np.ndarray,
    train_task_ids: np.ndarray,
    num_base_tasks: int,
    num_clients: int,
    alpha: float,
    seed: int,
    allocation_mode: str,
) -> List[List[List[int]]]:
    client_task_indices = [[[] for _ in range(num_base_tasks)] for _ in range(num_clients)]

    for task_id in range(num_base_tasks):
        domain_indices = np.where(train_task_ids == task_id)[0]
        ys = train_targets[domain_indices]
        split = dirichlet_split_indices(
            y=ys,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed + 10000 + task_id,
            allocation_mode=allocation_mode,
        )
        for client_id in range(num_clients):
            local_positions = np.asarray(split[client_id], dtype=np.int64)
            if local_positions.size == 0:
                client_task_indices[client_id][task_id] = []
            else:
                client_task_indices[client_id][task_id] = domain_indices[local_positions].tolist()

    return client_task_indices


def build_partition(dataset_bundle: DatasetBundle, args) -> PartitionBundle:
    scenario = str(args.scenario)

    if scenario in {"class-il", "task-il"}:
        task_labels = build_task_labels(
            num_classes=dataset_bundle.num_classes,
            classes_per_task=args.classes_per_task,
            seed=args.seed,
            order_mode=args.task_label_order,
        )
        num_base_tasks = len(task_labels)
        class_to_task = np.zeros((dataset_bundle.num_classes,), dtype=np.int64)
        for task_id, classes in enumerate(task_labels):
            for cls in classes:
                class_to_task[int(cls)] = int(task_id)

        test_task_indices = []
        for task_id in range(num_base_tasks):
            mask = np.isin(dataset_bundle.test_targets, task_labels[task_id])
            test_task_indices.append(np.where(mask)[0].tolist())

        client_train_indices = dirichlet_split_indices(
            y=dataset_bundle.train_targets,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            allocation_mode=args.dirichlet_allocation,
        )
        client_test_indices = dirichlet_split_indices(
            y=dataset_bundle.test_targets,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed + 9999,
            allocation_mode=args.dirichlet_allocation,
        )
        client_task_indices = [[[] for _ in range(num_base_tasks)] for _ in range(args.num_clients)]
        for client_id in range(args.num_clients):
            idxs_client = np.asarray(client_train_indices[client_id], dtype=np.int64)
            labels_client = dataset_bundle.train_targets[idxs_client]
            for task_id in range(num_base_tasks):
                mask = np.isin(labels_client, task_labels[task_id])
                client_task_indices[client_id][task_id] = idxs_client[mask].tolist()
        client_test_task_indices = [[[] for _ in range(num_base_tasks)] for _ in range(args.num_clients)]
        for client_id in range(args.num_clients):
            idxs_client = np.asarray(client_test_indices[client_id], dtype=np.int64)
            labels_client = dataset_bundle.test_targets[idxs_client]
            for task_id in range(num_base_tasks):
                mask = np.isin(labels_client, task_labels[task_id])
                client_test_task_indices[client_id][task_id] = idxs_client[mask].tolist()
        client_task_orders = [
            adjacent_swap_permutation(
                num_tasks=num_base_tasks,
                psi=args.order_psi,
                seed=args.seed + 123 + client_id,
                swap_mode=args.schedule_swap_mode,
            )
            for client_id in range(args.num_clients)
        ]
        client_stream_orders = [list(order) for order in client_task_orders]
        stream_length = num_base_tasks
        task_names = [f"task_{task_id}" for task_id in range(num_base_tasks)]

    elif scenario == "domain-il":
        if dataset_bundle.train_task_ids is None or dataset_bundle.test_task_ids is None or dataset_bundle.task_names is None:
            raise ValueError(
                "domain-il requires dataset_bundle.train_task_ids, test_task_ids, and task_names."
            )

        ordered_domain_ids = list(range(len(dataset_bundle.task_names)))
        if args.task_label_order == "random":
            rng = np.random.RandomState(args.seed)
            ordered_domain_ids = rng.permutation(np.asarray(ordered_domain_ids)).tolist()

        task_names = [dataset_bundle.task_names[domain_id] for domain_id in ordered_domain_ids]
        task_labels = [list(range(dataset_bundle.num_classes)) for _ in ordered_domain_ids]
        num_base_tasks = len(task_names)
        class_to_task = None

        test_task_indices = []
        for domain_id in ordered_domain_ids:
            mask = dataset_bundle.test_task_ids == int(domain_id)
            test_task_indices.append(np.where(mask)[0].tolist())

        reordered_train_task_ids = np.zeros_like(dataset_bundle.train_task_ids)
        reordered_test_task_ids = np.zeros_like(dataset_bundle.test_task_ids)
        for new_task_id, domain_id in enumerate(ordered_domain_ids):
            reordered_train_task_ids[dataset_bundle.train_task_ids == int(domain_id)] = int(new_task_id)
            reordered_test_task_ids[dataset_bundle.test_task_ids == int(domain_id)] = int(new_task_id)

        client_task_indices = _partition_domain_tasks(
            train_targets=dataset_bundle.train_targets,
            train_task_ids=reordered_train_task_ids,
            num_base_tasks=num_base_tasks,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            allocation_mode=args.dirichlet_allocation,
        )

        client_train_indices = []
        for client_id in range(args.num_clients):
            union_indices = []
            for task_id in range(num_base_tasks):
                union_indices.extend(client_task_indices[client_id][task_id])
            client_train_indices.append(sorted(union_indices))

        client_task_orders = [
            adjacent_swap_permutation(
                num_tasks=num_base_tasks,
                psi=args.order_psi,
                seed=args.seed + 123 + client_id,
                swap_mode=args.schedule_swap_mode,
            )
            for client_id in range(args.num_clients)
        ]

        requested_stream_length = int(args.num_tasks) if args.num_tasks is not None else num_base_tasks
        client_stream_orders = _extend_stream_orders(
            base_orders=client_task_orders,
            stream_length=requested_stream_length,
            repeat_mode=args.domain_repeat_mode,
            seed=args.seed,
        )
        stream_length = requested_stream_length

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    client_seen_tasks_by_pos, client_eval_indices_by_pos = _build_seen_task_cache(
        client_stream_orders=client_stream_orders,
        client_task_indices=client_task_indices,
        test_task_indices=test_task_indices,
        eval_filter_empty_tasks=args.eval_filter_empty_tasks,
    )

    return PartitionBundle(
        scenario=scenario,
        task_labels=task_labels,
        task_names=task_names,
        class_to_task=class_to_task,
        test_task_indices=test_task_indices,
        client_train_indices=client_train_indices,
        client_test_indices=client_test_indices,
        client_test_task_indices=client_test_task_indices,
        client_task_orders=client_task_orders,
        client_stream_orders=client_stream_orders,
        client_task_indices=client_task_indices,
        client_seen_tasks_by_pos=client_seen_tasks_by_pos,
        client_eval_indices_by_pos=client_eval_indices_by_pos,
        stream_length=stream_length,
    )


def summarize_partition(
    partition: PartitionBundle,
    train_targets: np.ndarray,
    rounds_per_task: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "scenario": partition.scenario,
        "num_base_tasks": partition.num_base_tasks,
        "stream_length": partition.stream_length,
        "task_labels": [list(map(int, classes)) for classes in partition.task_labels],
        "task_names": list(partition.task_names),
        "clients": [],
    }

    for client_id, task_order in enumerate(partition.client_task_orders):
        client_entry: Dict[str, Any] = {
            "client_id": int(client_id),
            "task_order": list(map(int, task_order)),
            "stream_order": list(map(int, partition.client_stream_orders[client_id])),
            "tasks": [],
        }
        for order_pos, task_id in enumerate(task_order):
            indices = partition.client_task_indices[client_id][task_id]
            counts = Counter(train_targets[np.asarray(indices, dtype=np.int64)].tolist())
            classes = partition.task_labels[task_id]
            class_counts = {str(int(cls)): int(counts.get(int(cls), 0)) for cls in classes}
            client_entry["tasks"].append(
                {
                    "order_pos": int(order_pos),
                    "task_id": int(task_id),
                    "task_name": partition.task_names[task_id],
                    "classes": list(map(int, classes)),
                    "total_samples": int(len(indices)),
                    "class_counts": class_counts,
                    "rounds_per_task": int(rounds_per_task),
                }
            )
        summary["clients"].append(client_entry)

    return summary