from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from validation_config import (
    DATASET_DEFAULT_BATCH_SIZE,
    DATASET_DEFAULT_CLASSES_PER_TASK,
    DATASET_DEFAULT_EVAL_BATCH_SIZE,
)
from validation_utils import (
    choose_classes_per_task,
    compute_delta_order,
    compute_jlabel,
    ensure_repo_on_path,
    flatten_state_dict_delta,
    mean_pairwise_cosine,
)

ensure_repo_on_path()

from backbones import build_backbone
from datasets import build_dataset
from evaluations import (
    compute_client_first_avg_acc,
    compute_client_first_forgetting,
    compute_local_global_gap,
    eval_taskwise_accuracy,
)
from methods import build_method
from utils import resolve_device, set_seed, state_dict_to_cpu


@dataclass
class ValidationPartition:
    task_labels: List[List[int]]
    class_to_task: np.ndarray
    test_task_indices: List[List[int]]
    client_train_indices: List[List[int]]
    client_task_orders: List[List[int]]
    client_task_indices: List[List[List[int]]]
    client_seen_tasks_by_pos: List[List[List[int]]]
    client_eval_indices_by_pos: List[List[List[int]]]

    @property
    def num_tasks(self) -> int:
        return len(self.task_labels)


def make_args(
    dataset: str,
    method: str,
    seed: int,
    alpha: float,
    psi: float,
    classes_per_task: Optional[int] = None,
    rounds_per_task: int = 50,
    local_epochs: int = 1,
    client_fraction: float = 1.0,
    num_clients: int = 5,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    task_label_order: str = "random",
    dirichlet_allocation: str = "floor_remainder",
    schedule_swap_mode: str = "scan",
    loss_mode: str = "full",
    device: str = "auto",
    download: bool = True,
    data_root: str = "data",
    output_root: str = "outputs",
    num_workers: int = 0,
    eval_filter_empty_tasks: bool = True,
    use_cifar100_tensor_cache: bool = True,
    _label_constructor: str = "default",
    _order_constructor: str = "default",
    _order_match_target: Optional[float] = None,
    _label_match_target: Optional[float] = None,
    _label_search_grid: Optional[List[float]] = None,
    _order_search_grid: Optional[List[float]] = None,
    _block_search_grid: Optional[List[int]] = None,
    **overrides,
):
    if batch_size is None:
        batch_size = DATASET_DEFAULT_BATCH_SIZE.get(dataset, 128)
    if eval_batch_size is None:
        eval_batch_size = DATASET_DEFAULT_EVAL_BATCH_SIZE.get(dataset, 256)

    args = dict(
        dataset=dataset,
        method=method,
        backbone="auto",
        data_root=data_root,
        output_root=output_root,
        download=download,
        device=device,
        log_level="INFO",
        seed=int(seed),
        num_clients=int(num_clients),
        classes_per_task=classes_per_task,
        dirichlet_alpha=float(alpha),
        order_psi=float(psi),
        task_label_order=task_label_order,
        dirichlet_allocation=dirichlet_allocation,
        schedule_swap_mode=schedule_swap_mode,
        rounds_per_task=int(rounds_per_task),
        client_fraction=float(client_fraction),
        local_epochs=int(local_epochs),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        num_workers=int(num_workers),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        loss_mode=loss_mode,
        eval_filter_empty_tasks=bool(eval_filter_empty_tasks),
        use_cifar100_tensor_cache=bool(use_cifar100_tensor_cache),
        _label_constructor=_label_constructor,
        _order_constructor=_order_constructor,
        _order_match_target=_order_match_target,
        _label_match_target=_label_match_target,
        _label_search_grid=_label_search_grid,
        _order_search_grid=_order_search_grid,
        _block_search_grid=_block_search_grid,
        fedprox_mu=0.01,
        fedala_top_p=2,
        fedala_weight_lr=1.0,
        fedala_sample_ratio=0.8,
        fedala_init_epochs=1,
        fedala_adapt_epochs=1,
        fedas_align_epochs=1,
        fedas_align_lr=0.01,
        fedas_align_ratio=1.0,
        fedas_fim_ratio=0.5,
        fedl2p_hidden_dim=100,
        fedl2p_max_lr_scale=5.0,
        fedl2p_meta_lr=1e-3,
        fedl2p_meta_steps=1,
        fedl2p_train_ratio=0.8,
        fedl2p_stats_ratio=0.5,
        target_client_kd_weight=25.0,
        target_client_kd_temperature=2.0,
        target_client_replay_batch_size=64,
        target_syn_rounds=10,
        target_g_steps=10,
        target_kd_steps=400,
        target_warmup_rounds=20,
        target_synthesis_batch_size=256,
        target_sample_batch_size=256,
        target_latent_dim=256,
        target_generator_lr=0.002,
        target_noise_lr=0.01,
        target_student_lr=0.2,
        target_ce_weight=0.5,
        target_div_weight=1.0,
        target_bn_weight=10.0,
        target_generator_kd_temperature=20.0,
        target_divergence_mask="official",
        target_use_fomaml=True,
        target_bn_momentum=0.9,
        target_max_replay_images=8000,
        enable_repeat_tasks=False,
        repeat_backtracking_region=0,
        repeat_backtracking_time=0,
        repeat_sampling="uniform",
        tagfed_alpha_c=0.9,
        tagfed_beta_c=0.1,
        tagfed_alpha_s=0.9,
        tagfed_beta_s=0.1,
        tagfed_temperature=5.0,
        tagfed_server_epochs=20,
        tagfed_server_lr=0.2,
        tagfed_server_batch_size=128,
        tagfed_message_ratio=1.0,
        tagfed_retrain_groups=2,
        fedewc_lambda=10.0,
        fedewc_fisher_max_samples=256,
        fedlwf_lambda=1.0,
        fedlwf_temperature=2.0,
        fedderpp_alpha=0.5,
        fedderpp_beta=0.5,
        fedderpp_buffer_size=500,
        fedderpp_store_per_task=200,
        fedderpp_replay_batch_size=64,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def build_task_labels(num_classes: int, classes_per_task: int, seed: int, order_mode: str) -> List[List[int]]:
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive.")
    if num_classes % classes_per_task != 0:
        raise ValueError(
            f"num_classes={num_classes} must be divisible by classes_per_task={classes_per_task}"
        )
    rng = np.random.RandomState(seed)
    if order_mode == "sequential":
        class_order = list(range(num_classes))
    elif order_mode == "random":
        class_order = rng.permutation(np.arange(num_classes)).tolist()
    else:
        raise ValueError(f"Unsupported task_label_order: {order_mode}")
    num_tasks = num_classes // classes_per_task
    return [class_order[i * classes_per_task : (i + 1) * classes_per_task] for i in range(num_tasks)]


def dirichlet_split_indices(y: np.ndarray, num_clients: int, alpha: float, seed: int, allocation_mode: str) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    idx_by_class = {int(c): np.where(y == c)[0] for c in np.unique(y)}
    client_indices = [[] for _ in range(num_clients)]
    for _, idxs in idx_by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        weights = rng.dirichlet(alpha * np.ones(num_clients))
        if allocation_mode == "multinomial":
            counts = rng.multinomial(len(idxs), weights)
        else:
            raw = weights * len(idxs)
            counts = np.floor(raw).astype(int)
            remainder = int(len(idxs) - counts.sum())
            if remainder > 0:
                frac = raw - counts
                top = np.argsort(-frac, kind="mergesort")[:remainder]
                counts[top] += 1
        start = 0
        for client_id in range(num_clients):
            take = int(counts[client_id])
            if take > 0:
                client_indices[client_id].extend(idxs[start : start + take].tolist())
            start += take
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])
    return client_indices


def sparse_balanced_split_indices(y: np.ndarray, num_clients: int, subset_fraction: float, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    idx_by_class = {int(c): np.where(y == c)[0] for c in np.unique(y)}
    client_indices = [[] for _ in range(num_clients)]
    subset_size = max(1, int(round(num_clients * subset_fraction)))
    subset_size = min(subset_size, num_clients)

    for _, idxs in idx_by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        chosen_clients = rng.choice(np.arange(num_clients), size=subset_size, replace=False)
        chunks = np.array_split(idxs, subset_size)
        for local_idx, client_id in enumerate(chosen_clients.tolist()):
            client_indices[client_id].extend(chunks[local_idx].tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])
    return client_indices


def adjacent_swap_permutation(num_tasks: int, psi: float, seed: int, mode: str) -> List[int]:
    rng = np.random.RandomState(seed)
    perm = list(range(num_tasks))
    if num_tasks <= 1 or psi <= 0.0:
        return perm

    if mode == "disjoint":
        idx = 0
        while idx < num_tasks - 1:
            if rng.rand() < psi:
                perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
                idx += 2
            else:
                idx += 1
        return perm

    if mode == "scan":
        if psi >= 1.0 - 1e-12:
            return rng.permutation(np.arange(num_tasks)).tolist()
        for idx in range(num_tasks - 1):
            if rng.rand() < psi:
                perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
        return perm

    raise ValueError(f"Unsupported schedule_swap_mode: {mode}")


def rank_jitter_permutation(num_tasks: int, noise_scale: float, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    base = np.arange(num_tasks, dtype=float)
    scores = base + rng.normal(loc=0.0, scale=noise_scale, size=num_tasks)
    return np.argsort(scores, kind="mergesort").tolist()


def block_shuffle_permutation(num_tasks: int, block_size: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    block_size = max(1, min(block_size, num_tasks))
    blocks = [list(range(i, min(i + block_size, num_tasks))) for i in range(0, num_tasks, block_size)]
    rng.shuffle(blocks)
    return [task for block in blocks for task in block]


def _build_client_task_indices(train_targets: np.ndarray, client_train_indices: List[List[int]], task_labels: List[List[int]]):
    num_clients = len(client_train_indices)
    num_tasks = len(task_labels)
    client_task_indices = [[[] for _ in range(num_tasks)] for _ in range(num_clients)]
    for client_id in range(num_clients):
        idxs = np.asarray(client_train_indices[client_id], dtype=np.int64)
        ys = train_targets[idxs]
        for task_id, classes in enumerate(task_labels):
            mask = np.isin(ys, np.asarray(classes, dtype=np.int64))
            client_task_indices[client_id][task_id] = idxs[mask].tolist()
    return client_task_indices


def _build_seen_eval_cache(client_task_orders, client_task_indices, test_task_indices, eval_filter_empty_tasks):
    num_clients = len(client_task_orders)
    num_tasks = len(client_task_orders[0]) if num_clients > 0 else 0
    client_seen_tasks_by_pos = [[[] for _ in range(num_tasks)] for _ in range(num_clients)]
    client_eval_indices_by_pos = [[[] for _ in range(num_tasks)] for _ in range(num_clients)]

    for client_id in range(num_clients):
        seen = []
        for pos in range(num_tasks):
            task_id = int(client_task_orders[client_id][pos])
            has_train = len(client_task_indices[client_id][task_id]) > 0
            if (not eval_filter_empty_tasks) or has_train:
                seen.append(task_id)
            unique_seen = []
            for t in seen:
                if t not in unique_seen:
                    unique_seen.append(t)
            client_seen_tasks_by_pos[client_id][pos] = list(unique_seen)
            eval_indices = []
            for t in unique_seen:
                eval_indices.extend(test_task_indices[t])
            client_eval_indices_by_pos[client_id][pos] = sorted(set(eval_indices))
    return client_seen_tasks_by_pos, client_eval_indices_by_pos


def _test_task_indices(test_targets: np.ndarray, task_labels: List[List[int]]) -> List[List[int]]:
    indices = []
    for classes in task_labels:
        mask = np.isin(test_targets, np.asarray(classes, dtype=np.int64))
        indices.append(np.where(mask)[0].tolist())
    return indices


def _class_to_task(num_classes: int, task_labels: List[List[int]]) -> np.ndarray:
    mapping = np.zeros((num_classes,), dtype=np.int64)
    for task_id, classes in enumerate(task_labels):
        for c in classes:
            mapping[int(c)] = int(task_id)
    return mapping


def build_custom_partition(dataset_bundle, args) -> tuple[ValidationPartition, Dict[str, float | str]]:
    num_classes = int(dataset_bundle.num_classes)
    classes_per_task = choose_classes_per_task(
        dataset_name=args.dataset,
        num_classes=num_classes,
        configured=args.classes_per_task,
    )
    task_labels = build_task_labels(
        num_classes=num_classes,
        classes_per_task=classes_per_task,
        seed=int(args.seed),
        order_mode=str(args.task_label_order),
    )
    test_task_indices = _test_task_indices(dataset_bundle.test_targets, task_labels)
    class_to_task = _class_to_task(num_classes, task_labels)

    label_constructor = getattr(args, "_label_constructor", "default")
    order_constructor = getattr(args, "_order_constructor", "default")

    if label_constructor == "default":
        client_train_indices = dirichlet_split_indices(
            y=dataset_bundle.train_targets,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            allocation_mode=args.dirichlet_allocation,
        )
    elif label_constructor == "sparse_balanced":
        grid = getattr(args, "_label_search_grid", None) or list(np.linspace(0.15, 1.0, 12))
        target_j = getattr(args, "_label_match_target", None)
        best = None
        best_gap = float("inf")
        for frac in grid:
            tmp = sparse_balanced_split_indices(
                y=dataset_bundle.train_targets,
                num_clients=args.num_clients,
                subset_fraction=float(frac),
                seed=args.seed,
            )
            gap = 0.0 if target_j is None else abs(compute_jlabel(dataset_bundle.train_targets, tmp, num_classes) - target_j)
            if gap < best_gap:
                best_gap = gap
                best = tmp
        client_train_indices = best
    else:
        raise ValueError(f"Unknown label constructor: {label_constructor}")

    client_task_indices = _build_client_task_indices(
        train_targets=dataset_bundle.train_targets,
        client_train_indices=client_train_indices,
        task_labels=task_labels,
    )

    num_tasks = len(task_labels)

    if order_constructor == "default":
        client_task_orders = [
            adjacent_swap_permutation(
                num_tasks=num_tasks,
                psi=args.order_psi,
                seed=args.seed + 123 + client_id,
                mode=args.schedule_swap_mode,
            )
            for client_id in range(args.num_clients)
        ]
    elif order_constructor == "rank_jitter":
        grid = getattr(args, "_order_search_grid", None) or list(np.linspace(0.0, max(1.0, num_tasks / 2), 12))
        target_delta = getattr(args, "_order_match_target", None)
        best = None
        best_gap = float("inf")
        for sigma in grid:
            orders = [
                rank_jitter_permutation(num_tasks=num_tasks, noise_scale=float(sigma), seed=args.seed + 321 + client_id)
                for client_id in range(args.num_clients)
            ]
            gap = 0.0 if target_delta is None else abs(compute_delta_order(orders, client_task_indices) - target_delta)
            if gap < best_gap:
                best_gap = gap
                best = orders
        client_task_orders = best
    elif order_constructor == "block_shuffle":
        grid = getattr(args, "_block_search_grid", None) or list(range(2, min(6, num_tasks + 1)))
        target_delta = getattr(args, "_order_match_target", None)
        best = None
        best_gap = float("inf")
        for block_size in grid:
            orders = [
                block_shuffle_permutation(num_tasks=num_tasks, block_size=int(block_size), seed=args.seed + 654 + client_id)
                for client_id in range(args.num_clients)
            ]
            gap = 0.0 if target_delta is None else abs(compute_delta_order(orders, client_task_indices) - target_delta)
            if gap < best_gap:
                best_gap = gap
                best = orders
        client_task_orders = best
    else:
        raise ValueError(f"Unknown order constructor: {order_constructor}")

    client_seen_tasks_by_pos, client_eval_indices_by_pos = _build_seen_eval_cache(
        client_task_orders=client_task_orders,
        client_task_indices=client_task_indices,
        test_task_indices=test_task_indices,
        eval_filter_empty_tasks=args.eval_filter_empty_tasks,
    )

    partition = ValidationPartition(
        task_labels=task_labels,
        class_to_task=class_to_task,
        test_task_indices=test_task_indices,
        client_train_indices=client_train_indices,
        client_task_orders=client_task_orders,
        client_task_indices=client_task_indices,
        client_seen_tasks_by_pos=client_seen_tasks_by_pos,
        client_eval_indices_by_pos=client_eval_indices_by_pos,
    )
    metadata = {
        "label_constructor": label_constructor,
        "order_constructor": order_constructor,
        "j_label": compute_jlabel(dataset_bundle.train_targets, client_train_indices, num_classes),
        "delta_order": compute_delta_order(client_task_orders, client_task_indices),
        "num_tasks": partition.num_tasks,
        "classes_per_task": classes_per_task,
    }
    return partition, metadata


def run_single_benchmark(
    dataset: str,
    method: str,
    seed: int,
    alpha: float,
    psi: float,
    label_constructor: str = "default",
    order_constructor: str = "default",
    matched_jlabel: Optional[float] = None,
    matched_delta: Optional[float] = None,
    extra_args: Optional[Dict] = None,
):
    extra_args = extra_args or {}
    set_seed(int(seed))
    args = make_args(
        dataset=dataset,
        method=method,
        seed=seed,
        alpha=alpha,
        psi=psi,
        _label_constructor=label_constructor,
        _order_constructor=order_constructor,
        _label_match_target=matched_jlabel,
        _order_match_target=matched_delta,
        **extra_args,
    )
    device = resolve_device(args.device)

    from datasets import list_datasets
    from methods import list_methods

    if dataset not in list_datasets():
        raise RuntimeError(f"Dataset '{dataset}' is not implemented in the current repo.")
    if method not in list_methods():
        raise RuntimeError(f"Method '{method}' is not implemented in the current repo.")

    dataset_bundle = build_dataset(args.dataset, args)

    if args.backbone == "auto":
        args.backbone = dataset_bundle.default_backbone

    if args.classes_per_task is None:
        args.classes_per_task = choose_classes_per_task(
            dataset_name=args.dataset,
            num_classes=int(dataset_bundle.num_classes),
            configured=DATASET_DEFAULT_CLASSES_PER_TASK.get(args.dataset, None),
        )

    partition, part_meta = build_custom_partition(dataset_bundle=dataset_bundle, args=args)

    server_cls, client_cls = build_method(args.method)
    model_builder = lambda: build_backbone(args.backbone, dataset_bundle.num_classes, args)
    global_model = model_builder().to(device)

    server = server_cls(args=args, device=device)
    if hasattr(server, "register_context"):
        server.register_context(
            model_builder=model_builder,
            dataset_bundle=dataset_bundle,
            partition=partition,
            logger=None,
        )

    client = client_cls(
        args=args,
        dataset_bundle=dataset_bundle,
        task_labels=partition.task_labels,
        device=device,
    )

    local_model_states = {
        client_id: state_dict_to_cpu(global_model.state_dict()) for client_id in range(args.num_clients)
    }

    task_acc_history = []
    round_rows = []
    total_rounds = partition.num_tasks * args.rounds_per_task
    global_rng = np.random.RandomState(args.seed)

    for task_pos in range(partition.num_tasks):
        if hasattr(server, "on_task_start"):
            server.on_task_start(global_model=global_model, task_pos=task_pos, partition=partition)

        current_task_for_client = [
            partition.client_task_orders[client_id][task_pos] for client_id in range(args.num_clients)
        ]

        for round_in_task in range(args.rounds_per_task):
            global_round = task_pos * args.rounds_per_task + round_in_task + 1
            num_selected = max(1, int(args.client_fraction * args.num_clients))
            selected_clients = global_rng.choice(np.arange(args.num_clients), size=num_selected, replace=False)

            local_updates = []
            payload = server.get_client_payload(global_model) if hasattr(server, "get_client_payload") else global_model

            global_state_cpu = state_dict_to_cpu(global_model.state_dict())
            delta_vectors = []

            for client_id in selected_clients:
                task_id = current_task_for_client[int(client_id)]
                train_indices = partition.client_task_indices[int(client_id)][task_id]
                update = client.fit(
                    global_payload=payload,
                    client_id=int(client_id),
                    task_id=int(task_id),
                    train_indices=train_indices,
                )
                if update is None:
                    continue
                local_updates.append(update)
                local_state = update.personalized_state_dict if update.personalized_state_dict is not None else update.state_dict
                local_model_states[int(client_id)] = local_state
                delta_vectors.append(flatten_state_dict_delta(local_state, global_state_cpu))

            update_alignment = mean_pairwise_cosine(delta_vectors)

            if len(local_updates) > 0:
                global_model = server.aggregate(global_model=global_model, local_updates=local_updates)

            task_accs, task_correct, task_total = eval_taskwise_accuracy(
                model=global_model,
                dataset_bundle=dataset_bundle,
                class_to_task=partition.class_to_task,
                num_tasks=partition.num_tasks,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
            )
            task_acc_history.append(task_accs)
            history_array = np.stack(task_acc_history, axis=0)

            client_seen_tasks = [
                partition.client_seen_tasks_by_pos[client_id][task_pos]
                for client_id in range(args.num_clients)
            ]
            client_eval_indices = [
                partition.client_eval_indices_by_pos[client_id][task_pos]
                for client_id in range(args.num_clients)
            ]

            avg_acc = compute_client_first_avg_acc(
                current_task_acc=task_accs,
                client_seen_tasks=client_seen_tasks,
            )
            forgetting = compute_client_first_forgetting(
                task_acc_history=history_array,
                client_seen_tasks=client_seen_tasks,
            )
            local_global_gap = compute_local_global_gap(
                global_task_correct=task_correct,
                global_task_total=task_total,
                local_model_states=local_model_states,
                model_builder=model_builder,
                dataset_bundle=dataset_bundle,
                client_seen_tasks=client_seen_tasks,
                client_eval_indices=client_eval_indices,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
            )

            round_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "psi": float(psi),
                    "label_constructor": label_constructor,
                    "order_constructor": order_constructor,
                    "global_round": int(global_round),
                    "task_pos": int(task_pos),
                    "round_in_task": int(round_in_task),
                    "avg_acc": float(avg_acc),
                    "forgetting": float(forgetting),
                    "local_global_gap": float(local_global_gap),
                    "update_alignment": float(update_alignment) if np.isfinite(update_alignment) else np.nan,
                    "j_label": float(part_meta["j_label"]),
                    "delta_order": float(part_meta["delta_order"]),
                }
            )

        if hasattr(server, "on_task_end"):
            server.on_task_end(global_model=global_model, task_pos=task_pos, partition=partition)

    rounds_df = pd.DataFrame(round_rows)
    if len(rounds_df) == 0:
        raise RuntimeError("No rounds were executed.")

    final_row = rounds_df.iloc[-1].to_dict()
    summary = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "alpha": float(alpha),
        "psi": float(psi),
        "label_constructor": label_constructor,
        "order_constructor": order_constructor,
        "avg_acc": float(final_row["avg_acc"]),
        "forgetting": float(final_row["forgetting"]),
        "local_global_gap": float(final_row["local_global_gap"]),
        "update_alignment_mean": float(rounds_df["update_alignment"].mean(skipna=True)),
        "j_label": float(part_meta["j_label"]),
        "delta_order": float(part_meta["delta_order"]),
        "num_tasks": int(partition.num_tasks),
        "classes_per_task": int(part_meta["classes_per_task"]),
        "matched_jlabel_target": float(matched_jlabel) if matched_jlabel is not None else np.nan,
        "matched_delta_target": float(matched_delta) if matched_delta is not None else np.nan,
    }
    return summary, rounds_df


def safe_run_single_benchmark(**kwargs):
    try:
        summary, rounds_df = run_single_benchmark(**kwargs)
        summary["status"] = "ok"
        summary["error"] = ""
        return summary, rounds_df
    except Exception as exc:
        summary = {k: kwargs.get(k, np.nan) for k in ["dataset", "method", "seed", "alpha", "psi"]}
        summary.update(
            {
                "label_constructor": kwargs.get("label_constructor", "default"),
                "order_constructor": kwargs.get("order_constructor", "default"),
                "status": "error",
                "error": repr(exc),
            }
        )
        return summary, pd.DataFrame()


def generate_instance_stats(
    dataset: str,
    seed: int,
    alpha: float,
    psi: float,
    label_constructor: str = "default",
    order_constructor: str = "default",
    matched_jlabel: Optional[float] = None,
    matched_delta: Optional[float] = None,
    extra_args: Optional[Dict] = None,
):
    extra_args = extra_args or {}
    args = make_args(
        dataset=dataset,
        method="fedavg",
        seed=seed,
        alpha=alpha,
        psi=psi,
        _label_constructor=label_constructor,
        _order_constructor=order_constructor,
        _label_match_target=matched_jlabel,
        _order_match_target=matched_delta,
        **extra_args,
    )
    from datasets import list_datasets

    if dataset not in list_datasets():
        raise RuntimeError(f"Dataset '{dataset}' is not implemented in the current repo.")

    dataset_bundle = build_dataset(args.dataset, args)
    if args.classes_per_task is None:
        args.classes_per_task = choose_classes_per_task(
            dataset_name=args.dataset,
            num_classes=int(dataset_bundle.num_classes),
            configured=DATASET_DEFAULT_CLASSES_PER_TASK.get(args.dataset, None),
        )

    _, part_meta = build_custom_partition(dataset_bundle=dataset_bundle, args=args)
    return {
        "dataset": dataset,
        "seed": int(seed),
        "alpha": float(alpha),
        "psi": float(psi),
        "label_constructor": label_constructor,
        "order_constructor": order_constructor,
        "j_label": float(part_meta["j_label"]),
        "delta_order": float(part_meta["delta_order"]),
        "num_tasks": int(part_meta["num_tasks"]),
        "classes_per_task": int(part_meta["classes_per_task"]),
    }


def safe_generate_instance_stats(**kwargs):
    try:
        stats = generate_instance_stats(**kwargs)
        stats["status"] = "ok"
        stats["error"] = ""
        return stats
    except Exception as exc:
        return {
            "dataset": kwargs.get("dataset"),
            "seed": kwargs.get("seed"),
            "alpha": kwargs.get("alpha"),
            "psi": kwargs.get("psi"),
            "label_constructor": kwargs.get("label_constructor", "default"),
            "order_constructor": kwargs.get("order_constructor", "default"),
            "status": "error",
            "error": repr(exc),
        }