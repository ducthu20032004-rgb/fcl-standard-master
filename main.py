from __future__ import annotations

import argparse
from pathlib import Path
import copy
from measure_gpu import (
    compute_feature_resnet18,
    get_resnet18_blocks,
    compute_eps,
    compute_alignment_from_arrays,
)
import csv
import numpy as np
import os
import torch
from torchvision.models import resnet18,resnet34
from backbones import build_backbone, list_backbones
from datasets import build_dataset, build_partition, list_datasets, summarize_partition
from evaluations import (
    RoundTracker,
    compute_client_first_avg_acc,
    compute_client_first_forgetting,
    compute_local_global_gap,
    eval_taskwise_accuracy,
    save_metric_curve,
)
from methods import build_method, list_methods
from utils import (
    build_setting_name,
    create_run_dirs,
    resolve_device,
    save_json,
    set_seed,
    setup_logger,
    state_dict_to_cpu,
    str2bool,
)


DATASET_DEFAULTS = {
    "mnist": {
        "batch_size": 128,
        "eval_batch_size": 256,
        "task_label_order": "sequential",
        "dirichlet_allocation": "multinomial",
        "schedule_swap_mode": "disjoint",
        "use_cifar100_tensor_cache": False,
    },
    "cifar10": {
        "batch_size": 64,
        "eval_batch_size": 256,
        "task_label_order": "sequential",
        "dirichlet_allocation": "multinomial",
        "schedule_swap_mode": "disjoint",
        "use_cifar100_tensor_cache": False,
    },
    "cifar100": {
        "batch_size": 128,
        "eval_batch_size": 256,
        "task_label_order": "random",
        "dirichlet_allocation": "floor_remainder",
        "schedule_swap_mode": "scan",
        "use_cifar100_tensor_cache": True,
    },
    "pacs": {
        "batch_size": 64,
        "eval_batch_size": 128,
        "task_label_order": "sequential",
        "dirichlet_allocation": "floor_remainder",
        "schedule_swap_mode": "scan",
        "use_cifar100_tensor_cache": False,
    },
    "domainnet": {
        "batch_size": 32,
        "eval_batch_size": 64,
        "task_label_order": "sequential",
        "dirichlet_allocation": "floor_remainder",
        "schedule_swap_mode": "scan",
        "use_cifar100_tensor_cache": False,
    },
    "thucnews": {
        "batch_size": 64,
        "eval_batch_size": 128,
        "task_label_order": "sequential",
        "dirichlet_allocation": "floor_remainder",
        "schedule_swap_mode": "disjoint",
        "use_cifar100_tensor_cache": False,
    },
    "cora": {
        "batch_size": 128,
        "eval_batch_size": 256,
        "task_label_order": "sequential",
        "dirichlet_allocation": "floor_remainder",
        "schedule_swap_mode": "disjoint",
        "use_cifar100_tensor_cache": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple and extensible FCL repo.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=list_datasets())
    parser.add_argument("--method", type=str, default="fedavg", choices=list_methods())
    parser.add_argument("--backbone", type=str, default="auto")
    parser.add_argument("--checkpoint_dir_round", type=str, default="C:\\Thu\\FCL-standard-master", help="Directory to save checkpoints for each round. If not specified, checkpoints will be saved in the main output directory.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="class-il",
        choices=["class-il", "task-il", "domain-il"],
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--domain-repeat-mode",
        type=str,
        default="cycle",
        choices=["cycle", "random"],
    )
    parser.add_argument("--use-pretrained-backbone", type=str2bool, default=False)

    # PACS
    parser.add_argument("--pacs-image-size", type=int, default=224)
    parser.add_argument("--pacs-test-ratio", type=float, default=0.1)
    parser.add_argument("--pacs-split-seed", type=int, default=2023)

    #DOMAINNET
    parser.add_argument("--domainnet-image-size", type=int, default=224)
    parser.add_argument("--domainnet-split-seed", type=int, default=2023)

    # THUCNews / text backbone
    parser.add_argument("--thucnews-max-length", type=int, default=256)
    parser.add_argument("--text-min-seq-len", type=int, default=5)
    parser.add_argument("--text-embed-dim", type=int, default=128)
    parser.add_argument("--text-num-filters", type=int, default=128)
    parser.add_argument("--text-feature-dim", type=int, default=256)
    parser.add_argument("--text-dropout", type=float, default=0.1)

    # Cora / graph backbone
    parser.add_argument(
        "--cora-train-pool",
        type=str,
        default="nontest",
        choices=["train", "trainval", "nontest"],
    )
    parser.add_argument("--gcn-hidden-dim", type=int, default=256)
    parser.add_argument("--gcn-dropout", type=float, default=0.5)


    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--download", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-level", type=str, default="INFO")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--order-psi", type=float, default=0.5)

    parser.add_argument(
        "--task-label-order",
        type=str,
        default="dataset_default",
        choices=["dataset_default", "sequential", "random"],
    )
    parser.add_argument(
        "--dirichlet-allocation",
        type=str,
        default="dataset_default",
        choices=["dataset_default", "multinomial", "floor_remainder"],
    )
    parser.add_argument(
        "--schedule-swap-mode",
        type=str,
        default="dataset_default",
        choices=["dataset_default", "disjoint", "scan"],
    )

    parser.add_argument("--rounds-per-task", type=int, default=25)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-mode", type=str, default="full", choices=["full", "partial"])

    parser.add_argument("--eval-filter-empty-tasks", type=str2bool, default=True)
    parser.add_argument("--use-cifar100-tensor-cache", type=str2bool, default=True)

    # FedProx
    parser.add_argument("--fedprox-mu", type=float, default=0.01)

    # FedALA
    parser.add_argument("--fedala-top-p", type=int, default=2)
    parser.add_argument("--fedala-weight-lr", type=float, default=1.0)
    parser.add_argument("--fedala-sample-ratio", type=float, default=0.8)
    parser.add_argument("--fedala-init-epochs", type=int, default=1)
    parser.add_argument("--fedala-adapt-epochs", type=int, default=1)

    # FedAS
    parser.add_argument("--fedas-align-epochs", type=int, default=1)
    parser.add_argument("--fedas-align-lr", type=float, default=0.01)
    parser.add_argument("--fedas-align-ratio", type=float, default=1.0)
    parser.add_argument("--fedas-fim-ratio", type=float, default=0.5)

    # FedL2P practical adaptation
    parser.add_argument("--fedl2p-hidden-dim", type=int, default=100)
    parser.add_argument("--fedl2p-max-lr-scale", type=float, default=5.0)
    parser.add_argument("--fedl2p-meta-lr", type=float, default=1e-3)
    parser.add_argument("--fedl2p-meta-steps", type=int, default=1)
    parser.add_argument("--fedl2p-train-ratio", type=float, default=0.8)
    parser.add_argument("--fedl2p-stats-ratio", type=float, default=0.5)

    # TARGET
    parser.add_argument("--target-client-kd-weight", type=float, default=25.0)
    parser.add_argument("--target-client-kd-temperature", type=float, default=2.0)
    parser.add_argument("--target-client-replay-batch-size", type=int, default=64)

    parser.add_argument("--target-syn-rounds", type=int, default=10)
    parser.add_argument("--target-g-steps", type=int, default=10)
    parser.add_argument("--target-kd-steps", type=int, default=400)
    parser.add_argument("--target-warmup-rounds", type=int, default=2)

    parser.add_argument("--target-synthesis-batch-size", type=int, default=256)
    parser.add_argument("--target-sample-batch-size", type=int, default=256)
    parser.add_argument("--target-latent-dim", type=int, default=256)

    parser.add_argument("--target-generator-lr", type=float, default=0.002)
    parser.add_argument("--target-noise-lr", type=float, default=0.01)
    parser.add_argument("--target-student-lr", type=float, default=0.2)

    parser.add_argument("--target-ce-weight", type=float, default=0.5)
    parser.add_argument("--target-div-weight", type=float, default=1.0)
    parser.add_argument("--target-bn-weight", type=float, default=10.0)
    parser.add_argument("--target-generator-kd-temperature", type=float, default=20.0)

    parser.add_argument(
        "--target-divergence-mask",
        type=str,
        default="official",
        choices=["official", "paper"],
    )
    parser.add_argument("--target-use-fomaml", type=str2bool, default=True)
    parser.add_argument("--target-bn-momentum", type=float, default=0.9)
    parser.add_argument("--target-max-replay-images", type=int, default=8000)

    # Repeat-task / TFCL schedule
    parser.add_argument("--enable-repeat-tasks", type=str2bool, default=False)
    parser.add_argument("--repeat-backtracking-region", type=int, default=0)
    parser.add_argument("--repeat-backtracking-time", type=int, default=0)
    parser.add_argument(
        "--repeat-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "recent", "cyclic"],
    )

    # TagFed
    parser.add_argument("--tagfed-alpha-c", type=float, default=0.9)
    parser.add_argument("--tagfed-beta-c", type=float, default=0.1)
    parser.add_argument("--tagfed-alpha-s", type=float, default=0.9)
    parser.add_argument("--tagfed-beta-s", type=float, default=0.1)
    parser.add_argument("--tagfed-temperature", type=float, default=5.0)

    parser.add_argument("--tagfed-server-epochs", type=int, default=20)
    parser.add_argument("--tagfed-server-lr", type=float, default=0.2)
    parser.add_argument("--tagfed-server-batch-size", type=int, default=128)

    parser.add_argument("--tagfed-message-ratio", type=float, default=1.0)
    parser.add_argument("--tagfed-retrain-groups", type=int, default=2)

    # Fed-EWC
    parser.add_argument("--fedewc-lambda", type=float, default=10.0)
    parser.add_argument("--fedewc-fisher-max-samples", type=int, default=256)

    # Fed-LwF
    parser.add_argument("--fedlwf-lambda", type=float, default=1.0)
    parser.add_argument("--fedlwf-temperature", type=float, default=2.0)

    # Fed-DER++
    parser.add_argument("--fedderpp-alpha", type=float, default=0.5)
    parser.add_argument("--fedderpp-beta", type=float, default=0.5)
    parser.add_argument("--fedderpp-buffer-size", type=int, default=500)
    parser.add_argument("--fedderpp-store-per-task", type=int, default=200)
    parser.add_argument("--fedderpp-replay-batch-size", type=int, default=64)

    # AF-FCL
    parser.add_argument("--affcl-flow-lr", type=float, default=1e-3)
    parser.add_argument("--affcl-local-iterations", type=int, default=-1)
    parser.add_argument("--affcl-flow-hidden-dim", type=int, default=512)
    parser.add_argument("--affcl-flow-layers", type=int, default=4)
    parser.add_argument("--affcl-temperature", type=float, default=2.0)

    parser.add_argument("--affcl-k-loss-flow", type=float, default=1.0)
    parser.add_argument("--affcl-k-kd-global-cls", type=float, default=1.0)
    parser.add_argument("--affcl-k-kd-last-cls", type=float, default=1.0)
    parser.add_argument("--affcl-k-kd-feature", type=float, default=1.0)
    parser.add_argument("--affcl-k-kd-output", type=float, default=1.0)
    parser.add_argument("--affcl-k-flow-lastflow", type=float, default=1.0)
    parser.add_argument("--affcl-flow-explore-theta", type=float, default=0.5)
    parser.add_argument("--affcl-use-lastflow-x", type=str2bool, default=True)

        # DDDR
    parser.add_argument("--dddr-syn-image-path", type=str, default=None)
    parser.add_argument("--dddr-require-syn-root", type=str2bool, default=True)
    parser.add_argument("--dddr-current-size", type=int, default=50)
    parser.add_argument("--dddr-prev-size", type=int, default=200)
    parser.add_argument("--dddr-replay-batch-size", type=int, default=128)

    parser.add_argument("--dddr-w-kd", type=float, default=10.0)
    parser.add_argument("--dddr-w-ce-pre", type=float, default=0.5)
    parser.add_argument("--dddr-w-scl", type=float, default=1.0)

    parser.add_argument("--dddr-temperature", type=float, default=2.0)
    parser.add_argument("--dddr-scl-temperature", type=float, default=0.07)

    parser.add_argument("--dddr-proj-hidden-dim", type=int, default=4096)
    parser.add_argument("--dddr-proj-dim", type=int, default=256)

    return parser.parse_args()


def apply_dataset_defaults(args):
    dataset_defaults = DATASET_DEFAULTS[args.dataset]
    if args.batch_size is None:
        args.batch_size = dataset_defaults["batch_size"]
    if args.eval_batch_size is None:
        args.eval_batch_size = dataset_defaults["eval_batch_size"]
    if args.task_label_order == "dataset_default":
        args.task_label_order = dataset_defaults["task_label_order"]
    if args.dirichlet_allocation == "dataset_default":
        args.dirichlet_allocation = dataset_defaults["dirichlet_allocation"]
    if args.schedule_swap_mode == "dataset_default":
        args.schedule_swap_mode = dataset_defaults["schedule_swap_mode"]
    if args.dataset != "cifar100":
        args.use_cifar100_tensor_cache = False

    # Helpful scenario defaults
    if args.scenario == "task-il" and args.loss_mode == "full":
        # task-il usually works best / most cleanly with partial training loss
        #args.loss_mode = "partial"
        args.loss_mode = "full"
    if args.dataset == "pacs" and args.scenario == "class-il":
        # PACS is intended here for domain-il benchmarking
        args.scenario = "domain-il"

    if args.dataset == "domainnet" and args.scenario == "class-il":
        # DomainNet is intended here for domain-il benchmarking
        args.scenario = "domain-il"

    return args


def log_partition(logger, partition_summary):
    logger.info("[Global tasks]")
    for task_id, classes in enumerate(partition_summary["task_labels"]):
        logger.info("  Task %s -> classes=%s", task_id, classes)

    logger.info("[Client schedules and task content]")
    for client_entry in partition_summary["clients"]:
        logger.info("Client %s | task_order=%s", client_entry["client_id"], client_entry["task_order"])
        for task_entry in client_entry["tasks"]:
            logger.info(
                "  order_pos=%s | task_id=%s | classes=%s | class_counts=%s | total_samples=%s | rounds=%s",
                task_entry["order_pos"],
                task_entry["task_id"],
                task_entry["classes"],
                task_entry["class_counts"],
                task_entry["total_samples"],
                task_entry["rounds_per_task"],
            )

def build_client_stream_orders(partition, args):
    base_orders = [list(map(int, order)) for order in partition.client_task_orders]
    if (not args.enable_repeat_tasks) or args.repeat_backtracking_region <= 0 or args.repeat_backtracking_time <= 0:
        return base_orders

    stream_orders = []
    for client_id, base_order in enumerate(base_orders):
        rng = np.random.RandomState(args.seed + 70000 + client_id)
        history = []
        stream = []
        for task_id in base_order:
            stream.append(int(task_id))
            history.append(int(task_id))

            region = history[-min(len(history), int(args.repeat_backtracking_region)) :]
            for back_idx in range(int(args.repeat_backtracking_time)):
                if len(region) == 0:
                    break
                if args.repeat_sampling == "recent":
                    repeated_task = int(region[-1])
                elif args.repeat_sampling == "cyclic":
                    repeated_task = int(region[back_idx % len(region)])
                else:
                    repeated_task = int(rng.choice(np.asarray(region)))
                stream.append(repeated_task)
        stream_orders.append(stream)
    return stream_orders


def build_stream_caches(partition, stream_orders, eval_filter_empty_tasks: bool):
    num_clients = len(stream_orders)
    stream_length = len(stream_orders[0]) if num_clients > 0 else partition.num_tasks

    client_seen_tasks_by_pos = [[[] for _ in range(stream_length)] for _ in range(num_clients)]
    client_eval_indices_by_pos = [[[] for _ in range(stream_length)] for _ in range(num_clients)]

    for client_id in range(num_clients):
        seen = []
        for pos, task_id in enumerate(stream_orders[client_id]):
            has_train_data = len(partition.client_task_indices[client_id][task_id]) > 0
            if (not eval_filter_empty_tasks) or has_train_data:
                if task_id not in seen:
                    seen.append(int(task_id))
            client_seen_tasks_by_pos[client_id][pos] = list(seen)

            eval_indices = []
            for seen_task_id in seen:
                eval_indices.extend(partition.test_task_indices[seen_task_id])
            client_eval_indices_by_pos[client_id][pos] = sorted(set(eval_indices))

    return stream_length, client_seen_tasks_by_pos, client_eval_indices_by_pos


def log_stream_orders(logger, stream_orders):
    logger.info("[Client stream orders | after repeat-task expansion]")
    for client_id, stream in enumerate(stream_orders):
        logger.info("Client %s | stream_order=%s", client_id, stream)
def save_checkpoint(state_dict, checkpoint_dir: Path, tag: str, client_id: int, task_id: int, round_in_task: int, logger):
    """tag: 'before_train' | 'after_train' | 'after_aggr'"""
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"client_{client_id}_task_{task_id}_round_{round_in_task}_{tag}.pt"
        #torch.save(state_dict, path)
        #logger.info("[CHECKPOINT][%s] client=%s | task=%s | round=%s", tag, client_id, task_id, round_in_task)
    except Exception as e:
        logger.error("[CHECKPOINT ERROR][%s] client=%s: %s", tag, client_id, str(e))
def main():
    args = parse_args()
    args = apply_dataset_defaults(args)

    repo_root = Path(__file__).resolve().parent
    args.data_root = str((repo_root / args.data_root).resolve())
    args.output_root = str((repo_root / args.output_root).resolve())

    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset_bundle = build_dataset(args.dataset, args)
    if args.backbone == "auto":
        args.backbone = dataset_bundle.default_backbone

    elif args.backbone not in list_backbones():
        raise ValueError(f"Unknown backbone '{args.backbone}'. Available: {list_backbones()}")

    if args.scenario == "domain-il" and args.dataset == "pacs":
        if dataset_bundle.train_task_ids is None or dataset_bundle.test_task_ids is None:
            raise ValueError("PACS domain-il requires train_task_ids/test_task_ids in the dataset bundle.")

    if args.scenario == "domain-il" and args.dataset == "domainnet":
        if dataset_bundle.train_task_ids is None or dataset_bundle.test_task_ids is None:
            raise ValueError("DomainNet domain-il requires train_task_ids/test_task_ids in the dataset bundle.")

    partition = build_partition(dataset_bundle, args)
    setting_name = build_setting_name(args, num_tasks=partition.stream_length)
    run_dirs = create_run_dirs(
        output_root=Path(args.output_root),
        dataset=args.dataset,
        method=args.method,
        setting_name=setting_name,
    )

    logger = setup_logger(run_dirs["logs_dir"] / "run.log", level=args.log_level)
    logger.info("=" * 100)
    logger.info("Dataset : %s", args.dataset)
    logger.info("Method  : %s", args.method)
    logger.info("Setting : %s", setting_name)
    logger.info("Device  : %s", device)
    logger.info("Output  : %s", run_dirs["run_dir"])
    logger.info("=" * 100)

    save_json(vars(args), run_dirs["run_dir"] / "args.json")
    partition_summary = summarize_partition(
        partition=partition,
        train_targets=dataset_bundle.train_targets,
        rounds_per_task=args.rounds_per_task,
    )

    client_stream_orders = build_client_stream_orders(partition, args)
    stream_length, client_seen_tasks_by_pos, client_eval_indices_by_pos = build_stream_caches(
        partition=partition,
        stream_orders=client_stream_orders,
        eval_filter_empty_tasks=bool(args.eval_filter_empty_tasks),
    )

    partition_summary["client_stream_orders"] = {
        str(client_id): stream
        for client_id, stream in enumerate(client_stream_orders)
    }
    partition_summary["stream_length"] = int(stream_length)

    save_json(partition_summary, run_dirs["tables_dir"] / "partition_summary.json")
    log_partition(logger, partition_summary)
    log_stream_orders(logger, client_stream_orders)

    server_cls, client_cls = build_method(args.method)
    model_builder = lambda: build_backbone(args.backbone, dataset_bundle.num_classes, args)
    global_model = model_builder().to(device)

    server = server_cls(args=args, device=device)
    if hasattr(server, "register_context"):
        server.register_context(
            model_builder=model_builder,
            dataset_bundle=dataset_bundle,
            partition=partition,
            logger=logger,
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

    tracker = RoundTracker()
    task_acc_history = []
    total_rounds = partition.stream_length * args.rounds_per_task
    global_rng = np.random.RandomState(args.seed)
    cp_round_dir = Path(args.checkpoint_dir_round) / f"round_checkpoints_{args.dirichlet_alpha}"
    for task_pos in range(partition.stream_length):
        current_task_for_client = [
            partition.client_stream_orders[client_id][task_pos] for client_id in range(args.num_clients)
        ]
        logger.info("\n=== Task position %s/%s ===", task_pos + 1, partition.stream_length)

        for round_in_task in range(args.rounds_per_task):
            global_round = task_pos * args.rounds_per_task + round_in_task + 1
            num_selected = max(1, int(args.client_fraction * args.num_clients))
            selected_clients = global_rng.choice(
                np.arange(args.num_clients), size=num_selected, replace=False
            )
            logger.info(
                "[Round %s/%s] task_pos=%s/%s | round_in_task=%s/%s | selected_clients=%s | current_tasks=%s",
                global_round,
                total_rounds,
                task_pos + 1,
                partition.stream_length,
                round_in_task + 1,
                args.rounds_per_task,
                selected_clients.tolist(),
                current_task_for_client,
            )

            local_updates = []
            broadcast_payload = (
                server.get_client_payload(global_model)
                if hasattr(server, "get_client_payload")
                else global_model
            )
            # ── Snapshot global model TRƯỚC khi bất kỳ client nào train (① before_train) ──
            global_state_before = state_dict_to_cpu(global_model.state_dict())
            # ── Khởi tạo container để gom drift data sau aggregate ──
            _drift_snapshots = {}   # client_id -> {"before": sd, "after_local": sd, "task_id": int}
            for client_id in selected_clients:
                task_id = current_task_for_client[int(client_id)]
                train_indices = partition.client_task_indices[int(client_id)][task_id]
                # ① Lưu checkpoint before_train
                save_checkpoint(
                    state_dict=global_state_before,
                    checkpoint_dir=cp_round_dir,
                    tag="before_train",
                    client_id=int(client_id),
                    task_id=task_id,
                    round_in_task=round_in_task,
                    logger=logger,
                )

                update = client.fit(
                    global_payload=broadcast_payload,
                    client_id=int(client_id),
                    task_id=int(task_id),
                    train_indices=train_indices,
                )

                if update is None:
                    logger.info(
                        "  - Client %s skipped because current task %s has 0 samples.",
                        client_id,
                        task_id,
                    )
                    continue
                local_state = (
                    update.personalized_state_dict
                    if update.personalized_state_dict is not None
                    else update.state_dict
                )

                # ② Lưu checkpoint after_train
                save_checkpoint(
                    state_dict=state_dict_to_cpu(local_state),
                    checkpoint_dir=cp_round_dir,
                    tag="after_train",
                    client_id=int(client_id),
                    task_id=task_id,
                    round_in_task=round_in_task,
                    logger=logger,
                )
                    # Ghi nhớ để tính drift sau aggregate
                _drift_snapshots[int(client_id)] = {
                    "before":      global_state_before,
                    "after_local": state_dict_to_cpu(local_state),
                    "task_id":     task_id,
                }
                local_updates.append(update)
                local_model_states[int(client_id)] = (
                    update.personalized_state_dict if update.personalized_state_dict is not None else update.state_dict
                )

                
                # Save client weights at the end of each task for offline drift analysis.
                if round_in_task == args.rounds_per_task - 1:
                    try:
                        checkpoint_dir = Path(run_dirs["run_dir"]) / f"checkpoints_client_{args.backbone}_hete_{args.dirichlet_alpha}"
                        
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        
                        state_to_save = (
                            update.personalized_state_dict 
                            if update.personalized_state_dict is not None 
                            else update.state_dict
                        )
                        
                        save_path = checkpoint_dir / f"client_{client_id}_task_{task_id}_round_{round_in_task}.pt"
                        torch.save(state_to_save, save_path)
                        
                        logger.info("[CHECKPOINT] client=%s | task=%s | round=%s", 
                                    client_id, task_id, round_in_task)
                    except Exception as e:
                        logger.error("[CHECKPOINT ERROR] client=%s: %s", client_id, str(e))
            

            server.aggregate(local_updates=local_updates, global_model=global_model)
            # ── ③ Snapshot AFTER aggregate + tính drift ──
            aggr_state = state_dict_to_cpu(global_model.state_dict())

            _drift_csv_path = cp_round_dir / f"drift_results_{args.dirichlet_alpha}.csv"
            _drift_fieldnames = [
                "global_round", "task_pos", "round_in_task",
                "client_id", "task_id", "block",
                "eps_trained", "eps_aggr", "eps_global",
                "cknna_trained", "cknna_aggr", "cknna_global",
            ]

            for client_id, snap in _drift_snapshots.items():
                task_id = snap["task_id"]

                #③ Lưu checkpoint after_aggr
                save_checkpoint(
                    state_dict=aggr_state,
                    checkpoint_dir=cp_round_dir,
                    tag="after_aggr",
                    client_id=client_id,
                    task_id=task_id,
                    round_in_task=round_in_task,
                    logger=logger,
                )

                # Dùng test set của toàn task để số mẫu đủ lớn cho đo drift block-wise.
                # Nếu dùng test split riêng theo client, alpha lớn sẽ khiến N nhỏ và
                # phép fit tuyến tính của eps dễ suy biến thành residual ~0 giả tạo.
                test_indices = partition.test_task_indices[task_id]
                if len(test_indices) == 0:
                    logger.warning(
                        "[DRIFT] client=%s task=%s: no test data, skip.", client_id, task_id
                    )
                    continue

                test_subset = torch.utils.data.Subset(
                    dataset_bundle.test_dataset, test_indices
                )

                def _load_model(sd):
                    m = model_builder().to(device)
                    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
                    m.eval()
                    return m

                try:
                    m_before      = _load_model(snap["before"])
                    m_after_local = _load_model(snap["after_local"])
                    m_after_aggr  = _load_model(aggr_state)
                except Exception as e:
                    logger.error("[DRIFT] load model error client=%s: %s", client_id, e)
                    continue

                num_blocks = len(get_resnet18_blocks(m_before))
                for block_idx in [4]:
                    target_layer = f"block{block_idx}"
                    try:
                        # feat_before = compute_feature_resnet18(
                        #     m_before,      client_id, test_subset, target_layer, args.seed, args
                        # )
                        # feat_local  = compute_feature_resnet18(
                        #     m_after_local, client_id, test_subset, target_layer, args.seed, args
                        # )
                        # feat_aggr   = compute_feature_resnet18(
                        #     m_after_aggr,  client_id, test_subset, target_layer, args.seed, args
                        # )

                        # eps_trained = compute_eps(feat_before, feat_local)
                        # eps_aggr    = compute_eps(feat_local,  feat_aggr)
                        # eps_global  = compute_eps(feat_before, feat_aggr)

                        # cknna_trained, _ = compute_alignment_from_arrays(
                        #     feat_before, feat_local, "mutual_knn", topk=10, precise=True
                        # )
                        # cknna_aggr, _    = compute_alignment_from_arrays(
                        #     feat_local,  feat_aggr,  "mutual_knn", topk=10, precise=True
                        # )
                        # cknna_global, _  = compute_alignment_from_arrays(
                        #     feat_before, feat_aggr,  "mutual_knn", topk=10, precise=True
                        # )

                        logger.info(
                            "[DRIFT] round=%s task_pos=%s client=%s %s | "
                            "eps: trained=%.4f aggr=%.4f global=%.4f | "
                            "cknna: trained=%.4f aggr=%.4f global=%.4f",
                            global_round, task_pos, client_id, target_layer,
                            eps_trained, eps_aggr, eps_global,
                            cknna_trained, cknna_aggr, cknna_global,
                        )

                        write_header = not _drift_csv_path.exists()
                        with open(_drift_csv_path, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=_drift_fieldnames)
                            if write_header:
                                writer.writeheader()
                            writer.writerow({
                                "global_round":  global_round,
                                "task_pos":      task_pos,
                                "round_in_task": round_in_task,
                                "client_id":     client_id,
                                "task_id":       task_id,
                                "block":         block_idx,
                                "eps_trained":   round(float(eps_trained),  6),
                                "eps_aggr":      round(float(eps_aggr),     6),
                                "eps_global":    round(float(eps_global),   6),
                                "cknna_trained": round(float(cknna_trained),6),
                                "cknna_aggr":    round(float(cknna_aggr),   6),
                                "cknna_global":  round(float(cknna_global), 6),
                            })

                    except Exception as e:
                        logger.error(
                            "[DRIFT] client=%s block%s error: %s", client_id, block_idx, e
                        )

                del m_before, m_after_local, m_after_aggr
                torch.cuda.empty_cache()
            task_accs, task_correct, task_total = eval_taskwise_accuracy(
                model=global_model,
                dataset_bundle=dataset_bundle,
                partition=partition,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
            )
            task_acc_history.append(task_accs)
            history_array = np.stack(task_acc_history, axis=0)
            client_seen_tasks = [
                client_seen_tasks_by_pos[client_id][task_pos]
                for client_id in range(args.num_clients)
            ]
            client_eval_indices = [
                client_eval_indices_by_pos[client_id][task_pos]
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
                global_model=global_model,
                local_model_states=local_model_states,
                model_builder=model_builder,
                dataset_bundle=dataset_bundle,
                partition=partition,
                client_seen_tasks=client_seen_tasks,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
            )

            round_row = {
                "global_round": int(global_round),
                "task_pos": int(task_pos),
                "round_in_task": int(round_in_task),
                "avg_acc": float(avg_acc),
                "forgetting": float(forgetting),
                "local_global_gap": float(local_global_gap),
            }
            tracker.log_round(round_row=round_row, task_accs=task_accs)
            logger.info(
                "[Metrics] task_pos=%s | round=%s | avg_acc=%.6f | forgetting=%.6f | local_global_gap=%.6f",
                task_pos,
                global_round,
                avg_acc,
                forgetting,
                local_global_gap,
            )

        if hasattr(server, "on_task_end"):
            server.on_task_end(global_model=global_model, task_pos=task_pos, partition=partition)

    saved_tables = tracker.save(run_dirs["tables_dir"])
    rounds = [row["global_round"] for row in tracker.round_rows]
    acc_curve = [row["avg_acc"] for row in tracker.round_rows]
    forgetting_curve = [row["forgetting"] for row in tracker.round_rows]
    gap_curve = [row["local_global_gap"] for row in tracker.round_rows]
    task_boundaries = [
        boundary for boundary in range(args.rounds_per_task, total_rounds, args.rounds_per_task)
    ]

    save_metric_curve(
        x_values=rounds,
        y_values=acc_curve,
        title=f"{args.method.upper()} | {args.dataset} | Average accuracy per round",
        y_label="Average accuracy",
        output_path=run_dirs["figures_dir"] / "avg_acc_per_round.png",
        task_boundaries=task_boundaries,
    )
    save_metric_curve(
        x_values=rounds,
        y_values=forgetting_curve,
        title=f"{args.method.upper()} | {args.dataset} | Forgetting per round",
        y_label="Forgetting",
        output_path=run_dirs["figures_dir"] / "forgetting_per_round.png",
        task_boundaries=task_boundaries,
    )
    save_metric_curve(
        x_values=rounds,
        y_values=gap_curve,
        title=f"{args.method.upper()} | {args.dataset} | Local-global gap per round",
        y_label="Local-global gap",
        output_path=run_dirs["figures_dir"] / "local_global_gap_per_round.png",
        task_boundaries=task_boundaries,
    )

    logger.info("\nTraining finished.")
    logger.info("Saved tables : %s", saved_tables)
    logger.info("Saved figures: %s", run_dirs["figures_dir"])
    logger.info("Run dir      : %s", run_dirs["run_dir"])


if __name__ == "__main__":
    main()
