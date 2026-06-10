from __future__ import annotations

from typing import Dict, List, Optional

from utils import state_dict_to_cpu

from ..affcl_modules import build_conditional_flow
from .fedavg_server import FedAvgServer, fedavg_state_dict


class AFFCLServer(FedAvgServer):
    def __init__(self, args, device) -> None:
        super().__init__(args=args, device=device)
        self.affcl_flow = None
        self.partition = None
        self.current_task_pos: Optional[int] = None
        self.affcl_seen_class_ids: List[int] = []
        self.affcl_past_class_ids: List[int] = []

    def register_context(self, model_builder, dataset_bundle, partition, logger) -> None:
        del model_builder, dataset_bundle, logger
        self.partition = partition

    def _ensure_flow_model(self, global_model):
        if self.affcl_flow is not None:
            return
        feature_dim = int(global_model.head.in_features)
        num_classes = int(global_model.head.out_features)
        self.affcl_flow = build_conditional_flow(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_features=int(self.args.affcl_flow_hidden_dim),
            num_layers=int(self.args.affcl_flow_layers),
        ).to(self.device)

    def on_task_start(self, global_model, task_pos: int, partition) -> None:
        del global_model
        self.current_task_pos = int(task_pos)
        self.partition = partition

        current_tasks = set()
        past_tasks = set()

        for client_id in range(self.args.num_clients):
            current_tasks.add(int(self.partition.client_task_orders[client_id][task_pos]))
            for old_pos in range(task_pos):
                past_tasks.add(int(self.partition.client_task_orders[client_id][old_pos]))

        seen_tasks = sorted(set(current_tasks) | set(past_tasks))

        past_classes = set()
        seen_classes = set()
        for task_id in past_tasks:
            past_classes.update(int(c) for c in self.partition.task_labels[task_id])
        for task_id in seen_tasks:
            seen_classes.update(int(c) for c in self.partition.task_labels[task_id])

        self.affcl_past_class_ids = sorted(past_classes)
        self.affcl_seen_class_ids = sorted(seen_classes)

    def get_client_payload(self, global_model):
        self._ensure_flow_model(global_model)
        return {
            "global_model": global_model,
            "affcl_flow_state": state_dict_to_cpu(self.affcl_flow.state_dict()),
            "affcl_seen_class_ids": list(self.affcl_seen_class_ids),
            "affcl_past_class_ids": list(self.affcl_past_class_ids),
        }

    def aggregate(self, global_model, local_updates):
        if len(local_updates) == 0:
            return global_model

        self._ensure_flow_model(global_model)

        classifier_weights = [float(update.num_samples) for update in local_updates]
        classifier_states = [update.state_dict for update in local_updates]
        global_classifier_state = state_dict_to_cpu(global_model.state_dict())
        aggregated_classifier = fedavg_state_dict(global_classifier_state, classifier_states, classifier_weights)
        global_model.load_state_dict(aggregated_classifier, strict=True)
        global_model = global_model.to(self.device)

        flow_states = [update.extra["affcl_flow_state"] for update in local_updates]
        global_flow_state = state_dict_to_cpu(self.affcl_flow.state_dict())
        aggregated_flow = fedavg_state_dict(global_flow_state, flow_states, classifier_weights)
        self.affcl_flow.load_state_dict(aggregated_flow, strict=True)

        return global_model