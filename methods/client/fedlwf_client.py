from __future__ import annotations

import copy
from typing import Dict, List, Set

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, compute_task_loss, state_from_model


def temperature_kldiv(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    q = F.log_softmax(student_logits / temperature, dim=1)
    p = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(q, p, reduction="batchmean") * (temperature ** 2)


class FedLwFClient(BaseClient):
    """
    Federated + local LwF:
    - server: FedAvg
    - client: CE(current task) + lambda * KD(old classes on current-task inputs)
    """

    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.client_last_task_id: Dict[int, int] = {}
        self.client_last_local_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_teacher_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_seen_classes: Dict[int, Set[int]] = {}

    def _maybe_refresh_teacher(self, client_id: int, current_task_id: int) -> None:
        prev_task_id = self.client_last_task_id.get(client_id, None)
        if prev_task_id is None or prev_task_id == current_task_id:
            return
        if client_id not in self.client_last_local_state:
            return

        if client_id not in self.client_seen_classes:
            self.client_seen_classes[client_id] = set()

        self.client_seen_classes[client_id].update(int(c) for c in self.task_labels[prev_task_id])
        self.client_teacher_state[client_id] = self.client_last_local_state[client_id]

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload

        self._maybe_refresh_teacher(client_id=int(client_id), current_task_id=int(task_id))

        local_model = copy.deepcopy(global_model).to(self.device)
        teacher_model = None
        current_task_classes = set(int(c) for c in self.task_labels[task_id])
        old_classes = sorted(
            c for c in self.client_seen_classes.get(int(client_id), set())
            if c not in current_task_classes
        )

        if int(client_id) in self.client_teacher_state and len(old_classes) > 0:
            teacher_model = copy.deepcopy(global_model).to(self.device)
            teacher_model.load_state_dict(self.client_teacher_state[int(client_id)], strict=True)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=train_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )
        current_classes, class_map = build_loss_context(
            args=self.args,
            task_labels=self.task_labels,
            num_classes=self.dataset_bundle.num_classes,
            task_id=task_id,
            device=self.device,
        )

        local_model.train()
        for _ in range(self.args.local_epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = local_model(x)
                loss_ce = compute_task_loss(logits, y, self.args, current_classes, class_map)

                if teacher_model is not None and len(old_classes) > 0:
                    with torch.no_grad():
                        teacher_logits = teacher_model(x)
                    loss_kd = temperature_kldiv(
                        student_logits=logits[:, old_classes],
                        teacher_logits=teacher_logits[:, old_classes],
                        temperature=float(self.args.fedlwf_temperature),
                    )
                    loss = loss_ce + float(self.args.fedlwf_lambda) * loss_kd
                else:
                    loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        local_state = state_from_model(local_model)
        self.client_last_task_id[int(client_id)] = int(task_id)
        self.client_last_local_state[int(client_id)] = local_state

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=local_state,
            personalized_state_dict=local_state,
        )