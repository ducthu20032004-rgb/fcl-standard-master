from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, compute_task_loss, state_from_model
from ..tagfed_modules import GroupLogitHead, build_tagfed_message


def temperature_kldiv(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    q = F.log_softmax(student_logits / temperature, dim=1)
    p = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(q, p, reduction="batchmean") * (temperature ** 2)


class TagFedClient(BaseClient):
    """
    Dense TagFed adaptation:
    - new task: start from current global model
    - repeated task: trace back to the stored local state of that task
    - repeated task only retrains the last few groups + head
    - client distills from the server group head of the same task
    """
    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.client_task_states: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    def _make_local_model(self, global_model, client_id: int, task_id: int):
        repeated = client_id in self.client_task_states and task_id in self.client_task_states[client_id]
        local_model = copy.deepcopy(global_model).to(self.device)
        if repeated:
            local_model.load_state_dict(self.client_task_states[client_id][task_id], strict=True)
        return local_model, repeated

    def _set_trainable_params(self, model, repeated: bool) -> None:
        if not repeated:
            for param in model.parameters():
                param.requires_grad = True
            return

        group_prefixes = list(model.get_trainable_block_prefixes())
        num_trainable = max(1, min(len(group_prefixes), int(self.args.tagfed_retrain_groups)))
        trainable_prefixes = group_prefixes[-num_trainable:]

        for name, param in model.named_parameters():
            allow = any(name == prefix or name.startswith(prefix + ".") for prefix in trainable_prefixes)
            param.requires_grad = allow

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload
        group_teacher_states = (
            global_payload.get("tagfed_group_teacher_states", {})
            if isinstance(global_payload, dict)
            else {}
        )

        local_model, repeated = self._make_local_model(
            global_model=global_model,
            client_id=int(client_id),
            task_id=int(task_id),
        )
        self._set_trainable_params(local_model, repeated=repeated)

        teacher_head = None
        if int(task_id) in group_teacher_states:
            teacher_head = GroupLogitHead(
                feature_dim=local_model.head.in_features,
                num_classes=self.dataset_bundle.num_classes,
            ).to(self.device)
            teacher_head.load_state_dict(group_teacher_states[int(task_id)], strict=True)
            teacher_head.eval()
            for param in teacher_head.parameters():
                param.requires_grad = False

        optimizer = optim.SGD(
            [param for param in local_model.parameters() if param.requires_grad],
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
        task_classes = list(self.task_labels[task_id])

        local_model.train()
        for _ in range(self.args.local_epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                features = local_model.extract_features(x)
                logits = local_model.forward_from_features(features)
                loss_ce = compute_task_loss(logits, y, self.args, current_classes, class_map)

                if teacher_head is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_head(features.detach())[:, task_classes]
                    loss_kd = temperature_kldiv(
                        student_logits=logits[:, task_classes],
                        teacher_logits=teacher_logits,
                        temperature=float(self.args.tagfed_temperature),
                    )
                    loss = (
                        float(self.args.tagfed_alpha_c) * loss_ce
                        + float(self.args.tagfed_beta_c) * loss_kd
                    )
                else:
                    loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        local_state = state_from_model(local_model)

        if client_id not in self.client_task_states:
            self.client_task_states[client_id] = {}
        self.client_task_states[client_id][task_id] = local_state

        tagfed_message = build_tagfed_message(
            model=local_model,
            dataset_bundle=self.dataset_bundle,
            train_indices=train_indices,
            ratio=float(self.args.tagfed_message_ratio),
            seed=int(self.args.seed + 9000 + 100 * client_id + task_id),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            device=self.device,
        )

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=local_state,
            personalized_state_dict=local_state,
            extra={
                "tagfed_message": tagfed_message,
                "tagfed_repeated": bool(repeated),
            },
        )