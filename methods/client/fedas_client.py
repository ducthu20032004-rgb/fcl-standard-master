from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_client import BaseClient, LocalUpdate
from .common import (
    build_loader,
    build_loss_context,
    clone_model,
    compute_task_loss,
    maybe_subsample_indices,
    state_from_model,
)


class FedASClient(BaseClient):
    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.local_states: Dict[int, Dict[str, torch.Tensor]] = {}

    def _parameter_alignment(self, local_model, prev_local_model, task_id: int, train_indices):
        if len(train_indices) == 0:
            return local_model

        align_ratio = float(self.args.fedas_align_ratio)
        align_indices = maybe_subsample_indices(
            train_indices,
            ratio=align_ratio,
            seed=int(self.args.seed + 2000 + task_id),
        )
        loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=align_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

        for param in local_model.head.parameters():
            param.requires_grad = False
        for _, param in local_model.named_backbone_parameters():
            param.requires_grad = True

        optimizer = optim.SGD(
            [param for _, param in local_model.named_backbone_parameters() if param.requires_grad],
            lr=float(self.args.fedas_align_lr),
            momentum=0.0,
            weight_decay=0.0,
        )

        local_model.train()
        prev_local_model.eval()
        for _ in range(max(1, int(self.args.fedas_align_epochs))):
            for x, _ in loader:
                x = x.to(self.device, non_blocking=True)
                with torch.no_grad():
                    target_features = prev_local_model.extract_features(x)
                current_features = local_model.extract_features(x)
                loss = F.mse_loss(current_features, target_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for param in local_model.head.parameters():
            param.requires_grad = True
        return local_model

    def _local_train(self, local_model, task_id: int, train_indices):
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
                loss = compute_task_loss(logits, y, self.args, current_classes, class_map)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return local_model

    def _estimate_tfim_alpha(self, local_model, task_id: int, train_indices) -> float:
        subset_indices = maybe_subsample_indices(
            train_indices,
            ratio=float(self.args.fedas_fim_ratio),
            seed=int(self.args.seed + 3000 + task_id),
        )
        loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=subset_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
        current_classes, class_map = build_loss_context(
            args=self.args,
            task_labels=self.task_labels,
            num_classes=self.dataset_bundle.num_classes,
            task_id=task_id,
            device=self.device,
        )

        local_model.eval()
        local_model.zero_grad(set_to_none=True)
        alpha = 0.0
        count = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = local_model(x)
            loss = compute_task_loss(logits, y, self.args, current_classes, class_map)
            local_model.zero_grad(set_to_none=True)
            loss.backward()
            sqnorm = 0.0
            for param in local_model.parameters():
                if param.grad is not None:
                    sqnorm += float(torch.sum(param.grad.detach() ** 2).item())
            alpha += sqnorm
            count += 1
        return float(alpha / max(count, 1))

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload
        local_model = clone_model(global_model, self.device)
        prev_local_model: Optional[torch.nn.Module] = None

        if client_id in self.local_states:
            prev_local_model = clone_model(global_model, self.device)
            prev_local_model.load_state_dict(self.local_states[client_id], strict=True)
            local_model.load_head_state_dict(prev_local_model.head_state_dict(), strict=True)
            local_model = self._parameter_alignment(
                local_model=local_model,
                prev_local_model=prev_local_model,
                task_id=task_id,
                train_indices=train_indices,
            )

        local_model = self._local_train(local_model, task_id=task_id, train_indices=train_indices)
        alpha_i = self._estimate_tfim_alpha(local_model, task_id=task_id, train_indices=train_indices)

        state_dict = state_from_model(local_model)
        self.local_states[client_id] = state_dict
        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=state_dict,
            personalized_state_dict=state_dict,
            aggregation_weight=float(alpha_i),
            extra={
                "backbone_state_dict": {
                    key: value.detach().cpu().clone() for key, value in local_model.backbone_state_dict().items()
                },
                "head_state_dict": {
                    key: value.detach().cpu().clone() for key, value in local_model.head_state_dict().items()
                },
            },
        )