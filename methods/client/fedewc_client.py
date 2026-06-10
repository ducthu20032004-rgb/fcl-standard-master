from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.optim as optim

from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, compute_task_loss, state_from_model


class FedEWCClient(BaseClient):
    """
    Federated + local EWC:
    - server: FedAvg
    - client: CE(current task) + lambda * quadratic Fisher penalty from past tasks
    """

    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.client_last_task_id: Dict[int, int] = {}
        self.client_last_task_indices: Dict[int, List[int]] = {}
        self.client_last_local_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_ewc_snapshots: Dict[int, List[Dict[str, Dict[str, torch.Tensor]]]] = {}

    def _estimate_fisher(
        self,
        model: torch.nn.Module,
        task_id: int,
        train_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
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

        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
        }

        model.eval()
        seen_samples = 0
        num_batches = 0
        max_samples = int(self.args.fedewc_fisher_max_samples)

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = model(x)
            loss = compute_task_loss(logits, y, self.args, current_classes, class_map)

            model.zero_grad(set_to_none=True)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            num_batches += 1
            seen_samples += int(y.size(0))
            if seen_samples >= max_samples:
                break

        if num_batches == 0:
            return {
                name: tensor.detach().cpu().clone()
                for name, tensor in fisher.items()
            }

        return {
            name: (tensor / float(num_batches)).detach().cpu().clone()
            for name, tensor in fisher.items()
        }

    def _maybe_consolidate_previous_task(
        self,
        client_id: int,
        current_task_id: int,
        global_model: torch.nn.Module,
    ) -> None:
        prev_task_id = self.client_last_task_id.get(client_id, None)
        if prev_task_id is None or prev_task_id == current_task_id:
            return
        if client_id not in self.client_last_local_state:
            return

        prev_indices = self.client_last_task_indices.get(client_id, [])
        if len(prev_indices) == 0:
            return

        prev_model = copy.deepcopy(global_model).to(self.device)
        prev_model.load_state_dict(self.client_last_local_state[client_id], strict=True)

        prev_params = {
            name: param.detach().cpu().clone()
            for name, param in prev_model.named_parameters()
        }
        prev_fisher = self._estimate_fisher(
            model=prev_model,
            task_id=prev_task_id,
            train_indices=prev_indices,
        )

        if client_id not in self.client_ewc_snapshots:
            self.client_ewc_snapshots[client_id] = []

        self.client_ewc_snapshots[client_id].append(
            {
                "params": prev_params,
                "fisher": prev_fisher,
            }
        )

    def _ewc_penalty(self, model: torch.nn.Module, client_id: int) -> torch.Tensor:
        snapshots = self.client_ewc_snapshots.get(client_id, [])
        if len(snapshots) == 0:
            return next(model.parameters()).new_zeros(())

        named_params = dict(model.named_parameters())
        penalty = next(model.parameters()).new_zeros(())

        for snapshot in snapshots:
            ref_params = snapshot["params"]
            ref_fisher = snapshot["fisher"]
            for name, param in named_params.items():
                fisher_t = ref_fisher[name].to(self.device)
                param_ref = ref_params[name].to(self.device)
                penalty = penalty + (fisher_t * (param - param_ref).pow(2)).sum()

        return 0.5 * float(self.args.fedewc_lambda) * penalty

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload

        self._maybe_consolidate_previous_task(
            client_id=int(client_id),
            current_task_id=int(task_id),
            global_model=global_model,
        )

        local_model = copy.deepcopy(global_model).to(self.device)
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
                loss_reg = self._ewc_penalty(local_model, client_id=int(client_id))
                loss = loss_ce + loss_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        local_state = state_from_model(local_model)
        self.client_last_task_id[int(client_id)] = int(task_id)
        self.client_last_task_indices[int(client_id)] = list(train_indices)
        self.client_last_local_state[int(client_id)] = local_state

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=local_state,
            personalized_state_dict=local_state,
        )