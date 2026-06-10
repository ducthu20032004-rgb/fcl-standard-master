from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore

from .base_client import BaseClient, LocalUpdate
from .common import (
    build_loader,
    build_loss_context,
    clone_model,
    compute_task_loss,
    maybe_subsample_indices,
    state_from_model,
    train_standard_local_model,
)


class FedALAClient(BaseClient):
    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.local_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.gate_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_rounds: Dict[int, int] = {}

    def _selected_gate_keys(self, model) -> List[str]:
        prefixes = list(model.get_trainable_block_prefixes())
        top_p = max(1, min(int(self.args.fedala_top_p), len(prefixes)))
        selected_prefixes = prefixes[-top_p:]
        selected_keys: List[str] = []
        for key, tensor in model.state_dict().items():
            if not torch.is_floating_point(tensor):
                continue
            if any(key == prefix or key.startswith(prefix + ".") for prefix in selected_prefixes):
                selected_keys.append(key)
        return selected_keys

    def _build_mixed_state(
        self,
        model,
        global_state: Dict[str, torch.Tensor],
        local_prev_state: Dict[str, torch.Tensor],
        gate_keys: List[str],
        gate_params: List[nn.Parameter],
    ) -> OrderedDict:
        gate_map = {key: gate_params[idx].clamp(0.0, 1.0) for idx, key in enumerate(gate_keys)}
        mixed: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, _ in model.state_dict().items():
            g = global_state[key].to(self.device)
            if key in gate_map:
                l = local_prev_state[key].to(self.device)
                mixed[key] = l + (g - l) * gate_map[key]
            else:
                mixed[key] = g
        return mixed

    def _initialize_gates(self, client_id: int, gate_keys: List[str], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if client_id not in self.gate_states:
            self.gate_states[client_id] = {
                key: torch.ones_like(global_state[key], dtype=torch.float32)
                for key in gate_keys
            }
        return self.gate_states[client_id]

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload
        local_model = clone_model(global_model, self.device)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        if client_id not in self.local_states:
            local_model = train_standard_local_model(
                local_model=local_model,
                args=self.args,
                dataset_bundle=self.dataset_bundle,
                task_labels=self.task_labels,
                device=self.device,
                task_id=task_id,
                train_indices=train_indices,
            )
            state_dict = state_from_model(local_model)
            self.local_states[client_id] = state_dict
            self.client_rounds[client_id] = 1
            return LocalUpdate(
                client_id=int(client_id),
                task_id=int(task_id),
                num_samples=int(len(train_indices)),
                state_dict=state_dict,
                personalized_state_dict=state_dict,
            )

        prev_local_state = self.local_states[client_id]
        gate_keys = self._selected_gate_keys(local_model)
        stored_gates = self._initialize_gates(client_id, gate_keys, global_state)
        gate_params = [nn.Parameter(stored_gates[key].to(self.device)) for key in gate_keys]
        gate_optimizer = optim.SGD(gate_params, lr=float(self.args.fedala_weight_lr), momentum=0.0)

        subset_indices = maybe_subsample_indices(
            train_indices,
            ratio=float(self.args.fedala_sample_ratio),
            seed=int(self.args.seed + 1000 * client_id + self.client_rounds.get(client_id, 0)),
        )
        gate_loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=subset_indices,
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

        is_initial_adaptation = self.client_rounds.get(client_id, 0) <= 1
        gate_epochs = int(self.args.fedala_init_epochs if is_initial_adaptation else self.args.fedala_adapt_epochs)
        for _ in range(max(1, gate_epochs)):
            for x, y in gate_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mixed_state = self._build_mixed_state(
                    model=local_model,
                    global_state=global_state,
                    local_prev_state=prev_local_state,
                    gate_keys=gate_keys,
                    gate_params=gate_params,
                )
                logits = functional_call(local_model, mixed_state, (x,))
                loss = compute_task_loss(logits, y, self.args, current_classes, class_map)
                gate_optimizer.zero_grad()
                loss.backward()
                gate_optimizer.step()
                for gate in gate_params:
                    gate.data.clamp_(0.0, 1.0)

        mixed_state = self._build_mixed_state(
            model=local_model,
            global_state=global_state,
            local_prev_state=prev_local_state,
            gate_keys=gate_keys,
            gate_params=gate_params,
        )
        local_model.load_state_dict(mixed_state, strict=True)

        self.gate_states[client_id] = {
            key: gate_params[idx].detach().cpu().clone()
            for idx, key in enumerate(gate_keys)
        }

        local_model = train_standard_local_model(
            local_model=local_model,
            args=self.args,
            dataset_bundle=self.dataset_bundle,
            task_labels=self.task_labels,
            device=self.device,
            task_id=task_id,
            train_indices=train_indices,
        )
        state_dict = state_from_model(local_model)
        self.local_states[client_id] = state_dict
        self.client_rounds[client_id] = self.client_rounds.get(client_id, 0) + 1

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=state_dict,
            personalized_state_dict=state_dict,
        )