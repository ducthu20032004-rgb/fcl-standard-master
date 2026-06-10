from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, compute_task_loss, state_from_model


class ReservoirReplayBuffer:
    def __init__(self, max_size: int, seed: int) -> None:
        self.max_size = int(max_size)
        self.rng = np.random.RandomState(seed)
        self.num_seen = 0
        self.examples: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.examples)

    def add_batch(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu().long()
        logits = logits.detach().cpu()

        for idx in range(x.size(0)):
            self.num_seen += 1
            if len(self.examples) < self.max_size:
                self.examples.append(x[idx].clone())
                self.labels.append(y[idx].clone())
                self.logits.append(logits[idx].clone())
            else:
                replace_idx = self.rng.randint(0, self.num_seen)
                if replace_idx < self.max_size:
                    self.examples[replace_idx] = x[idx].clone()
                    self.labels[replace_idx] = y[idx].clone()
                    self.logits[replace_idx] = logits[idx].clone()

    def sample(self, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(self.examples) == 0:
            return None

        take = min(int(batch_size), len(self.examples))
        indices = self.rng.choice(len(self.examples), size=take, replace=False)

        x = torch.stack([self.examples[i] for i in indices], dim=0).to(device, non_blocking=True)
        y = torch.stack([self.labels[i] for i in indices], dim=0).long().to(device, non_blocking=True)
        z = torch.stack([self.logits[i] for i in indices], dim=0).to(device, non_blocking=True)
        return x, y, z


class FedDERPPClient(BaseClient):
    """
    Federated + local DER++:
    - server: FedAvg
    - client: CE(current batch) + alpha * replay-logit MSE + beta * replay-label CE
    """

    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.client_last_task_id: Dict[int, int] = {}
        self.client_last_task_indices: Dict[int, List[int]] = {}
        self.client_last_local_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_buffers: Dict[int, ReservoirReplayBuffer] = {}

    def _get_buffer(self, client_id: int) -> ReservoirReplayBuffer:
        if client_id not in self.client_buffers:
            self.client_buffers[client_id] = ReservoirReplayBuffer(
                max_size=int(self.args.fedderpp_buffer_size),
                seed=int(self.args.seed + 10000 + client_id),
            )
        return self.client_buffers[client_id]

    def _maybe_update_buffer(
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
        prev_model.eval()

        loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=prev_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

        buffer = self._get_buffer(client_id)
        target_store = int(self.args.fedderpp_store_per_task)
        stored = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = prev_model(x)

                remain = target_store - stored
                if remain <= 0:
                    break

                take = min(int(x.size(0)), remain)
                buffer.add_batch(x[:take], y[:take], logits[:take])
                stored += take

                if stored >= target_store:
                    break

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload

        self._maybe_update_buffer(
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
        buffer = self._get_buffer(int(client_id))

        local_model.train()
        for _ in range(self.args.local_epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = local_model(x)
                loss = compute_task_loss(logits, y, self.args, current_classes, class_map)

                replay_batch = buffer.sample(
                    batch_size=int(self.args.fedderpp_replay_batch_size),
                    device=self.device,
                )
                if replay_batch is not None:
                    mem_x, mem_y, mem_z = replay_batch
                    replay_logits = local_model(mem_x)
                    loss = loss + float(self.args.fedderpp_alpha) * F.mse_loss(replay_logits, mem_z)
                    loss = loss + float(self.args.fedderpp_beta) * F.cross_entropy(replay_logits, mem_y)

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