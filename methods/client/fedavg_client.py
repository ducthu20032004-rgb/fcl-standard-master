from __future__ import annotations

import copy

from .base_client import BaseClient, LocalUpdate
from .common import clone_model, state_from_model, train_standard_local_model


class FedAvgClient(BaseClient):
    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None
        # Đủ data mới train, tránh lỗi BatchNorm
        if len(train_indices) < self.args.batch_size:
            print(f"[WARN] Skip client {client_id} task {task_id}: only {len(train_indices)} samples")
            return None
        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload
        local_model = copy.deepcopy(global_model).to(self.device)
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
        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=state_dict,
            personalized_state_dict=state_dict,
        )