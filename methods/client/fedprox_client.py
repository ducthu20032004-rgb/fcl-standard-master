from __future__ import annotations

import torch
import torch.optim as optim

from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, clone_model, compute_task_loss, state_from_model


class FedProxClient(BaseClient):
    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload
        local_model = clone_model(global_model, self.device)
        global_params = {
            name: param.detach().clone().to(self.device)
            for name, param in global_model.named_parameters()
        }

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
                task_loss = compute_task_loss(logits, y, self.args, current_classes, class_map)

                prox_reg = 0.0
                for name, param in local_model.named_parameters():
                    prox_reg = prox_reg + torch.sum((param - global_params[name]) ** 2)
                loss = task_loss + 0.5 * float(self.args.fedprox_mu) * prox_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        state_dict = state_from_model(local_model)
        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=state_dict,
            personalized_state_dict=state_dict,
        )