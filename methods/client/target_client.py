from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.misc import state_dict_to_cpu

from ..target_modules import CyclingDataIter, augment_and_normalize, infer_image_stats, temperature_kldiv
from .base_client import BaseClient, LocalUpdate


class TARGETClient(BaseClient):
    def _build_real_loader(self, train_indices):
        return DataLoader(
            Subset(self.dataset_bundle.train_dataset, train_indices),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.dataset_bundle.collate_train_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def _real_ce_loss(self, logits: torch.Tensor, y: torch.Tensor, task_id: int) -> torch.Tensor:
        if self.args.loss_mode == "partial":
            current_classes = self.task_labels[task_id]
            class_map = torch.full(
                (self.dataset_bundle.num_classes,),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for new_idx, cls in enumerate(current_classes):
                class_map[int(cls)] = int(new_idx)
            return F.cross_entropy(logits[:, current_classes], class_map[y])
        return F.cross_entropy(logits, y)

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload
        teacher_model = None
        replay_buffer = None
        if isinstance(global_payload, dict):
            teacher_model = global_payload.get("teacher_model", None)
            replay_buffer = global_payload.get("replay_buffer", None)

        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        real_loader = self._build_real_loader(train_indices)

        use_replay = (
            teacher_model is not None
            and replay_buffer is not None
            and replay_buffer.num_samples() > 0
        )

        if use_replay:
            teacher_model = copy.deepcopy(teacher_model).to(self.device)
            teacher_model.eval()
            replay_loader = replay_buffer.build_loader(
                batch_size=int(self.args.target_client_replay_batch_size),
                num_workers=self.args.num_workers,
                shuffle=True,
            )
            replay_iter = CyclingDataIter(replay_loader)
            replay_class_ids = list(map(int, replay_buffer.class_ids))
            mean, std, _ = infer_image_stats(self.dataset_bundle.name)
        else:
            teacher_model = None
            replay_iter = None
            replay_class_ids = None
            mean, std = None, None

        for _ in range(self.args.local_epochs):
            for x, y in real_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = local_model(x)
                loss_ce = self._real_ce_loss(logits, y, task_id=task_id)
                loss = loss_ce

                if use_replay and replay_iter is not None and replay_class_ids is not None:
                    syn_x_01 = replay_iter.next().to(self.device, non_blocking=True)
                    syn_x = augment_and_normalize(syn_x_01, mean=mean, std=std, do_augment=True)

                    student_syn_logits = local_model(syn_x)
                    with torch.no_grad():
                        teacher_syn_logits = teacher_model(syn_x)

                    loss_kd = temperature_kldiv(
                        student_syn_logits[:, replay_class_ids],
                        teacher_syn_logits[:, replay_class_ids],
                        temperature=float(self.args.target_client_kd_temperature),
                        reduction="batchmean",
                    )
                    loss = loss + float(self.args.target_client_kd_weight) * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=state_dict_to_cpu(local_model.state_dict()),
        )