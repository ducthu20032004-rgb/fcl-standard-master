from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

from utils import state_dict_to_cpu

from .base_client import BaseClient, LocalUpdate
from .common import build_loss_context, compute_task_loss
from ..dddr_modules import (
    CyclingDataIter,
    DDDRProjectionHead,
    SupConLoss,
    build_concat_syn_dataset,
    build_task_syn_dataset,
    kd_loss,
)


class DDDRClient(BaseClient):
    """
    Benchmark-friendly DDDR adaptation.

    Exact from DDDR paper:
    - CE on current task real + current task generated images
    - supervised contrastive loss on current task real/generated images
    - CE on previous generated images
    - KD from previous-task model on previous generated images
    - FedAvg classifier aggregation

    Adaptation:
    - The Federated Class Inversion phase is externalized as a synthetic image cache.
    - This client only implements the replay-augmented classifier training phase.
    """

    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.client_last_task_id: Dict[int, int] = {}
        self.client_last_round_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_old_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_seen_task_ids: Dict[int, List[int]] = {}
        self.scl_criterion = SupConLoss(temperature=float(self.args.dddr_scl_temperature))

    def _refresh_old_model_if_task_changed(self, client_id: int, current_task_id: int) -> None:
        prev_task_id = self.client_last_task_id.get(client_id, None)
        if prev_task_id is None or prev_task_id == current_task_id:
            return
        if client_id not in self.client_last_round_state:
            return

        self.client_old_state[client_id] = self.client_last_round_state[client_id]
        if client_id not in self.client_seen_task_ids:
            self.client_seen_task_ids[client_id] = []
        if prev_task_id not in self.client_seen_task_ids[client_id]:
            self.client_seen_task_ids[client_id].append(prev_task_id)

    def _build_current_loader(self, train_indices, current_task_id: int):
        real_subset = Subset(self.dataset_bundle.train_dataset, list(train_indices))
        datasets = [real_subset]

        syn_root = self.args.dddr_syn_image_path
        if syn_root is not None:
            current_syn_dataset = build_task_syn_dataset(
                syn_root=syn_root,
                dataset_name=self.dataset_bundle.name,
                task_id=current_task_id,
                size_per_cls=int(self.args.dddr_current_size),
            )
            if current_syn_dataset is not None:
                datasets.append(current_syn_dataset)

        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.dataset_bundle.collate_train_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def _build_previous_replay_loader(self, client_id: int, current_task_id: int):
        syn_root = self.args.dddr_syn_image_path
        if syn_root is None:
            return None

        prev_tasks = [
            int(t) for t in self.client_seen_task_ids.get(client_id, [])
            if int(t) != int(current_task_id)
        ]
        if len(prev_tasks) == 0:
            return None

        prev_syn_dataset = build_concat_syn_dataset(
            syn_root=syn_root,
            dataset_name=self.dataset_bundle.name,
            task_ids=prev_tasks,
            size_per_cls=int(self.args.dddr_prev_size),
        )
        if prev_syn_dataset is None:
            return None

        loader = DataLoader(
            prev_syn_dataset,
            batch_size=int(self.args.dddr_replay_batch_size),
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.dataset_bundle.collate_train_fn,
            pin_memory=torch.cuda.is_available(),
        )
        return CyclingDataIter(loader)

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        if self.dataset_bundle.collate_train_fn is None:
            raise RuntimeError(
                "DDDR current integration expects a dataset bundle with collate_train_fn "
                "(currently intended for CIFAR-style datasets)."
            )

        self._refresh_old_model_if_task_changed(client_id=int(client_id), current_task_id=int(task_id))

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload
        proj_state = global_payload.get("dddr_proj_state", None) if isinstance(global_payload, dict) else None

        if self.args.dddr_require_syn_root and self.args.dddr_syn_image_path is None:
            raise RuntimeError(
                "DDDR requires --dddr-syn-image-path pointing to a synthetic image cache "
                "generated by the official DDDR pipeline."
            )

        local_model = copy.deepcopy(global_model).to(self.device)

        feat_dim = int(local_model.head.in_features)
        proj_head = DDDRProjectionHead(
            in_dim=feat_dim,
            hidden_dim=int(self.args.dddr_proj_hidden_dim),
            out_dim=int(self.args.dddr_proj_dim),
        ).to(self.device)
        if proj_state is not None:
            proj_head.load_state_dict(proj_state, strict=True)

        old_model = None
        old_class_ids: List[int] = []
        if int(client_id) in self.client_old_state:
            old_model = copy.deepcopy(global_model).to(self.device)
            old_model.load_state_dict(self.client_old_state[int(client_id)], strict=True)
            old_model.eval()
            for param in old_model.parameters():
                param.requires_grad = False

            for seen_task_id in self.client_seen_task_ids.get(int(client_id), []):
                old_class_ids.extend(int(c) for c in self.task_labels[int(seen_task_id)])
            old_class_ids = sorted(set(old_class_ids))

        optimizer = optim.SGD(
            list(local_model.parameters()) + list(proj_head.parameters()),
            lr=float(self.args.lr),
            momentum=float(self.args.momentum),
            weight_decay=float(self.args.weight_decay),
        )

        current_loader = self._build_current_loader(train_indices=train_indices, current_task_id=int(task_id))
        previous_replay_iter = self._build_previous_replay_loader(
            client_id=int(client_id),
            current_task_id=int(task_id),
        )

        current_classes, class_map = build_loss_context(
            args=self.args,
            task_labels=self.task_labels,
            num_classes=self.dataset_bundle.num_classes,
            task_id=task_id,
            device=self.device,
        )

        local_model.train()
        proj_head.train()

        for _ in range(self.args.local_epochs):
            for x, y in current_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                outputs = local_model(x)
                loss = compute_task_loss(outputs, y, self.args, current_classes, class_map)

                if float(self.args.dddr_w_scl) > 0:
                    features = local_model.extract_features(x)
                    proj = proj_head(features)
                    proj = F.normalize(proj, dim=1).unsqueeze(1)
                    loss = loss + float(self.args.dddr_w_scl) * self.scl_criterion(proj, labels=y)

                if previous_replay_iter is not None:
                    pre_x, pre_y = previous_replay_iter.next()
                    pre_x = pre_x.to(self.device, non_blocking=True)
                    pre_y = pre_y.to(self.device, non_blocking=True)

                    pre_logits = local_model(pre_x)

                    if float(self.args.dddr_w_ce_pre) > 0:
                        loss = loss + float(self.args.dddr_w_ce_pre) * F.cross_entropy(pre_logits, pre_y)

                    if old_model is not None and len(old_class_ids) > 0 and float(self.args.dddr_w_kd) > 0:
                        with torch.no_grad():
                            old_logits = old_model(pre_x)
                        loss = loss + float(self.args.dddr_w_kd) * kd_loss(
                            student_logits=pre_logits[:, old_class_ids],
                            teacher_logits=old_logits[:, old_class_ids],
                            temperature=float(self.args.dddr_temperature),
                        )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        local_state = state_dict_to_cpu(local_model.state_dict())
        proj_head_state = state_dict_to_cpu(proj_head.state_dict())

        self.client_last_round_state[int(client_id)] = local_state
        self.client_last_task_id[int(client_id)] = int(task_id)
        if int(client_id) not in self.client_seen_task_ids:
            self.client_seen_task_ids[int(client_id)] = []
        if int(task_id) not in self.client_seen_task_ids[int(client_id)]:
            self.client_seen_task_ids[int(client_id)].append(int(task_id))

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=local_state,
            personalized_state_dict=local_state,
            extra={
                "dddr_proj_state": proj_head_state,
            },
        )