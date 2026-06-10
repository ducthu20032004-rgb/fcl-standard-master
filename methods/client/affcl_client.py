from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import state_dict_to_cpu

from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, compute_task_loss
from ..affcl_modules import (
    build_conditional_flow,
    flow_log_prob_and_latent,
    multiclass_cross_entropy_probs,
    probability_in_localdata,
    probs_from_logits,
    sample_flow_features,
)


class AFFCLClient(BaseClient):
    """
    AF-FCL adaptation for the current benchmark codebase.

    What stays faithful to the paper:
    - global conditional flow in feature space
    - flow replay + last-flow replay
    - feature/output KD to previous and global classifier
    - correlation-weighted replay objective
    - server aggregates classifier and flow

    What is adapted:
    - current benchmark backbone exposes a single extract_features() stage,
      so the flow is trained on the penultimate feature vector directly.
    """

    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.client_last_task_id: Dict[int, int] = {}
        self.client_last_classifier_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_last_flow_state: Dict[int, Dict[str, torch.Tensor]] = {}

    def _build_flow(self, model) -> torch.nn.Module:
        feature_dim = int(model.head.in_features)
        num_classes = int(self.dataset_bundle.num_classes)
        flow = build_conditional_flow(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_features=int(self.args.affcl_flow_hidden_dim),
            num_layers=int(self.args.affcl_flow_layers),
        )
        return flow.to(self.device)

    def _maybe_refresh_last_state(self, client_id: int, current_task_id: int, local_model, local_flow) -> None:
        prev_task = self.client_last_task_id.get(client_id, None)
        if prev_task is None or prev_task == current_task_id:
            return
        self.client_last_classifier_state[client_id] = state_dict_to_cpu(local_model.state_dict())
        self.client_last_flow_state[client_id] = state_dict_to_cpu(local_flow.state_dict())

    def _build_teacher_models(
        self,
        client_id: int,
        global_model,
        global_flow_state: Optional[Dict[str, torch.Tensor]],
    ):
        last_classifier = None
        last_flow = None

        if client_id in self.client_last_classifier_state:
            last_classifier = copy.deepcopy(global_model).to(self.device)
            last_classifier.load_state_dict(self.client_last_classifier_state[client_id], strict=True)
            last_classifier.eval()
            for param in last_classifier.parameters():
                param.requires_grad_(False)

        if client_id in self.client_last_flow_state:
            last_flow = self._build_flow(global_model)
            last_flow.load_state_dict(self.client_last_flow_state[client_id], strict=True)
            last_flow.eval()
            for param in last_flow.parameters():
                param.requires_grad_(False)

        global_flow = None
        if global_flow_state is not None:
            global_flow = self._build_flow(global_model)
            global_flow.load_state_dict(global_flow_state, strict=True)
            global_flow.train()

        return last_classifier, last_flow, global_flow

    def _kd_to_models(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        logits: torch.Tensor,
        last_classifier,
        global_classifier,
    ):
        kd_feature_last = features.new_zeros(())
        kd_output_last = features.new_zeros(())
        kd_feature_global = features.new_zeros(())
        kd_output_global = features.new_zeros(())

        if last_classifier is not None and float(self.args.affcl_k_kd_last_cls) > 0:
            with torch.no_grad():
                last_features = last_classifier.extract_features(x)
                last_logits = last_classifier.forward_from_features(last_features)
            kd_feature_last = (
                float(self.args.affcl_k_kd_last_cls)
                * torch.mean((features - last_features) ** 2)
            )
            kd_output_last = (
                float(self.args.affcl_k_kd_last_cls)
                * multiclass_cross_entropy_probs(
                    probs_from_logits(logits),
                    probs_from_logits(last_logits),
                    temperature=float(self.args.affcl_temperature),
                )
            )

        if global_classifier is not None and float(self.args.affcl_k_kd_global_cls) > 0:
            with torch.no_grad():
                global_features = global_classifier.extract_features(x)
                global_logits = global_classifier.forward_from_features(global_features)
            kd_feature_global = (
                float(self.args.affcl_k_kd_global_cls)
                * torch.mean((features - global_features) ** 2)
            )
            kd_output_global = (
                float(self.args.affcl_k_kd_global_cls)
                * multiclass_cross_entropy_probs(
                    probs_from_logits(logits),
                    probs_from_logits(global_logits),
                    temperature=float(self.args.affcl_temperature),
                )
            )

        kd_feature = float(self.args.affcl_k_kd_feature) * (kd_feature_last + kd_feature_global)
        kd_output = float(self.args.affcl_k_kd_output) * (kd_output_last + kd_output_global)
        return kd_feature, kd_output

    def _train_flow_branch(
        self,
        local_model,
        local_flow,
        last_flow,
        x: torch.Tensor,
        y: torch.Tensor,
        past_class_ids: List[int],
        flow_optimizer,
    ):
        local_model.eval()
        with torch.no_grad():
            xa = local_model.extract_features(x).reshape(x.size(0), -1)

        loss_data = -local_flow.log_prob(inputs=xa, context=F.one_hot(y, num_classes=self.dataset_bundle.num_classes).float()).mean()

        loss_last_flow = xa.new_zeros(())
        if last_flow is not None and len(past_class_ids) > 0 and float(self.args.affcl_k_flow_lastflow) > 0:
            sampled_xa, sampled_y, _ = sample_flow_features(
                flow=last_flow,
                labels_pool=past_class_ids,
                batch_size=x.size(0),
                num_classes=self.dataset_bundle.num_classes,
                feature_dim=xa.size(1),
                device=self.device,
            )
            sampled_context = F.one_hot(sampled_y, num_classes=self.dataset_bundle.num_classes).float()
            loss_last_flow = -local_flow.log_prob(inputs=sampled_xa, context=sampled_context).mean()
            loss_last_flow = float(self.args.affcl_k_flow_lastflow) * loss_last_flow

        total_flow_loss = loss_data + loss_last_flow
        flow_optimizer.zero_grad()
        total_flow_loss.backward()
        flow_optimizer.step()

        return {
            "flow_loss": float(loss_data.item()),
            "flow_loss_last": float(loss_last_flow.item()) if torch.is_tensor(loss_last_flow) else float(loss_last_flow),
        }

    def _train_classifier_branch(
        self,
        local_model,
        replay_flow,
        last_classifier,
        global_classifier,
        seen_class_ids: List[int],
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        cls_optimizer,
    ):
        current_classes, class_map = build_loss_context(
            args=self.args,
            task_labels=self.task_labels,
            num_classes=self.dataset_bundle.num_classes,
            task_id=task_id,
            device=self.device,
        )

        local_model.train()
        features = local_model.extract_features(x)
        logits = local_model.forward_from_features(features)

        c_loss = compute_task_loss(logits, y, self.args, current_classes, class_map)

        kd_feature, kd_output = self._kd_to_models(
            x=x,
            features=features,
            logits=logits,
            last_classifier=last_classifier,
            global_classifier=global_classifier,
        )

        c_loss_flow = logits.new_zeros(())
        flow_prob_mean = 0.0

        if replay_flow is not None and float(self.args.affcl_k_loss_flow) > 0 and len(seen_class_ids) > 0:
            replay_flow.eval()
            with torch.no_grad():
                xa = local_model.extract_features(x).reshape(x.size(0), -1)
                log_prob, xa_u = flow_log_prob_and_latent(
                    flow=replay_flow,
                    features=xa,
                    labels=y,
                    num_classes=self.dataset_bundle.num_classes,
                )
                prob_mean = torch.exp(log_prob / xa.size(1)).mean() + 1e-30
                flow_prob_mean = float(prob_mean.item())

                sampled_xa, sampled_y, sampled_u = sample_flow_features(
                    flow=replay_flow,
                    labels_pool=seen_class_ids,
                    batch_size=x.size(0),
                    num_classes=self.dataset_bundle.num_classes,
                    feature_dim=xa.size(1),
                    device=self.device,
                )
                sample_prob = probability_in_localdata(
                    local_latents=xa_u,
                    local_labels=y,
                    fallback_prob=prob_mean,
                    sampled_latents=sampled_u,
                    sampled_labels=sampled_y,
                )

            replay_logits = local_model.forward_from_features(sampled_xa)
            per_sample_ce = F.cross_entropy(replay_logits, sampled_y, reduction="none")
            c_loss_flow_generate = (per_sample_ce * sample_prob).mean()

            explore_factor = (1.0 - float(self.args.affcl_flow_explore_theta)) * float(prob_mean.item()) + float(self.args.affcl_flow_explore_theta)
            c_loss_flow = float(self.args.affcl_k_loss_flow) * c_loss_flow_generate * explore_factor

        total_loss = c_loss + kd_feature + kd_output + c_loss_flow

        cls_optimizer.zero_grad()
        total_loss.backward()
        cls_optimizer.step()

        return {
            "c_loss": float(c_loss.item()),
            "kd_feature": float(kd_feature.item()) if torch.is_tensor(kd_feature) else float(kd_feature),
            "kd_output": float(kd_output.item()) if torch.is_tensor(kd_output) else float(kd_output),
            "c_loss_flow": float(c_loss_flow.item()) if torch.is_tensor(c_loss_flow) else float(c_loss_flow),
            "flow_prob_mean": float(flow_prob_mean),
        }

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"] if isinstance(global_payload, dict) else global_payload
        global_flow_state = global_payload.get("affcl_flow_state", None) if isinstance(global_payload, dict) else None
        seen_class_ids = global_payload.get("affcl_seen_class_ids", []) if isinstance(global_payload, dict) else []
        past_class_ids = global_payload.get("affcl_past_class_ids", []) if isinstance(global_payload, dict) else []

        local_model = copy.deepcopy(global_model).to(self.device)
        last_classifier, last_flow, local_flow = self._build_teacher_models(
            client_id=int(client_id),
            global_model=global_model,
            global_flow_state=global_flow_state,
        )

        if local_flow is None:
            local_flow = self._build_flow(global_model)

        global_classifier = copy.deepcopy(global_model).to(self.device)
        global_classifier.eval()
        for param in global_classifier.parameters():
            param.requires_grad_(False)

        cls_optimizer = optim.Adam(
            local_model.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )
        flow_optimizer = optim.Adam(
            local_flow.parameters(),
            lr=float(self.args.affcl_flow_lr),
            weight_decay=float(self.args.weight_decay),
        )

        loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=train_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )
        train_iter = iter(loader)

        local_iters = int(self.args.affcl_local_iterations) if int(self.args.affcl_local_iterations) > 0 else None
        total_steps = local_iters if local_iters is not None else (self.args.local_epochs * len(loader))

        metrics = {
            "c_loss": 0.0,
            "kd_feature": 0.0,
            "kd_output": 0.0,
            "c_loss_flow": 0.0,
            "flow_loss": 0.0,
            "flow_loss_last": 0.0,
            "flow_prob_mean": 0.0,
        }
        counted = 0

        for _ in range(max(1, total_steps)):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                x, y = next(train_iter)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            flow_stats = self._train_flow_branch(
                local_model=local_model,
                local_flow=local_flow,
                last_flow=last_flow,
                x=x,
                y=y,
                past_class_ids=past_class_ids,
                flow_optimizer=flow_optimizer,
            )

            replay_flow = last_flow if (bool(self.args.affcl_use_lastflow_x) and last_flow is not None) else local_flow
            cls_stats = self._train_classifier_branch(
                local_model=local_model,
                replay_flow=replay_flow,
                last_classifier=last_classifier,
                global_classifier=global_classifier,
                seen_class_ids=seen_class_ids,
                x=x,
                y=y,
                task_id=task_id,
                cls_optimizer=cls_optimizer,
            )

            for key in metrics:
                metrics[key] += float(flow_stats.get(key, 0.0)) + float(cls_stats.get(key, 0.0))
            counted += 1

        local_state = state_dict_to_cpu(local_model.state_dict())
        flow_state = state_dict_to_cpu(local_flow.state_dict())

        self._maybe_refresh_last_state(
            client_id=int(client_id),
            current_task_id=int(task_id),
            local_model=local_model,
            local_flow=local_flow,
        )
        self.client_last_task_id[int(client_id)] = int(task_id)

        if counted > 0:
            metrics = {k: v / counted for k, v in metrics.items()}

        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=local_state,
            personalized_state_dict=local_state,
            extra={
                "affcl_flow_state": flow_state,
                **metrics,
            },
        )