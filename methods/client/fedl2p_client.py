from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import copy
import numpy as np
import torch

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore

from utils.misc import state_dict_to_cpu

from ..fedl2p_modules import (
    LRMetaNet,
    blend_bn_running_stats,
    build_optimizer_with_block_lrs,
    collect_block_stats,
    compute_bn_divergences,
    divergence_to_beta,
    get_param_group_index,
)
from .base_client import BaseClient, LocalUpdate
from .common import build_loader, build_loss_context, compute_task_loss, maybe_subsample_indices


class FedL2PClient(BaseClient):
    """Practical FedL2P adaptation for this simple continual FL codebase.

    The original paper learns BNNet + LRNet with hypergradients/IFT on a fixed
    pretrained global model. Here we keep the core idea of client-conditioned
    personalization strategy learning, but use:
    1) a learned LR meta-net with a first-order one-step meta-update; and
    2) a deterministic BN blending heuristic from local/global BN divergence.
    """

    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        super().__init__(args, dataset_bundle, task_labels, device)
        self.local_states: Dict[int, Dict[str, torch.Tensor]] = {}

    def _make_lr_meta_net(self, model) -> LRMetaNet:
        num_blocks = len(model.get_trainable_block_prefixes())
        stats_dim = 2 * (num_blocks - 1)
        return LRMetaNet(
            input_dim=stats_dim,
            output_dim=num_blocks,
            hidden_dim=int(self.args.fedl2p_hidden_dim),
            max_scale=float(self.args.fedl2p_max_lr_scale),
        )

    def _split_train_val(self, train_indices, client_id: int) -> Tuple[List[int], List[int]]:
        if len(train_indices) <= 1:
            return list(train_indices), list(train_indices)
        rng = np.random.RandomState(self.args.seed + 4000 + client_id)
        shuffled = np.asarray(train_indices).copy()
        rng.shuffle(shuffled)
        train_ratio = float(self.args.fedl2p_train_ratio)
        split = max(1, int(round(len(shuffled) * train_ratio)))
        split = min(split, len(shuffled) - 1)
        return shuffled[:split].tolist(), shuffled[split:].tolist()

    def _meta_update_lr_net(self, global_model, lr_meta_net, task_id: int, inner_train_indices, inner_val_indices):
        if len(inner_train_indices) == 0 or len(inner_val_indices) == 0:
            stats_loader = build_loader(
                dataset_bundle=self.dataset_bundle,
                indices=inner_train_indices if len(inner_train_indices) > 0 else inner_val_indices,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )
            stats_vec = collect_block_stats(global_model, stats_loader, self.device)
            return lr_meta_net, lr_meta_net(stats_vec).detach()

        stats_indices = maybe_subsample_indices(
            inner_train_indices,
            ratio=float(self.args.fedl2p_stats_ratio),
            seed=int(self.args.seed + 5000 + task_id),
        )
        stats_loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=stats_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
        stats_vec = collect_block_stats(global_model, stats_loader, self.device)

        lr_meta_net.train()
        meta_optimizer = torch.optim.Adam(lr_meta_net.parameters(), lr=float(self.args.fedl2p_meta_lr))

        train_loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=inner_train_indices,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )
        val_loader = build_loader(
            dataset_bundle=self.dataset_bundle,
            indices=inner_val_indices,
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

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        for _ in range(max(1, int(self.args.fedl2p_meta_steps))):
            try:
                x_tr, y_tr = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_tr, y_tr = next(train_iter)
            try:
                x_val, y_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_val, y_val = next(val_iter)

            x_tr = x_tr.to(self.device, non_blocking=True)
            y_tr = y_tr.to(self.device, non_blocking=True)
            x_val = x_val.to(self.device, non_blocking=True)
            y_val = y_val.to(self.device, non_blocking=True)

            lr_scales = lr_meta_net(stats_vec)
            params = OrderedDict((name, param) for name, param in global_model.named_parameters())
            logits_tr = functional_call(global_model, params, (x_tr,))
            train_loss = compute_task_loss(logits_tr, y_tr, self.args, current_classes, class_map)
            grads = torch.autograd.grad(train_loss, list(params.values()), create_graph=False, retain_graph=False)

            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                group_idx = get_param_group_index(global_model, name)
                updated_params[name] = param - float(self.args.lr) * lr_scales[group_idx] * grad

            logits_val = functional_call(global_model, updated_params, (x_val,))
            val_loss = compute_task_loss(logits_val, y_val, self.args, current_classes, class_map)
            meta_optimizer.zero_grad()
            val_loss.backward()
            meta_optimizer.step()

        lr_meta_net.eval()
        with torch.no_grad():
            final_lr_scales = lr_meta_net(stats_vec).detach()
        return lr_meta_net, final_lr_scales

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        if len(train_indices) == 0:
            return None

        global_model = global_payload["global_model"]
        lr_meta_net = self._make_lr_meta_net(global_model).to(self.device)
        lr_meta_net.load_state_dict(global_payload["lr_meta_state"], strict=True)

        local_model = copy.deepcopy(global_model).to(self.device)

        if client_id in self.local_states:
            prev_local_model = copy.deepcopy(global_model).to(self.device)
            prev_local_model.load_state_dict(self.local_states[client_id], strict=True)
            bn_divs = compute_bn_divergences(global_model, prev_local_model)
            beta = divergence_to_beta(bn_divs)
            local_model.load_state_dict(global_model.state_dict(), strict=True)
            temp_local = prev_local_model
            blend_bn_running_stats(global_model=global_model, local_model=temp_local, betas=beta)
            for (_, temp_bn), (_, cur_bn) in zip(temp_local.get_bn_layers(), local_model.get_bn_layers()):
                cur_bn.running_mean.data.copy_(temp_bn.running_mean.data)
                cur_bn.running_var.data.copy_(temp_bn.running_var.data)
        else:
            local_model.load_state_dict(global_model.state_dict(), strict=True)

        inner_train_indices, inner_val_indices = self._split_train_val(train_indices, client_id=client_id)
        lr_meta_net, lr_scales = self._meta_update_lr_net(
            global_model=local_model,
            lr_meta_net=lr_meta_net,
            task_id=task_id,
            inner_train_indices=inner_train_indices,
            inner_val_indices=inner_val_indices,
        )

        optimizer = build_optimizer_with_block_lrs(
            model=local_model,
            base_lr=float(self.args.lr),
            lr_scales=lr_scales,
            momentum=float(self.args.momentum),
            weight_decay=float(self.args.weight_decay),
        )
        train_loader = build_loader(
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
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = local_model(x)
                loss = compute_task_loss(logits, y, self.args, current_classes, class_map)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        state_dict = state_dict_to_cpu(local_model.state_dict())
        self.local_states[client_id] = state_dict
        return LocalUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            num_samples=int(len(train_indices)),
            state_dict=state_dict,
            personalized_state_dict=state_dict,
            extra={
                "lr_meta_state": state_dict_to_cpu(lr_meta_net.state_dict()),
            },
        )