from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class LRMetaNet(nn.Module):
    """Practical FedL2P meta-net for block-wise learning-rate multipliers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100, max_scale: float = 5.0) -> None:
        super().__init__()
        self.max_scale = float(max_scale)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 1.0)

    def forward(self, stats_vector: torch.Tensor) -> torch.Tensor:
        raw = self.net(stats_vector)
        return torch.clamp(raw, min=0.0, max=self.max_scale)


def collect_block_stats(model, loader, device: torch.device) -> torch.Tensor:
    model.eval()
    sums: List[torch.Tensor] = []
    sq_sums: List[torch.Tensor] = []
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _, block_outputs = model.extract_features(x, return_block_outputs=True)
            flat_outputs = []
            for feat in block_outputs:
                feat_flat = feat.float().view(feat.size(0), -1)
                flat_outputs.append(feat_flat)
            if not sums:
                sums = [torch.zeros((), device=device) for _ in flat_outputs]
                sq_sums = [torch.zeros((), device=device) for _ in flat_outputs]
            for idx, feat in enumerate(flat_outputs):
                sums[idx] += feat.mean()
                sq_sums[idx] += feat.std(unbiased=False)
            count += 1
    if count == 0:
        num_stats_blocks = max(1, len(model.get_trainable_block_prefixes()) - 1)
        return torch.zeros((2 * num_stats_blocks,), device=device)
    vec = []
    for mean_sum, std_sum in zip(sums, sq_sums):
        vec.append(mean_sum / count)
        vec.append(std_sum / count)
    return torch.stack(vec).float()


def compute_bn_divergences(global_model, local_model, eps: float = 1e-6) -> List[float]:
    global_bn_layers = global_model.get_bn_layers()
    local_bn_layers = local_model.get_bn_layers()
    divergences: List[float] = []
    for (_, g_bn), (_, l_bn) in zip(global_bn_layers, local_bn_layers):
        g_mean = g_bn.running_mean.detach().float()
        g_var = g_bn.running_var.detach().float().clamp_min(eps)
        l_mean = l_bn.running_mean.detach().float()
        l_var = l_bn.running_var.detach().float().clamp_min(eps)

        kl_gl = 0.5 * torch.mean(
            torch.log(l_var / g_var) + (g_var + (g_mean - l_mean) ** 2) / l_var - 1.0
        )
        kl_lg = 0.5 * torch.mean(
            torch.log(g_var / l_var) + (l_var + (l_mean - g_mean) ** 2) / g_var - 1.0
        )
        divergences.append(float((0.5 * (kl_gl + kl_lg)).item()))
    return divergences


def divergence_to_beta(divergences: Sequence[float]) -> List[float]:
    betas: List[float] = []
    for div in divergences:
        beta = float(div / (1.0 + div))
        betas.append(max(0.0, min(1.0, beta)))
    return betas


def blend_bn_running_stats(global_model, local_model, betas: Sequence[float]) -> None:
    global_bn_layers = global_model.get_bn_layers()
    local_bn_layers = local_model.get_bn_layers()
    for idx, ((_, g_bn), (_, l_bn)) in enumerate(zip(global_bn_layers, local_bn_layers)):
        beta = float(betas[idx])
        l_bn.running_mean.data.copy_((1.0 - beta) * g_bn.running_mean.data + beta * l_bn.running_mean.data)
        l_bn.running_var.data.copy_((1.0 - beta) * g_bn.running_var.data + beta * l_bn.running_var.data)


def build_optimizer_with_block_lrs(model, base_lr: float, lr_scales: torch.Tensor, momentum: float, weight_decay: float):
    param_groups = []
    prefixes = model.get_trainable_block_prefixes()
    for idx, prefix in enumerate(prefixes):
        params = []
        for name, param in model.named_parameters():
            if name == prefix or name.startswith(prefix + "."):
                params.append(param)
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "lr": float(base_lr * lr_scales[idx].item()),
                "momentum": momentum,
                "weight_decay": weight_decay,
            }
        )
    return torch.optim.SGD(param_groups)


def get_param_group_index(model, parameter_name: str) -> int:
    prefixes = model.get_trainable_block_prefixes()
    for idx, prefix in enumerate(prefixes):
        if parameter_name == prefix or parameter_name.startswith(prefix + "."):
            return idx
    return len(prefixes) - 1