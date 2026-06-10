from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import RandomPermutation, ReversePermutation


EPS = 1e-30


class ConditionedMLP(nn.Module):
    """
    Small context-conditioned MLP used inside affine coupling.
    It is intentionally lightweight and benchmark-friendly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        hidden_features: int = 512,
    ) -> None:
        super().__init__()
        self.context_features = int(context_features)
        self.net = nn.Sequential(
            nn.Linear(in_features + context_features, hidden_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("ConditionedMLP requires context for AF-FCL conditional flow.")
        h = torch.cat([x, context], dim=1)
        return self.net(h)


def build_conditional_flow(
    feature_dim: int,
    num_classes: int,
    hidden_features: int = 512,
    num_layers: int = 4,
) -> Flow:
    transforms = []
    for layer_idx in range(num_layers):
        if layer_idx < max(1, num_layers // 2):
            transforms.append(ReversePermutation(features=feature_dim))
        else:
            transforms.append(RandomPermutation(features=feature_dim))

        mask = (torch.arange(0, feature_dim) >= (feature_dim // 2)).float()

        def _make_net(in_d, out_d, context_features=num_classes, hidden=hidden_features):
            return ConditionedMLP(
                in_features=in_d,
                out_features=out_d,
                context_features=context_features,
                hidden_features=hidden,
            )

        transforms.append(
            AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=_make_net,
            )
        )

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal(shape=[feature_dim])
    return Flow(transform, base_dist)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def multiclass_cross_entropy_probs(student_probs: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    student_probs = torch.pow(student_probs + EPS, 1.0 / temperature)
    student_probs = student_probs / (student_probs.sum(dim=1, keepdim=True) + EPS)
    teacher_probs = torch.pow(teacher_probs + EPS, 1.0 / temperature)
    teacher_probs = teacher_probs / (teacher_probs.sum(dim=1, keepdim=True) + EPS)

    outputs = torch.log(student_probs + EPS)
    outputs = torch.sum(outputs * teacher_probs, dim=1)
    return -torch.mean(outputs)


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def flow_log_prob_and_latent(flow: Flow, features: torch.Tensor, labels: torch.Tensor, num_classes: int):
    context = one_hot(labels, num_classes).to(features.device)
    z, logabsdet = flow._transform(features, context=context)
    log_prob = flow._distribution.log_prob(z) + logabsdet
    return log_prob, z


def sample_flow_features(
    flow: Flow,
    labels_pool: Sequence[int],
    batch_size: int,
    num_classes: int,
    feature_dim: int,
    device: torch.device,
):
    sampled_labels_np = np.random.choice(np.asarray(labels_pool, dtype=np.int64), size=batch_size)
    sampled_labels = torch.tensor(sampled_labels_np, device=device, dtype=torch.long)
    context = one_hot(sampled_labels, num_classes).to(device)

    z = torch.randn((batch_size, feature_dim), device=device)
    sampled_features, _ = flow._transform.inverse(z, context=context)
    return sampled_features.detach(), sampled_labels, z.detach()


def probability_in_localdata(
    local_latents: torch.Tensor,
    local_labels: torch.Tensor,
    fallback_prob: torch.Tensor,
    sampled_latents: torch.Tensor,
    sampled_labels: torch.Tensor,
) -> torch.Tensor:
    probs = torch.zeros(sampled_latents.size(0), device=sampled_latents.device)
    unique_labels = sampled_labels.unique().tolist()

    for label_i in unique_labels:
        label_i = int(label_i)
        local_mask = (local_labels == label_i)
        sampled_mask = (sampled_labels == label_i)

        if local_mask.sum() > 0:
            z_local = local_latents[local_mask]
            mean = z_local.mean(dim=0, keepdim=True)
            var = ((z_local - mean) ** 2).mean(dim=0, keepdim=True)

            z_sample = sampled_latents[sampled_mask]
            prob_per_dim = (
                (1.0 / math.sqrt(2.0 * math.pi))
                * torch.pow(var + EPS, -0.5)
                * torch.exp(-0.5 * torch.pow(z_sample - mean, 2) * torch.pow(var + EPS, -1.0))
            )
            probs[sampled_mask] = prob_per_dim.mean(dim=1)
        else:
            probs[sampled_mask] = fallback_prob

    return probs


def mean_nonzero_or_zero(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros(())
    return x.mean()