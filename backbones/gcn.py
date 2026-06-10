from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_backbone


try:
    from torch_geometric.nn import GCNConv
except ImportError as exc:
    raise ImportError(
        "GCN backbone requires torch-geometric. Please install torch-geometric first."
    ) from exc


class CoraGCNBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = float(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, batch, return_block_outputs: bool = False):
        x = batch.x
        edge_index = batch.edge_index
        target_nodes = batch.target_nodes

        block_outputs: List[torch.Tensor] = []

        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        block_outputs.append(h[target_nodes])

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        feat = h[target_nodes]
        block_outputs.append(feat)

        if return_block_outputs:
            return feat, block_outputs
        return feat

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, batch) -> torch.Tensor:
        features = self.extract_features(batch)
        return self.forward_from_features(features)

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        modules = OrderedDict(
            conv1=self.conv1,
            bn1=self.bn1,
            conv2=self.conv2,
            bn2=self.bn2,
        )
        state: Dict[str, torch.Tensor] = {}
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value
        return state

    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        return {f"head.{key}": value for key, value in self.head.state_dict().items()}

    def load_backbone_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        grouped: Dict[str, Dict[str, torch.Tensor]] = {
            "conv1": {},
            "bn1": {},
            "conv2": {},
            "bn2": {},
        }
        for key, value in state_dict.items():
            prefix, rest = key.split(".", 1)
            if prefix in grouped:
                grouped[prefix][rest] = value
        self.conv1.load_state_dict(grouped["conv1"], strict=strict)
        self.bn1.load_state_dict(grouped["bn1"], strict=strict)
        self.conv2.load_state_dict(grouped["conv2"], strict=strict)
        self.bn2.load_state_dict(grouped["bn2"], strict=strict)

    def load_head_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                head_state[key[len("head.") :]] = value
        self.head.load_state_dict(head_state, strict=strict)

    def named_backbone_parameters(self):
        for module_name in ["conv1", "bn1", "conv2", "bn2"]:
            module = getattr(self, module_name)
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def named_head_parameters(self):
        for name, param in self.head.named_parameters():
            yield f"head.{name}", param

    def get_trainable_block_prefixes(self) -> List[str]:
        return ["conv1", "bn1", "conv2", "bn2", "head"]

    def get_block_parameter_groups(self) -> List[Tuple[str, Iterable[nn.Parameter]]]:
        return [
            ("conv1", self.conv1.parameters()),
            ("bn1", self.bn1.parameters()),
            ("conv2", self.conv2.parameters()),
            ("bn2", self.bn2.parameters()),
            ("head", self.head.parameters()),
        ]

    def get_bn_layers(self):
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                layers.append((name, module))
        return layers


@register_backbone("cora_gcn")
def make_cora_gcn(num_classes: int, args=None) -> nn.Module:
    if args is None:
        raise ValueError("cora_gcn requires args.graph_input_dim.")
    input_dim = int(getattr(args, "graph_input_dim"))
    return CoraGCNBackbone(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=int(getattr(args, "gcn_hidden_dim", 256)),
        dropout=float(getattr(args, "gcn_dropout", 0.5)),
    )