from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_backbone


class MNISTCNN(nn.Module):
    """
    Simple backbone for MNIST that is:
    - fast
    - stable
    - compatible with current methods that expect:
        * extract_features(...)
        * forward_from_features(...)
        * head.in_features
        * backbone/head state helpers
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.feature_proj = nn.Linear(64 * 7 * 7, 256)
        self.head = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor, return_block_outputs: bool = False):
        block_outputs: List[torch.Tensor] = []

        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        block_outputs.append(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))
        block_outputs.append(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.feature_proj(x), inplace=True)
        block_outputs.append(x)

        if return_block_outputs:
            return x, block_outputs
        return x

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.forward_from_features(features)

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        modules = OrderedDict(
            conv1=self.conv1,
            bn1=self.bn1,
            conv2=self.conv2,
            bn2=self.bn2,
            feature_proj=self.feature_proj,
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
            "feature_proj": {},
        }
        for key, value in state_dict.items():
            prefix, rest = key.split(".", 1)
            if prefix in grouped:
                grouped[prefix][rest] = value

        self.conv1.load_state_dict(grouped["conv1"], strict=strict)
        self.bn1.load_state_dict(grouped["bn1"], strict=strict)
        self.conv2.load_state_dict(grouped["conv2"], strict=strict)
        self.bn2.load_state_dict(grouped["bn2"], strict=strict)
        self.feature_proj.load_state_dict(grouped["feature_proj"], strict=strict)

    def load_head_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                head_state[key[len("head.") :]] = value
        self.head.load_state_dict(head_state, strict=strict)

    def named_backbone_parameters(self):
        for module_name in ["conv1", "bn1", "conv2", "bn2", "feature_proj"]:
            module = getattr(self, module_name)
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def named_head_parameters(self):
        for name, param in self.head.named_parameters():
            yield f"head.{name}", param

    def get_trainable_block_prefixes(self) -> List[str]:
        return ["conv1", "bn1", "conv2", "bn2", "feature_proj", "head"]

    def get_block_parameter_groups(self) -> List[Tuple[str, Iterable[nn.Parameter]]]:
        return [
            ("conv1", self.conv1.parameters()),
            ("bn1", self.bn1.parameters()),
            ("conv2", self.conv2.parameters()),
            ("bn2", self.bn2.parameters()),
            ("feature_proj", self.feature_proj.parameters()),
            ("head", self.head.parameters()),
        ]

    def get_bn_layers(self):
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                layers.append((name, module))
        return layers


@register_backbone("mnist_cnn")
def make_mnist_cnn(num_classes: int, args=None) -> nn.Module:
    del args
    return MNISTCNN(num_classes=num_classes)