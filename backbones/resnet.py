from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch

import torch.nn as nn

from torchvision.models import resnet18 as tv_resnet18
from torchvision.models import resnet34 as tv_resnet34
from torchvision.models import resnet50 as tv_resnet50
from torchvision.models import resnet101 as tv_resnet101
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, VisionTransformer
from .registry import register_backbone


class CIFARResNet18(nn.Module):
    """ResNet-18 wrapper exposing backbone/head utilities.

    These helpers keep the backbone registry simple while enabling methods that
    need decoupled backbone/head states (FedAS), feature extraction (FedAS),
    block-wise control (FedALA, FedL2P), and BN inspection (FedL2P).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.head = nn.Linear(base.fc.in_features, num_classes)

    def extract_features(self, x: torch.Tensor, return_block_outputs: bool = False):
        block_outputs: List[torch.Tensor] = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        block_outputs.append(x)

        x = self.layer1(x)
        block_outputs.append(x)
        x = self.layer2(x)
        block_outputs.append(x)
        x = self.layer3(x)
        block_outputs.append(x)
        x = self.layer4(x)
        block_outputs.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            layer1=self.layer1,
            layer2=self.layer2,
            layer3=self.layer3,
            layer4=self.layer4,
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
            "layer1": {},
            "layer2": {},
            "layer3": {},
            "layer4": {},
        }
        for key, value in state_dict.items():
            prefix, rest = key.split(".", 1)
            if prefix in grouped:
                grouped[prefix][rest] = value
        self.conv1.load_state_dict(grouped["conv1"], strict=strict)
        self.bn1.load_state_dict(grouped["bn1"], strict=strict)
        self.layer1.load_state_dict(grouped["layer1"], strict=strict)
        self.layer2.load_state_dict(grouped["layer2"], strict=strict)
        self.layer3.load_state_dict(grouped["layer3"], strict=strict)
        self.layer4.load_state_dict(grouped["layer4"], strict=strict)

    def load_head_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                head_state[key[len("head.") :]] = value
        self.head.load_state_dict(head_state, strict=strict)

    def named_backbone_parameters(self):
        for module_name in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]:
            module = getattr(self, module_name)
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def named_head_parameters(self):
        for name, param in self.head.named_parameters():
            yield f"head.{name}", param

    def get_trainable_block_prefixes(self) -> List[str]:
        return ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "head"]

    def get_block_parameter_groups(self) -> List[Tuple[str, Iterable[nn.Parameter]]]:
        return [
            ("conv1", self.conv1.parameters()),
            ("bn1", self.bn1.parameters()),
            ("layer1", self.layer1.parameters()),
            ("layer2", self.layer2.parameters()),
            ("layer3", self.layer3.parameters()),
            ("layer4", self.layer4.parameters()),
            ("head", self.head.parameters()),
        ]

    def get_bn_layers(self) -> List[Tuple[str, nn.BatchNorm2d]]:
        layers: List[Tuple[str, nn.BatchNorm2d]] = []
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                layers.append((name, module))
        return layers


@register_backbone("cifar_resnet18")
def make_cifar_resnet18(num_classes: int, args=None) -> nn.Module:
    del args
    return CIFARResNet18(num_classes=num_classes)

@register_backbone("resnet18")
def make_resnet18(num_classes: int, args=None) -> nn.Module:
    model = tv_resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@register_backbone("resnet34")
def make_resnet34(num_classes: int, args=None) -> nn.Module:
    model = tv_resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@register_backbone("resnet50")
def make_resnet50(num_classes: int, args=None) -> nn.Module:
    model = tv_resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@register_backbone("resnet101")
def make_resnet101(num_classes: int, args=None) -> nn.Module:
    model = tv_resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class CIFARViT(nn.Module):
    """
    ViT cho CIFAR (32x32), wrapper tương thích 100% với get_resnet18_blocks()
    và compute_feature_resnet18() mà không cần sửa code đó.

    get_resnet18_blocks() chạy:
        for block_name, ops in blocks.items():
            features = ops(features)
            if block_name == target: break
        output = flatten(features, 1)

    Nên mỗi block phải:
        - Nhận tensor từ block trước
        - Trả tensor cho block sau
        - Block cuối (block4) trả (B, hidden_dim) để flatten = chính nó
    """

    # ── Internal block modules ────────────────────────────────────────────

    class _PatchEmbedBlock(nn.Module):
        """
        block0: ảnh (B,3,32,32) → token sequence (B, 1+64, 512)
        Tự chứa conv_proj + class_token + pos_embedding.
        """
        def __init__(self, conv_proj, class_token, pos_embedding):
            super().__init__()
            self.conv_proj     = conv_proj
            self.class_token   = class_token   # nn.Parameter — register thủ công
            self.pos_embedding = pos_embedding  # nn.Parameter

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            x = self.conv_proj(x)                        # (B, 512, 8, 8)
            x = x.flatten(2).transpose(1, 2)             # (B, 64, 512)
            cls = self.class_token.expand(B, -1, -1)     # (B, 1, 512)
            x = torch.cat([cls, x], dim=1)               # (B, 65, 512)
            x = x + self.pos_embedding                   # (B, 65, 512)
            return x

    class _EncoderLayersBlock(nn.Module):
        """
        block1-3: (B, 65, 512) → (B, 65, 512)
        Gom một hoặc nhiều encoder TransformerEncoderLayer.
        """
        def __init__(self, layers: list):
            super().__init__()
            self.layers = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    class _FinalEncoderBlock(nn.Module):
        """
        block4: (B, 65, 512) → (B, 512)  ← CLS token sau LayerNorm
        flatten(x, 1) trên (B, 512) = chính nó → compute_feature hoạt động đúng.
        """
        def __init__(self, layers: list, ln: nn.Module):
            super().__init__()
            self.layers = nn.Sequential(*layers)
            self.ln     = ln

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers(x)   # (B, 65, 512)
            x = self.ln(x)       # (B, 65, 512)
            return x[:, 0]       # (B, 512) — CLS token

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        vit = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=2048,
            num_classes=num_classes,
        )

        enc_layers = list(vit.encoder.layers.children())  # 6 TransformerEncoderLayer

        # block0: patch embed
        self.conv1 = self._PatchEmbedBlock(
            conv_proj     = vit.conv_proj,
            class_token   = vit.class_token,
            pos_embedding = vit.encoder.pos_embedding,
        )

        # block1-3: mỗi block 1-2 encoder layers
        # Phân chia: [0], [1], [2,3], [4,5]
        # → 4 nhóm tương ứng layer1..layer4
        self.layer1 = self._EncoderLayersBlock([enc_layers[0]])
        self.layer2 = self._EncoderLayersBlock([enc_layers[1]])
        self.layer3 = self._EncoderLayersBlock([enc_layers[2], enc_layers[3]])

        # block4: 2 layer cuối + ln + lấy CLS → (B, 512)
        self.layer4 = self._FinalEncoderBlock(
            layers = [enc_layers[4], enc_layers[5]],
            ln     = vit.encoder.ln,
        )

        # Alias để get_resnet18_blocks() không bị KeyError
        self.bn1     = nn.Identity()
        self.relu    = nn.Identity()
        self.maxpool = nn.Identity()

        # Head
        self.head = vit.heads.head  # nn.Linear(512, num_classes)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)    # (B, 65, 512)
        x = self.layer1(x)   # (B, 65, 512)
        x = self.layer2(x)   # (B, 65, 512)
        x = self.layer3(x)   # (B, 65, 512)
        x = self.layer4(x)   # (B, 512)
        return self.head(x)


@register_backbone("cifar_vit")
def make_cifar_vit(num_classes: int, args=None) -> nn.Module:
    return CIFARViT(num_classes=num_classes)