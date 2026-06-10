from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_backbone


class TextCNNBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_idx: int = 0,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        feature_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=k)
                for k in kernel_sizes
            ]
        )
        self.feature_proj = nn.Linear(num_filters * len(kernel_sizes), feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor, return_block_outputs: bool = False):
        # x: [B, L]
        emb = self.embedding(x)                          # [B, L, D]
        emb_t = emb.transpose(1, 2)                     # [B, D, L]

        pooled = []
        block_outputs: List[torch.Tensor] = []
        for conv in self.convs:
            h = F.relu(conv(emb_t), inplace=True)
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            pooled.append(h)
            block_outputs.append(h)

        feat = torch.cat(pooled, dim=1)
        feat = self.dropout(feat)
        feat = F.relu(self.feature_proj(feat), inplace=True)
        block_outputs.append(feat)

        if return_block_outputs:
            return feat, block_outputs
        return feat

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.forward_from_features(features)

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        modules = OrderedDict(
            embedding=self.embedding,
            convs=self.convs,
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
            "embedding": {},
            "convs": {},
            "feature_proj": {},
        }
        for key, value in state_dict.items():
            prefix, rest = key.split(".", 1)
            if prefix in grouped:
                grouped[prefix][rest] = value
        self.embedding.load_state_dict(grouped["embedding"], strict=strict)
        self.convs.load_state_dict(grouped["convs"], strict=strict)
        self.feature_proj.load_state_dict(grouped["feature_proj"], strict=strict)

    def load_head_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                head_state[key[len("head.") :]] = value
        self.head.load_state_dict(head_state, strict=strict)

    def named_backbone_parameters(self):
        for module_name in ["embedding", "convs", "feature_proj"]:
            module = getattr(self, module_name)
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def named_head_parameters(self):
        for name, param in self.head.named_parameters():
            yield f"head.{name}", param

    def get_trainable_block_prefixes(self) -> List[str]:
        return ["embedding", "convs", "feature_proj", "head"]

    def get_block_parameter_groups(self) -> List[Tuple[str, Iterable[nn.Parameter]]]:
        return [
            ("embedding", self.embedding.parameters()),
            ("convs", self.convs.parameters()),
            ("feature_proj", self.feature_proj.parameters()),
            ("head", self.head.parameters()),
        ]

    def get_bn_layers(self):
        return []


@register_backbone("text_cnn")
def make_text_cnn(num_classes: int, args=None) -> nn.Module:
    if args is None:
        raise ValueError("text_cnn requires args with text_vocab_size and text_pad_idx.")
    vocab_size = int(getattr(args, "text_vocab_size"))
    pad_idx = int(getattr(args, "text_pad_idx", 0))
    return TextCNNBackbone(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
        embed_dim=int(getattr(args, "text_embed_dim", 128)),
        num_filters=int(getattr(args, "text_num_filters", 128)),
        kernel_sizes=(3, 4, 5),
        feature_dim=int(getattr(args, "text_feature_dim", 256)),
        dropout=float(getattr(args, "text_dropout", 0.1)),
    )