from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class LocalUpdate:
    client_id: int
    task_id: int
    num_samples: int
    state_dict: Dict[str, torch.Tensor]
    personalized_state_dict: Optional[Dict[str, torch.Tensor]] = None
    aggregation_weight: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseClient:
    def __init__(self, args, dataset_bundle, task_labels, device: torch.device) -> None:
        self.args = args
        self.dataset_bundle = dataset_bundle
        self.task_labels = task_labels
        self.device = device

    def fit(self, global_payload, client_id: int, task_id: int, train_indices):
        raise NotImplementedError