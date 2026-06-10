from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    name: str
    modality: str
    train_dataset: Dataset
    test_dataset: Dataset
    train_targets: np.ndarray
    test_targets: np.ndarray
    num_classes: int
    class_names: List[str]
    collate_train_fn: Optional[Callable] = None
    collate_test_fn: Optional[Callable] = None
    default_backbone: str = "cifar_resnet18"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional task/domain metadata for non-class scenarios
    train_task_ids: Optional[np.ndarray] = None
    test_task_ids: Optional[np.ndarray] = None
    task_names: Optional[List[str]] = None
    default_scenario: str = "class-il"

    @property
    def num_train_samples(self) -> int:
        return int(len(self.train_targets))

    @property
    def num_test_samples(self) -> int:
        return int(len(self.test_targets))