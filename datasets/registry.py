from __future__ import annotations

from typing import Callable, Dict

from .base import DatasetBundle

DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str) -> Callable:
    def decorator(builder: Callable) -> Callable:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASET_REGISTRY[name] = builder
        return builder
    return decorator


def build_dataset(name: str, args) -> DatasetBundle:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](args)


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)