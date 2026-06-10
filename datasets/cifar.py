from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.nn import functional as F

from .base import DatasetBundle
from .registry import register_dataset


class U8TensorDataset(Dataset):
    def __init__(self, x_u8: torch.Tensor, y: torch.Tensor) -> None:
        self.x_u8 = x_u8.contiguous()
        self.y = y.long().contiguous()

    def __len__(self) -> int:
        return int(self.x_u8.size(0))

    def __getitem__(self, idx: int):
        return self.x_u8[idx], self.y[idx]


_norm_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_crop_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def normalize_batch(x01: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    device_key = (x01.device.type, x01.device.index if x01.device.type == "cuda" else -1, tuple(mean), tuple(std))
    if device_key not in _norm_cache:
        mean_t = torch.tensor(mean, device=x01.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=x01.device).view(1, 3, 1, 1)
        _norm_cache[device_key] = (mean_t, std_t)
    mean_t, std_t = _norm_cache[device_key]
    return (x01 - mean_t) / std_t


def random_crop_batch(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    batch, _, height, width = x.shape

    x_pad = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0.0)
    device_key = (
        x.device.type,
        x.device.index if x.device.type == "cuda" else -1,
        height,
        width,
        padding,
    )
    if device_key not in _crop_cache:
        _crop_cache[device_key] = (
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device),
        )
    ar_h, ar_w = _crop_cache[device_key]
    i = torch.randint(0, 2 * padding + 1, (batch,), device=x.device)
    j = torch.randint(0, 2 * padding + 1, (batch,), device=x.device)
    b = torch.arange(batch, device=x.device)[:, None, None]
    hh = i[:, None, None] + ar_h[None, :, None]
    ww = j[:, None, None] + ar_w[None, None, :]
    out = x_pad[b, :, hh, ww].permute(0, 3, 1, 2).contiguous()
    return out


def random_flip_batch(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if p <= 0.0:
        return x
    mask = torch.rand(x.size(0), device=x.device) < p
    if not mask.any():
        return x
    x = x.clone()
    x[mask] = torch.flip(x[mask], dims=[3])
    return x


def make_u8_collate_train(mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> Callable:
    def collate(batch):
        xs = torch.stack([item[0] for item in batch], dim=0)
        ys = torch.stack([item[1] for item in batch], dim=0).long()
        x01 = xs.float().div(255.0)
        x01 = random_crop_batch(x01, padding=4)
        x01 = random_flip_batch(x01, p=0.5)
        x = normalize_batch(x01, mean=mean, std=std)
        return x, ys

    return collate


def make_u8_collate_test(mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> Callable:
    def collate(batch):
        xs = torch.stack([item[0] for item in batch], dim=0)
        ys = torch.stack([item[1] for item in batch], dim=0).long()
        x01 = xs.float().div(255.0)
        x = normalize_batch(x01, mean=mean, std=std)
        return x, ys

    return collate


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _build_standard_cifar10(root: Path, download: bool) -> DatasetBundle:
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=True,
        download=download,
        transform=train_tf,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=False,
        download=download,
        transform=test_tf,
    )
    return DatasetBundle(
        name="cifar10",
        modality="image",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=np.asarray(train_dataset.targets, dtype=np.int64),
        test_targets=np.asarray(test_dataset.targets, dtype=np.int64),
        num_classes=10,
        class_names=list(train_dataset.classes),
        default_backbone="cifar_resnet18",
        metadata={"image_size": 32, "tensor_cache": False},
    )


def _ensure_cifar100_tensor_cache(root: Path, download: bool) -> tuple[Path, Path]:
    cache_dir = root / "cifar100_tensor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "cifar100_train_u8.pt"
    test_path = cache_dir / "cifar100_test_u8.pt"

    if train_path.exists() and test_path.exists():
        return train_path, test_path

    train_raw = torchvision.datasets.CIFAR100(root=str(root), train=True, download=download)
    test_raw = torchvision.datasets.CIFAR100(root=str(root), train=False, download=download)
    x_train = torch.from_numpy(train_raw.data).permute(0, 3, 1, 2).contiguous()
    y_train = torch.tensor(train_raw.targets, dtype=torch.long)
    x_test = torch.from_numpy(test_raw.data).permute(0, 3, 1, 2).contiguous()
    y_test = torch.tensor(test_raw.targets, dtype=torch.long)
    torch.save({"x_u8": x_train, "y": y_train}, train_path)
    torch.save({"x_u8": x_test, "y": y_test}, test_path)
    return train_path, test_path


def _build_cached_cifar100(root: Path, download: bool) -> DatasetBundle:
    train_path, test_path = _ensure_cifar100_tensor_cache(root=root, download=download)
    train_pack = torch.load(train_path, map_location="cpu")
    test_pack = torch.load(test_path, map_location="cpu")
    x_train = train_pack["x_u8"].contiguous()
    y_train = train_pack["y"].long().contiguous()
    x_test = test_pack["x_u8"].contiguous()
    y_test = test_pack["y"].long().contiguous()
    train_dataset = U8TensorDataset(x_train, y_train)
    test_dataset = U8TensorDataset(x_test, y_test)
    class_names = [str(i) for i in range(100)]
    return DatasetBundle(
        name="cifar100",
        modality="image",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=y_train.numpy(),
        test_targets=y_test.numpy(),
        num_classes=100,
        class_names=class_names,
        collate_train_fn=make_u8_collate_train(CIFAR100_MEAN, CIFAR100_STD),
        collate_test_fn=make_u8_collate_test(CIFAR100_MEAN, CIFAR100_STD),
        default_backbone="cifar_resnet18",
        metadata={"image_size": 32, "tensor_cache": True},
    )


def _build_standard_cifar100(root: Path, download: bool) -> DatasetBundle:
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(root),
        train=True,
        download=download,
        transform=train_tf,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(root),
        train=False,
        download=download,
        transform=test_tf,
    )
    return DatasetBundle(
        name="cifar100",
        modality="image",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=np.asarray(train_dataset.targets, dtype=np.int64),
        test_targets=np.asarray(test_dataset.targets, dtype=np.int64),
        num_classes=100,
        class_names=list(train_dataset.classes),
        default_backbone="cifar_resnet18",
        metadata={"image_size": 32, "tensor_cache": False},
    )


@register_dataset("cifar10")
def build_cifar10(args) -> DatasetBundle:
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    return _build_standard_cifar10(root=root, download=bool(args.download))


@register_dataset("cifar100")
def build_cifar100(args) -> DatasetBundle:
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    if bool(args.use_cifar100_tensor_cache):
        return _build_cached_cifar100(root=root, download=bool(args.download))
    return _build_standard_cifar100(root=root, download=bool(args.download))
