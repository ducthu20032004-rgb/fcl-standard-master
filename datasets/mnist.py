from __future__ import annotations

import os

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

import numpy as np
import torch

try:
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

import torchvision
import torchvision.transforms as T

from .base import DatasetBundle
from .registry import register_dataset


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@register_dataset("mnist")
def build_mnist(args) -> DatasetBundle:
    train_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=args.data_root,
        train=True,
        download=bool(args.download),
        transform=train_tf,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_root,
        train=False,
        download=bool(args.download),
        transform=test_tf,
    )

    return DatasetBundle(
        name="mnist",
        modality="image",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=np.asarray(train_dataset.targets, dtype=np.int64),
        test_targets=np.asarray(test_dataset.targets, dtype=np.int64),
        num_classes=10,
        class_names=[str(i) for i in range(10)],
        collate_train_fn=None,
        collate_test_fn=None,
        default_backbone="mnist_cnn",
        metadata={
            "image_size": 28,
            "channels": 1,
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
            "source": "torchvision",
        },
        train_task_ids=None,
        test_task_ids=None,
        task_names=None,
        default_scenario="class-il",
    )