from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

from .base import DatasetBundle
from .registry import register_dataset


PACS_CANONICAL_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_name(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def _match_domain_dirs(root: Path) -> List[Tuple[str, Path]]:
    children = [p for p in root.iterdir() if p.is_dir()]
    normalized = {_normalize_name(p.name): p for p in children}
    matched = []
    for domain in PACS_CANONICAL_DOMAINS:
        if domain not in normalized:
            raise FileNotFoundError(
                f"Could not find PACS domain folder '{domain}' under {root}. "
                f"Expected domains: {PACS_CANONICAL_DOMAINS}"
            )
        matched.append((domain, normalized[domain]))
    return matched


def _discover_class_names(domain_dirs: Sequence[Tuple[str, Path]]) -> List[str]:
    sets = []
    for _, domain_dir in domain_dirs:
        class_names = sorted([p.name for p in domain_dir.iterdir() if p.is_dir()])
        sets.append(set(class_names))
    class_intersection = sorted(set.intersection(*sets))
    if len(class_intersection) == 0:
        raise RuntimeError("PACS class intersection across domains is empty.")
    return class_intersection


def _stratified_split_indices(
    samples: List[Tuple[Path, int, int]],
    test_ratio: float,
    seed: int,
) -> tuple[List[Tuple[Path, int, int]], List[Tuple[Path, int, int]]]:
    rng = np.random.RandomState(seed)
    by_group = defaultdict(list)
    for item in samples:
        _, class_id, domain_id = item
        by_group[(domain_id, class_id)].append(item)

    train_items = []
    test_items = []
    for _, items in by_group.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            train_items.extend(items)
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)
        test_items.extend(items[:n_test])
        train_items.extend(items[n_test:])
    return train_items, test_items


class PACSListDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int, int]], transform=None) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, label, domain_id = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@register_dataset("pacs")
def build_pacs(args) -> DatasetBundle:
    root = Path(args.data_root)

    pacs_candidates = [
        root / "PACS",
        root / "pacs",
    ]
    pacs_root = None
    for candidate in pacs_candidates:
        if candidate.exists():
            pacs_root = candidate
            break
    if pacs_root is None:
        raise FileNotFoundError(
            f"Could not find PACS under {root}. "
            "Please manually place the PACS folder at data/PACS or data/pacs."
        )

    domain_dirs = _match_domain_dirs(pacs_root)
    class_names = _discover_class_names(domain_dirs)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    items: List[Tuple[Path, int, int]] = []
    for domain_id, (_, domain_dir) in enumerate(domain_dirs):
        for class_name in class_names:
            class_dir = domain_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    items.append((image_path, class_to_idx[class_name], domain_id))

    if len(items) == 0:
        raise RuntimeError(f"No PACS images found under {pacs_root}")

    train_items, test_items = _stratified_split_indices(
        samples=items,
        test_ratio=float(args.pacs_test_ratio),
        seed=int(args.pacs_split_seed),
    )

    image_size = int(args.pacs_image_size)
    train_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = PACSListDataset(train_items, transform=train_tf)
    test_dataset = PACSListDataset(test_items, transform=test_tf)

    train_targets = np.asarray([label for _, label, _ in train_items], dtype=np.int64)
    test_targets = np.asarray([label for _, label, _ in test_items], dtype=np.int64)
    train_task_ids = np.asarray([domain_id for _, _, domain_id in train_items], dtype=np.int64)
    test_task_ids = np.asarray([domain_id for _, _, domain_id in test_items], dtype=np.int64)
    task_names = [domain_name for domain_name, _ in domain_dirs]

    return DatasetBundle(
        name="pacs",
        modality="image",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=train_targets,
        test_targets=test_targets,
        num_classes=len(class_names),
        class_names=class_names,
        default_backbone="resnet18_imagenet",
        metadata={
            "image_size": image_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "domains": task_names,
            "source_root": str(pacs_root),
        },
        train_task_ids=train_task_ids,
        test_task_ids=test_task_ids,
        task_names=task_names,
        default_scenario="domain-il",
    )