from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

from .base import DatasetBundle
from .registry import register_dataset


DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DomainNetListDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int, int]], transform=None) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, label, _domain_id = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _normalize_name(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def _find_domainnet_root(root: Path) -> Path:
    candidates = [
        root / "DomainNet",
        root / "domainnet",
        root,
    ]

    for candidate in candidates:
        has_all_train = all((candidate / f"{d}_train.txt").exists() for d in DOMAINNET_DOMAINS)
        has_all_test = all((candidate / f"{d}_test.txt").exists() for d in DOMAINNET_DOMAINS)
        if has_all_train and has_all_test:
            return candidate

    raise FileNotFoundError(
        f"Could not find DomainNet split files under {root}. "
        "Expected either data/DomainNet, data/domainnet, or data/ itself."
    )


def _parse_split_line(line: str) -> Tuple[Path, Optional[int]]:
    """
    Supports both:
      1) 'relative/path/to/image.jpg 123'
      2) 'relative/path/to/image.jpg'
    """
    parts = line.split()
    if len(parts) == 0:
        raise ValueError("Empty split-file line encountered.")

    # Try: last token is an integer label
    try:
        raw_label = int(parts[-1])
        rel_path_str = " ".join(parts[:-1]).strip()
        if rel_path_str == "":
            raise ValueError("Missing image path before label.")
        return Path(rel_path_str), raw_label
    except ValueError:
        # Fallback: whole line is just the relative path
        return Path(line.strip()), None


def _infer_class_name(rel_path: Path, domain_name: str) -> str:
    """
    Handles cases like:
      - real/shark/img.jpg                -> shark
      - sketch/shark/img.jpg              -> shark
      - clipart/train/trunk13/img.jpg     -> trunk13
      - clipart/test/trunk13/img.jpg      -> trunk13
      - shark/img.jpg                     -> shark
      - train/trunk13/img.jpg             -> trunk13
    """
    parts = list(rel_path.parts)
    if len(parts) < 2:
        return rel_path.parent.name

    # Remove domain prefix if present
    if _normalize_name(parts[0]) == _normalize_name(domain_name):
        parts = parts[1:]

    if len(parts) == 0:
        return rel_path.parent.name

    # Skip split directory if present
    if _normalize_name(parts[0]) in {"train", "test", "val", "valid", "validation"}:
        if len(parts) >= 2:
            return parts[1]
        return rel_path.parent.name

    return parts[0]


def _resolve_image_path(dataset_root: Path, rel_path: Path, domain_name: str) -> Path:
    """
    Tries several likely layouts without requiring you to move files around.
    """
    candidates: List[Path] = []

    # As written in split file
    candidates.append(dataset_root / rel_path)

    parts = list(rel_path.parts)

    # If split path does not begin with the domain, prepend it
    if len(parts) == 0 or _normalize_name(parts[0]) != _normalize_name(domain_name):
        candidates.append(dataset_root / domain_name / rel_path)
        candidates.append(dataset_root / domain_name / "train" / rel_path)
        candidates.append(dataset_root / domain_name / "test" / rel_path)
    else:
        # If split path begins with domain, also try inserting train/test after domain
        tail = Path(*parts[1:]) if len(parts) > 1 else Path()
        candidates.append(dataset_root / domain_name / "train" / tail)
        candidates.append(dataset_root / domain_name / "test" / tail)

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path for domain '{domain_name}' and relative path '{rel_path}'. "
        f"Tried: {[str(p) for p in unique_candidates]}"
    )


def _read_split_file(
    split_file: Path,
    dataset_root: Path,
    domain_name: str,
    domain_id: int,
) -> List[Tuple[Path, Union[int, str], int, str]]:
    """
    Returns tuples:
      (image_path, raw_label_key, domain_id, class_name)

    raw_label_key is:
      - int if split file provides numeric labels
      - str (class name) if split file only provides paths
    """
    items: List[Tuple[Path, Union[int, str], int, str]] = []

    with split_file.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            rel_path, raw_label = _parse_split_line(line)
            image_path = _resolve_image_path(dataset_root, rel_path, domain_name)

            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            class_name = _infer_class_name(rel_path, domain_name)
            label_key: Union[int, str] = raw_label if raw_label is not None else class_name

            items.append((image_path, label_key, domain_id, class_name))

    return items


@register_dataset("domainnet")
def build_domainnet(args) -> DatasetBundle:
    root = Path(args.data_root)
    domainnet_root = _find_domainnet_root(root)

    image_size = int(
        getattr(args, "domainnet_image_size", getattr(args, "pacs_image_size", 224))
    )

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

    train_raw: List[Tuple[Path, Union[int, str], int, str]] = []
    test_raw: List[Tuple[Path, Union[int, str], int, str]] = []

    for domain_id, domain_name in enumerate(DOMAINNET_DOMAINS):
        train_raw.extend(
            _read_split_file(
                split_file=domainnet_root / f"{domain_name}_train.txt",
                dataset_root=domainnet_root,
                domain_name=domain_name,
                domain_id=domain_id,
            )
        )
        test_raw.extend(
            _read_split_file(
                split_file=domainnet_root / f"{domain_name}_test.txt",
                dataset_root=domainnet_root,
                domain_name=domain_name,
                domain_id=domain_id,
            )
        )

    if len(train_raw) == 0:
        raise RuntimeError("No DomainNet training images found.")
    if len(test_raw) == 0:
        raise RuntimeError("No DomainNet test images found.")

    # Build a stable label mapping
    all_label_keys = [label_key for _, label_key, _, _ in (train_raw + test_raw)]
    all_are_int = all(isinstance(k, int) for k in all_label_keys)

    if all_are_int:
        ordered_keys = sorted(set(int(k) for k in all_label_keys))
    else:
        ordered_keys = []
        seen = set()
        for k in all_label_keys:
            if k not in seen:
                seen.add(k)
                ordered_keys.append(k)

    key_to_idx = {k: idx for idx, k in enumerate(ordered_keys)}

    class_name_by_key: Dict[Union[int, str], str] = {}
    for _, label_key, _, class_name in (train_raw + test_raw):
        if label_key not in class_name_by_key:
            class_name_by_key[label_key] = class_name

    class_names = [
        class_name_by_key.get(k, f"class_{i}") for i, k in enumerate(ordered_keys)
    ]

    train_items: List[Tuple[Path, int, int]] = [
        (image_path, key_to_idx[label_key], domain_id)
        for image_path, label_key, domain_id, _ in train_raw
    ]
    test_items: List[Tuple[Path, int, int]] = [
        (image_path, key_to_idx[label_key], domain_id)
        for image_path, label_key, domain_id, _ in test_raw
    ]

    train_dataset = DomainNetListDataset(train_items, transform=train_tf)
    test_dataset = DomainNetListDataset(test_items, transform=test_tf)

    train_targets = np.asarray([label for _, label, _ in train_items], dtype=np.int64)
    test_targets = np.asarray([label for _, label, _ in test_items], dtype=np.int64)
    train_task_ids = np.asarray([domain_id for _, _, domain_id in train_items], dtype=np.int64)
    test_task_ids = np.asarray([domain_id for _, _, domain_id in test_items], dtype=np.int64)

    return DatasetBundle(
        name="domainnet",
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
            "domains": DOMAINNET_DOMAINS,
            "source_root": str(domainnet_root),
        },
        train_task_ids=train_task_ids,
        test_task_ids=test_task_ids,
        task_names=DOMAINNET_DOMAINS,
        default_scenario="domain-il",
    )