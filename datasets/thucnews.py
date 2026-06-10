from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import DatasetBundle
from .registry import register_dataset


class THUCNewsDataset(Dataset):
    def __init__(self, records: List[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        return item["text"], int(item["label"])


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _encode_text(
    text: str,
    stoi: dict,
    unk_idx: int,
    max_len: int,
) -> List[int]:
    chars = [ch for ch in str(text).strip() if not ch.isspace()]
    if len(chars) == 0:
        chars = ["<empty>"]
    token_ids = [int(stoi.get(ch, unk_idx)) for ch in chars[:max_len]]
    if len(token_ids) == 0:
        token_ids = [unk_idx]
    return token_ids


@register_dataset("thucnews")
def build_thucnews(args) -> DatasetBundle:
    root = Path(args.data_root) / "thucnews" / "processed"
    train_path = root / "train.jsonl"
    test_path = root / "test.jsonl"
    vocab_path = root / "vocab.json"
    class_names_path = root / "class_names.json"

    missing = [p for p in [train_path, test_path, vocab_path, class_names_path] if not p.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(
            "THUCNews processed files are missing. "
            "Please run: python scripts/download_thucnews.py"
        )

    train_records = _load_jsonl(train_path)
    test_records = _load_jsonl(test_path)
    vocab_meta = _load_json(vocab_path)
    class_names = _load_json(class_names_path)

    stoi = vocab_meta["stoi"]
    pad_idx = int(vocab_meta["pad_idx"])
    unk_idx = int(vocab_meta["unk_idx"])
    vocab_size = int(vocab_meta["vocab_size"])

    args.text_vocab_size = vocab_size
    args.text_pad_idx = pad_idx

    max_length = int(args.thucnews_max_length)

    train_dataset = THUCNewsDataset(train_records)
    test_dataset = THUCNewsDataset(test_records)

    train_targets = np.asarray([int(item["label"]) for item in train_records], dtype=np.int64)
    test_targets = np.asarray([int(item["label"]) for item in test_records], dtype=np.int64)

    def collate_fn(batch: List[Tuple[str, int]]):
        texts, labels = zip(*batch)
        token_lists = [
            _encode_text(
                text=text,
                stoi=stoi,
                unk_idx=unk_idx,
                max_len=max_length,
            )
            for text in texts
        ]
        seq_len = max(int(args.text_min_seq_len), max(len(tokens) for tokens in token_lists))
        input_ids = torch.full((len(token_lists), seq_len), pad_idx, dtype=torch.long)
        for row_id, token_ids in enumerate(token_lists):
            take = min(len(token_ids), seq_len)
            input_ids[row_id, :take] = torch.tensor(token_ids[:take], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels

    return DatasetBundle(
        name="thucnews",
        modality="text",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_targets=train_targets,
        test_targets=test_targets,
        num_classes=len(class_names),
        class_names=list(class_names),
        collate_train_fn=collate_fn,
        collate_test_fn=collate_fn,
        default_backbone="text_cnn",
        metadata={
            "vocab_size": vocab_size,
            "pad_idx": pad_idx,
            "unk_idx": unk_idx,
            "max_length": max_length,
            "tokenizer": "character",
            "source_root": str(root),
        },
        train_task_ids=None,
        test_task_ids=None,
        task_names=None,
        default_scenario="class-il",
    )