from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import py7zr
import requests


FIGSHARE_ARTICLE_ID = 28279964
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare THUCNews.")
    parser.add_argument("--output-dir", type=str, default="data/thucnews")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=-1)
    return parser.parse_args()


def _read_text(path: Path) -> str:
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
        try:
            return path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            continue
    raise RuntimeError(f"Failed to read text file: {path}")


def _get_figshare_download_url() -> tuple[str, str]:
    resp = requests.get(FIGSHARE_API, timeout=60)
    resp.raise_for_status()
    meta = resp.json()
    files = meta.get("files", [])
    if len(files) == 0:
        raise RuntimeError("No downloadable files found on Figshare article.")
    file_meta = None
    for f in files:
        if str(f.get("name", "")).endswith(".7z"):
            file_meta = f
            break
    if file_meta is None:
        file_meta = files[0]
    return file_meta["download_url"], file_meta["name"]


def _download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _extract_7z(archive_path: Path, extract_dir: Path) -> None:
    with py7zr.SevenZipFile(archive_path, mode="r") as zf:
        zf.extractall(path=extract_dir)


def _find_class_root(root: Path) -> Path:
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for candidate in candidates:
        child_dirs = [p for p in candidate.iterdir() if p.is_dir()]
        if len(child_dirs) < 5:
            continue
        num_valid_children = 0
        for child in child_dirs:
            if any(fp.suffix.lower() == ".txt" for fp in child.rglob("*.txt")):
                num_valid_children += 1
        if num_valid_children >= 5:
            return candidate
    raise RuntimeError("Could not find THUCNews class root after extraction.")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.output_dir).resolve()
    raw_dir = out_root / "raw"
    extract_dir = out_root / "extracted"
    processed_dir = out_root / "processed"
    archive_dir = out_root / "archives"

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print("[THUCNews] Querying Figshare API...")
    download_url, file_name = _get_figshare_download_url()
    archive_path = archive_dir / file_name

    if not archive_path.exists():
        print(f"[THUCNews] Downloading archive to {archive_path} ...")
        _download_file(download_url, archive_path)
    else:
        print(f"[THUCNews] Reusing cached archive: {archive_path}")

    print("[THUCNews] Extracting archive...")
    _extract_7z(archive_path, extract_dir)

    class_root = _find_class_root(extract_dir)
    class_dirs = sorted([p for p in class_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    class_names = [p.name for p in class_dirs]

    rng = np.random.RandomState(args.seed)
    train_rows = []
    test_rows = []
    vocab_counter = Counter()

    for label_id, class_dir in enumerate(class_dirs):
        files = sorted(class_dir.rglob("*.txt"))
        if args.max_per_class > 0:
            files = files[: int(args.max_per_class)]

        files = list(files)
        rng.shuffle(files)
        n = len(files)
        if n == 0:
            continue
        n_test = max(1, int(round(n * float(args.test_ratio))))
        n_test = min(n_test, n - 1) if n > 1 else 0

        test_files = files[:n_test]
        train_files = files[n_test:] if n_test > 0 else files

        for path in train_files:
            text = _read_text(path).strip()
            if not text:
                continue
            train_rows.append({"text": text, "label": label_id})
            vocab_counter.update([ch for ch in text if not ch.isspace()])

        for path in test_files:
            text = _read_text(path).strip()
            if not text:
                continue
            test_rows.append({"text": text, "label": label_id})

    if len(train_rows) == 0 or len(test_rows) == 0:
        raise RuntimeError("Prepared THUCNews split is empty.")

    specials = ["<pad>", "<unk>"]
    vocab_items = [
        token for token, freq in vocab_counter.most_common()
        if int(freq) >= int(args.min_freq)
    ]
    max_vocab_body = max(0, int(args.max_vocab_size) - len(specials))
    vocab_items = vocab_items[:max_vocab_body]
    itos = specials + vocab_items
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    vocab_meta = {
        "itos": itos,
        "stoi": stoi,
        "pad_idx": 0,
        "unk_idx": 1,
        "vocab_size": len(itos),
    }

    _write_jsonl(processed_dir / "train.jsonl", train_rows)
    _write_jsonl(processed_dir / "test.jsonl", test_rows)

    with (processed_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    with (processed_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, indent=2, ensure_ascii=False)

    stats = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "vocab_size": len(itos),
        "figshare_article_id": FIGSHARE_ARTICLE_ID,
        "class_root": str(class_root),
    }
    with (processed_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[THUCNews] Done.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import numpy as np
    main()