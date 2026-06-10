from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "This script requires the Hugging Face 'datasets' package. "
        "Install it with: pip install datasets pillow"
    ) from exc


EXPECTED_DOMAIN_COUNTS = {
    "art_painting": 2048,
    "cartoon": 2344,
    "photo": 1670,
    "sketch": 3929,
}
EXPECTED_NUM_CLASSES = 7
EXPECTED_TOTAL = 9991


def _sanitize_name(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Download PACS into data/pacs/")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pacs",
        help="Target directory where PACS will be exported.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="flwrlabs/pacs",
        help="Hugging Face dataset id for PACS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported images if they already exist.",
    )
    parser.add_argument(
        "--strict-checks",
        action="store_true",
        help="Fail if domain counts / class count do not match the expected PACS metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PACS] Loading dataset '{args.dataset_id}' from Hugging Face...")
    ds = load_dataset(args.dataset_id, split="train")

    label_feature = ds.features["label"]
    label_names = list(label_feature.names) if hasattr(label_feature, "names") else None

    domain_counts = defaultdict(int)
    class_counts = defaultdict(int)

    print(f"[PACS] Exporting to: {output_dir}")
    for idx, sample in enumerate(ds):
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        domain_name = _sanitize_name(sample["domain"])
        label_idx = int(sample["label"])

        if label_names is not None:
            class_name = _sanitize_name(label_names[label_idx])
        else:
            class_name = str(label_idx)

        save_dir = output_dir / domain_name / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{idx:05d}.png"
        if save_path.exists() and not args.overwrite:
            domain_counts[domain_name] += 1
            class_counts[class_name] += 1
            continue

        image.save(save_path, format="PNG")
        domain_counts[domain_name] += 1
        class_counts[class_name] += 1

        if (idx + 1) % 500 == 0:
            print(f"[PACS] Exported {idx + 1}/{len(ds)} images...")

    total_images = sum(domain_counts.values())
    exported_domains = sorted(domain_counts.keys())
    exported_classes = sorted(class_counts.keys())

    metadata = {
        "dataset_id": args.dataset_id,
        "output_dir": str(output_dir),
        "num_images": total_images,
        "num_domains": len(exported_domains),
        "num_classes": len(exported_classes),
        "domains": exported_domains,
        "classes": exported_classes,
        "domain_counts": dict(sorted(domain_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n[PACS] Export finished.")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    problems = []

    if total_images != EXPECTED_TOTAL:
        problems.append(
            f"Expected total images = {EXPECTED_TOTAL}, got {total_images}"
        )

    if len(exported_classes) != EXPECTED_NUM_CLASSES:
        problems.append(
            f"Expected num_classes = {EXPECTED_NUM_CLASSES}, got {len(exported_classes)}"
        )

    for domain_name, expected_count in EXPECTED_DOMAIN_COUNTS.items():
        actual_count = domain_counts.get(domain_name, 0)
        if actual_count != expected_count:
            problems.append(
                f"Domain '{domain_name}' expected {expected_count}, got {actual_count}"
            )

    if len(problems) == 0:
        print("[PACS] Integrity check passed.")
    else:
        print("[PACS] Integrity check found mismatches:")
        for p in problems:
            print(f"  - {p}")
        if args.strict_checks:
            raise RuntimeError("PACS integrity checks failed.")

    print("\n[PACS] Done. Your benchmark loader can now read from:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()