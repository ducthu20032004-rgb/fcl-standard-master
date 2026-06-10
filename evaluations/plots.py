from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def save_metric_curve(
    x_values: Sequence[int],
    y_values: Sequence[float],
    title: str,
    y_label: str,
    output_path: Path,
    task_boundaries: Iterable[int] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, y_values, marker="o", markersize=2)
    if task_boundaries is not None:
        for boundary in task_boundaries:
            plt.axvline(boundary, linestyle="--", alpha=0.25)
    plt.title(title)
    plt.xlabel("Global round")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
