from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> Path:
    root = get_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_slug(text: str) -> str:
    return (
        str(text)
        .replace("/", "-")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def choose_classes_per_task(dataset_name: str, num_classes: int, configured: int | None = None) -> int:
    if configured is not None and configured > 0 and num_classes % configured == 0:
        return int(configured)
    for candidate in [10, 7, 5, 4, 3, 2, 1]:
        if num_classes % candidate == 0:
            return candidate
    return 1


def _rankdata(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for uidx, count in enumerate(counts):
            if count > 1:
                idx = np.where(inverse == uidx)[0]
                ranks[idx] = ranks[idx].mean()
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson_corr(_rankdata(x), _rankdata(y))


def kendall_tau(x_ranked: Sequence[float], y_ranked: Sequence[float]) -> float:
    x = np.asarray(x_ranked, dtype=float)
    y = np.asarray(y_ranked, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def ci95(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    half = 1.96 * std / math.sqrt(len(arr))
    return mean, mean - half, mean + half


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def shannon_entropy(prob: np.ndarray, eps: float = 1e-12) -> float:
    prob = np.asarray(prob, dtype=float)
    prob = prob / max(prob.sum(), eps)
    prob = np.clip(prob, eps, 1.0)
    return float(-(prob * np.log(prob)).sum())


def compute_jlabel(train_targets: np.ndarray, client_train_indices: List[List[int]], num_classes: int) -> float:
    num_clients = len(client_train_indices)
    client_class_counts = np.zeros((num_classes, num_clients), dtype=float)
    for client_id, indices in enumerate(client_train_indices):
        ys = train_targets[np.asarray(indices, dtype=np.int64)]
        values, counts = np.unique(ys, return_counts=True)
        for value, count in zip(values, counts):
            client_class_counts[int(value), client_id] = float(count)

    per_class = []
    for class_id in range(num_classes):
        q = client_class_counts[class_id]
        if q.sum() <= 0:
            continue
        per_class.append(1.0 - shannon_entropy(q / q.sum()) / max(math.log(num_clients), 1e-12))
    if len(per_class) == 0:
        return 0.0
    return float(np.mean(per_class))


def normalized_inversion_distance(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    if len(order_a) < 2:
        return 0.0
    pos_a = {task: idx for idx, task in enumerate(order_a)}
    pos_b = {task: idx for idx, task in enumerate(order_b)}
    shared = [task for task in order_a if task in pos_b]
    m = len(shared)
    if m < 2:
        return 0.0
    inv = 0
    total = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += 1
            ta, tb = shared[i], shared[j]
            if (pos_a[ta] - pos_a[tb]) * (pos_b[ta] - pos_b[tb]) < 0:
                inv += 1
    return float(inv / max(total, 1))


def compute_delta_order(client_task_orders: List[List[int]], client_task_indices: List[List[List[int]]]) -> float:
    num_clients = len(client_task_orders)
    pair_values = []
    for c1 in range(num_clients):
        for c2 in range(c1 + 1, num_clients):
            shared_tasks = [
                t for t in range(len(client_task_orders[c1]))
                if len(client_task_indices[c1][t]) > 0 and len(client_task_indices[c2][t]) > 0
            ]
            order_1 = [t for t in client_task_orders[c1] if t in shared_tasks]
            order_2 = [t for t in client_task_orders[c2] if t in shared_tasks]
            pair_values.append(normalized_inversion_distance(order_1, order_2))
    if len(pair_values) == 0:
        return 0.0
    return float(np.mean(pair_values))


def flatten_state_dict_delta(local_state: Dict[str, object], global_state: Dict[str, object]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    import torch

    for key, g in global_state.items():
        l = local_state[key]
        if torch.is_tensor(g):
            if not torch.is_floating_point(g):
                continue
            delta = (l.detach().cpu().float() - g.detach().cpu().float()).reshape(-1).numpy()
            chunks.append(delta)
    if len(chunks) == 0:
        return np.zeros((1,), dtype=float)
    return np.concatenate(chunks, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), eps)
    return float(np.dot(a, b) / denom)


def mean_pairwise_cosine(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return float("nan")
    sims = []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        sims.append(cosine_similarity(vectors[i], vectors[j]))
    return float(np.mean(sims)) if sims else float("nan")


def fit_interaction_ols(df: pd.DataFrame, y_col: str, x1_col: str, x2_col: str) -> Dict[str, float]:
    sub = df[[y_col, x1_col, x2_col]].dropna().copy()
    if len(sub) < 4:
        return {
            "beta0": float("nan"),
            "beta1": float("nan"),
            "beta2": float("nan"),
            "beta3": float("nan"),
            "r2": float("nan"),
            "n": int(len(sub)),
        }
    x1 = sub[x1_col].to_numpy(dtype=float)
    x2 = sub[x2_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x1), x1, x2, x1 * x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "beta2": float(beta[2]),
        "beta3": float(beta[3]),
        "r2": r2,
        "n": int(len(sub)),
    }


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def get_available_repo_methods() -> List[str]:
    ensure_repo_on_path()
    try:
        from methods import list_methods
        return list_methods()
    except Exception:
        return []


def get_available_repo_datasets() -> List[str]:
    ensure_repo_on_path()
    try:
        from datasets import list_datasets
        return list_datasets()
    except Exception:
        return []


def resolve_method_panel(primary_panel: Dict[str, str], fallback_panel: Dict[str, str]) -> List[tuple[str, str]]:
    available = set(get_available_repo_methods())
    resolved = [(display, method_id) for display, method_id in primary_panel.items() if method_id in available]
    if len(resolved) > 0:
        return resolved
    return [(display, method_id) for display, method_id in fallback_panel.items() if method_id in available]


def regime_name_from_values(alpha: float, psi: float, regime_grid: Dict[str, Dict[str, float]], tol: float = 1e-12) -> str:
    for name, spec in regime_grid.items():
        if abs(alpha - float(spec["alpha"])) <= tol and abs(psi - float(spec["psi"])) <= tol:
            return name
    return f"a={alpha}_psi={psi}"


def normalize_minmax(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)