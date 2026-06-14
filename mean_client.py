#!/usr/bin/env python3
"""
analyze_blocks.py
-----------------
Pipeline:
  1. Với mỗi (client, block): tính trung bình qua tất cả các cặp (t, tprime)
  2. Với mỗi block: tính trung bình + std qua tất cả các client
     (tức là avg across clients, per block)

Sử dụng từ R:
  system("python3 analyze_blocks.py input.csv")
"""

import sys
from typing import Optional
import pandas as pd
import numpy as np

METRICS = ["sigma_old", "eps_old", "linear_cka_old", "align10"]


def load_data(source: Optional[str]) -> pd.DataFrame:
    if source:
        return pd.read_csv(source)
    return pd.read_csv(sys.stdin)


def step1_per_client_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: Trung bình các cặp (t, tprime) cho mỗi (client, block).
    Kết quả: một hàng cho mỗi (client, block).
    """
    return (
        df.groupby(["client", "block"])[METRICS]
        .mean()
        .reset_index()
    )


def step2_per_block_across_clients(per_cb: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 2: Với mỗi block, tính mean và std qua tất cả client.
    Kết quả: một hàng cho mỗi block, cột mean + std cho 4 metrics.
    """
    rows = []
    for block, grp in per_cb.groupby("block"):
        row = {"block": block}
        for m in METRICS:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_std"]  = grp[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def print_step1(per_cb: pd.DataFrame) -> None:
    print("=" * 90)
    print("BƯỚC 1 — Trung bình (t,tprime) cho từng (client, block)")
    print("=" * 90)
    for client, grp in per_cb.groupby("client"):
        print(f"\n  Client {client}:")
        hdr = f"  {'Block':>6}" + "".join(f"{m:>18}" for m in METRICS)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, r in grp.iterrows():
            line = f"  {int(r['block']):>6}"
            for m in METRICS:
                line += f"{r[m]:>18.6f}"
            print(line)


def print_step2(block_summary: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("BƯỚC 2 — Trung bình các client theo block (mean ± std across clients)")
    print("=" * 90)
    col_w = 28
    hdr = f"{'Block':>7}" + "".join(f"{m:>{col_w}}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for _, r in block_summary.iterrows():
        line = f"{int(r['block']):>7}"
        for m in METRICS:
            cell = f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"
            line += f"{cell:>{col_w}}"
        print(line)


def export_csv(per_cb: pd.DataFrame, block_summary: pd.DataFrame) -> None:
    per_cb.to_csv("per_client_block.csv", index=False, float_format="%.6f")
    block_summary.to_csv("block_summary_across_clients.csv", index=False, float_format="%.6f")
    print("\n[Đã xuất] per_client_block.csv  và  block_summary_across_clients.csv")


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(source)

    required = {"client", "block", "t", "tprime"} | set(METRICS)
    missing = required - set(df.columns)
    if missing:
        print(f"[Lỗi] Thiếu cột: {missing}", file=sys.stderr)
        sys.exit(1)

    per_cb        = step1_per_client_block(df)
    block_summary = step2_per_block_across_clients(per_cb)

    print_step1(per_cb)
    print_step2(block_summary)
    export_csv(per_cb, block_summary)


if __name__ == "__main__":
    main()