import pandas as pd
import numpy as np
import sys


METRICS = ["eps_trained", "eps_aggr", "eps_global",
           "cknna_trained", "cknna_aggr", "cknna_global"]


def compute_stats(filepath: str) -> None:
    df = pd.read_csv(filepath)

    # ── 1. Per-client mean for each metric ──────────────────────────────────
    per_client = (
        df.groupby("client_id")[METRICS]
        .mean()
        .rename(columns={m: f"{m}_mean" for m in METRICS})
    )

    print("=" * 60)
    print("PER-CLIENT MEAN")
    print("=" * 60)
    print(per_client.to_string())
    print()

    # ── 2. Overall mean ± std across ALL rows ────────────────────────────────
    overall_mean = df[METRICS].mean()
    overall_std  = df[METRICS].std()

    summary = pd.DataFrame({
        "mean": overall_mean,
        "std":  overall_std,
        "mean±std": [
            f"{m:.6f} ± {s:.6f}"
            for m, s in zip(overall_mean, overall_std)
        ]
    })

    print("=" * 60)
    print("OVERALL MEAN ± STD  (across all clients & rounds)")
    print("=" * 60)
    print(summary.to_string())
    print()

    # ── 3. Per-client mean ± std ─────────────────────────────────────────────
    grp = df.groupby("client_id")[METRICS]
    client_mean = grp.mean()
    client_std  = grp.std()

    rows = []
    for cid in client_mean.index:
        for m in METRICS:
            rows.append({
                "client_id": cid,
                "metric":    m,
                "mean":      client_mean.loc[cid, m],
                "std":       client_std.loc[cid, m],
                "mean±std":  f"{client_mean.loc[cid, m]:.6f} ± {client_std.loc[cid, m]:.6f}",
            })

    client_summary = pd.DataFrame(rows).set_index(["client_id", "metric"])

    print("=" * 60)
    print("PER-CLIENT MEAN ± STD")
    print("=" * 60)
    print(client_summary.to_string())


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    compute_stats(path)