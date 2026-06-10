from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation_config import DEFAULT_OUTPUT_ROOT, LAMBDA_GRID, REGIME_GRID
from validation_utils import kendall_tau, normalize_minmax, regime_name_from_values, save_figure, save_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--interaction-csv",
        type=str,
        default=None,
        help="Default: scripts/outputs/validation_53_interaction/interaction_results.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_root) / "validation_54_lambda_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.interaction_csv is None:
        interaction_csv = Path(args.output_root) / "validation_53_interaction" / "interaction_results.csv"
    else:
        interaction_csv = Path(args.interaction_csv)

    df = pd.read_csv(interaction_csv)
    df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        save_table(pd.DataFrame(), out_dir / "lambda_rank_stability.csv")
        save_table(pd.DataFrame(), out_dir / "lambda_regime_ordering.csv")
        return

    cell_rows = []
    for dataset, group in df.groupby("dataset"):
        agg = (
            group.groupby(["alpha", "psi"], as_index=False)[["j_label", "delta_order"]]
            .mean()
        )
        agg["j_norm"] = normalize_minmax(agg["j_label"])
        agg["delta_norm"] = normalize_minmax(agg["delta_order"])
        for lam in LAMBDA_GRID:
            tmp = agg.copy()
            tmp["lambda"] = lam
            tmp["h_joint"] = lam * tmp["j_norm"] + (1.0 - lam) * tmp["delta_norm"]
            tmp["regime"] = [
                regime_name_from_values(alpha=a, psi=p, regime_grid=REGIME_GRID, tol=1e-9)
                for a, p in zip(tmp["alpha"], tmp["psi"])
            ]
            cell_rows.append(tmp.assign(dataset=dataset))

    cell_df = pd.concat(cell_rows, ignore_index=True)

    rank_rows = []
    for dataset, group in cell_df.groupby("dataset"):
        ref = group[group["lambda"] == 0.5].copy()
        ref = ref.sort_values("h_joint", ascending=False).reset_index(drop=True)
        ref_rank = {f"{r.alpha}_{r.psi}": idx + 1 for idx, r in ref.iterrows()}

        for lam, sub in group.groupby("lambda"):
            sub = sub.sort_values("h_joint", ascending=False).reset_index(drop=True)
            shared = []
            ref_vals = []
            cur_vals = []
            for idx, row in sub.iterrows():
                key = f"{row.alpha}_{row.psi}"
                if key in ref_rank:
                    shared.append(key)
                    ref_vals.append(ref_rank[key])
                    cur_vals.append(idx + 1)
            rank_rows.append(
                {
                    "dataset": dataset,
                    "lambda": lam,
                    "kendall_tau_vs_0p5": kendall_tau(ref_vals, cur_vals) if len(shared) >= 2 else np.nan,
                }
            )

    rank_df = pd.DataFrame(rank_rows)
    save_table(rank_df, out_dir / "lambda_rank_stability.csv")

    regime_rows = []
    for dataset, group in cell_df.groupby("dataset"):
        for lam, sub in group.groupby("lambda"):
            sub = sub.copy()
            sub["rank"] = sub["h_joint"].rank(ascending=False, method="average")
            for _, row in sub.iterrows():
                regime_rows.append(
                    {
                        "dataset": dataset,
                        "lambda": lam,
                        "regime": row["regime"],
                        "alpha": row["alpha"],
                        "psi": row["psi"],
                        "h_joint": row["h_joint"],
                        "rank": row["rank"],
                    }
                )

    regime_df = pd.DataFrame(regime_rows)
    save_table(regime_df, out_dir / "lambda_regime_ordering.csv")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    curve = rank_df.groupby("lambda")["kendall_tau_vs_0p5"].mean().reset_index()
    ax.plot(curve["lambda"], curve["kendall_tau_vs_0p5"], marker="o")
    ax.set_title(r"Rank stability of $H_{joint}(\lambda)$ vs $\lambda=0.5$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Mean Kendall $\tau$")
    ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(out_dir / "lambda_rank_stability.png")


if __name__ == "__main__":
    main()