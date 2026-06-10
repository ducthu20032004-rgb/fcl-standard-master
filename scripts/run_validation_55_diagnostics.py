from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_runner import safe_run_single_benchmark
from validation_config import (
    BASELINE_PANEL,
    DEFAULT_OUTPUT_ROOT,
    FALLBACK_BASELINE_PANEL,
    FULL_SEEDS,
    REGIME_GRID,
    VALIDATION_DATASETS_CORE,
)
from validation_utils import (
    ensure_dir,
    get_available_repo_datasets,
    pearson_corr,
    resolve_method_panel,
    save_figure,
    save_table,
    spearman_corr,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=VALIDATION_DATASETS_CORE)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--source-csv", type=str, default=None)
    parser.add_argument("--gradient-similarity-csv", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=40)
    return parser.parse_args()


def _build_or_load_results(args, out_dir: Path) -> pd.DataFrame:
    if args.source_csv is not None and Path(args.source_csv).exists():
        return pd.read_csv(args.source_csv)

    available_datasets = set(get_available_repo_datasets())
    datasets = [d for d in args.datasets if d in available_datasets]
    if args.methods is None:
        method_panel = resolve_method_panel(BASELINE_PANEL, FALLBACK_BASELINE_PANEL)
    else:
        method_panel = [(m, m) for m in args.methods]

    rows = []
    run_budget = 0
    for dataset in datasets:
        for regime_name, spec in REGIME_GRID.items():
            for seed in FULL_SEEDS[:2]:
                for _, method in method_panel[:4]:
                    if run_budget >= args.max_runs:
                        break
                    summary, _ = safe_run_single_benchmark(
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        alpha=spec["alpha"],
                        psi=spec["psi"],
                    )
                    summary["regime"] = regime_name
                    rows.append(summary)
                    run_budget += 1

    df = pd.DataFrame(rows)
    save_table(df, out_dir / "diagnostic_source_results.csv")
    return df


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "validation_55_diagnostics")
    df = _build_or_load_results(args, out_dir)
    ok = df[df["status"] == "ok"].copy()

    corr_rows = []
    for group_name, sub in [("overall", ok)] + [(f"dataset::{d}", g) for d, g in ok.groupby("dataset")]:
        if len(sub) == 0:
            continue
        for y_col, expected_sign in [
            ("avg_acc", "positive"),
            ("forgetting", "negative"),
            ("local_global_gap", "negative"),
        ]:
            corr_rows.append(
                {
                    "group": group_name,
                    "metric_x": "update_alignment_mean",
                    "metric_y": y_col,
                    "pearson": pearson_corr(sub["update_alignment_mean"], sub[y_col]),
                    "spearman": spearman_corr(sub["update_alignment_mean"], sub[y_col]),
                    "expected_direction": expected_sign,
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    save_table(corr_df, out_dir / "diagnostic_correlation_summary.csv")

    if len(ok) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        panels = [
            ("avg_acc", "Accuracy"),
            ("forgetting", "Forgetting"),
            ("local_global_gap", "Local-global gap"),
        ]
        for ax, (col, title) in zip(axes, panels):
            ax.scatter(ok["update_alignment_mean"], ok[col], alpha=0.8)
            if len(ok) >= 2:
                x = ok["update_alignment_mean"].to_numpy(dtype=float)
                y = ok[col].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 2:
                    coef = np.polyfit(x[mask], y[mask], deg=1)
                    xx = np.linspace(x[mask].min(), x[mask].max(), 100)
                    yy = coef[0] * xx + coef[1]
                    ax.plot(xx, yy, linestyle="--")
            ax.set_xlabel("Mean update alignment")
            ax.set_ylabel(col)
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.3)

        save_figure(out_dir / "diagnostic_scatter_panels.png")

    if len(corr_df) > 0:
        direction_rows = []
        for _, row in corr_df.iterrows():
            observed = "positive" if float(row["spearman"]) >= 0 else "negative"
            direction_rows.append(
                {
                    **row.to_dict(),
                    "observed_direction": observed,
                    "direction_match": observed == row["expected_direction"],
                }
            )
        direction_df = pd.DataFrame(direction_rows)
    else:
        direction_df = pd.DataFrame()

    save_table(direction_df, out_dir / "diagnostic_directional_sanity.csv")

    if args.gradient_similarity_csv is not None and Path(args.gradient_similarity_csv).exists():
        grad_df = pd.read_csv(args.gradient_similarity_csv)
        if "update_alignment_mean" in grad_df.columns and "gradient_similarity" in grad_df.columns:
            merged = grad_df[["update_alignment_mean", "gradient_similarity"]].dropna()
            if len(merged) > 0:
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.scatter(merged["update_alignment_mean"], merged["gradient_similarity"], alpha=0.8)
                ax.set_xlabel("Update alignment")
                ax.set_ylabel("Gradient similarity")
                ax.set_title("Appendix: update alignment vs gradient similarity")
                ax.grid(True, linestyle="--", alpha=0.3)
                save_figure(out_dir / "appendix_update_vs_gradient_similarity.png")

                appendix_df = pd.DataFrame(
                    [
                        {
                            "pearson": pearson_corr(merged["update_alignment_mean"], merged["gradient_similarity"]),
                            "spearman": spearman_corr(merged["update_alignment_mean"], merged["gradient_similarity"]),
                            "n": len(merged),
                        }
                    ]
                )
                save_table(appendix_df, out_dir / "appendix_update_vs_gradient_similarity.csv")


if __name__ == "__main__":
    main()