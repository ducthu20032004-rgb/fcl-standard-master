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
    INTERACTION_ALPHA_GRID,
    INTERACTION_DATASETS,
    INTERACTION_PSI_GRID,
)
from validation_utils import (
    ensure_dir,
    get_available_repo_datasets,
    fit_interaction_ols,
    kendall_tau,
    resolve_method_panel,
    save_figure,
    save_table,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=INTERACTION_DATASETS)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--max-runs", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "validation_53_interaction")

    available_datasets = set(get_available_repo_datasets())
    datasets = [d for d in args.datasets if d in available_datasets]
    if args.methods is None:
        method_panel = resolve_method_panel(BASELINE_PANEL, FALLBACK_BASELINE_PANEL)
    else:
        method_panel = [(m, m) for m in args.methods]

    plan_rows = []
    result_rows = []
    run_budget = 0

    for dataset in datasets:
        for alpha in INTERACTION_ALPHA_GRID:
            for psi in INTERACTION_PSI_GRID:
                for seed in FULL_SEEDS:
                    for _, method in method_panel:
                        if args.max_runs > 0 and run_budget >= args.max_runs:
                            break
                        plan_rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "seed": seed,
                                "alpha": alpha,
                                "psi": psi,
                            }
                        )
                        summary, _ = safe_run_single_benchmark(
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            alpha=alpha,
                            psi=psi,
                        )
                        result_rows.append(summary)
                        run_budget += 1

    plan_df = pd.DataFrame(plan_rows)
    save_table(plan_df, out_dir / "interaction_plan.csv")

    results_df = pd.DataFrame(result_rows)
    save_table(results_df, out_dir / "interaction_results.csv")

    ok = results_df[results_df["status"] == "ok"].copy()
    if len(ok) > 0:
        metrics = ["avg_acc", "forgetting", "local_global_gap"]
        fig, axes = plt.subplots(
            len(metrics),
            len(datasets),
            figsize=(4.2 * max(1, len(datasets)), 3.6 * len(metrics)),
            squeeze=False,
        )
        for r, metric in enumerate(metrics):
            for c, dataset in enumerate(datasets):
                sub = ok[ok["dataset"] == dataset].copy()
                if len(sub) == 0:
                    axes[r, c].axis("off")
                    continue
                pivot = (
                    sub.groupby(["alpha", "psi"])[metric]
                    .mean()
                    .reset_index()
                    .pivot(index="alpha", columns="psi", values=metric)
                )
                im = axes[r, c].imshow(pivot.values, aspect="auto")
                axes[r, c].set_xticks(range(len(pivot.columns)))
                axes[r, c].set_xticklabels([str(x) for x in pivot.columns])
                axes[r, c].set_yticks(range(len(pivot.index)))
                axes[r, c].set_yticklabels([str(x) for x in pivot.index])
                axes[r, c].set_xlabel("psi")
                axes[r, c].set_ylabel("alpha")
                axes[r, c].set_title(f"{dataset} | {metric}")
                plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        save_figure(out_dir / "interaction_heatmaps.png")

        coef_rows = []
        for dataset in sorted(ok["dataset"].unique()):
            for method in sorted(ok["method"].unique()):
                sub = ok[(ok["dataset"] == dataset) & (ok["method"] == method)]
                if len(sub) < 4:
                    continue
                for metric in ["avg_acc", "forgetting", "local_global_gap"]:
                    fit = fit_interaction_ols(
                        df=sub,
                        y_col=metric,
                        x1_col="j_label",
                        x2_col="delta_order",
                    )
                    fit.update(
                        {
                            "dataset": dataset,
                            "method": method,
                            "metric": metric,
                        }
                    )
                    coef_rows.append(fit)

        coef_df = pd.DataFrame(coef_rows)
        save_table(coef_df, out_dir / "interaction_coefficients.csv")

        rank_rows = []
        mild = ok[(ok["alpha"] == 1.0) & (ok["psi"] == 0.0)]
        joint = ok[(ok["alpha"] == 0.05) & (ok["psi"] == 0.75)]

        for dataset in sorted(ok["dataset"].unique()):
            mild_mean = (
                mild[mild["dataset"] == dataset]
                .groupby("method")["avg_acc"]
                .mean()
                .reset_index()
            )
            joint_mean = (
                joint[joint["dataset"] == dataset]
                .groupby("method")["avg_acc"]
                .mean()
                .reset_index()
            )
            if len(mild_mean) < 2 or len(joint_mean) < 2:
                continue

            mild_mean["rank_mild"] = mild_mean["avg_acc"].rank(ascending=False, method="average")
            joint_mean["rank_joint"] = joint_mean["avg_acc"].rank(ascending=False, method="average")
            merged = mild_mean.merge(joint_mean, on="method", how="inner")
            if len(merged) < 2:
                continue

            tau = kendall_tau(merged["rank_mild"], merged["rank_joint"])
            for _, row in merged.iterrows():
                rank_rows.append(
                    {
                        "dataset": dataset,
                        "method": row["method"],
                        "rank_mild": float(row["rank_mild"]),
                        "rank_joint": float(row["rank_joint"]),
                        "rank_shift": float(row["rank_joint"] - row["rank_mild"]),
                        "kendall_tau_dataset": tau,
                    }
                )

        rank_df = pd.DataFrame(rank_rows)
        save_table(rank_df, out_dir / "interaction_rank_changes.csv")

        if len(rank_df) > 0:
            datasets_for_plot = sorted(rank_df["dataset"].unique())
            fig, axes = plt.subplots(1, len(datasets_for_plot), figsize=(5.2 * len(datasets_for_plot), 5), squeeze=False)
            for col, dataset in enumerate(datasets_for_plot):
                ax = axes[0, col]
                sub = rank_df[rank_df["dataset"] == dataset].copy()
                for _, row in sub.iterrows():
                    ax.plot(["mild", "joint-hard"], [row["rank_mild"], row["rank_joint"]], marker="o")
                    ax.text("mild", row["rank_mild"], row["method"], fontsize=8, va="center")
                    ax.text("joint-hard", row["rank_joint"], row["method"], fontsize=8, va="center")
                ax.invert_yaxis()
                ax.set_title(f"{dataset} bump chart")
                ax.set_ylabel("Method rank (lower is better)")
                ax.grid(True, linestyle="--", alpha=0.3)
            save_figure(out_dir / "interaction_bump_chart.png")


if __name__ == "__main__":
    main()