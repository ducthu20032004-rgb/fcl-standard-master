from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_runner import safe_generate_instance_stats, safe_run_single_benchmark
from validation_config import (
    ALPHA_SWEEP,
    BASELINE_PANEL,
    DEFAULT_OUTPUT_ROOT,
    FALLBACK_BASELINE_PANEL,
    GENERATOR_ONLY_SEEDS,
    PSI_SWEEP,
    REGIME_GRID,
    VALIDATION_DATASETS_CORE,
)
from validation_utils import (
    ensure_dir,
    get_available_repo_datasets,
    mean_std,
    resolve_method_panel,
    save_figure,
    save_table,
    spearman_corr,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=VALIDATION_DATASETS_CORE)
    parser.add_argument("--run-sanity", action="store_true")
    parser.add_argument("--sanity-max-runs", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "validation_51_controls")

    available_datasets = set(get_available_repo_datasets())
    datasets = [d for d in args.datasets if d in available_datasets]

    rows = []
    for dataset in datasets:
        for seed in GENERATOR_ONLY_SEEDS:
            for alpha in ALPHA_SWEEP:
                row = safe_generate_instance_stats(
                    dataset=dataset,
                    seed=seed,
                    alpha=alpha,
                    psi=0.0,
                    label_constructor="default",
                    order_constructor="default",
                )
                row["control_type"] = "alpha_to_jlabel"
                row["x_value"] = alpha
                rows.append(row)

            for psi in PSI_SWEEP:
                row = safe_generate_instance_stats(
                    dataset=dataset,
                    seed=seed,
                    alpha=1.0,
                    psi=psi,
                    label_constructor="default",
                    order_constructor="default",
                )
                row["control_type"] = "psi_to_delta"
                row["x_value"] = psi
                rows.append(row)

    control_df = pd.DataFrame(rows)
    save_table(control_df, out_dir / "control_parameter_behavior.csv")

    summary_rows = []
    for dataset in sorted(control_df["dataset"].dropna().unique()):
        sub_alpha = control_df[
            (control_df["dataset"] == dataset)
            & (control_df["control_type"] == "alpha_to_jlabel")
            & (control_df["status"] == "ok")
        ]
        if len(sub_alpha) > 0:
            mean_j, std_j = mean_std(sub_alpha["j_label"])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "control_type": "alpha_to_jlabel",
                    "target_metric": "j_label",
                    "spearman": spearman_corr(sub_alpha["alpha"], sub_alpha["j_label"]),
                    "mean": mean_j,
                    "std": std_j,
                    "spread": float(sub_alpha["j_label"].max() - sub_alpha["j_label"].min()),
                }
            )

        sub_psi = control_df[
            (control_df["dataset"] == dataset)
            & (control_df["control_type"] == "psi_to_delta")
            & (control_df["status"] == "ok")
        ]
        if len(sub_psi) > 0:
            mean_d, std_d = mean_std(sub_psi["delta_order"])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "control_type": "psi_to_delta",
                    "target_metric": "delta_order",
                    "spearman": spearman_corr(sub_psi["psi"], sub_psi["delta_order"]),
                    "mean": mean_d,
                    "std": std_d,
                    "spread": float(sub_psi["delta_order"].max() - sub_psi["delta_order"].min()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df, out_dir / "control_parameter_behavior_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for dataset in datasets:
        sub = control_df[
            (control_df["dataset"] == dataset)
            & (control_df["control_type"] == "alpha_to_jlabel")
            & (control_df["status"] == "ok")
        ]
        if len(sub) == 0:
            continue
        curve = sub.groupby("alpha")["j_label"].agg(["mean", "std"]).reset_index()
        axes[0].plot(curve["alpha"], curve["mean"], marker="o", label=dataset)
        axes[0].fill_between(
            curve["alpha"],
            curve["mean"] - curve["std"].fillna(0.0),
            curve["mean"] + curve["std"].fillna(0.0),
            alpha=0.15,
        )

    axes[0].set_xscale("log")
    axes[0].set_title(r"Control sanity: $\alpha \rightarrow J_{label}$")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"Realized $J_{label}$")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=8)

    for dataset in datasets:
        sub = control_df[
            (control_df["dataset"] == dataset)
            & (control_df["control_type"] == "psi_to_delta")
            & (control_df["status"] == "ok")
        ]
        if len(sub) == 0:
            continue
        curve = sub.groupby("psi")["delta_order"].agg(["mean", "std"]).reset_index()
        axes[1].plot(curve["psi"], curve["mean"], marker="o", label=dataset)
        axes[1].fill_between(
            curve["psi"],
            curve["mean"] - curve["std"].fillna(0.0),
            curve["mean"] + curve["std"].fillna(0.0),
            alpha=0.15,
        )

    axes[1].set_title(r"Control sanity: $\psi \rightarrow \Delta_{order}$")
    axes[1].set_xlabel(r"$\psi$")
    axes[1].set_ylabel(r"Realized $\Delta_{order}$")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=8)

    save_figure(out_dir / "control_parameter_behavior.png")

    if args.run_sanity:
        resolved_methods = resolve_method_panel(BASELINE_PANEL, FALLBACK_BASELINE_PANEL)
        resolved_methods = resolved_methods[:2]
        sanity_rows = []
        run_budget = 0
        for dataset in datasets[:2]:
            for _, method in resolved_methods:
                for regime_name, spec in REGIME_GRID.items():
                    for seed in GENERATOR_ONLY_SEEDS[:2]:
                        if run_budget >= args.sanity_max_runs:
                            break
                        summary, _ = safe_run_single_benchmark(
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            alpha=spec["alpha"],
                            psi=spec["psi"],
                        )
                        summary["regime"] = regime_name
                        sanity_rows.append(summary)
                        run_budget += 1

        sanity_df = pd.DataFrame(sanity_rows)
        if len(sanity_df) > 0:
            save_table(sanity_df, out_dir / "downstream_sanity_summary.csv")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            ok = sanity_df[sanity_df["status"] == "ok"].copy()
            if len(ok) > 0:
                axes[0].scatter(ok["j_label"], ok["avg_acc"], alpha=0.8)
                axes[0].set_xlabel(r"Realized $J_{label}$")
                axes[0].set_ylabel("Final avg_acc")
                axes[0].set_title(r"Downstream sanity: $J_{label}$ vs performance")
                axes[0].grid(True, linestyle="--", alpha=0.3)

                axes[1].scatter(ok["delta_order"], ok["avg_acc"], alpha=0.8)
                axes[1].set_xlabel(r"Realized $\Delta_{order}$")
                axes[1].set_ylabel("Final avg_acc")
                axes[1].set_title(r"Downstream sanity: $\Delta_{order}$ vs performance")
                axes[1].grid(True, linestyle="--", alpha=0.3)

                save_figure(out_dir / "downstream_sanity.png")


if __name__ == "__main__":
    main()