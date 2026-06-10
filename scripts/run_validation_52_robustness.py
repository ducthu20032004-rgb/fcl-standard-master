from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_runner import safe_generate_instance_stats, safe_run_single_benchmark
from validation_config import (
    BASELINE_PANEL,
    DEFAULT_OUTPUT_ROOT,
    FALLBACK_BASELINE_PANEL,
    FULL_SEEDS,
    MATCH_TOLERANCE,
    REGIME_GRID,
    VALIDATION_DATASETS_CORE,
)
from validation_utils import (
    ensure_dir,
    get_available_repo_datasets,
    kendall_tau,
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
    parser.add_argument("--max-runs", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "validation_52_robustness")

    available_datasets = set(get_available_repo_datasets())
    datasets = [d for d in args.datasets if d in available_datasets]

    if args.methods is None:
        method_panel = resolve_method_panel(BASELINE_PANEL, FALLBACK_BASELINE_PANEL)
    else:
        method_panel = [(m, m) for m in args.methods]

    plan_rows = []
    result_rows = []
    run_budget = 0

    constructors = [
        ("default", "default"),
        ("sparse_balanced", "default"),
        ("default", "rank_jitter"),
        ("default", "block_shuffle"),
    ]

    for dataset in datasets:
        for regime_name, spec in REGIME_GRID.items():
            for seed in FULL_SEEDS:
                base_stats = safe_generate_instance_stats(
                    dataset=dataset,
                    seed=seed,
                    alpha=spec["alpha"],
                    psi=spec["psi"],
                    label_constructor="default",
                    order_constructor="default",
                )

                if base_stats["status"] != "ok":
                    plan_rows.append(
                        {
                            "dataset": dataset,
                            "regime": regime_name,
                            "seed": seed,
                            "status": "skip",
                            "reason": base_stats["error"],
                        }
                    )
                    continue

                target_j = float(base_stats["j_label"])
                target_d = float(base_stats["delta_order"])

                for label_constructor, order_constructor in constructors:
                    plan_rows.append(
                        {
                            "dataset": dataset,
                            "regime": regime_name,
                            "seed": seed,
                            "label_constructor": label_constructor,
                            "order_constructor": order_constructor,
                            "matched_j_target": target_j,
                            "matched_delta_target": target_d,
                        }
                    )

                    for _, method in method_panel:
                        if args.max_runs > 0 and run_budget >= args.max_runs:
                            break

                        matched_j = target_j if label_constructor != "default" else None
                        matched_d = target_d if order_constructor != "default" else None

                        summary, _ = safe_run_single_benchmark(
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            alpha=spec["alpha"],
                            psi=spec["psi"],
                            label_constructor=label_constructor,
                            order_constructor=order_constructor,
                            matched_jlabel=matched_j,
                            matched_delta=matched_d,
                        )
                        summary["regime"] = regime_name
                        result_rows.append(summary)
                        run_budget += 1

    plan_df = pd.DataFrame(plan_rows)
    save_table(plan_df, out_dir / "robustness_plan.csv")

    results_df = pd.DataFrame(result_rows)
    save_table(results_df, out_dir / "robustness_results.csv")

    rank_rows = []
    ok = results_df[results_df["status"] == "ok"].copy()
    if len(ok) > 0:
        for dataset in sorted(ok["dataset"].unique()):
            for regime in sorted(ok["regime"].unique()):
                for seed in sorted(ok["seed"].unique()):
                    ref = ok[
                        (ok["dataset"] == dataset)
                        & (ok["regime"] == regime)
                        & (ok["seed"] == seed)
                        & (ok["label_constructor"] == "default")
                        & (ok["order_constructor"] == "default")
                    ][["method", "avg_acc"]].copy()
                    if len(ref) < 2:
                        continue
                    ref["ref_rank"] = ref["avg_acc"].rank(ascending=False, method="average")
                    ref_rank = dict(zip(ref["method"], ref["ref_rank"]))

                    alt_sub = ok[
                        (ok["dataset"] == dataset)
                        & (ok["regime"] == regime)
                        & (ok["seed"] == seed)
                        & ~(
                            (ok["label_constructor"] == "default")
                            & (ok["order_constructor"] == "default")
                        )
                    ]
                    for (label_c, order_c), group in alt_sub.groupby(["label_constructor", "order_constructor"]):
                        if len(group) < 2:
                            continue
                        group = group[["method", "avg_acc", "j_label", "delta_order", "matched_jlabel_target", "matched_delta_target"]].copy()
                        group["alt_rank"] = group["avg_acc"].rank(ascending=False, method="average")

                        shared = sorted(set(ref_rank) & set(group["method"]))
                        if len(shared) < 2:
                            continue

                        ref_vals = [ref_rank[m] for m in shared]
                        alt_vals = [float(group.loc[group["method"] == m, "alt_rank"].iloc[0]) for m in shared]

                        realized_j = float(group["j_label"].mean())
                        realized_d = float(group["delta_order"].mean())
                        matched_j = float(group["matched_jlabel_target"].mean())
                        matched_d = float(group["matched_delta_target"].mean())

                        rank_rows.append(
                            {
                                "dataset": dataset,
                                "regime": regime,
                                "seed": seed,
                                "label_constructor": label_c,
                                "order_constructor": order_c,
                                "kendall_tau": kendall_tau(ref_vals, alt_vals),
                                "spearman": spearman_corr(ref_vals, alt_vals),
                                "j_gap": abs(realized_j - matched_j) if np.isfinite(matched_j) else np.nan,
                                "delta_gap": abs(realized_d - matched_d) if np.isfinite(matched_d) else np.nan,
                                "within_tolerance": (
                                    (abs(realized_j - matched_j) <= MATCH_TOLERANCE if np.isfinite(matched_j) else True)
                                    and (abs(realized_d - matched_d) <= MATCH_TOLERANCE if np.isfinite(matched_d) else True)
                                ),
                            }
                        )

    summary_df = pd.DataFrame(rank_rows)
    if len(summary_df) > 0:
        robustness_summary = (
            summary_df.groupby(["dataset", "regime", "label_constructor", "order_constructor"], as_index=False)
            .agg(
                kendall_tau_mean=("kendall_tau", "mean"),
                spearman_mean=("spearman", "mean"),
                j_gap_mean=("j_gap", "mean"),
                delta_gap_mean=("delta_gap", "mean"),
                tolerance_rate=("within_tolerance", "mean"),
            )
        )
    else:
        robustness_summary = pd.DataFrame()

    save_table(robustness_summary, out_dir / "robustness_summary.csv")

    if len(ok) > 0:
        plot_df = ok.copy()
        plot_df["constructor"] = plot_df["label_constructor"] + "|" + plot_df["order_constructor"]
        regime_order = list(REGIME_GRID.keys())
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for constructor, group in plot_df.groupby("constructor"):
            values = (
                group.groupby("regime")["avg_acc"]
                .mean()
                .reindex(regime_order)
            )
            ax.plot(regime_order, values.values, marker="o", label=constructor)
        ax.set_title("Robustness overview: mean performance trend under alternative constructors")
        ax.set_ylabel("Mean avg_acc")
        ax.set_xlabel("Regime")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        save_figure(out_dir / "robustness_overview.png")


if __name__ == "__main__":
    main()