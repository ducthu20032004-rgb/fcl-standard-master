from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_runner import safe_run_single_benchmark
from validation_config import (
    DEFAULT_OUTPUT_ROOT,
    FULL_MAIN_DATASETS,
    FULL_SEEDS,
    REGIME_GRID,
)
from validation_utils import (
    ci95,
    ensure_dir,
    get_available_repo_datasets,
    get_available_repo_methods,
    kendall_tau,
    regime_name_from_values,
    save_figure,
    save_table,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=FULL_MAIN_DATASETS)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--source-csv", type=str, default=None)
    parser.add_argument("--rank-metric", type=str, default="avg_acc", choices=["avg_acc", "forgetting", "local_global_gap"])
    parser.add_argument("--max-runs", type=int, default=-1)
    return parser.parse_args()


def _build_or_load_results(args, out_dir: Path) -> pd.DataFrame:
    if args.source_csv is not None and Path(args.source_csv).exists():
        return pd.read_csv(args.source_csv)

    available_datasets = set(get_available_repo_datasets())
    datasets = [d for d in args.datasets if d in available_datasets]
    if args.methods is None:
        methods = sorted(get_available_repo_methods())
    else:
        methods = [m for m in args.methods if m in get_available_repo_methods()]

    rows = []
    run_budget = 0
    for dataset in datasets:
        for regime_name, spec in REGIME_GRID.items():
            for seed in FULL_SEEDS:
                for method in methods:
                    if args.max_runs > 0 and run_budget >= args.max_runs:
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
    save_table(df, out_dir / "statistical_results.csv")
    return df


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "validation_56_rank_stability")
    raw_df = _build_or_load_results(args, out_dir)
    ok = raw_df[raw_df["status"] == "ok"].copy()

    appendix_rows = []
    for (dataset, method, regime), sub in ok.groupby(["dataset", "method", "regime"]):
        for metric in ["avg_acc", "forgetting", "local_global_gap"]:
            mean, lo, hi = ci95(sub[metric])
            appendix_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "regime": regime,
                    "metric": metric,
                    "mean": mean,
                    "std": float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_seeds": int(len(sub)),
                }
            )

    appendix_df = pd.DataFrame(appendix_rows)
    save_table(appendix_df, out_dir / "appendix_mean_std_ci.csv")

    rank_rows = []
    rank_metric = args.rank_metric
    ascending = rank_metric != "avg_acc"

    for dataset, dataset_df in ok.groupby("dataset"):
        # within-regime seed stability
        for regime, regime_df in dataset_df.groupby("regime"):
            seed_rankings = {}
            for seed, seed_df in regime_df.groupby("seed"):
                if len(seed_df) < 2:
                    continue
                ranked = seed_df[["method", rank_metric]].copy()
                ranked["rank"] = ranked[rank_metric].rank(ascending=ascending, method="average")
                seed_rankings[seed] = dict(zip(ranked["method"], ranked["rank"]))

            taus = []
            for s1, s2 in itertools.combinations(sorted(seed_rankings), 2):
                shared = sorted(set(seed_rankings[s1]) & set(seed_rankings[s2]))
                if len(shared) < 2:
                    continue
                r1 = [seed_rankings[s1][m] for m in shared]
                r2 = [seed_rankings[s2][m] for m in shared]
                taus.append(kendall_tau(r1, r2))

            rank_rows.append(
                {
                    "dataset": dataset,
                    "comparison_type": "within_regime_seed_stability",
                    "regime_a": regime,
                    "regime_b": regime,
                    "kendall_tau_mean": float(np.nanmean(taus)) if len(taus) > 0 else np.nan,
                    "n_pairs": int(len(taus)),
                }
            )

        # across-regime shift using seed-averaged rankings
        if "mild" in set(dataset_df["regime"]) and "joint-hard" in set(dataset_df["regime"]):
            mild = dataset_df[dataset_df["regime"] == "mild"].groupby("method")[rank_metric].mean().reset_index()
            joint = dataset_df[dataset_df["regime"] == "joint-hard"].groupby("method")[rank_metric].mean().reset_index()
            if len(mild) >= 2 and len(joint) >= 2:
                mild["rank_mild"] = mild[rank_metric].rank(ascending=ascending, method="average")
                joint["rank_joint"] = joint[rank_metric].rank(ascending=ascending, method="average")
                merged = mild.merge(joint, on="method", how="inner")
                if len(merged) >= 2:
                    tau = kendall_tau(merged["rank_mild"], merged["rank_joint"])
                    rank_rows.append(
                        {
                            "dataset": dataset,
                            "comparison_type": "across_regime_shift",
                            "regime_a": "mild",
                            "regime_b": "joint-hard",
                            "kendall_tau_mean": tau,
                            "n_pairs": int(len(merged)),
                        }
                    )

    rank_df = pd.DataFrame(rank_rows)
    save_table(rank_df, out_dir / "rank_stability_vs_shift.csv")

    # Overview figure with mean ± CI on avg_acc
    avg_acc_df = appendix_df[appendix_df["metric"] == "avg_acc"].copy()
    if len(avg_acc_df) > 0:
        regimes = list(REGIME_GRID.keys())
        datasets = sorted(avg_acc_df["dataset"].unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.8), squeeze=False)

        for ax, dataset in zip(axes[0], datasets):
            sub = avg_acc_df[avg_acc_df["dataset"] == dataset].copy()
            top_methods = (
                sub.groupby("method")["mean"].mean().sort_values(ascending=False).head(5).index.tolist()
            )
            for method in top_methods:
                msub = sub[sub["method"] == method].set_index("regime")
                means = []
                lows = []
                highs = []
                for regime in regimes:
                    if regime in msub.index:
                        row = msub.loc[regime]
                        means.append(float(row["mean"]))
                        lows.append(float(row["ci95_low"]))
                        highs.append(float(row["ci95_high"]))
                    else:
                        means.append(np.nan)
                        lows.append(np.nan)
                        highs.append(np.nan)
                x = np.arange(len(regimes))
                ax.plot(x, means, marker="o", label=method)
                ax.fill_between(x, lows, highs, alpha=0.15)

            ax.set_xticks(np.arange(len(regimes)))
            ax.set_xticklabels(regimes, rotation=20)
            ax.set_title(dataset)
            ax.set_ylabel("avg_acc mean ± 95% CI")
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[0][-1].legend(fontsize=8, loc="best")
        save_figure(out_dir / "core_mean_ci_overview.png")


if __name__ == "__main__":
    main()