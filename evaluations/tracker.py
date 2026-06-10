from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.io import save_json


@dataclass
class RoundTracker:
    round_rows: List[Dict] = field(default_factory=list)
    task_accuracy_rows: List[Dict] = field(default_factory=list)

    def log_round(self, round_row: Dict, task_accs) -> None:
        self.round_rows.append(dict(round_row))
        task_row = {
            "global_round": round_row["global_round"],
            "task_pos": round_row["task_pos"],
            "round_in_task": round_row["round_in_task"],
        }
        for task_id, acc in enumerate(task_accs):
            task_row[f"acc_task_{task_id}"] = float(acc)
        self.task_accuracy_rows.append(task_row)

    def save(self, tables_dir: Path) -> Dict[str, Path]:
        tables_dir.mkdir(parents=True, exist_ok=True)
        round_path = tables_dir / "round_metrics.csv"
        task_path = tables_dir / "task_accuracy_history.csv"
        summary_path = tables_dir / "summary.json"

        round_df = pd.DataFrame(self.round_rows)
        task_df = pd.DataFrame(self.task_accuracy_rows)
        round_df.to_csv(round_path, index=False)
        task_df.to_csv(task_path, index=False)

        if len(round_df) > 0:
            last_row = round_df.iloc[-1].to_dict()
            best_acc = float(round_df["avg_acc"].max())
            best_gap = float(round_df["local_global_gap"].min())
            summary = {
                "num_logged_rounds": int(len(round_df)),
                "best_avg_acc": best_acc,
                "best_min_gap": best_gap,
                "final": {
                    "avg_acc": float(last_row["avg_acc"]),
                    "forgetting": float(last_row["forgetting"]),
                    "local_global_gap": float(last_row["local_global_gap"]),
                },
            }
        else:
            summary = {"num_logged_rounds": 0}

        save_json(summary, summary_path)
        return {
            "round_metrics": round_path,
            "task_history": task_path,
            "summary": summary_path,
        }
