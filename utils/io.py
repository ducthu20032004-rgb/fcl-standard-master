from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _sanitize_value(value) -> str:
    text = str(value)
    text = text.replace("/", "-").replace(" ", "")
    return text


def build_setting_name(args, num_tasks: int) -> str:
    scenario = getattr(args, "scenario", "class-il")
    parts = [
        f"scn{scenario}",
        f"C{args.num_clients}",
        f"T{num_tasks}",
        f"P{args.classes_per_task}",
        f"a{_sanitize_value(args.dirichlet_alpha)}",
        f"psi{_sanitize_value(args.order_psi)}",
        f"rpt{args.rounds_per_task}",
        f"frac{_sanitize_value(args.client_fraction)}",
        f"le{args.local_epochs}",
        f"bs{args.batch_size}",
        f"lr{_sanitize_value(args.lr)}",
        f"loss{args.loss_mode}",
        f"seed{args.seed}",
    ]
    if getattr(args, "task_label_order", None) is not None:
        parts.append(f"taskorder{args.task_label_order}")
    if getattr(args, "dirichlet_allocation", None) is not None:
        parts.append(f"alloc{args.dirichlet_allocation}")
    if getattr(args, "schedule_swap_mode", None) is not None:
        parts.append(f"swap{args.schedule_swap_mode}")
    if getattr(args, "num_tasks", None) is not None:
        parts.append(f"stream{args.num_tasks}")

    method = getattr(args, "method", None)
    if method == "fedprox":
        parts.append(f"mu{_sanitize_value(args.fedprox_mu)}")
    elif method == "fedala":
        parts.extend(
            [
                f"top{args.fedala_top_p}",
                f"wlr{_sanitize_value(args.fedala_weight_lr)}",
                f"sr{_sanitize_value(args.fedala_sample_ratio)}",
            ]
        )
    elif method == "fedas":
        parts.extend(
            [
                f"aligne{args.fedas_align_epochs}",
                f"alignlr{_sanitize_value(args.fedas_align_lr)}",
                f"alignr{_sanitize_value(args.fedas_align_ratio)}",
                f"fimr{_sanitize_value(args.fedas_fim_ratio)}",
            ]
        )
    elif method == "fedl2p":
        parts.extend(
            [
                f"meta{_sanitize_value(args.fedl2p_meta_lr)}",
                f"msteps{args.fedl2p_meta_steps}",
                f"h{args.fedl2p_hidden_dim}",
                f"mxlr{_sanitize_value(args.fedl2p_max_lr_scale)}",
            ]
        )
    elif method == "target":
        parts.extend(
            [
                f"tkd{_sanitize_value(args.target_client_kd_weight)}",
                f"tt{_sanitize_value(args.target_client_kd_temperature)}",
                f"synr{args.target_syn_rounds}",
                f"g{args.target_g_steps}",
                f"ks{args.target_kd_steps}",
                f"div{args.target_divergence_mask}",
            ]
        )
    elif method == "tagfed":
        parts.extend(
            [
                f"ac{_sanitize_value(args.tagfed_alpha_c)}",
                f"bc{_sanitize_value(args.tagfed_beta_c)}",
                f"as{_sanitize_value(args.tagfed_alpha_s)}",
                f"bs{_sanitize_value(args.tagfed_beta_s)}",
                f"temp{_sanitize_value(args.tagfed_temperature)}",
            ]
        )
    elif method in {"affcl", "af_fcl"}:
        parts.extend(
            [
                f"flr{_sanitize_value(args.affcl_flow_lr)}",
                f"fiter{args.affcl_local_iterations}",
                f"fhd{args.affcl_flow_hidden_dim}",
                f"fL{args.affcl_flow_layers}",
                f"flow{_sanitize_value(args.affcl_k_loss_flow)}",
                f"kdg{_sanitize_value(args.affcl_k_kd_global_cls)}",
                f"kdl{_sanitize_value(args.affcl_k_kd_last_cls)}",
            ]
        )
    elif method == "dddr":
        parts.extend(
            [
                f"wk{_sanitize_value(args.dddr_w_kd)}",
                f"wcp{_sanitize_value(args.dddr_w_ce_pre)}",
                f"wscl{_sanitize_value(args.dddr_w_scl)}",
                f"cur{args.dddr_current_size}",
                f"pre{args.dddr_prev_size}",
            ]
        )
    
    return "_".join(parts)


def create_run_dirs(output_root: Path, dataset: str, method: str, setting_name: str) -> Dict[str, Path]:
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_root / dataset / method / setting_name / time_id)
    tables_dir = ensure_dir(run_dir / "tables")
    figures_dir = ensure_dir(run_dir / "figures")
    logs_dir = ensure_dir(run_dir / "logs")
    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
        "time_id": Path(time_id),
    }