from __future__ import annotations

import copy
from typing import List, Optional

import torch

from ..target_modules import TargetReplayBuffer, TargetSynthesizer
from .fedavg_server import FedAvgServer


class TARGETServer(FedAvgServer):
    def __init__(self, args, device) -> None:
        super().__init__(args=args, device=device)
        self.teacher_model: Optional[torch.nn.Module] = None
        self.replay_buffer: Optional[TargetReplayBuffer] = None
        self.model_builder = None
        self.dataset_bundle = None
        self.partition = None
        self.logger = None
        self.current_task_pos: Optional[int] = None

    def register_context(self, model_builder, dataset_bundle, partition, logger) -> None:
        self.model_builder = model_builder
        self.dataset_bundle = dataset_bundle
        self.partition = partition
        self.logger = logger

    def get_client_payload(self, global_model):
        return {
            "global_model": global_model,
            "teacher_model": self.teacher_model,
            "replay_buffer": self.replay_buffer,
            "current_task_pos": self.current_task_pos,
        }

    def on_task_start(self, global_model, task_pos: int, partition) -> None:
        del global_model, partition
        self.current_task_pos = int(task_pos)

    def _collect_seen_task_ids(self, upto_task_pos: int) -> List[int]:
        assert self.partition is not None
        seen = set()
        for client_id in range(self.args.num_clients):
            seen.update(self.partition.client_task_orders[client_id][: upto_task_pos + 1])
        return sorted(int(task_id) for task_id in seen)

    def _collect_seen_class_ids(self, upto_task_pos: int) -> List[int]:
        assert self.partition is not None
        class_ids = set()
        for task_id in self._collect_seen_task_ids(upto_task_pos):
            class_ids.update(int(cls) for cls in self.partition.task_labels[task_id])
        return sorted(class_ids)

    def on_task_end(self, global_model, task_pos: int, partition) -> None:
        del partition

        if self.partition is None or self.dataset_bundle is None or self.model_builder is None:
            return

        if task_pos >= self.partition.num_tasks - 1:
            self.teacher_model = copy.deepcopy(global_model).cpu()
            return

        if self.dataset_bundle.modality != "image":
            raise ValueError("TARGET in this codebase currently supports image datasets only.")

        seen_class_ids = self._collect_seen_class_ids(task_pos)
        if len(seen_class_ids) == 0:
            self.teacher_model = copy.deepcopy(global_model).cpu()
            self.replay_buffer = None
            return

        if self.logger is not None:
            self.logger.info(
                "[TARGET] start server-side synthesis after task_pos=%s | replay_classes=%s",
                task_pos,
                seen_class_ids,
            )

        synthesizer = TargetSynthesizer(
            teacher_model=copy.deepcopy(global_model).to(self.device),
            student_model=self.model_builder().to(self.device),
            dataset_name=self.dataset_bundle.name,
            class_ids=seen_class_ids,
            device=self.device,
            syn_rounds=int(self.args.target_syn_rounds),
            g_steps=int(self.args.target_g_steps),
            kd_steps=int(self.args.target_kd_steps),
            warmup_rounds=int(self.args.target_warmup_rounds),
            synthesis_batch_size=int(self.args.target_synthesis_batch_size),
            sample_batch_size=int(self.args.target_sample_batch_size),
            latent_dim=int(self.args.target_latent_dim),
            generator_lr=float(self.args.target_generator_lr),
            noise_lr=float(self.args.target_noise_lr),
            student_lr=float(self.args.target_student_lr),
            bn_weight=float(self.args.target_bn_weight),
            ce_weight=float(self.args.target_ce_weight),
            div_weight=float(self.args.target_div_weight),
            kd_temperature=float(self.args.target_generator_kd_temperature),
            use_fomaml=bool(self.args.target_use_fomaml),
            divergence_mask_mode=str(self.args.target_divergence_mask),
            bn_momentum=float(self.args.target_bn_momentum),
            max_images=(
                None
                if int(self.args.target_max_replay_images) <= 0
                else int(self.args.target_max_replay_images)
            ),
        )

        self.replay_buffer = synthesizer.run()
        self.teacher_model = copy.deepcopy(global_model).cpu()

        if self.logger is not None:
            self.logger.info(
                "[TARGET] finish synthesis after task_pos=%s | replay_samples=%s | replay_classes=%s",
                task_pos,
                self.replay_buffer.num_samples() if self.replay_buffer is not None else 0,
                seen_class_ids,
            )