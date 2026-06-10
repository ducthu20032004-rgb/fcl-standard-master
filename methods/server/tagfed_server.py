from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import state_dict_to_cpu
from ..tagfed_modules import GroupLogitHead, build_group_train_loader
from .fedavg_server import FedAvgServer, fedavg_state_dict


def temperature_kldiv(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    q = F.log_softmax(student_logits / temperature, dim=1)
    p = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(q, p, reduction="batchmean") * (temperature ** 2)


class TagFedServer(FedAvgServer):
    """
    Dense TagFed adaptation:
    - global reference model is still updated by FedAvg-style averaging
      so your benchmark can keep the same global metrics.
    - task-group teachers are trained separately from uploaded
      (feature, logit, label) messages.
    """
    def __init__(self, args, device) -> None:
        super().__init__(args=args, device=device)
        self.tagfed_group_teacher_states = {}

    def get_client_payload(self, global_model):
        return {
            "global_model": global_model,
            "tagfed_group_teacher_states": self.tagfed_group_teacher_states,
        }

    def _train_group_teacher(self, task_id: int, messages):
        # giữ trên CPU trước, DataLoader sẽ pin CPU tensor bình thường
        features = torch.cat([msg["features"] for msg in messages], dim=0).detach().cpu().contiguous()
        labels = torch.cat([msg["labels"] for msg in messages], dim=0).detach().cpu().long().contiguous()
        client_logits = torch.cat([msg["logits"] for msg in messages], dim=0).detach().cpu().contiguous()

        feature_dim = int(features.size(1))
        teacher_head = GroupLogitHead(
            feature_dim=feature_dim,
            num_classes=client_logits.size(1),
        ).to(self.device)

        if task_id in self.tagfed_group_teacher_states:
            teacher_head.load_state_dict(self.tagfed_group_teacher_states[task_id], strict=True)

        optimizer = optim.SGD(
            teacher_head.parameters(),
            lr=float(self.args.tagfed_server_lr),
            momentum=float(self.args.momentum),
            weight_decay=0.0,
        )

        loader = build_group_train_loader(
            features=features,
            labels=labels,
            logits=client_logits,
            batch_size=int(self.args.tagfed_server_batch_size),
        )

        teacher_head.train()
        for _ in range(max(1, int(self.args.tagfed_server_epochs))):
            for feat, y, z_client in loader:
                feat = feat.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                z_client = z_client.to(self.device, non_blocking=True)

                z_server = teacher_head(feat)
                loss_ce = F.cross_entropy(z_server, y)
                loss_kd = temperature_kldiv(
                    student_logits=z_server,
                    teacher_logits=z_client,
                    temperature=float(self.args.tagfed_temperature),
                )
                loss = (
                    float(self.args.tagfed_alpha_s) * loss_ce
                    + float(self.args.tagfed_beta_s) * loss_kd
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.tagfed_group_teacher_states[task_id] = state_dict_to_cpu(teacher_head.state_dict())

    def aggregate(self, global_model, local_updates):
        if len(local_updates) == 0:
            return global_model

        reference_state = state_dict_to_cpu(global_model.state_dict())
        local_states = [update.state_dict for update in local_updates]
        weights = [float(update.num_samples) for update in local_updates]

        aggregated_state = fedavg_state_dict(reference_state, local_states, weights)
        global_model.load_state_dict(aggregated_state, strict=True)
        global_model = global_model.to(self.device)

        grouped_messages = defaultdict(list)
        for update in local_updates:
            message = update.extra.get("tagfed_message", None)
            if message is None:
                continue
            grouped_messages[int(update.task_id)].append(message)

        for task_id, messages in grouped_messages.items():
            self._train_group_teacher(task_id=task_id, messages=messages)

        return global_model