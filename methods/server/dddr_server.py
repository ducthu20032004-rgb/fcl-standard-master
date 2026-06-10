from __future__ import annotations

from typing import Optional

from utils import state_dict_to_cpu

from .fedavg_server import fedavg_state_dict

from ..dddr_modules import DDDRProjectionHead
from .base_server import BaseServer


class DDDRServer(BaseServer):
    """
    Server side of the benchmark-friendly DDDR integration.

    It still uses FedAvg for the classifier, but also aggregates the
    supervised-contrastive projection head used by the client.
    """
    def __init__(self, args, device) -> None:
        super().__init__(args, device)
        self.proj_head = None

    def _ensure_proj_head(self, global_model) -> None:
        if self.proj_head is not None:
            return
        feature_dim = int(global_model.head.in_features)
        self.proj_head = DDDRProjectionHead(
            in_dim=feature_dim,
            hidden_dim=int(self.args.dddr_proj_hidden_dim),
            out_dim=int(self.args.dddr_proj_dim),
        ).to(self.device)

    def get_client_payload(self, global_model):
        self._ensure_proj_head(global_model)
        return {
            "global_model": global_model,
            "dddr_proj_state": state_dict_to_cpu(self.proj_head.state_dict()),
        }

    def aggregate(self, global_model, local_updates):
        if len(local_updates) == 0:
            return global_model

        # Aggregate classifier
        reference_state = state_dict_to_cpu(global_model.state_dict())
        local_states = [update.state_dict for update in local_updates]
        weights = [float(update.aggregation_weight if update.aggregation_weight is not None else update.num_samples)
                   for update in local_updates]
        classifier_state = fedavg_state_dict(reference_state, local_states, weights)
        global_model.load_state_dict(classifier_state, strict=True)
        global_model = global_model.to(self.device)

        # Aggregate projection head
        self._ensure_proj_head(global_model)
        proj_states = [update.extra["dddr_proj_state"] for update in local_updates if "dddr_proj_state" in update.extra]
        if len(proj_states) > 0:
            proj_reference = state_dict_to_cpu(self.proj_head.state_dict())
            proj_state = fedavg_state_dict(proj_reference, proj_states, weights[: len(proj_states)])
            self.proj_head.load_state_dict(proj_state, strict=True)

        return global_model