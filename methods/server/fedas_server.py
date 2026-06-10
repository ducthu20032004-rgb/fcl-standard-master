from __future__ import annotations

from .base_server import BaseServer
from .fedavg_server import fedavg_state_dict


class FedASServer(BaseServer):
    """FedAS server adaptation for this benchmark.

    Paper-faithful part: aggregate shared backbone using client-synchronization
    weights (t-FIM based). Practical benchmark adaptation: also maintain a
    sample-weighted global head as a reference global model for your existing
    global curves and local-global gap logging.
    """

    def aggregate(self, global_model, local_updates):
        if len(local_updates) == 0:
            return global_model

        backbone_states = [update.extra["backbone_state_dict"] for update in local_updates]
        head_states = [update.extra["head_state_dict"] for update in local_updates]

        alpha_weights = [float(update.aggregation_weight or 0.0) for update in local_updates]
        if sum(alpha_weights) <= 0.0:
            alpha_weights = [float(update.num_samples) for update in local_updates]
        sample_weights = [int(update.num_samples) for update in local_updates]

        global_backbone = {
            key: value.detach().cpu().clone() for key, value in global_model.backbone_state_dict().items()
        }
        global_head = {
            key: value.detach().cpu().clone() for key, value in global_model.head_state_dict().items()
        }

        aggregated_backbone = fedavg_state_dict(global_backbone, backbone_states, alpha_weights)
        aggregated_head = fedavg_state_dict(global_head, head_states, sample_weights)

        global_model.load_backbone_state_dict(aggregated_backbone, strict=True)
        global_model.load_head_state_dict(aggregated_head, strict=True)
        return global_model.to(self.device)