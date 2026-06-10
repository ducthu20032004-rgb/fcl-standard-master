from __future__ import annotations

from utils.misc import state_dict_to_cpu

from ..fedl2p_modules import LRMetaNet
from .base_server import BaseServer
from .fedavg_server import fedavg_state_dict


class FedL2PServer(BaseServer):
    def __init__(self, args, device) -> None:
        super().__init__(args, device)
        self.lr_meta_net = None

    def _ensure_meta_net(self, global_model):
        if self.lr_meta_net is None:
            num_blocks = len(global_model.get_trainable_block_prefixes())
            stats_dim = 2 * (num_blocks - 1)
            self.lr_meta_net = LRMetaNet(
                input_dim=stats_dim,
                output_dim=num_blocks,
                hidden_dim=int(self.args.fedl2p_hidden_dim),
                max_scale=float(self.args.fedl2p_max_lr_scale),
            ).to(self.device)

    def get_client_payload(self, global_model):
        self._ensure_meta_net(global_model)
        return {
            "global_model": global_model,
            "lr_meta_state": state_dict_to_cpu(self.lr_meta_net.state_dict()),
        }

    def aggregate(self, global_model, local_updates):
        if len(local_updates) == 0:
            return global_model

        self._ensure_meta_net(global_model)

        weights = [update.num_samples for update in local_updates]

        global_state = state_dict_to_cpu(global_model.state_dict())
        model_states = [update.state_dict for update in local_updates]
        aggregated_model_state = fedavg_state_dict(global_state, model_states, weights)
        global_model.load_state_dict(aggregated_model_state, strict=True)

        meta_global_state = state_dict_to_cpu(self.lr_meta_net.state_dict())
        meta_states = [update.extra["lr_meta_state"] for update in local_updates]
        aggregated_meta_state = fedavg_state_dict(meta_global_state, meta_states, weights)
        self.lr_meta_net.load_state_dict(aggregated_meta_state, strict=True)

        return global_model.to(self.device)