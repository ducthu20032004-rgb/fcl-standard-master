from __future__ import annotations

from typing import Dict, List

import torch

from utils.misc import state_dict_to_cpu

from .base_server import BaseServer


class FedAvgServer(BaseServer):
    def aggregate(self, global_model, local_updates):
        if len(local_updates) == 0:
            return global_model

        global_state = state_dict_to_cpu(global_model.state_dict())
        local_states = [update.state_dict for update in local_updates]
        weights = [update.num_samples for update in local_updates]
        aggregated_state = fedavg_state_dict(global_state, local_states, weights)
        global_model.load_state_dict(aggregated_state, strict=True)
        return global_model.to(self.device)


def fedavg_state_dict(
    global_state: Dict[str, torch.Tensor],
    local_states: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    keys = list(global_state.keys())
    total_weight = float(sum(weights)) if sum(weights) > 0 else 1.0
    aggregated: Dict[str, torch.Tensor] = {}

    for key in keys:
        if torch.is_floating_point(global_state[key]):
            aggregated[key] = torch.zeros_like(global_state[key])
        else:
            aggregated[key] = None

    for local_state, weight in zip(local_states, weights):
        coeff = float(weight) / total_weight
        for key in keys:
            if torch.is_floating_point(local_state[key]):
                aggregated[key] += local_state[key] * coeff

    first_state = local_states[0]
    for key in keys:
        if not torch.is_floating_point(global_state[key]):
            aggregated[key] = first_state[key].clone()

    return aggregated
