from __future__ import annotations


class BaseServer:
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device

    def get_client_payload(self, global_model):
        return global_model

    def on_task_start(self, global_model, task_pos: int, partition) -> None:
        del global_model, task_pos, partition

    def on_task_end(self, global_model, task_pos: int, partition) -> None:
        del global_model, task_pos, partition

    def aggregate(self, global_model, local_updates):
        raise NotImplementedError