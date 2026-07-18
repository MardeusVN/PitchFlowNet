"""Small torch.distributed lifecycle helpers for the custom training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import TypeVar

import torch
import torch.distributed as dist


MetricsT = TypeVar("MetricsT")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, *, force_cpu: bool = False) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        initialized_here = False
        if force_cpu or not torch.cuda.is_available():
            device = torch.device("cpu")
            backend = "gloo"
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
            initialized_here = True
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        return cls(rank, local_rank, world_size, device, initialized_here)

    def barrier(self) -> None:
        if self.enabled:
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            dist.barrier(device_ids=device_ids)

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()

    def average_metrics(self, metrics: MetricsT) -> MetricsT:
        if not self.enabled:
            return metrics
        values = asdict(metrics)
        keys = [key for key in values if key != "global_step"]
        tensor = torch.tensor(
            [float(values[key]) for key in keys],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        averaged = dict(values)
        averaged.update(dict(zip(keys, tensor.cpu().tolist())))
        return type(metrics)(**averaged)

    def broadcast_float(self, value: float) -> float:
        if not self.enabled:
            return value
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        dist.broadcast(tensor, src=0)
        return float(tensor.cpu())

    def broadcast_bool(self, value: bool) -> bool:
        if not self.enabled:
            return value
        tensor = torch.tensor(int(value), dtype=torch.int64, device=self.device)
        dist.broadcast(tensor, src=0)
        return bool(tensor.item())
