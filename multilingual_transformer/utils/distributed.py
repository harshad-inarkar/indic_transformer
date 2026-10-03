from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_distributed_process(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize process group on rank's dedicated GPU."""
    torch.cuda.set_device(rank)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def run_auto_distributed(target_fn: Callable[[int, int], None]) -> None:
    """
    Auto-detects GPU environment:
    - If WORLD_SIZE is set (launched via torchrun), executes directly.
    - If multiple GPUs exist, automatically spawns processes via mp.spawn.
    - If 1 GPU exists, runs single-process directly.
    """
    if "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        init_distributed_process(rank, world_size)
        try:
            target_fn(rank, world_size)
        finally:
            cleanup()
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        mp.spawn(_worker_entry, args=(num_gpus, target_fn), nprocs=num_gpus, join=True)
    else:
        target_fn(0, 1)


def _worker_entry(rank: int, world_size: int, target_fn: Callable[[int, int], None]) -> None:
    init_distributed_process(rank, world_size)
    try:
        target_fn(rank, world_size)
    finally:
        cleanup()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def gather_all(data: Any, world_size: int) -> list[Any]:
    """Gather arbitrary picklable objects from all ranks to Rank 0."""
    if world_size <= 1 or not dist.is_initialized():
        return [data]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data)
    return output