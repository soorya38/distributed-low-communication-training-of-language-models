"""Process-group setup that works under torchrun or mp.spawn, on NVIDIA GPUs or CPU."""

import os

import torch
import torch.distributed as dist


def pick_device(local_rank):
    """CUDA + NCCL when a GPU is visible, else CPU + gloo. DILOCO_DEVICE=cpu forces CPU."""
    if torch.cuda.is_available() and os.environ.get("DILOCO_DEVICE", "").lower() != "cpu":
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        return device, "nccl"
    return torch.device("cpu"), "gloo"


def launched_by_torchrun():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed(rank=None, world_size=None, init_method=None):
    """Join the process group and return (rank, world_size, device).

    Under torchrun, call with no arguments: everything comes from the environment.
    Under mp.spawn, pass rank, world_size and an init_method such as tcp://host:port.
    """
    if rank is None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        init_method = "env://"
    else:
        local_rank = rank

    device, backend = pick_device(local_rank)
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
    return rank, world_size, device
