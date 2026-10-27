"""
Adding DiLoCo to an existing PyTorch training loop: any model, any inner optimizer.

The model here is a small MLP fitting a nonlinear function. Each worker draws its
own data. The DiLoCo-specific lines are marked with  # <- DiLoCo.

Multi-process:   torchrun --nproc-per-node=2 examples/custom_model.py
Single process:  python examples/custom_model.py   (no communication, still runs the outer optimizer)
(On macOS add --master-addr=127.0.0.1 to torchrun if your hostname doesn't resolve.)
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diloco import DiLoCo, init_distributed, launched_by_torchrun, pick_device  # noqa: E402  # <- DiLoCo


def make_batch(generator, true_w, device, n=64):
    x = torch.randn(n, 16, generator=generator)
    y = torch.sin(x @ true_w / 4) + 0.05 * torch.randn(n, 1, generator=generator)
    return x.to(device), y.to(device)


def main():
    if launched_by_torchrun():
        rank, world_size, device = init_distributed()  # <- DiLoCo: NCCL on GPUs, gloo on CPU
    else:
        rank, world_size, (device, _) = 0, 1, pick_device(0)

    torch.manual_seed(0)  # every worker must start from the same weights
    model = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)).to(device)
    inner_opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    diloco = DiLoCo(model, inner_steps=25, quantize_bits=4, num_fragments=2)  # <- DiLoCo

    true_w = torch.randn(16, 1, generator=torch.Generator().manual_seed(1))
    train_gen = torch.Generator().manual_seed(100 + rank)  # each worker sees different data
    x_val, y_val = make_batch(torch.Generator().manual_seed(999), true_w, device, n=2048)

    for step in range(1, 1001):
        x, y = make_batch(train_gen, true_w, device)
        F.mse_loss(model(x), y).backward()
        inner_opt.step()
        inner_opt.zero_grad()
        diloco.step()  # <- DiLoCo: syncs every 25 steps (two staggered fragments)

        if rank == 0 and step % 200 == 0:
            with torch.no_grad(), diloco.global_weights():  # <- DiLoCo: evaluate the shared model
                val = F.mse_loss(model(x_val), y_val).item()
            print(f"step {step:>4}  val mse {val:.4f}")

    if rank == 0:
        s = diloco.stats
        print(f"workers={world_size} device={device} syncs={s['syncs']} "
              f"sent={s['bytes_sent'] / 1e6:.2f} MB/worker peak={s['peak_bytes'] / 1e3:.1f} KB/sync")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
