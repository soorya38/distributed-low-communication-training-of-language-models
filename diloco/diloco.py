"""
DiLoCo for any PyTorch model.

Wrap the model, keep your own inner optimizer and training loop, and call
`diloco.step()` after every inner optimizer step:

    diloco = DiLoCo(model, inner_steps=50)
    for x, y in loader:
        loss_fn(model(x), y).backward()
        inner_opt.step()
        inner_opt.zero_grad()
        diloco.step()

Every `inner_steps` steps the workers average their pseudo-gradients
(global params - local params) and an outer SGD with Nesterov momentum updates
the shared global params, which are then copied back into the model.

Options:
- quantize_bits: send pseudo-gradients as 8/4/2/1-bit blockwise codes instead
  of float32, with per-worker error feedback.
- num_fragments: Streaming DiLoCo. Parameters are split into P fragments that
  sync on a staggered schedule, so only ~1/P of the model is on the wire at once.
"""

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

from .compression import dequantize, payload_bytes, quantize
from .fragments import fragment_offsets, fragments_due, split_into_fragments


class DiLoCo:
    def __init__(self, model, inner_steps, outer_lr=0.7, outer_momentum=0.9, nesterov=True,
                 quantize_bits=None, error_feedback=True, num_fragments=1, group=None):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.inner_steps = inner_steps
        self.quantize_bits = quantize_bits
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1

        self.global_params = [p.detach().clone() for p in self.params]
        self.fragments = split_into_fragments(self.params, num_fragments)
        self.offsets = fragment_offsets(inner_steps, num_fragments)
        self.outer_optimizers = [
            torch.optim.SGD([self.global_params[i] for i in fragment], lr=outer_lr,
                            momentum=outer_momentum, nesterov=nesterov)
            for fragment in self.fragments
        ]
        self.residuals = None
        if quantize_bits is not None and error_feedback:
            self.residuals = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]

        self.step_count = 0
        self.stats = {"syncs": 0, "fragment_syncs": 0, "bytes_sent": 0, "peak_bytes": 0, "comm_seconds": 0.0}

    @torch.no_grad()
    def step(self):
        """Count one inner step; sync whichever fragments are due. Returns True if anything synced."""
        self.step_count += 1
        due = fragments_due(self.step_count, self.inner_steps, self.offsets)
        if due:
            self._sync(due)
        return bool(due)

    @torch.no_grad()
    def sync_all(self):
        """Sync every fragment now, e.g. once at the end of training."""
        self._sync(range(len(self.fragments)))

    @contextmanager
    def global_weights(self):
        """Temporarily load the global params into the model, e.g. for evaluation."""
        with torch.no_grad():
            local = [p.detach().clone() for p in self.params]
            for p, g in zip(self.params, self.global_params):
                p.copy_(g)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, saved in zip(self.params, local):
                    p.copy_(saved)

    def _sync(self, fragment_ids):
        indices = [i for f in fragment_ids for i in self.fragments[f]]
        flat = torch.cat([(self.global_params[i] - self.params[i]).float().flatten() for i in indices])

        start = time.perf_counter()
        if self.quantize_bits is None:
            num_bytes = flat.numel() * flat.element_size()
            if self.world_size > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.group)
            flat /= self.world_size
        else:
            flat, num_bytes = self._quantized_average(flat, indices)
        if flat.is_cuda:
            torch.cuda.synchronize(flat.device)
        self.stats["comm_seconds"] += time.perf_counter() - start

        for i, chunk in zip(indices, flat.split([self.global_params[i].numel() for i in indices])):
            self.global_params[i].grad = chunk.view_as(self.global_params[i]).to(self.global_params[i].dtype).clone()
        for f in fragment_ids:
            self.outer_optimizers[f].step()
        for i in indices:
            self.global_params[i].grad = None
            self.params[i].copy_(self.global_params[i])

        self.stats["syncs"] += 1
        self.stats["fragment_syncs"] += len(fragment_ids)
        self.stats["bytes_sent"] += num_bytes
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], num_bytes)

    def _quantized_average(self, flat, indices):
        """Quantize this worker's pseudo-gradient, all-gather every worker's codes, average locally.

        Integer codes can't be summed by all_reduce, so each worker decodes all
        messages in rank order; every rank computes the identical average.
        """
        bits = self.quantize_bits
        if self.residuals is not None:
            flat = flat + torch.cat([self.residuals[i].flatten() for i in indices])

        packed, scales = quantize(flat, bits)
        if self.residuals is not None:
            residual = flat - dequantize(packed, scales, bits, flat.numel())
            for i, chunk in zip(indices, residual.split([self.params[i].numel() for i in indices])):
                self.residuals[i].copy_(chunk.view_as(self.residuals[i]))

        if self.world_size > 1:
            all_packed = [torch.empty_like(packed) for _ in range(self.world_size)]
            all_scales = [torch.empty_like(scales) for _ in range(self.world_size)]
            dist.all_gather(all_packed, packed, group=self.group)
            dist.all_gather(all_scales, scales, group=self.group)
        else:
            all_packed, all_scales = [packed], [scales]

        total = torch.zeros_like(flat)
        for p, s in zip(all_packed, all_scales):
            total += dequantize(p, s, bits, flat.numel())

        return total / self.world_size, payload_bytes(flat.numel(), bits)
