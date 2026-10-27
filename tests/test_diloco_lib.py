"""Tests for the diloco PyTorch package (skipped when PyTorch is not installed)."""

import importlib.util
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn

    from diloco import (DiLoCo, dequantize, fragment_offsets, fragments_due, init_distributed, payload_bytes,
                        quantize, split_into_fragments)
    from diloco.compression import BLOCK_SIZE, pack_bits, unpack_bits


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3))


def train_steps(model, diloco, steps, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(8, 6, generator=gen)
        model(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
        diloco.step()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sync_worker(rank, world_size, init_method, kwargs, results):
    torch.set_num_threads(1)
    init_distributed(rank, world_size, init_method)
    model = tiny_model()
    diloco = DiLoCo(model, inner_steps=4, **kwargs)
    train_steps(model, diloco, steps=12, seed=rank)  # different data per rank
    flat = torch.cat([g.flatten() for g in diloco.global_params])
    gathered = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)
    if rank == 0:
        results["identical"] = all(torch.equal(gathered[0], g) for g in gathered[1:])
        results["moved"] = not torch.equal(flat, torch.cat([p.detach().flatten() for p in tiny_model().parameters()]))
        results["stats"] = dict(diloco.stats)
        results["numel"] = flat.numel()
    dist.destroy_process_group()


def run_two_workers(**kwargs):
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_sync_worker, args=(2, f"tcp://127.0.0.1:{_free_port()}", kwargs, results), nprocs=2, join=True)
        return dict(results)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestCompression(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        for bits in (1, 2, 4, 8):
            codes = torch.randint(0, 2 ** bits, (BLOCK_SIZE,))
            packed = pack_bits(codes, bits)
            self.assertEqual(packed.dtype, torch.uint8)
            self.assertEqual(packed.numel(), BLOCK_SIZE * bits // 8)
            self.assertTrue(torch.equal(unpack_bits(packed, bits), codes))

    def test_quantization_error_is_bounded_per_block(self):
        x = torch.randn(3000) * torch.linspace(0.01, 10, 3000)  # scales vary across blocks
        for bits in (8, 4, 2):
            levels = 2 ** (bits - 1) - 1
            packed, scales = quantize(x, bits)
            err = (dequantize(packed, scales, bits, x.numel()) - x).abs()
            padded = torch.zeros(len(scales) * BLOCK_SIZE)
            padded[:3000] = err
            self.assertTrue(torch.all(padded.view(-1, BLOCK_SIZE).amax(1) <= scales / levels / 2 + 1e-6))

    def test_one_bit_is_sign_times_block_mean(self):
        x = torch.tensor([1.0, -3.0, 2.0])
        packed, scales = quantize(x, 1)
        mean = x.abs().sum() / BLOCK_SIZE  # padding zeros count towards the block mean
        self.assertTrue(torch.allclose(dequantize(packed, scales, 1, 3), torch.tensor([1.0, -1.0, 1.0]) * mean))

    def test_payload_bytes(self):
        packed, scales = quantize(torch.randn(2500), 4)
        self.assertEqual(payload_bytes(2500, 4), packed.numel() + scales.numel() * 4)
        self.assertEqual(payload_bytes(2500, 4), 3 * 512 + 3 * 4)
        with self.assertRaises(ValueError):
            payload_bytes(10, 3)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestFragments(unittest.TestCase):
    def test_fragments_cover_every_param_once_and_are_balanced(self):
        params = list(tiny_model().parameters())
        for p_count in (1, 2, 3, len(params)):
            fragments = split_into_fragments(params, p_count)
            self.assertEqual(len(fragments), p_count)
            self.assertEqual(sorted(i for f in fragments for i in f), list(range(len(params))))
            self.assertTrue(all(f == sorted(f) and f for f in fragments))  # contiguous and non-empty
        sizes = [sum(params[i].numel() for i in f) for f in split_into_fragments(params, 2)]
        self.assertLess(max(sizes) / sum(sizes), 0.8)

    def test_staggered_schedule(self):
        offsets = fragment_offsets(inner_steps=12, num_fragments=4)
        self.assertEqual(offsets, [0, 3, 6, 9])
        schedule = {step: fragments_due(step, 12, offsets) for step in range(1, 25)}
        for f in range(4):
            self.assertEqual(sum(f in due for due in schedule.values()), 2)  # once per H steps
        self.assertTrue(all(len(due) <= 1 for due in schedule.values()))  # one fragment at a time
        self.assertEqual(schedule[12], [0])
        self.assertEqual(schedule[3], [1])
        with self.assertRaises(ValueError):
            fragment_offsets(2, 4)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestDiLoCoWrapper(unittest.TestCase):
    def test_single_process_matches_manual_nesterov_step(self):
        start = [p.detach().clone() for p in tiny_model().parameters()]
        model = tiny_model()
        diloco = DiLoCo(model, inner_steps=5, outer_lr=0.7, outer_momentum=0.9)
        train_steps(model, diloco, steps=5, seed=0)

        reference = tiny_model()  # same init, same data, never syncs: the local params before the sync
        train_steps(reference, DiLoCo(reference, inner_steps=1000), steps=5, seed=0)

        self.assertEqual(diloco.stats["syncs"], 1)
        for p, s, r in zip(model.parameters(), start, reference.parameters()):
            delta = s - r.detach()
            # first Nesterov SGD step: buf = delta, update = delta + momentum * buf
            self.assertTrue(torch.allclose(p, s - 0.7 * (delta + 0.9 * delta), atol=1e-6))

    def test_global_weights_context_restores_local_params(self):
        model = tiny_model()
        diloco = DiLoCo(model, inner_steps=100)
        train_steps(model, diloco, steps=3, seed=0)
        local = [p.detach().clone() for p in model.parameters()]
        with diloco.global_weights():
            self.assertTrue(all(torch.equal(p, g) for p, g in zip(model.parameters(), diloco.global_params)))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(model.parameters(), local)))

    def test_streaming_syncs_one_fragment_at_a_time(self):
        model = tiny_model()
        numel = sum(p.numel() for p in model.parameters())
        diloco = DiLoCo(model, inner_steps=8, num_fragments=2)
        train_steps(model, diloco, steps=16, seed=0)
        self.assertEqual(diloco.stats["syncs"], 4)  # steps 4, 8, 12, 16
        self.assertEqual(diloco.stats["bytes_sent"], 2 * numel * 4)  # whole model twice in total
        self.assertLess(diloco.stats["peak_bytes"], numel * 4)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestMultiProcess(unittest.TestCase):
    def test_float32_global_params_identical_across_ranks(self):
        r = run_two_workers()
        self.assertTrue(r["identical"] and r["moved"])
        self.assertEqual(r["stats"]["syncs"], 3)
        self.assertEqual(r["stats"]["bytes_sent"], 3 * r["numel"] * 4)

    def test_quantized_global_params_identical_across_ranks(self):
        for error_feedback in (True, False):
            r = run_two_workers(quantize_bits=4, error_feedback=error_feedback)
            self.assertTrue(r["identical"] and r["moved"])
            self.assertEqual(r["stats"]["bytes_sent"], 3 * payload_bytes(r["numel"], 4))

    def test_streaming_quantized_global_params_identical_across_ranks(self):
        r = run_two_workers(quantize_bits=2, num_fragments=2)
        self.assertTrue(r["identical"] and r["moved"])
        self.assertEqual(r["stats"]["fragment_syncs"], 6)


if __name__ == "__main__":
    unittest.main()
