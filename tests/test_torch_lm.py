"""Smoke tests for torch_diloco/train_lm.py (skipped when PyTorch is not installed)."""

import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch_diloco"))

HAS_TORCH = importlib.util.find_spec("torch") is not None

TINY = ["--world-size", "2", "--steps", "4", "--batch-size", "2", "--context", "16", "--d-model", "32",
        "--n-layers", "1", "--n-heads", "2", "--eval-batches", "1", "--threads", "1"]


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchLM(unittest.TestCase):
    def test_model_shapes_and_weight_tying(self):
        import torch
        from train_lm import CharGPT

        model = CharGPT(vocab_size=10, context=8, d_model=16, n_layers=1, n_heads=2)
        idx = torch.randint(10, (3, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(tuple(logits.shape), (3, 8, 10))
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(model.head.weight, model.tok_emb.weight)
        self.assertEqual(model.generate(idx[:1, :2], 5).shape[1], 7)

    def test_diloco_syncs_every_h_steps(self):
        from train_lm import parse_args, run

        r = run(parse_args(TINY + ["--mode", "diloco", "--inner-steps", "2", "--eval-interval", "2"]))
        self.assertEqual(r["syncs"], 2)
        self.assertEqual(r["bytes_sent_per_worker"], 2 * r["params"] * 4)  # float32 pseudo-gradients

    def test_ddp_syncs_every_step(self):
        from train_lm import parse_args, run

        r = run(parse_args(TINY + ["--mode", "ddp", "--eval-interval", "2"]))
        self.assertEqual(r["syncs"], 4)
        self.assertEqual(r["bytes_sent_per_worker"], 4 * r["params"] * 4)


if __name__ == "__main__":
    unittest.main()
