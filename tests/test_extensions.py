"""Unit tests for extensions.py. Run with: python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extensions import *  # noqa: F401, F403
from model import *  # noqa: F401, F403

HPARAMS = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4)


def make_data(n=120, input_dim=5, num_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, input_dim))
    y = np.argmax(x @ rng.standard_normal((input_dim, num_classes)), axis=1)
    return x, y


def assert_params_close(test, a, b, **kwargs):
    test.assertEqual(set(a), set(b))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], err_msg=k, **kwargs)


class TestSharding(unittest.TestCase):
    def test_dirichlet_shards_partition_the_data(self):
        x, y = make_data()
        shards = dirichlet_shard_dataset(x, y, num_workers=3, alpha=0.5, seed=0, min_size=5)
        self.assertEqual(len(shards), 3)
        self.assertEqual(sum(len(sy) for _, sy in shards), len(y))
        self.assertTrue(all(len(sy) >= 5 for _, sy in shards))
        rows = np.concatenate([sx for sx, _ in shards])
        self.assertEqual(len(np.unique(rows, axis=0)), len(x))  # every example used exactly once

    def test_small_alpha_is_more_skewed_than_large_alpha(self):
        x, y = make_data(n=400)
        near_iid = label_skew(dirichlet_shard_dataset(x, y, 4, alpha=100.0, seed=0), 4)
        skewed = label_skew(dirichlet_shard_dataset(x, y, 4, alpha=0.05, seed=0, min_size=1), 4)
        self.assertLess(near_iid, 0.1)
        self.assertGreater(skewed, 0.4)

    def test_label_skew_of_disjoint_shards(self):
        y = np.array([0, 0, 1, 1])
        x = np.zeros((4, 1))
        shards = [(x[:2], y[:2]), (x[2:], y[2:])]
        self.assertAlmostEqual(label_skew(shards, 2), 0.5)
        self.assertAlmostEqual(label_skew([(x, y), (x, y)], 2), 0.0)


class TestCompression(unittest.TestCase):
    def test_quantize_error_is_bounded(self):
        x = np.random.default_rng(0).standard_normal(1000)
        for bits in (8, 4, 2):
            levels = 2 ** (bits - 1) - 1
            err = np.abs(quantize_tensor(x, bits) - x).max()
            self.assertLessEqual(err, np.abs(x).max() / levels / 2 + 1e-12)

    def test_one_bit_quantization_is_scaled_sign(self):
        x = np.array([1.0, -3.0, 2.0])
        np.testing.assert_allclose(quantize_tensor(x, 1), [2.0, -2.0, 2.0])
        np.testing.assert_array_equal(quantize_tensor(np.zeros(3), 4), np.zeros(3))

    def test_topk_keeps_largest_magnitudes(self):
        x = np.array([[0.1, -5.0], [3.0, 0.2]])
        out, k = topk_tensor(x, 0.5)
        self.assertEqual(k, 2)
        np.testing.assert_array_equal(out, [[0.0, -5.0], [3.0, 0.0]])
        self.assertEqual(topk_tensor(x, 0.01)[1], 1)  # always keeps at least one

    def test_compress_params_byte_accounting(self):
        params = init_model_params(4, 6, 3, seed=0)
        n = param_count(params)
        self.assertEqual(compress_params(params, None)[1], n * 4)
        self.assertEqual(compress_params(params, {"method": "quantize", "bits": 8})[1], n + 4 * len(params))
        kept = sum(max(1, round(0.25 * v.size)) for v in params.values())
        self.assertEqual(compress_params(params, {"method": "topk", "fraction": 0.25})[1], kept * 8)

    def test_uncompressed_matches_train_diloco(self):
        x, y = make_data()
        shards = iid_shard_dataset(x, y, 3, seed=0)
        init = init_model_params(5, 8, 4, seed=0)
        expected, _ = train_diloco(init, shards, 3, 5, 8, HPARAMS, 0.7, 0.9, seed=1)
        actual, stats = train_compressed_diloco(init, shards, 3, 5, 8, HPARAMS, 0.7, 0.9, compressor=None, seed=1)
        assert_params_close(self, actual, expected)
        self.assertEqual(stats["bytes_up"], 3 * 3 * param_count(init) * 4)
        self.assertEqual(stats["bytes_down"], stats["bytes_up"])

    def test_error_feedback_keeps_what_compression_dropped(self):
        x, y = make_data()
        shards = iid_shard_dataset(x, y, 2, seed=0)
        params = init_model_params(5, 8, 4, seed=0)
        errors = [scale_params(params, 0.0) for _ in shards]
        compressor = {"method": "topk", "fraction": 0.1}
        _, _, errors, _ = run_compressed_diloco_round(
            params, init_outer_optimizer(params), errors, shards, 5, 8, HPARAMS, 0.7, 0.9, compressor, seed=0)

        worker_params, _ = inner_train_worker(params, *shards[0], 5, 8, *HPARAMS.values(), 0)
        delta = subtract_params(params, worker_params)
        sent, _ = compress_params(delta, compressor)
        assert_params_close(self, add_params(sent, errors[0]), delta, atol=1e-12)


class TestStragglersAndAsync(unittest.TestCase):
    def setUp(self):
        x, y = make_data()
        self.shards = iid_shard_dataset(x, y, 3, seed=0)
        self.init = init_model_params(5, 8, 4, seed=0)

    def test_waiting_for_slowest_worker_is_plain_diloco(self):
        step_times = [1.0, 1.0, 3.0]
        params, hist = train_timed_diloco(self.init, self.shards, step_times, time_budget=60, num_inner_steps=5,
                                          round_time=15, batch_size=8, inner_hparams=HPARAMS, outer_lr=0.7,
                                          momentum_coef=0.9, seed=2)
        expected, _ = train_diloco(self.init, self.shards, 4, 5, 8, HPARAMS, 0.7, 0.9, seed=2)
        assert_params_close(self, params, expected)
        self.assertEqual(hist["time"], [15, 30, 45, 60])
        self.assertAlmostEqual(hist["utilization"], (5 + 5 + 15) / (3 * 15))

    def test_deadline_lets_stragglers_send_partial_work(self):
        _, hist = train_timed_diloco(self.init, self.shards, [1.0, 1.0, 3.0], time_budget=50, num_inner_steps=5,
                                     round_time=5, batch_size=8, inner_hparams=HPARAMS, outer_lr=0.7,
                                     momentum_coef=0.9)
        self.assertEqual(len(hist["time"]), 10)
        self.assertEqual(hist["steps"], 10 * (5 + 5 + 1))  # the straggler fits 1 step per 5 s round

    def test_async_applies_every_push_without_waiting(self):
        evals = []
        _, hist = train_async_diloco(self.init, self.shards, [1.0, 1.0, 3.0], time_budget=30, num_inner_steps=5,
                                     batch_size=8, inner_hparams=HPARAMS, outer_lr=0.7, momentum_coef=0.9,
                                     eval_fn=lambda p: evals.append(1) or 0.0)
        # fast workers finish at 5, 10, ..., 30 (6 pushes each); the straggler at 15 and 30
        self.assertEqual(len(hist["time"]), 6 + 6 + 2)
        self.assertEqual(len(evals), 14)
        self.assertEqual(hist["time"], sorted(hist["time"]))
        self.assertEqual(hist["staleness"][:2], [0, 1])  # second push at t=5 is one update behind
        self.assertGreater(max(hist["staleness"]), 2)  # the straggler's pushes are the stalest
        self.assertAlmostEqual(hist["utilization"], 1.0)


if __name__ == "__main__":
    unittest.main()
