"""Unit tests for model.py. Run with: python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import *  # noqa: F401, F403


HPARAMS = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4)


def make_data(n=60, input_dim=5, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, input_dim))
    y = np.argmax(x @ rng.standard_normal((input_dim, num_classes)), axis=1)
    return x, y


def assert_params_close(test, a, b, **kwargs):
    test.assertEqual(set(a), set(b))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], err_msg=k, **kwargs)


class TestModel(unittest.TestCase):
    def test_init_model_params_shapes_and_determinism(self):
        params = init_model_params(4, 6, 3, seed=1)
        self.assertEqual(params["W1"].shape, (4, 6))
        self.assertEqual(params["b1"].shape, (6,))
        self.assertEqual(params["W2"].shape, (6, 3))
        self.assertEqual(params["b2"].shape, (3,))
        self.assertTrue(np.all(params["b1"] == 0))
        self.assertTrue(np.all(params["b2"] == 0))
        assert_params_close(self, params, init_model_params(4, 6, 3, seed=1))
        self.assertFalse(np.allclose(params["W1"], init_model_params(4, 6, 3, seed=2)["W1"]))

    def test_relu(self):
        np.testing.assert_array_equal(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])

    def test_model_forward(self):
        params = init_model_params(4, 6, 3, seed=0)
        x = np.random.default_rng(0).standard_normal((5, 4))
        logits, cache = model_forward(params, x)
        self.assertEqual(logits.shape, (5, 3))
        z1 = x @ params["W1"] + params["b1"]
        np.testing.assert_allclose(cache["z1"], z1)
        np.testing.assert_allclose(cache["h1"], np.maximum(0, z1))
        np.testing.assert_allclose(logits, np.maximum(0, z1) @ params["W2"] + params["b2"])

    def test_softmax_rows_sum_to_one_and_is_stable(self):
        probs = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
        np.testing.assert_allclose(probs[1], [1 / 3] * 3)
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_cross_entropy_loss(self):
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        labels = np.array([0, 1])
        expected = -np.mean(np.log(softmax(logits)[np.arange(2), labels]))
        self.assertAlmostEqual(cross_entropy_loss(logits, labels), expected)
        # Uniform logits give log(num_classes), even when very large
        self.assertAlmostEqual(cross_entropy_loss(np.full((2, 4), 1e4), labels), np.log(4))

    def test_model_backward_matches_numerical_gradient(self):
        params = init_model_params(4, 5, 3, seed=0)
        x, y = make_data(n=8, input_dim=4, num_classes=3, seed=1)
        _, cache = model_forward(params, x)
        grads = model_backward(params, cache, y)
        h = 1e-6
        for k, p in params.items():
            num = np.zeros_like(p)
            for idx in np.ndindex(p.shape):
                orig = p[idx]
                p[idx] = orig + h
                plus = cross_entropy_loss(model_forward(params, x)[0], y)
                p[idx] = orig - h
                minus = cross_entropy_loss(model_forward(params, x)[0], y)
                p[idx] = orig
                num[idx] = (plus - minus) / (2 * h)
            np.testing.assert_allclose(grads[k], num, atol=1e-6, err_msg=k)


class TestAdamW(unittest.TestCase):
    def setUp(self):
        self.params = {"w": np.array([1.0, -2.0]), "b": np.array([0.5])}
        self.grads = {"w": np.array([0.1, -0.3]), "b": np.array([2.0])}

    def test_init_adamw_state(self):
        state = init_adamw_state(self.params)
        self.assertEqual(state["t"], 0)
        for k, v in self.params.items():
            np.testing.assert_array_equal(state["m"][k], np.zeros_like(v))
            np.testing.assert_array_equal(state["v"][k], np.zeros_like(v))

    def test_update_adam_moments(self):
        state = update_adam_moments(init_adamw_state(self.params), self.grads, 0.9, 0.99)
        self.assertEqual(state["t"], 1)
        np.testing.assert_allclose(state["m"]["w"], 0.1 * self.grads["w"])
        np.testing.assert_allclose(state["v"]["w"], 0.01 * self.grads["w"] ** 2)

    def test_bias_correct_moments_recovers_first_gradient(self):
        state = update_adam_moments(init_adamw_state(self.params), self.grads, 0.9, 0.99)
        m_hat, v_hat = bias_correct_moments(state, 0.9, 0.99)
        assert_params_close(self, m_hat, self.grads)
        assert_params_close(self, v_hat, {k: g ** 2 for k, g in self.grads.items()})

    def test_adam_param_step(self):
        m_hat = {"w": np.array([1.0, -1.0]), "b": np.array([0.0])}
        v_hat = {"w": np.array([4.0, 1.0]), "b": np.array([1.0])}
        new = adam_param_step(self.params, m_hat, v_hat, lr=0.1, eps=0.0)
        np.testing.assert_allclose(new["w"], [1.0 - 0.05, -2.0 + 0.1])
        np.testing.assert_allclose(new["b"], [0.5])

    def test_decoupled_weight_decay(self):
        new = decoupled_weight_decay(self.params, lr=0.1, weight_decay=0.5)
        assert_params_close(self, new, {k: v * 0.95 for k, v in self.params.items()})


class TestParamOps(unittest.TestCase):
    def setUp(self):
        self.a = {"w": np.array([1.0, 2.0]), "b": np.array([3.0])}
        self.b = {"w": np.array([0.5, -1.0]), "b": np.array([1.0])}

    def test_clone_params_is_deep_copy(self):
        clone = clone_params(self.a)
        assert_params_close(self, clone, self.a)
        clone["w"][0] = 99.0
        self.assertEqual(self.a["w"][0], 1.0)

    def test_scale_params(self):
        assert_params_close(self, scale_params(self.a, 2.0), {"w": [2.0, 4.0], "b": [6.0]})

    def test_subtract_params(self):
        assert_params_close(self, subtract_params(self.a, self.b), {"w": [0.5, 3.0], "b": [2.0]})

    def test_average_params(self):
        assert_params_close(self, average_params([self.a, self.b]), {"w": [0.75, 0.5], "b": [2.0]})


class TestData(unittest.TestCase):
    def test_iid_shard_dataset_partitions_all_samples(self):
        x = np.arange(10, dtype=float).reshape(10, 1)
        y = np.arange(10)
        shards = iid_shard_dataset(x, y, num_workers=3, seed=0)
        self.assertEqual([len(sy) for _, sy in shards], [4, 3, 3])
        all_y = np.concatenate([sy for _, sy in shards])
        np.testing.assert_array_equal(np.sort(all_y), y)
        for sx, sy in shards:
            np.testing.assert_array_equal(sx[:, 0], sy)

    def test_noniid_shard_dataset_splits_by_label(self):
        x, y = make_data(n=60, num_classes=4)
        shards = noniid_shard_dataset(x, y, num_workers=2, num_classes=4, seed=0)
        self.assertEqual(len(shards), 2)
        self.assertEqual(sum(len(sy) for _, sy in shards), len(y))
        for worker, (sx, sy) in enumerate(shards):
            self.assertTrue(np.all(sy % 2 == worker))
            self.assertEqual(len(sx), len(sy))

    def test_sample_worker_batch(self):
        x = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10)
        xb, yb = sample_worker_batch(x, y, 4, np.random.default_rng(0))
        self.assertEqual(xb.shape, (4, 2))
        self.assertEqual(len(np.unique(yb)), 4)  # no replacement when batch fits
        np.testing.assert_array_equal(xb, x[yb])
        xb, yb = sample_worker_batch(x, y, 15, np.random.default_rng(0))
        self.assertEqual(len(yb), 15)  # replacement when batch is larger than shard


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.x, self.y = make_data(n=120, input_dim=5, num_classes=3)
        self.params = init_model_params(5, 8, 3, seed=0)
        self.shards = iid_shard_dataset(self.x, self.y, num_workers=3, seed=0)

    def test_local_train_step(self):
        state = init_adamw_state(self.params)
        new_params, new_state, loss = local_train_step(
            self.params, state, self.x[:16], self.y[:16], **HPARAMS
        )
        self.assertEqual(new_state["t"], 1)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.allclose(new_params["W1"], self.params["W1"]))

    def test_inner_train_worker_reduces_loss_and_keeps_input(self):
        original = clone_params(self.params)
        x_shard, y_shard = self.shards[0]
        worker_params, mean_loss = inner_train_worker(
            self.params, x_shard, y_shard, num_inner_steps=50, batch_size=16, seed=0, **HPARAMS
        )
        assert_params_close(self, self.params, original)
        self.assertIsInstance(mean_loss, float)
        self.assertLess(
            evaluate_loss(worker_params, x_shard, y_shard),
            evaluate_loss(self.params, x_shard, y_shard),
        )

    def test_inner_train_worker_zero_steps(self):
        x_shard, y_shard = self.shards[0]
        worker_params, mean_loss = inner_train_worker(
            self.params, x_shard, y_shard, num_inner_steps=0, batch_size=16, seed=0, **HPARAMS
        )
        assert_params_close(self, worker_params, self.params)
        self.assertEqual(mean_loss, 0.0)


class TestOuterOptimizer(unittest.TestCase):
    def setUp(self):
        self.params = {"w": np.array([1.0, 2.0])}
        self.grad = {"w": np.array([0.5, -1.0])}

    def test_init_outer_optimizer(self):
        state = init_outer_optimizer(self.params)
        np.testing.assert_array_equal(state["momentum"]["w"], [0.0, 0.0])

    def test_update_outer_momentum(self):
        state = init_outer_optimizer(self.params)
        update_outer_momentum(state, self.grad, 0.9)
        update_outer_momentum(state, self.grad, 0.9)
        np.testing.assert_allclose(state["momentum"]["w"], 1.9 * self.grad["w"])

    def test_nesterov_param_update(self):
        momentum = {"w": np.array([1.0, 1.0])}
        new = nesterov_param_update(self.params, momentum, self.grad, outer_lr=0.1, momentum_coef=0.9)
        np.testing.assert_allclose(new["w"], [1.0 - 0.1 * 1.4, 2.0 - 0.1 * -0.1])

    def test_compute_outer_gradient(self):
        workers = [{"w": np.array([0.0, 1.0])}, {"w": np.array([1.0, 1.0])}]
        grad = compute_outer_gradient(self.params, workers)
        np.testing.assert_allclose(grad["w"], [0.5, 1.0])


class TestDiLoCo(unittest.TestCase):
    def setUp(self):
        x, y = make_data(n=200, input_dim=5, num_classes=3)
        self.x_train, self.y_train = x[:160], y[:160]
        self.params = init_model_params(5, 8, 3, seed=0)
        self.shards = iid_shard_dataset(self.x_train, self.y_train, num_workers=4, seed=1)

    def test_run_diloco_round(self):
        state = init_outer_optimizer(self.params)
        new_params, new_state, losses = run_diloco_round(
            self.params, state, self.shards, num_inner_steps=5, batch_size=8,
            inner_hparams=HPARAMS, outer_lr=0.7, momentum_coef=0.9, seed=0,
        )
        self.assertEqual(len(losses), len(self.shards))
        self.assertEqual(set(new_params), set(self.params))
        self.assertFalse(np.allclose(new_state["momentum"]["W1"], 0.0))

    def test_outer_lr_one_no_momentum_equals_worker_average(self):
        state = init_outer_optimizer(self.params)
        new_params, _, _ = run_diloco_round(
            self.params, state, self.shards, num_inner_steps=3, batch_size=8,
            inner_hparams=HPARAMS, outer_lr=1.0, momentum_coef=0.0, seed=0,
        )
        workers = [
            inner_train_worker(self.params, sx, sy, 3, 8, seed=i, **HPARAMS)[0]
            for i, (sx, sy) in enumerate(self.shards)
        ]
        assert_params_close(self, new_params, average_params(workers), atol=1e-12)

    def test_train_diloco_improves_loss(self):
        params, history = train_diloco(
            self.params, self.shards, num_rounds=6, num_inner_steps=10, batch_size=16,
            inner_hparams=HPARAMS, outer_lr=0.7, momentum_coef=0.9, seed=0,
        )
        self.assertEqual(len(history["round_losses"]), 6)
        self.assertLess(
            evaluate_loss(params, self.x_train, self.y_train),
            evaluate_loss(self.params, self.x_train, self.y_train),
        )

    def test_train_synchronous_baseline_improves_loss(self):
        result = train_synchronous_baseline(
            self.params, self.shards, num_steps=60, batch_size=16, inner_hparams=HPARAMS, seed=0,
        )
        params = result[0] if isinstance(result, tuple) else result
        self.assertLess(
            evaluate_loss(params, self.x_train, self.y_train),
            evaluate_loss(self.params, self.x_train, self.y_train),
        )


class TestMetrics(unittest.TestCase):
    def test_evaluate_loss(self):
        params = init_model_params(5, 8, 3, seed=0)
        x, y = make_data(n=20)
        expected = cross_entropy_loss(model_forward(params, x)[0], y)
        loss = evaluate_loss(params, x, y)
        self.assertIsInstance(loss, float)
        self.assertAlmostEqual(loss, expected)

    def test_classification_accuracy(self):
        params = {
            "W1": np.eye(2), "b1": np.zeros(2),
            "W2": np.eye(2), "b2": np.zeros(2),
        }
        x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        self.assertEqual(classification_accuracy(params, x, np.array([0, 1, 0, 1])), 1.0)
        self.assertEqual(classification_accuracy(params, x, np.array([0, 1, 1, 0])), 0.5)

    def test_communication_savings(self):
        out = communication_savings(num_rounds=10, num_inner_steps=50, num_workers=4, param_count=100)
        self.assertEqual(out["diloco_scalars"], 2 * 4 * 100 * 10)
        self.assertEqual(out["sync_scalars"], 2 * 4 * 100 * 10 * 50)
        self.assertAlmostEqual(out["ratio"], 1 / 50)
        self.assertAlmostEqual(out["savings_factor"], 50)

    def test_communication_savings_zero(self):
        out = communication_savings(0, 50, 4, 100)
        self.assertEqual(out["ratio"], 0.0)
        self.assertEqual(out["savings_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
