"""
DiLoCo: Distributed Low-Communication Training of Language Models scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of DiLoCo: distributed low-communication training vs. sync baseline."""

import numpy as np


def main():
    np.random.seed(0)

    # Synthetic classification dataset
    num_samples, input_dim, hidden_dim, num_classes = 400, 8, 16, 3
    X = np.random.randn(num_samples, input_dim).astype(np.float64)
    true_W = np.random.randn(input_dim, num_classes)
    y = np.argmax(X @ true_W + 0.1 * np.random.randn(num_samples, num_classes), axis=1)

    # Held-out eval split
    n_train = 320
    x_train, y_train = X[:n_train], y[:n_train]
    x_eval, y_eval = X[n_train:], y[n_train:]

    # Initial (shared) model params
    init_params = init_model_params(input_dim, hidden_dim, num_classes, seed=0)
    print("Initialized 2-layer MLP with param keys:", sorted(init_params.keys()))

    # Sanity-check forward + loss
    logits, _ = model_forward(init_params, x_train[:4])
    print("Initial logits shape:", logits.shape)
    print("Initial train loss:", round(evaluate_loss(init_params, x_train, y_train), 4))
    print("Initial train acc :", round(classification_accuracy(init_params, x_train, y_train), 4))

    # Shard the data across workers (IID)
    num_workers = 4
    worker_shards = iid_shard_dataset(x_train, y_train, num_workers, seed=1)
    print(f"\nIID shards: {[len(sy) for _, sy in worker_shards]}")

    # DiLoCo hyperparameters
    inner_hparams = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4)
    num_rounds = 8
    num_inner_steps = 20
    batch_size = 16
    outer_lr = 0.7
    momentum_coef = 0.9

    # --- Run DiLoCo ---
    diloco_params, diloco_log = train_diloco(
        clone_params(init_params),
        worker_shards,
        num_rounds=num_rounds,
        num_inner_steps=num_inner_steps,
        batch_size=batch_size,
        inner_hparams=inner_hparams,
        outer_lr=outer_lr,
        momentum_coef=momentum_coef,
        seed=42,
    )
    print("\n=== DiLoCo training log (last few rounds) ===")
    if isinstance(diloco_log, dict):
        log_entries = [{k: diloco_log[k][i] for k in diloco_log} for i in range(len(next(iter(diloco_log.values()))))]
    else:
        log_entries = list(diloco_log)
    for entry in log_entries[-3:]:
        print(entry)
    print("DiLoCo eval loss:", round(evaluate_loss(diloco_params, x_eval, y_eval), 4))
    print("DiLoCo eval acc :", round(classification_accuracy(diloco_params, x_eval, y_eval), 4))

    # --- Synchronous baseline: communicates every step ---
    total_sync_steps = num_rounds * num_inner_steps
    sync_result = train_synchronous_baseline(
        clone_params(init_params),
        worker_shards,
        num_steps=total_sync_steps,
        batch_size=batch_size,
        inner_hparams=inner_hparams,
        seed=42,
    )
    if isinstance(sync_result, tuple):
        sync_params = sync_result[0]
    else:
        sync_params = sync_result
    print("\n=== Synchronous baseline ===")
    print("Sync   eval loss:", round(evaluate_loss(sync_params, x_eval, y_eval), 4))
    print("Sync   eval acc :", round(classification_accuracy(sync_params, x_eval, y_eval), 4))

    # --- Communication accounting ---
    param_count = sum(int(np.asarray(v).size) for v in init_params.values())
    savings = communication_savings(num_rounds, num_inner_steps, num_workers, param_count)
    print("\n=== Communication accounting ===")
    print(f"Total params per worker : {param_count}")
    print(f"Communication summary   : {savings}")


if __name__ == "__main__":
    main()
