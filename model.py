"""
DiLoCo: Distributed Low-Communication Training of Language Models.

A from-scratch NumPy implementation: a 2-layer MLP with hand-written backprop,
AdamW as the inner optimizer, Nesterov momentum as the outer optimizer, and a
synchronous data-parallel baseline for comparison.

Parameters are plain dicts of arrays, e.g. {"W1", "b1", "W2", "b2"}.
"""

import numpy as np


# --------------------------------------------------------------------------
# Model, forward pass & loss
# --------------------------------------------------------------------------


def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.standard_normal((input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.standard_normal((hidden_dim, output_dim)).astype(np.float64),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }


def relu(x):
    return np.maximum(0, x)


def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    z1 = x @ params["W1"] + params["b1"]
    h1 = relu(z1)
    logits = h1 @ params["W2"] + params["b2"]

    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "logits": logits,
    }

    return logits, cache


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(logits, labels):
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = (
        np.log(np.sum(np.exp(logits - max_logits), axis=1))
        + max_logits.ravel()
    )
    true_logits = logits[np.arange(len(labels)), labels]
    return np.mean(log_sum_exp - true_logits)


def model_backward(params, cache, labels):
    x = cache["x"]
    z1 = cache["z1"]
    h1 = cache["h1"]
    logits = cache["logits"]

    N = x.shape[0]

    # Softmax cross-entropy gradient
    dlogits = softmax(logits)
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N

    # Output layer gradients
    dW2 = h1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)

    # Backpropagate through ReLU
    dh1 = dlogits @ params["W2"].T
    dz1 = dh1 * (z1 > 0)

    # Input layer gradients
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }


# --------------------------------------------------------------------------
# Inner optimizer: AdamW
# --------------------------------------------------------------------------


def init_adamw_state(params):
    return {
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
        "t": 0,
    }


def update_adam_moments(state, grads, beta1, beta2):
    state["t"] += 1

    for k, g in grads.items():
        state["m"][k] = beta1 * state["m"][k] + (1 - beta1) * g
        state["v"][k] = beta2 * state["v"][k] + (1 - beta2) * (g ** 2)

    return state


def bias_correct_moments(state, beta1, beta2):
    t = state["t"]

    m_hat = {
        k: m / (1 - beta1 ** t)
        for k, m in state["m"].items()
    }

    v_hat = {
        k: v / (1 - beta2 ** t)
        for k, v in state["v"].items()
    }

    return m_hat, v_hat


def adam_param_step(params, m_hat, v_hat, lr, eps):
    return {
        k: params[k] - lr * m_hat[k] / (np.sqrt(v_hat[k]) + eps)
        for k in params
    }


def decoupled_weight_decay(params, lr, weight_decay):
    return {
        k: v * (1 - lr * weight_decay)
        for k, v in params.items()
    }


# --------------------------------------------------------------------------
# Parameter arithmetic
# --------------------------------------------------------------------------


def clone_params(params):
    return {
        k: v.copy()
        for k, v in params.items()
    }


def scale_params(params, scalar):
    return {
        k: v * scalar
        for k, v in params.items()
    }


def subtract_params(params_a, params_b):
    return {
        k: params_a[k] - params_b[k]
        for k in params_a
    }


def average_params(params_list):
    return {
        k: np.mean([params[k] for params in params_list], axis=0)
        for k in params_list[0]
    }


# --------------------------------------------------------------------------
# Data sharding & local worker training
# --------------------------------------------------------------------------


def iid_shard_dataset(x, y, num_workers, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))

    x_shards = np.array_split(indices, num_workers)

    return [
        (x[shard], y[shard])
        for shard in x_shards
    ]


def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    rng = np.random.default_rng(seed)
    shards = []

    for worker in range(num_workers):
        worker_indices = np.where(
            (y % num_workers) == worker
        )[0]

        rng.shuffle(worker_indices)

        shards.append((
            x[worker_indices],
            y[worker_indices]
        ))

    return shards


def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    n = len(x_shard)
    indices = rng.choice(
        n,
        size=batch_size,
        replace=(batch_size > n)
    )

    return x_shard[indices], y_shard[indices]


def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    # Forward pass
    logits, cache = model_forward(params, x_batch)

    # Compute loss
    loss = cross_entropy_loss(logits, y_batch)

    # Backpropagation
    grads = model_backward(params, cache, y_batch)

    # Update AdamW moments
    new_adam_state = update_adam_moments(
        adam_state, grads, beta1, beta2
    )

    # Bias correction
    m_hat, v_hat = bias_correct_moments(
        new_adam_state, beta1, beta2
    )

    # Adaptive Adam parameter update
    updated_params = adam_param_step(
        params, m_hat, v_hat, lr, eps
    )

    # Apply decoupled weight decay
    new_params = decoupled_weight_decay(
        updated_params, lr, weight_decay
    )

    return new_params, new_adam_state, loss


def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    rng = np.random.default_rng(seed)
    worker_params = clone_params(params)
    adam_state = init_adamw_state(worker_params)
    total_loss = 0.0

    for _ in range(num_inner_steps):
        x_batch, y_batch = sample_worker_batch(
            x_shard, y_shard, batch_size, rng
        )

        worker_params, adam_state, loss = local_train_step(
            worker_params, adam_state,
            x_batch, y_batch,
            lr, beta1, beta2, eps, weight_decay
        )

        total_loss += loss

    mean_loss = float(total_loss / num_inner_steps) if num_inner_steps > 0 else 0.0

    return worker_params, mean_loss


# --------------------------------------------------------------------------
# Outer optimizer: Nesterov momentum
# --------------------------------------------------------------------------


def init_outer_optimizer(params):
    return {
        "momentum": {
            k: np.zeros_like(v)
            for k, v in params.items()
        }
    }


def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    for k in outer_state["momentum"]:
        outer_state["momentum"][k] = (
            momentum_coef * outer_state["momentum"][k]
            + outer_grad[k]
        )

    return outer_state


def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    return {
        k: params[k] - outer_lr * (
            outer_grad[k] + momentum_coef * outer_state[k]
        )
        for k in params
    }


def compute_outer_gradient(global_params, worker_params_list):
    avg_worker_params = average_params(worker_params_list)
    return subtract_params(global_params, avg_worker_params)


# --------------------------------------------------------------------------
# DiLoCo training loop & synchronous baseline
# --------------------------------------------------------------------------


def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    worker_params_list = []
    worker_losses = []

    for i, (x_shard, y_shard) in enumerate(worker_shards):
        worker_params, mean_loss = inner_train_worker(
            global_params,
            x_shard,
            y_shard,
            num_inner_steps,
            batch_size,
            inner_hparams["lr"],
            inner_hparams["beta1"],
            inner_hparams["beta2"],
            inner_hparams["eps"],
            inner_hparams["weight_decay"],
            seed + i
        )
        worker_params_list.append(worker_params)
        worker_losses.append(mean_loss)

    outer_grad = compute_outer_gradient(
        global_params, worker_params_list
    )

    update_outer_momentum(
        outer_state, outer_grad, momentum_coef
    )

    new_global_params = nesterov_param_update(
        global_params,
        outer_state["momentum"],
        outer_grad,
        outer_lr,
        momentum_coef
    )

    return new_global_params, outer_state, worker_losses


def train_diloco(init_params, worker_shards, num_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed=0):
    params = clone_params(init_params)
    outer_state = init_outer_optimizer(params)
    history = {"round_losses": []}

    for round_idx in range(num_rounds):
        params, outer_state, worker_losses = run_diloco_round(
            params,
            outer_state,
            worker_shards,
            num_inner_steps,
            batch_size,
            inner_hparams,
            outer_lr,
            momentum_coef,
            seed + round_idx
        )

        mean_loss = (
            float(sum(worker_losses) / len(worker_losses))
            if worker_losses else 0.0
        )
        history["round_losses"].append(mean_loss)

    return params, history


def train_synchronous_baseline(init_params, worker_shards, num_steps, batch_size, inner_hparams, seed=0):
    params = clone_params(init_params)
    adam_state = init_adamw_state(params)
    rng = np.random.default_rng(seed)
    history = {"step_losses": []}

    lr = inner_hparams["lr"]
    beta1 = inner_hparams["beta1"]
    beta2 = inner_hparams["beta2"]
    eps = inner_hparams["eps"]
    weight_decay = inner_hparams["weight_decay"]

    for _ in range(num_steps):
        worker_grads = []
        worker_losses = []

        for x_shard, y_shard in worker_shards:
            x_batch, y_batch = sample_worker_batch(
                x_shard, y_shard, batch_size, rng
            )

            logits, cache = model_forward(params, x_batch)
            loss = cross_entropy_loss(logits, y_batch)
            grads = model_backward(params, cache, y_batch)

            worker_grads.append(grads)
            worker_losses.append(loss)

        if not worker_grads:
            history["step_losses"].append(0.0)
            continue

        avg_grads = average_params(worker_grads)

        update_adam_moments(adam_state, avg_grads, beta1, beta2)
        m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)

        params = adam_param_step(params, m_hat, v_hat, lr, eps)
        params = decoupled_weight_decay(params, lr, weight_decay)

        history["step_losses"].append(
            float(np.mean(worker_losses))
        )

    return params, history


# --------------------------------------------------------------------------
# Evaluation & communication analysis
# --------------------------------------------------------------------------


def evaluate_loss(params, x, y):
    logits, _ = model_forward(params, x)
    return float(cross_entropy_loss(logits, y))


def classification_accuracy(params, x, y):
    logits, _ = model_forward(params, x)
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == y))


def communication_savings(num_rounds, num_inner_steps, num_workers, param_count):
    diloco_scalars = 2 * num_workers * param_count * num_rounds
    sync_scalars = 2 * num_workers * param_count * num_rounds * num_inner_steps

    ratio = diloco_scalars / sync_scalars if sync_scalars else 0.0
    savings_factor = sync_scalars / diloco_scalars if diloco_scalars else 0.0

    return {
        "diloco_scalars": diloco_scalars,
        "sync_scalars": sync_scalars,
        "ratio": ratio,
        "savings_factor": savings_factor,
    }

