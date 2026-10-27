"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.standard_normal((input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.standard_normal((hidden_dim, output_dim)).astype(np.float64),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }

# Step 2 - relu
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Step 3 - model_forward
import numpy as np

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

# Step 4 - softmax
import numpy as np

def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Step 5 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(logits, labels):
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = (
        np.log(np.sum(np.exp(logits - max_logits), axis=1))
        + max_logits.ravel()
    )
    true_logits = logits[np.arange(len(labels)), labels]
    return np.mean(log_sum_exp - true_logits)

# Step 6 - model_backward
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

# Step 7 - init_adamw_state
def init_adamw_state(params):
    return {
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
        "t": 0,
    }

# Step 8 - update_adam_moments
def update_adam_moments(state, grads, beta1, beta2):
    state["t"] += 1

    for k, g in grads.items():
        state["m"][k] = beta1 * state["m"][k] + (1 - beta1) * g
        state["v"][k] = beta2 * state["v"][k] + (1 - beta2) * (g ** 2)

    return state

# Step 9 - bias_correct_moments
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

# Step 10 - adam_param_step
def adam_param_step(params, m_hat, v_hat, lr, eps):
    return {
        k: params[k] - lr * m_hat[k] / (np.sqrt(v_hat[k]) + eps)
        for k in params
    }

# Step 11 - decoupled_weight_decay
import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    return {
        k: v * (1 - lr * weight_decay)
        for k, v in params.items()
    }

# Step 12 - clone_params
def clone_params(params):
    return {
        k: v.copy()
        for k, v in params.items()
    }

# Step 13 - scale_params
def scale_params(params, scalar):
    return {
        k: v * scalar
        for k, v in params.items()
    }

# Step 14 - subtract_params
def subtract_params(params_a, params_b):
    return {
        k: params_a[k] - params_b[k]
        for k in params_a
    }

# Step 15 - average_params
def average_params(params_list):
    return {
        k: np.mean([params[k] for params in params_list], axis=0)
        for k in params_list[0]
    }

# Step 16 - iid_shard_dataset
def iid_shard_dataset(x, y, num_workers, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))

    x_shards = np.array_split(indices, num_workers)

    return [
        (x[shard], y[shard])
        for shard in x_shards
    ]

# Step 17 - noniid_shard_dataset
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

# Step 18 - sample_worker_batch
def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    n = len(x_shard)
    indices = rng.choice(
        n,
        size=batch_size,
        replace=(batch_size > n)
    )

    return x_shard[indices], y_shard[indices]

# Step 19 - local_train_step
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

# Step 20 - inner_train_worker
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

# Step 21 - init_outer_optimizer
def init_outer_optimizer(params):
    return {
        "momentum": {
            k: np.zeros_like(v)
            for k, v in params.items()
        }
    }

# Step 22 - update_outer_momentum
import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    for k in outer_state["momentum"]:
        outer_state["momentum"][k] = (
            momentum_coef * outer_state["momentum"][k]
            + outer_grad[k]
        )

    return outer_state

# Step 23 - nesterov_param_update (not yet solved)
# TODO: implement

# Step 24 - compute_outer_gradient (not yet solved)
# TODO: implement

# Step 25 - run_diloco_round (not yet solved)
# TODO: implement

# Step 26 - train_diloco (not yet solved)
# TODO: implement

# Step 27 - train_synchronous_baseline (not yet solved)
# TODO: implement

# Step 28 - evaluate_loss (not yet solved)
# TODO: implement

# Step 29 - classification_accuracy (not yet solved)
# TODO: implement

# Step 30 - communication_savings (not yet solved)
# TODO: implement

