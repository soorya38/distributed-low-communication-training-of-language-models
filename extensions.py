"""
Extensions on top of the core DiLoCo implementation in model.py.

- Dirichlet label-skew sharding: a dial between IID and fully non-IID data.
- Pseudo-gradient compression: k-bit quantization and top-k sparsification,
  with optional error feedback.
- Straggler-tolerant and asynchronous outer updates, run against a simulated
  wall clock where every worker has its own step time.

Communication is accounted as float32 on the wire (4 bytes per value), the
usual precision for distributed training, even though the NumPy math runs in
float64.
"""

import heapq

import numpy as np

from model import (
    average_params,
    clone_params,
    init_outer_optimizer,
    inner_train_worker,
    nesterov_param_update,
    scale_params,
    subtract_params,
    update_outer_momentum,
)

BYTES_PER_FLOAT = 4
BYTES_PER_INDEX = 4


def add_params(params_a, params_b):
    return {k: params_a[k] + params_b[k] for k in params_a}


def param_count(params):
    return sum(int(v.size) for v in params.values())


# --------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------


def dirichlet_shard_dataset(x, y, num_workers, alpha, seed=0, min_size=1, max_tries=100):
    """Split each class across workers with Dirichlet(alpha) proportions.

    Large alpha gives near-IID shards; small alpha gives each worker only a few
    classes. Resamples until every worker has at least `min_size` examples.
    """
    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        worker_indices = [[] for _ in range(num_workers)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            proportions = rng.dirichlet(alpha * np.ones(num_workers))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            for worker, part in enumerate(np.split(idx, cuts)):
                worker_indices[worker].extend(part.tolist())

        if min(len(idx) for idx in worker_indices) >= min_size:
            break
    else:
        raise ValueError(f"could not give every worker {min_size} examples with alpha={alpha}")

    shards = []
    for idx in worker_indices:
        idx = rng.permutation(np.array(idx, dtype=int))
        shards.append((x[idx], y[idx]))

    return shards


def label_skew(worker_shards, num_classes):
    """Mean total-variation distance between each worker's label mix and the global mix.

    0 means every worker sees the global label distribution; values near 1 mean
    workers see disjoint labels.
    """
    all_labels = np.concatenate([sy for _, sy in worker_shards])
    global_dist = np.bincount(all_labels, minlength=num_classes) / len(all_labels)

    distances = [
        0.5 * np.abs(np.bincount(sy, minlength=num_classes) / len(sy) - global_dist).sum()
        for _, sy in worker_shards
    ]

    return float(np.mean(distances))


# --------------------------------------------------------------------------
# Pseudo-gradient compression
# --------------------------------------------------------------------------


def quantize_tensor(x, bits):
    """Uniform symmetric per-tensor quantization; 1 bit means sign * mean(|x|)."""
    scale = np.max(np.abs(x))
    if scale == 0:
        return np.zeros_like(x)

    if bits == 1:
        return np.sign(x) * np.mean(np.abs(x))

    levels = 2 ** (bits - 1) - 1
    return np.round(x / scale * levels) / levels * scale


def topk_tensor(x, fraction):
    """Keep the `fraction` largest-magnitude entries (at least one), zero the rest."""
    k = max(1, int(round(fraction * x.size)))
    flat = x.ravel()
    keep = np.argpartition(np.abs(flat), -k)[-k:]

    out = np.zeros_like(flat)
    out[keep] = flat[keep]

    return out.reshape(x.shape), k


def compress_params(params, compressor):
    """Compress every tensor and return (decompressed params, bytes on the wire).

    compressor: None for float32, {"method": "quantize", "bits": b},
    or {"method": "topk", "fraction": f}.
    """
    if compressor is None:
        return clone_params(params), param_count(params) * BYTES_PER_FLOAT

    compressed, num_bytes = {}, 0.0
    for k, v in params.items():
        if compressor["method"] == "quantize":
            compressed[k] = quantize_tensor(v, compressor["bits"])
            num_bytes += v.size * compressor["bits"] / 8 + BYTES_PER_FLOAT  # values + scale
        elif compressor["method"] == "topk":
            compressed[k], kept = topk_tensor(v, compressor["fraction"])
            num_bytes += kept * (BYTES_PER_FLOAT + BYTES_PER_INDEX)
        else:
            raise ValueError(f"unknown compression method {compressor['method']!r}")

    return compressed, num_bytes


def run_compressed_diloco_round(global_params, outer_state, error_state, worker_shards, num_inner_steps,
                                batch_size, inner_hparams, outer_lr, momentum_coef, compressor, seed):
    """One DiLoCo round where each worker compresses its pseudo-gradient before sending.

    With error feedback (error_state is a list, one residual per worker), whatever
    compression drops this round is added back into the next round's message.
    """
    sent, bytes_up = [], 0.0

    for i, (x_shard, y_shard) in enumerate(worker_shards):
        worker_params, _ = inner_train_worker(
            global_params, x_shard, y_shard, num_inner_steps, batch_size,
            inner_hparams["lr"], inner_hparams["beta1"], inner_hparams["beta2"],
            inner_hparams["eps"], inner_hparams["weight_decay"], seed + i,
        )
        delta = subtract_params(global_params, worker_params)
        if error_state is not None:
            delta = add_params(delta, error_state[i])

        compressed, num_bytes = compress_params(delta, compressor)
        if error_state is not None:
            error_state[i] = subtract_params(delta, compressed)

        sent.append(compressed)
        bytes_up += num_bytes

    outer_grad = average_params(sent)
    update_outer_momentum(outer_state, outer_grad, momentum_coef)
    new_global_params = nesterov_param_update(
        global_params, outer_state["momentum"], outer_grad, outer_lr, momentum_coef
    )

    return new_global_params, outer_state, error_state, bytes_up


def train_compressed_diloco(init_params, worker_shards, num_rounds, num_inner_steps, batch_size, inner_hparams,
                            outer_lr, momentum_coef, compressor=None, error_feedback=True, seed=0):
    """DiLoCo with compressed worker-to-server messages.

    Returns (params, stats). bytes_up counts worker->server traffic, which is what
    compression shrinks; bytes_down counts the float32 broadcast of the new global
    params back to every worker.
    """
    params = clone_params(init_params)
    outer_state = init_outer_optimizer(params)
    error_state = None
    if error_feedback and compressor is not None:
        error_state = [scale_params(params, 0.0) for _ in worker_shards]

    bytes_up = 0.0
    for round_idx in range(num_rounds):
        params, outer_state, error_state, round_bytes = run_compressed_diloco_round(
            params, outer_state, error_state, worker_shards, num_inner_steps, batch_size,
            inner_hparams, outer_lr, momentum_coef, compressor, seed + round_idx,
        )
        bytes_up += round_bytes

    bytes_down = num_rounds * len(worker_shards) * param_count(params) * BYTES_PER_FLOAT

    return params, {"bytes_up": bytes_up, "bytes_down": bytes_down}


# --------------------------------------------------------------------------
# Stragglers and asynchronous outer updates (simulated wall clock)
# --------------------------------------------------------------------------


def train_timed_diloco(init_params, worker_shards, step_times, time_budget, num_inner_steps, round_time,
                       batch_size, inner_hparams, outer_lr, momentum_coef, seed=0, eval_fn=None):
    """Synchronous DiLoCo where every round ends after `round_time` simulated seconds.

    Worker i fits min(H, floor(round_time / step_times[i])) inner steps into a
    round. With round_time = H * max(step_times) this is plain DiLoCo, which waits
    for the slowest worker. A shorter round_time is a deadline: stragglers send
    whatever progress they have, and pseudo-gradients are averaged weighted by
    the number of steps each worker completed.
    """
    params = clone_params(init_params)
    outer_state = init_outer_optimizer(params)
    history = {"time": [], "eval": [], "busy": 0.0, "steps": 0}
    t, round_idx = 0.0, 0

    while t + round_time <= time_budget + 1e-9:
        deltas, weights = [], []
        for i, (x_shard, y_shard) in enumerate(worker_shards):
            steps = min(num_inner_steps, int(round_time / step_times[i] + 1e-9))
            if steps == 0:
                continue
            worker_params, _ = inner_train_worker(
                params, x_shard, y_shard, steps, batch_size,
                inner_hparams["lr"], inner_hparams["beta1"], inner_hparams["beta2"],
                inner_hparams["eps"], inner_hparams["weight_decay"], seed + round_idx + i,
            )
            deltas.append(subtract_params(params, worker_params))
            weights.append(steps)
            history["busy"] += steps * step_times[i]
            history["steps"] += steps

        total = float(sum(weights))
        outer_grad = average_params([scale_params(d, len(deltas) * w / total) for d, w in zip(deltas, weights)])
        update_outer_momentum(outer_state, outer_grad, momentum_coef)
        params = nesterov_param_update(params, outer_state["momentum"], outer_grad, outer_lr, momentum_coef)

        t += round_time
        round_idx += 1
        history["time"].append(t)
        history["eval"].append(eval_fn(params) if eval_fn else None)

    history["utilization"] = history["busy"] / (len(worker_shards) * t) if t else 0.0
    return params, history


def train_async_diloco(init_params, worker_shards, step_times, time_budget, num_inner_steps, batch_size,
                       inner_hparams, outer_lr, momentum_coef, staleness_discount=True, seed=0, eval_fn=None):
    """Asynchronous DiLoCo: no rounds, nobody waits.

    Each worker pulls the current global params, runs H inner steps (taking
    H * step_times[i] simulated seconds), and pushes its pseudo-gradient. The
    server applies it right away as an outer Nesterov step scaled by 1/k, so k
    pushes add up to about one synchronous round. A push is stale by the number
    of outer updates applied since that worker pulled; with staleness_discount
    it is further scaled by 1 / (1 + staleness).
    """
    num_workers = len(worker_shards)
    params = clone_params(init_params)
    outer_state = init_outer_optimizer(params)
    version = 0
    history = {"time": [], "eval": [], "staleness": [], "busy": 0.0, "steps": 0}

    # (finish_time, worker, job, version pulled, params pulled)
    events = [(num_inner_steps * step_times[i], i, 0, 0, clone_params(params)) for i in range(num_workers)]
    heapq.heapify(events)

    while events and events[0][0] <= time_budget + 1e-9:
        t, i, job, pulled_version, pulled_params = heapq.heappop(events)
        x_shard, y_shard = worker_shards[i]
        worker_params, _ = inner_train_worker(
            pulled_params, x_shard, y_shard, num_inner_steps, batch_size,
            inner_hparams["lr"], inner_hparams["beta1"], inner_hparams["beta2"],
            inner_hparams["eps"], inner_hparams["weight_decay"], seed + 1000 * i + job,
        )

        staleness = version - pulled_version
        weight = 1.0 / (1 + staleness) if staleness_discount else 1.0
        outer_grad = scale_params(subtract_params(pulled_params, worker_params), weight / num_workers)
        update_outer_momentum(outer_state, outer_grad, momentum_coef)
        params = nesterov_param_update(params, outer_state["momentum"], outer_grad, outer_lr, momentum_coef)
        version += 1

        history["time"].append(t)
        history["eval"].append(eval_fn(params) if eval_fn else None)
        history["staleness"].append(staleness)
        history["busy"] += num_inner_steps * step_times[i]
        history["steps"] += num_inner_steps

        heapq.heappush(events, (t + num_inner_steps * step_times[i], i, job + 1, version, clone_params(params)))

    history["utilization"] = history["busy"] / (num_workers * time_budget)
    return params, history
