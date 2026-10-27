"""
Stragglers and asynchronous outer updates.

Four IID workers where one is 4x slower than the others (step times 1, 1, 1, 4
simulated seconds), H = 20, and a 640-second budget: exactly 8 rounds of plain
DiLoCo, which waits for the straggler every round. We compare:

- no straggler (all step times 1): the upper bound
- synchronous DiLoCo: every round waits for the slowest worker
- deadline DiLoCo: rounds end after 20 s; the straggler sends the 5 steps it
  finished and its pseudo-gradient is down-weighted by its step count
- async DiLoCo: nobody waits; each push is applied as it arrives, with and
  without a 1 / (1 + staleness) discount

Accuracy is tracked against simulated time and averaged over 5 seeds.

Run with: python experiments/stragglers.py   (writes docs/assets/stragglers.png)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (BATCH_SIZE, INNER_HPARAMS, MOMENTUM_COEF, NUM_WORKERS, OUTER_LR, SEEDS,
                    make_data, save)
from extensions import train_async_diloco, train_timed_diloco
from model import *  # noqa: F401, F403

H = 20
STEP_TIMES = [1.0, 1.0, 1.0, 4.0]
TIME_BUDGET = 640.0
TARGET_ACC = 90.0
GRID = np.linspace(0, TIME_BUDGET, 641)

METHODS = {
    "No straggler (upper bound)": dict(kind="timed", step_times=[1.0] * NUM_WORKERS, round_time=H * 1.0),
    "Sync DiLoCo (waits for straggler)": dict(kind="timed", step_times=STEP_TIMES, round_time=H * max(STEP_TIMES)),
    "Deadline DiLoCo (20 s rounds)": dict(kind="timed", step_times=STEP_TIMES, round_time=H * min(STEP_TIMES)),
    "Async DiLoCo": dict(kind="async", step_times=STEP_TIMES, staleness_discount=False),
    "Async DiLoCo + staleness discount": dict(kind="async", step_times=STEP_TIMES, staleness_discount=True),
}
COLORS = ["#888888", "#dc2626", "#d97706", "#2563eb", "#16a34a"]


def on_grid(times, values, initial):
    """Step-function interpolation: the accuracy of the latest model at each grid time."""
    idx = np.searchsorted(times, GRID, side="right") - 1
    return np.where(idx >= 0, np.asarray(values)[np.maximum(idx, 0)], initial)


def run(method):
    curves, finals, utilization, updates, steps, staleness = [], [], [], [], [], []
    for seed in SEEDS:
        x_train, y_train, x_eval, y_eval = make_data(seed)
        init_params = init_model_params(x_train.shape[1], 16, 3, seed=seed)
        shards = iid_shard_dataset(x_train, y_train, NUM_WORKERS, seed=seed)
        eval_fn = lambda p: classification_accuracy(p, x_eval, y_eval) * 100  # noqa: E731
        common = dict(batch_size=BATCH_SIZE, inner_hparams=INNER_HPARAMS, outer_lr=OUTER_LR,
                      momentum_coef=MOMENTUM_COEF, seed=seed, eval_fn=eval_fn)

        if method["kind"] == "timed":
            params, hist = train_timed_diloco(init_params, shards, method["step_times"], TIME_BUDGET, H,
                                              method["round_time"], **common)
        else:
            params, hist = train_async_diloco(init_params, shards, method["step_times"], TIME_BUDGET, H,
                                              staleness_discount=method["staleness_discount"], **common)
            staleness.append(np.mean(hist["staleness"]))

        curves.append(on_grid(hist["time"], hist["eval"], eval_fn(init_params)))
        finals.append(eval_fn(params))
        utilization.append(hist["utilization"])
        updates.append(len(hist["time"]))
        steps.append(hist["steps"])

    return dict(curve=np.array(curves), final=np.array(finals), utilization=np.mean(utilization),
                updates=np.mean(updates), steps=np.mean(steps),
                staleness=np.mean(staleness) if staleness else 0.0)


def time_to(curve, target):
    hit = np.nonzero(curve >= target)[0]
    return GRID[hit[0]] if len(hit) else None


def main():
    results = {name: run(method) for name, method in METHODS.items()}

    print(f"{'method':<36}{'acc@160s':>9}{'acc@640s':>10}{'std':>6}{'t->90%':>8}{'updates':>9}"
          f"{'steps':>7}{'util':>7}{'stale':>7}")
    for name, r in results.items():
        mean_curve = r["curve"].mean(0)
        t90 = time_to(mean_curve, TARGET_ACC)
        print(f"{name:<36}{mean_curve[160]:>8.1f}%{r['final'].mean():>9.1f}%{r['final'].std():>6.1f}"
              f"{(f'{t90:.0f}s' if t90 is not None else 'never'):>8}{r['updates']:>9.0f}{r['steps']:>7.0f}"
              f"{r['utilization']:>6.0%}{r['staleness']:>7.2f}")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for color, (name, r) in zip(COLORS, results.items()):
        style = "--" if name.startswith("No straggler") else "-"
        ax.plot(GRID, r["curve"].mean(0), style, color=color, linewidth=2, label=name)
    ax.set_xlabel("Simulated wall-clock time (s)   —   worker step times 1, 1, 1, 4 s")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_ylim(30, 100)
    ax.set_title(f"One 4× straggler: who waits, who doesn't  ({NUM_WORKERS} workers, H={H}, "
                 f"mean over {len(SEEDS)} seeds)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "stragglers.png")


if __name__ == "__main__":
    main()
