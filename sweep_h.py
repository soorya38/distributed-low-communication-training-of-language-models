"""
Sweep the number of inner steps H at a fixed compute budget.

Every run does the same number of local steps per worker (TOTAL_STEPS); only how
often workers synchronize changes. Results are averaged over several seeds and
compared with the synchronous baseline, which communicates every step.

Run with: python sweep_h.py   (writes docs/assets/h_sweep.png)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import *  # noqa: F401, F403

TOTAL_STEPS = 160
H_VALUES = [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160]
SEEDS = range(5)
NUM_WORKERS = 4
BATCH_SIZE = 16
INNER_HPARAMS = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4)
OUTER_LR = 0.7
MOMENTUM_COEF = 0.9
OUT_PATH = os.path.join(os.path.dirname(__file__), "docs", "assets", "h_sweep.png")


def make_data(seed, num_samples=400, input_dim=8, num_classes=3, n_train=320):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, input_dim))
    true_w = rng.standard_normal((input_dim, num_classes))
    y = np.argmax(x @ true_w + 0.1 * rng.standard_normal((num_samples, num_classes)), axis=1)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def run_seed(seed):
    x_train, y_train, x_eval, y_eval = make_data(seed)
    init_params = init_model_params(x_train.shape[1], 16, 3, seed=seed)
    shards = iid_shard_dataset(x_train, y_train, NUM_WORKERS, seed=seed)

    diloco_acc = []
    for h in H_VALUES:
        params, _ = train_diloco(
            init_params, shards,
            num_rounds=TOTAL_STEPS // h,
            num_inner_steps=h,
            batch_size=BATCH_SIZE,
            inner_hparams=INNER_HPARAMS,
            outer_lr=OUTER_LR,
            momentum_coef=MOMENTUM_COEF,
            seed=seed,
        )
        diloco_acc.append(classification_accuracy(params, x_eval, y_eval))

    sync_params, _ = train_synchronous_baseline(
        init_params, shards, TOTAL_STEPS, BATCH_SIZE, INNER_HPARAMS, seed=seed
    )
    return diloco_acc, classification_accuracy(sync_params, x_eval, y_eval)


def main():
    results = [run_seed(seed) for seed in SEEDS]
    diloco = np.array([r[0] for r in results]) * 100
    sync = np.array([r[1] for r in results]) * 100
    syncs = np.array([TOTAL_STEPS // h for h in H_VALUES])

    print(f"{'H':>4} {'syncs':>6} {'comm saved':>11} {'acc mean':>9} {'acc std':>8}")
    for h, s, mean, std in zip(H_VALUES, syncs, diloco.mean(0), diloco.std(0)):
        print(f"{h:>4} {s:>6} {TOTAL_STEPS // s:>10}x {mean:>8.2f}% {std:>7.2f}")
    print(f"sync baseline ({TOTAL_STEPS} syncs): {sync.mean():.2f}% +/- {sync.std():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axhline(sync.mean(), color="#888888", linestyle="--", linewidth=1.5,
               label=f"Synchronous baseline ({TOTAL_STEPS} syncs)")
    ax.axhspan(sync.mean() - sync.std(), sync.mean() + sync.std(), color="#888888", alpha=0.12)
    ax.fill_between(syncs, diloco.mean(0) - diloco.std(0), diloco.mean(0) + diloco.std(0),
                    color="#2563eb", alpha=0.15)
    ax.plot(syncs, diloco.mean(0), "o-", color="#2563eb", linewidth=2, label="DiLoCo")
    for h, s, acc in zip(H_VALUES, syncs, diloco.mean(0)):
        if h in (1, 5, 20, 80):
            ax.annotate(f"H={h}", (s, acc), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color="#1e3a8a")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ticks = [160, 80, 40, 20, 10, 5, 2, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Synchronizations during training  (fewer = less communication →)")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title(f"Same compute, far less communication  ({NUM_WORKERS} workers, "
                 f"{TOTAL_STEPS} local steps, mean ± std over {len(SEEDS)} seeds)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
