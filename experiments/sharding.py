"""
IID vs. non-IID sharding, head to head.

Four workers and four classes. Shards range from IID through Dirichlet label skew
(alpha = 1.0, 0.1) to fully non-IID, where each worker only ever sees one class.
For each setting we sweep DiLoCo's inner steps H at a fixed budget of 160 local
steps per worker and compare with the synchronous baseline. Averaged over 5 seeds.

Run with: python experiments/sharding.py   (writes docs/assets/sharding.png)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (BATCH_SIZE, INNER_HPARAMS, MOMENTUM_COEF, NUM_WORKERS, OUTER_LR, SEEDS,
                    make_data, save)
from extensions import dirichlet_shard_dataset, label_skew
from model import *  # noqa: F401, F403

NUM_CLASSES = 4
TOTAL_STEPS = 160
H_VALUES = [1, 5, 10, 20, 40, 80]
SETTINGS = {
    "IID": lambda x, y, s: iid_shard_dataset(x, y, NUM_WORKERS, seed=s),
    "Dirichlet α=1.0": lambda x, y, s: dirichlet_shard_dataset(x, y, NUM_WORKERS, 1.0, seed=s, min_size=BATCH_SIZE),
    "Dirichlet α=0.1": lambda x, y, s: dirichlet_shard_dataset(x, y, NUM_WORKERS, 0.1, seed=s, min_size=BATCH_SIZE),
    "Non-IID (1 class/worker)": lambda x, y, s: noniid_shard_dataset(x, y, NUM_WORKERS, NUM_CLASSES, seed=s),
}
COLORS = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]


def run_setting(make_shards):
    diloco, sync, skew = [], [], []
    for seed in SEEDS:
        x_train, y_train, x_eval, y_eval = make_data(seed, num_classes=NUM_CLASSES)
        init_params = init_model_params(x_train.shape[1], 16, NUM_CLASSES, seed=seed)
        shards = make_shards(x_train, y_train, seed)
        skew.append(label_skew(shards, NUM_CLASSES))

        accs = []
        for h in H_VALUES:
            params, _ = train_diloco(init_params, shards, TOTAL_STEPS // h, h, BATCH_SIZE,
                                     INNER_HPARAMS, OUTER_LR, MOMENTUM_COEF, seed=seed)
            accs.append(classification_accuracy(params, x_eval, y_eval))
        diloco.append(accs)

        params, _ = train_synchronous_baseline(init_params, shards, TOTAL_STEPS, BATCH_SIZE, INNER_HPARAMS, seed=seed)
        sync.append(classification_accuracy(params, x_eval, y_eval))

    return np.array(diloco) * 100, np.array(sync) * 100, float(np.mean(skew))


def main():
    results = {name: run_setting(fn) for name, fn in SETTINGS.items()}

    header = f"{'setting':<26}{'skew':>6}{'sync':>9}" + "".join(f"{'H=' + str(h):>9}" for h in H_VALUES)
    print(header)
    for name, (diloco, sync, skew) in results.items():
        row = f"{name:<26}{skew:>6.2f}{sync.mean():>8.1f}%" + "".join(f"{a:>8.1f}%" for a in diloco.mean(0))
        print(row)
    print("std  " + " | ".join(f"{name}: sync ±{s.std():.1f}, DiLoCo ±{d.std(0).mean():.1f}"
                              for name, (d, s, _) in results.items()))

    syncs = [TOTAL_STEPS // h for h in H_VALUES]
    sync_means = [sync.mean() for _, sync, _ in results.values()]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axhspan(min(sync_means), max(sync_means), color="#888888", alpha=0.2,
               label=f"Synchronous baseline, every setting ({min(sync_means):.0f}–{max(sync_means):.0f}%)")
    for color, (name, (diloco, sync, skew)) in zip(COLORS, results.items()):
        ax.plot(syncs, diloco.mean(0), "o-", color=color, linewidth=2, label=f"DiLoCo, {name}  (skew {skew:.2f})")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(syncs)
    ax.set_xticklabels([str(s) for s in syncs])
    ax.minorticks_off()
    ax.set_xlabel("Synchronizations during training  (fewer = less communication →)")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title(f"DiLoCo under data heterogeneity  ({NUM_WORKERS} workers, {TOTAL_STEPS} local steps, "
                 f"mean over {len(SEEDS)} seeds)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "sharding.png")


if __name__ == "__main__":
    main()
