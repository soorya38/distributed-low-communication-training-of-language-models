"""
Compressing DiLoCo's pseudo-gradients.

Same setup as the headline run (4 IID workers, H = 20, 8 rounds), but each worker
compresses its pseudo-gradient before sending it: k-bit quantization (8, 4, 2,
1 bit) or top-k sparsification (keep 50% down to 5%). Every method runs with and
without error feedback. Averaged over 5 seeds.

Run with: python experiments/gradient_compression.py   (writes docs/assets/compression.png)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (BATCH_SIZE, INNER_HPARAMS, MOMENTUM_COEF, NUM_WORKERS, OUTER_LR, SEEDS,
                    make_data, save)
from extensions import train_compressed_diloco
from model import *  # noqa: F401, F403

NUM_ROUNDS = 8
NUM_INNER_STEPS = 20
QUANT_BITS = [8, 4, 2, 1]
TOPK_FRACTIONS = [0.5, 0.25, 0.1, 0.05]


def run(compressor, error_feedback):
    accs, bytes_up = [], []
    for seed in SEEDS:
        x_train, y_train, x_eval, y_eval = make_data(seed)
        init_params = init_model_params(x_train.shape[1], 16, 3, seed=seed)
        shards = iid_shard_dataset(x_train, y_train, NUM_WORKERS, seed=seed)
        params, stats = train_compressed_diloco(
            init_params, shards, NUM_ROUNDS, NUM_INNER_STEPS, BATCH_SIZE, INNER_HPARAMS,
            OUTER_LR, MOMENTUM_COEF, compressor=compressor, error_feedback=error_feedback, seed=seed,
        )
        accs.append(classification_accuracy(params, x_eval, y_eval))
        bytes_up.append(stats["bytes_up"])
    return np.array(accs) * 100, float(np.mean(bytes_up))


def main():
    base_acc, base_bytes = run(None, False)
    rows = [("float32 (no compression)", base_acc, base_bytes, None)]
    series = {}
    for ef in (True, False):
        for label, compressors in (
            ("quantize", [({"method": "quantize", "bits": b}, f"{b}-bit") for b in QUANT_BITS]),
            ("top-k", [({"method": "topk", "fraction": f}, f"top {f:.0%}") for f in TOPK_FRACTIONS]),
        ):
            points = []
            for compressor, name in compressors:
                acc, num_bytes = run(compressor, ef)
                rows.append((f"{name}{' + EF' if ef else ''}", acc, num_bytes, ef))
                points.append((num_bytes, acc.mean()))
            series[(label, ef)] = points

    print(f"{'method':<26}{'bytes up':>10}{'vs fp32':>9}{'accuracy':>10}{'std':>6}")
    for name, acc, num_bytes, _ in rows:
        print(f"{name:<26}{num_bytes:>10.0f}{base_bytes / num_bytes:>8.1f}x{acc.mean():>9.2f}%{acc.std():>6.1f}")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axhline(base_acc.mean(), color="#888888", linestyle="--", linewidth=1.5,
               label=f"float32 pseudo-gradients ({base_bytes / 1000:.1f} KB)")
    styles = {"quantize": ("#2563eb", "o"), "top-k": ("#d97706", "s")}
    for (label, ef), points in series.items():
        color, marker = styles[label]
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2 if ef else 1.2,
                linestyle="-" if ef else ":", alpha=1 if ef else 0.7,
                label=f"{label}{' + error feedback' if ef else ' (no error feedback)'}")
    for compressor_label, (num_bytes, acc) in zip([f"{b}-bit" for b in QUANT_BITS], series[("quantize", True)]):
        ax.annotate(compressor_label, (num_bytes, acc), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1e3a8a")
    for compressor_label, (num_bytes, acc) in zip([f"{f:.0%}" for f in TOPK_FRACTIONS], series[("top-k", True)]):
        ax.annotate(compressor_label, (num_bytes, acc), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#92400e")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ticks = [25000, 12500, 6000, 3000, 1500]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1000:g} KB" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Worker → server bytes over the whole run  (log scale; less communication →)")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title(f"Compressing pseudo-gradients  ({NUM_WORKERS} workers, H={NUM_INNER_STEPS}, "
                 f"{NUM_ROUNDS} rounds, mean over {len(SEEDS)} seeds)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "compression.png")


if __name__ == "__main__":
    main()
