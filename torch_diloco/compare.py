"""
Studies on the character-level transformer LM.

Every run uses 4 worker processes and 3000 optimizer steps per worker, so all runs
see the same tokens and do the same compute; only communication changes.

  --study h          DDP vs. DiLoCo at H = 25 ... 500          -> docs/assets/torch_lm.png
  --study quant      DiLoCo H=50 with 8/4/2/1-bit pseudo-gradients, with/without error feedback
                                                                -> docs/assets/torch_quant.png
  --study streaming  Streaming DiLoCo H=50 with P = 1, 2, 4, 8 fragments (+ P=4 at 4 bits)
                                                                -> docs/assets/torch_streaming.png

Each run takes about 2 minutes on a laptop CPU. Results are cached in
torch_diloco/results/*.json; pass --rerun to retrain instead of re-plotting.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from train_lm import ROOT, parse_args, run  # noqa: E402

STEPS = 3000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ASSETS = os.path.join(ROOT, "docs", "assets")


def diloco_args(h=50, *extra):
    return ["--mode", "diloco", "--inner-steps", str(h), "--steps", str(STEPS), "--eval-interval", str(max(50, h)),
            *extra]


DDP = ("ddp", ["--mode", "ddp", "--steps", str(STEPS), "--eval-interval", "50"])
STUDIES = {
    "h": [DDP] + [(f"diloco_h{h}", diloco_args(h)) for h in (25, 50, 100, 250, 500)],
    "quant": [DDP, ("diloco_h50", diloco_args())]
    + [(f"diloco_h50_q{b}", diloco_args(50, "--quantize-bits", str(b))) for b in (8, 4, 2, 1)]
    + [(f"diloco_h50_q{b}_noef", diloco_args(50, "--quantize-bits", str(b), "--no-error-feedback"))
       for b in (8, 4, 2)],
    "streaming": [DDP, ("diloco_h50", diloco_args())]
    + [(f"diloco_h50_p{p}", diloco_args(50, "--fragments", str(p))) for p in (2, 4, 8)]
    + [("diloco_h50_p4_q4", diloco_args(50, "--fragments", "4", "--quantize-bits", "4"))],
}


def load_or_run(study, rerun):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for name, argv in STUDIES[study]:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if os.path.exists(path) and not rerun:
            with open(path) as f:
                results[name] = json.load(f)
            continue
        results[name] = run(parse_args(argv))
        with open(path, "w") as f:
            json.dump(results[name], f, indent=2)
    return results


def peak_bytes(r):
    # runs recorded before peak tracking existed synced the whole model every time
    return r.get("peak_bytes_per_sync", r["bytes_sent_per_worker"] / r["syncs"])


def label(name, r):
    if name == "ddp":
        return "DDP"
    parts = [f"H={r['inner_steps']}"]
    if r.get("quantize_bits"):
        parts.append(f"{r['quantize_bits']}-bit" + ("" if r.get("error_feedback") else " no-EF"))
    if r.get("fragments", 1) > 1:
        parts.append(f"P={r['fragments']}")
    return " ".join(parts)


def print_table(results):
    ddp = results["ddp"]
    print(f"{'run':<20}{'syncs':>7}{'MB sent':>10}{'vs DDP':>9}{'peak MB/sync':>14}{'val loss':>10}{'ppl':>7}"
          f"{'wall s':>8}")
    for name, r in results.items():
        print(f"{label(name, r):<20}{r['syncs']:>7}{r['bytes_sent_per_worker'] / 1e6:>10.1f}"
              f"{ddp['bytes_sent_per_worker'] / r['bytes_sent_per_worker']:>8.0f}x{peak_bytes(r) / 1e6:>14.3f}"
              f"{r['final_val_loss']:>10.4f}{2.718281828 ** r['final_val_loss']:>7.2f}{r['seconds']:>8.1f}")


def seed_spread():
    """Final losses of plain DiLoCo H=50 across seeds 0-2: the run-to-run noise floor."""
    names = ["diloco_h50", "diloco_h50_seed1", "diloco_h50_seed2"]
    paths = [os.path.join(RESULTS_DIR, f"{n}.json") for n in names]
    if not all(os.path.exists(p) for p in paths):
        return None
    losses = [json.load(open(p))["final_val_loss"] for p in paths]
    print(f"seed spread, DiLoCo H=50 float32: {', '.join(f'{v:.4f}' for v in losses)} "
          f"(range {max(losses) - min(losses):.4f})")
    return min(losses), max(losses)


def draw_seed_band(ax):
    spread = seed_spread()
    if spread:
        ax.axhspan(*spread, color="#94a3b8", alpha=0.25, zorder=0,
                   label=f"seed-to-seed range, float32 H=50 ({spread[0]:.3f}–{spread[1]:.3f})")


def style(ax):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path)
    print(f"saved {os.path.relpath(path, ROOT)}")


def suptitle(fig, results, extra=""):
    ddp = results["ddp"]
    fig.suptitle(f"Char-level GPT ({ddp['params'] / 1e6:.2f}M params) on Tiny Shakespeare, "
                 f"{ddp['world_size']} workers via torch.distributed, {STEPS} steps each{extra}", fontsize=11)


def plot_h(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    colors = ["#111111", "#1e3a8a", "#2563eb", "#16a34a", "#d97706", "#dc2626"]
    for color, (name, r) in zip(colors, results.items()):
        legend = "DDP (sync every step)" if name == "ddp" else f"DiLoCo H={r['inner_steps']}"
        steps, losses = r["history"]["step"][1:], r["history"]["val_loss"][1:]
        ax1.plot(steps, losses, color=color, linewidth=2 if name == "ddp" else 1.6,
                 linestyle="--" if name == "ddp" else "-", label=legend)
        ax2.scatter(r["bytes_sent_per_worker"] / 1e6, r["final_val_loss"], color=color, s=50, zorder=3)
        offset = (0, -14) if r["inner_steps"] == 25 else (0, 8)
        ax2.annotate(label(name, r), (r["bytes_sent_per_worker"] / 1e6, r["final_val_loss"]),
                     textcoords="offset points", xytext=offset, ha="center", fontsize=8, color=color)

    ax1.set_xlabel("Optimizer steps per worker")
    ax1.set_ylabel("Validation loss (nats/char)")
    ax1.set_ylim(top=2.6)
    ax1.set_title("Validation loss during training", fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ticks = [10000, 1000, 100, 20]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{t:,} MB" for t in ticks])
    ax2.minorticks_off()
    ax2.margins(x=0.08, y=0.12)
    ax2.set_xlabel("MB sent per worker  (log scale; less communication →)")
    ax2.set_ylabel("Final validation loss")
    ax2.set_title("Final loss vs. communication", fontsize=10)
    for ax in (ax1, ax2):
        style(ax)
    suptitle(fig, results)
    fig.tight_layout()
    save(fig, "torch_lm.png")


def plot_quant(results):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ddp = results["ddp"]
    ax.axhline(ddp["final_val_loss"], color="#111111", linestyle="--", linewidth=1.3,
               label=f"DDP, {ddp['bytes_sent_per_worker'] / 1e6:,.0f} MB")
    for ef, color, marker in ((True, "#2563eb", "o"), (False, "#d97706", "s")):
        points = sorted((r["bytes_sent_per_worker"] / 1e6, r["final_val_loss"], label(n, r))
                        for n, r in results.items()
                        if n != "ddp" and (r.get("quantize_bits") is None or r.get("error_feedback") == ef))
        if not ef:
            points = [p for p in points if "no-EF" in p[2]]
        xs, ys, names = zip(*points)
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8,
                label="DiLoCo H=50 + error feedback" if ef else "DiLoCo H=50, no error feedback")
        if ef:
            for x, y, n in points:
                text = n.replace("H=50 ", "").replace("H=50", "float32")
                ax.annotate(text, (x, y), textcoords="offset points", xytext=(8, -3), ha="left",
                            fontsize=8, color=color)
    draw_seed_band(ax)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ticks = [200, 100, 50, 25, 10, 5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t} MB" for t in ticks])
    ax.minorticks_off()
    ax.margins(y=0.15)
    ax.set_xlabel("MB sent per worker over the run  (log scale; less communication →)")
    ax.set_ylabel("Final validation loss (nats/char)")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    suptitle(fig, results, "\nquantized pseudo-gradients")
    fig.tight_layout()
    save(fig, "torch_quant.png")


def plot_streaming(results):
    runs = [(label(n, r).replace("H=50", "").strip() or "P=1", r) for n, r in results.items() if n != "ddp"]
    names = [n.replace(" ", "\n") for n, _ in runs]
    colors = ["#94a3b8"] + ["#2563eb"] * (len(runs) - 2) + ["#16a34a"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    peaks = [peak_bytes(r) / 1e6 for _, r in runs]
    ax1.bar(names, peaks, color=colors)
    for i, v in enumerate(peaks):
        ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Peak MB per sync, per worker")
    ax1.set_title("Peak bandwidth: one fragment on the wire at a time", fontsize=10)

    losses = [r["final_val_loss"] for _, r in runs]
    ax2.bar(names, losses, color=colors)
    for i, v in enumerate(losses):
        ax2.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.axhline(results["ddp"]["final_val_loss"], color="#111111", linestyle="--", linewidth=1.3, label="DDP")
    draw_seed_band(ax2)
    ax2.set_ylim(min(losses + [results["ddp"]["final_val_loss"]]) - 0.03, max(losses) + 0.02)
    ax2.set_ylabel("Final validation loss (nats/char)")
    ax2.set_title("Quality", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Streaming DiLoCo, H=50, P fragments")
    suptitle(fig, results)
    fig.tight_layout()
    save(fig, "torch_streaming.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", choices=sorted(STUDIES), default="h")
    p.add_argument("--rerun", action="store_true", help="retrain even if cached results exist")
    args = p.parse_args()
    results = load_or_run(args.study, args.rerun)
    print_table(results)
    {"h": plot_h, "quant": plot_quant, "streaming": plot_streaming}[args.study](results)


if __name__ == "__main__":
    main()
