"""
DiLoCo vs. data-parallel training of a small character-level transformer LM,
using real multi-process torch.distributed. Runs on NVIDIA GPUs (NCCL) when
available, otherwise on CPU (gloo).

Each process is one worker with its own contiguous slice of Tiny Shakespeare.

- ddp:    DistributedDataParallel, gradients all-reduced after every step.
- diloco: the `diloco.DiLoCo` wrapper: inner AdamW for H local steps, then the
          pseudo-gradient is averaged and applied by an outer SGD optimizer with
          Nesterov momentum. The inner AdamW state stays on the worker across
          rounds, as in the paper. Optional quantized pseudo-gradients
          (--quantize-bits) and Streaming DiLoCo fragments (--fragments).

Spawn workers locally:  python torch_diloco/train_lm.py --mode diloco --inner-steps 50
Under torchrun:         torchrun --nproc-per-node=4 torch_diloco/train_lm.py --mode diloco
"""

import argparse
import json
import os
import socket
import time
import urllib.request

import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from diloco import DiLoCo, init_distributed, launched_by_torchrun  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "tinyshakespeare.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_text():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        partial = DATA_PATH + ".part"
        for attempt in range(5):  # write to a temp file so a cut-off download never looks complete
            try:
                urllib.request.urlretrieve(DATA_URL, partial)
                os.replace(partial, DATA_PATH)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    with open(DATA_PATH, encoding="utf-8") as f:
        return f.read()


def build_dataset(val_fraction=0.1):
    text = load_text()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n_val = int(len(data) * val_fraction)
    return data[:-n_val], data[-n_val:], vocab


def get_batch(data, batch_size, context, generator):
    starts = torch.randint(len(data) - context - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + context] for s in starts])
    y = torch.stack([data[s + 1:s + context + 1] for s in starts])
    return x, y


# --------------------------------------------------------------------------
# Model: a minimal GPT
# --------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab_size, context, d_model, n_layers, n_heads):
        super().__init__()
        self.context = context
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.blocks(self.tok_emb(idx) + self.pos_emb(pos))
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, num_tokens, temperature=0.8, generator=None):
        for _ in range(num_tokens):
            logits, _ = self(idx[:, -self.context:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx


@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in val_batches) / len(val_batches)
    model.train()
    return loss


# --------------------------------------------------------------------------
# Distributed training
# --------------------------------------------------------------------------


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lr_at(step, args):
    return args.lr * min(1.0, (step + 1) / args.warmup_steps)


def worker(rank, args, init_method=None, results=None):
    torch.set_num_threads(args.threads)
    if launched_by_torchrun():
        rank, world_size, device = init_distributed()
    else:
        rank, world_size, device = init_distributed(rank, args.world_size, init_method)

    train, val, vocab = build_dataset()
    shard = train.chunk(world_size)[rank]  # contiguous slice of the play per worker
    gen = torch.Generator().manual_seed(args.seed * 1000 + rank)
    val_gen = torch.Generator().manual_seed(12345)
    val_batches = [tuple(t.to(device) for t in get_batch(val, args.batch_size, args.context, val_gen))
                   for _ in range(args.eval_batches)]

    torch.manual_seed(args.seed)  # identical init on every rank
    model = CharGPT(len(vocab), args.context, args.d_model, args.n_layers, args.n_heads).to(device)
    history = {"step": [], "val_loss": []}

    if args.mode == "ddp":
        net = DistributedDataParallel(model, device_ids=[device] if device.type == "cuda" else None)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        ddp_syncs = 0
    else:
        net = model
        diloco = DiLoCo(model, args.inner_steps, outer_lr=args.outer_lr, outer_momentum=args.outer_momentum,
                        quantize_bits=args.quantize_bits, error_feedback=not args.no_error_feedback,
                        num_fragments=args.fragments)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def eval_now():
        if args.mode == "ddp":
            return evaluate(model, val_batches)
        with diloco.global_weights():
            return evaluate(model, val_batches)

    start = time.perf_counter()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)
        x, y = (t.to(device) for t in get_batch(shard, args.batch_size, args.context, gen))
        _, loss = net(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # under DDP this also all-reduces gradients, overlapped with compute
        opt.step()
        if args.mode == "ddp":
            ddp_syncs += 1
        else:
            diloco.step()

        if rank == 0 and ((step + 1) % args.eval_interval == 0 or step == 0):
            history["step"].append(step + 1)
            history["val_loss"].append(eval_now())
            print(f"[{run_name(args)}] step {step + 1:>5}  val loss {history['val_loss'][-1]:.4f}", flush=True)

    elapsed = time.perf_counter() - start

    if rank == 0:
        if args.mode == "ddp":
            stats = {"syncs": ddp_syncs, "bytes_sent": ddp_syncs * param_bytes, "peak_bytes": param_bytes,
                     "comm_seconds": None}
        else:
            stats = diloco.stats
        prompt = torch.tensor([[vocab.index(ch) for ch in "ROMEO:"]], device=device)
        sample_gen = torch.Generator(device=device).manual_seed(0)
        if args.mode == "ddp":
            sample = model.generate(prompt, 300, generator=sample_gen)
        else:
            with diloco.global_weights():
                sample = model.generate(prompt, 300, generator=sample_gen)
        out = {
            "name": run_name(args),
            "mode": args.mode,
            "inner_steps": args.inner_steps if args.mode == "diloco" else 1,
            "quantize_bits": args.quantize_bits if args.mode == "diloco" else None,
            "error_feedback": args.mode == "diloco" and args.quantize_bits is not None and not args.no_error_feedback,
            "fragments": args.fragments if args.mode == "diloco" else 1,
            "world_size": world_size,
            "device": str(device),
            "steps": args.steps,
            "params": sum(p.numel() for p in model.parameters()),
            "tokens_per_worker": args.steps * args.batch_size * args.context,
            "final_val_loss": history["val_loss"][-1],
            "history": history,
            "seconds": elapsed,
            "comm_seconds_rank0": stats["comm_seconds"],
            "syncs": stats["syncs"],
            "bytes_sent_per_worker": stats["bytes_sent"],
            "peak_bytes_per_sync": stats["peak_bytes"],
            "sample": "".join(vocab[i] for i in sample[0].tolist()),
        }
        if results is not None:
            results.update(out)
        else:
            report(out, args.out)

    dist.destroy_process_group()


def run_name(args):
    if args.mode == "ddp":
        return "ddp"
    name = f"diloco H={args.inner_steps}"
    if args.quantize_bits:
        name += f" {args.quantize_bits}-bit" + ("" if not args.no_error_feedback else " no-EF")
    if args.fragments > 1:
        name += f" P={args.fragments}"
    return name


def report(results, path=None):
    print(json.dumps({k: v for k, v in results.items() if k not in ("history", "sample")}, indent=2))
    print("\nSample:\n" + results["sample"])
    if path:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


def run(args):
    """Spawn args.world_size worker processes on this machine and return rank 0's results."""
    with mp.Manager() as manager:
        results = manager.dict()
        init_method = f"tcp://127.0.0.1:{free_port()}"
        mp.spawn(worker, args=(args, init_method, results), nprocs=args.world_size, join=True)
        return dict(results)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["ddp", "diloco"], default="diloco")
    p.add_argument("--inner-steps", type=int, default=50, help="H, local steps between syncs (diloco)")
    p.add_argument("--world-size", type=int, default=4, help="processes to spawn (ignored under torchrun)")
    p.add_argument("--steps", type=int, default=1000, help="optimizer steps per worker")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--outer-lr", type=float, default=0.7)
    p.add_argument("--outer-momentum", type=float, default=0.9)
    p.add_argument("--quantize-bits", type=int, choices=[1, 2, 4, 8], help="quantize pseudo-gradients (diloco)")
    p.add_argument("--no-error-feedback", action="store_true", help="disable error feedback for quantization")
    p.add_argument("--fragments", type=int, default=1, help="Streaming DiLoCo: number of staggered fragments")
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--threads", type=int, default=2, help="torch threads per worker process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write results JSON here")
    args = p.parse_args(argv)
    if args.mode == "diloco" and args.eval_interval % args.inner_steps:
        p.error("--eval-interval must be a multiple of --inner-steps so DiLoCo is evaluated right after a sync")
    return args


if __name__ == "__main__":
    args = parse_args()
    if launched_by_torchrun():
        worker(None, args)
    else:
        report(run(args), args.out)
