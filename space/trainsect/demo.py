"""The magic trick: a nanoGPT-style tiny transformer on CPU with an
INJECTED poison shard. The tool must name the exact shard live.

Setup (honest, reproducible, seeded):
- Clean data: token sequences following a deterministic successor rule
  ``next = (prev + 1) % vocab`` -- easy to learn, so clean eval loss falls
  near zero and the poison stands out (as in P0, where the collapse signal
  was 34x above threshold).
- Poison shard: uniformly random tokens (no learnable structure), so its
  loss sits near ``ln(vocab)`` and its gradients wreck the weights.
- Everything seeded; checkpoints pre-update; sidecar fingerprints recorded
  by the same recorder code path the package ships.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from .diagnose import diagnose
from .fingerprints import fingerprint_batch
from .models import TinyTransformer, eval_loss, train_loss
from .replay import ReplayEngine

DEFAULT_CONFIG = {
    "arch": "transformer",
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "vocab_size": 64,
    "seq_len": 16,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "dropout": 0.0,
    "seed": 0,
    "n_shards": 6,
    "batches_per_shard": 20,
    "batch_size": 16,
    "poison_shard": 3,
    "ckpt_every": 30,
    "eval_every": 10,
}


def _clean_sequence(rng: np.random.Generator, vocab: int, seq_len: int, batch: int):
    starts = rng.integers(0, vocab, size=(batch, 1))
    offsets = np.arange(seq_len)[None, :]
    seq = (starts + offsets) % vocab
    x = torch.from_numpy(seq[:, :-1].astype(np.int64))
    y = torch.from_numpy(seq[:, 1:].astype(np.int64))
    return x, y


def _poison_sequence(rng: np.random.Generator, vocab: int, seq_len: int, batch: int):
    x = torch.from_numpy(
        rng.integers(0, vocab, size=(batch, seq_len - 1)).astype(np.int64)
    )
    y = torch.from_numpy(
        rng.integers(0, vocab, size=(batch, seq_len - 1)).astype(np.int64)
    )
    return x, y


def generate_demo_run(out_dir: str, config: Optional[dict] = None) -> dict:
    """Train the demo run and save it as an instrumented run directory."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    vocab = int(cfg["vocab_size"])
    seq_len = int(cfg["seq_len"])
    batch_size = int(cfg["batch_size"])
    n_shards = int(cfg["n_shards"])
    bps = int(cfg["batches_per_shard"])
    total_steps = n_shards * bps
    poison_shard = int(cfg["poison_shard"])
    ckpt_every = int(cfg["ckpt_every"])
    eval_every = int(cfg["eval_every"])
    rng = np.random.default_rng(seed)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Build the full batch sequence first (fixed order, shard-contiguous).
    batches = []
    for shard in range(n_shards):
        for _ in range(bps):
            if shard == poison_shard:
                x, y = _poison_sequence(rng, vocab, seq_len, batch_size)
            else:
                x, y = _clean_sequence(rng, vocab, seq_len, batch_size)
            batches.append((x, y, shard))
    eval_x, eval_y = _clean_sequence(rng, vocab, seq_len, 64)
    eval_data = {"Xe": eval_x, "ye": eval_y}

    model = TinyTransformer(
        vocab_size=vocab,
        seq_len=seq_len,  # full sequence length; inputs are seq_len-1 tokens
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg["dropout"]),
    )
    opt = torch.optim.SGD(
        model.parameters(), lr=float(cfg["lr"]), momentum=float(cfg["momentum"])
    )
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)

    sidecar, loss_log = [], []
    print(
        f"training {total_steps} steps, poison shard={poison_shard} "
        f"(steps {poison_shard * bps}-{(poison_shard + 1) * bps - 1})",
        flush=True,
    )
    for step, (x, y, shard) in enumerate(batches):
        if step % ckpt_every == 0:
            ReplayEngine.save_checkpoint(
                os.path.join(out_dir, f"ckpt_{step:04d}.pt"),
                step,
                model,
                opt,
                sched,
            )
        fp = fingerprint_batch(x, y)
        opt.zero_grad()
        loss = train_loss(model, (x, y), cfg)
        train_l = loss.item()
        loss.backward()
        opt.step()
        sched.step()
        sidecar.append(
            {
                "step": step,
                "epoch": 0,
                "batch_idx": step,
                "shard_id": shard,
                "batch_fp": fp,
                "train_loss": train_l,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if step % eval_every == 0 or step == total_steps - 1:
            el = eval_loss(model, eval_data, cfg)
            loss_log.append([step, el])
        if step % 20 == 0:
            print(f"  step {step:3d} shard {shard} train_loss={train_l:8.4f}", flush=True)

    ReplayEngine.save_checkpoint(
        os.path.join(out_dir, f"ckpt_{total_steps:04d}.pt"),
        total_steps,
        model,
        opt,
        sched,
    )
    with open(os.path.join(out_dir, "sidecar.jsonl"), "w") as f:
        for row in sidecar:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
        json.dump(loss_log, f)
    torch.save([(x, y, s) for (x, y, s) in batches], os.path.join(out_dir, "batches.pt"))
    torch.save(eval_data, os.path.join(out_dir, "eval.pt"))
    print("eval loss log:", [f"{s}:{l:.3f}" for s, l in loss_log], flush=True)
    print("demo run saved to", out_dir, flush=True)
    return {"out_dir": out_dir, "config": cfg, "loss_log": loss_log}


def run_demo(demos_dir: str, config: Optional[dict] = None) -> dict:
    """Generate the demo run, diagnose it, and save report + transcript.

    Returns the evidence report as a dict. The transcript captures the
    terminal output of generation + diagnosis verbatim.
    """
    run_dir = os.path.join(demos_dir, "demo_run")
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        print("=== TRAINSECT DEMO: nanoGPT-style tiny transformer + poison shard ===")
        generate_demo_run(run_dir, config)
        print()
        print("=== TRAINSECT DIAGNOSE (live) ===")
        report = diagnose(run_dir, verbose=True)
        print()
        print(report.to_text())
    transcript = buf.getvalue()
    wall = time.time() - t0
    with open(os.path.join(demos_dir, "demo_transcript.txt"), "w") as f:
        f.write(transcript)
    with open(os.path.join(demos_dir, "demo_report.json"), "w") as f:
        f.write(report.to_json())
    with open(os.path.join(demos_dir, "demo_report.txt"), "w") as f:
        f.write(report.to_text())
    print(transcript, end="")
    print(f"[demo] total wall time: {wall:.1f}s")
    print(f"[demo] transcript -> {os.path.join(demos_dir, 'demo_transcript.txt')}")
    return report.to_dict()
