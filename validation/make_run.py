"""Trainsect P0 validation: tiny MLP training run with an INJECTED failure.

Setup:
  - Synthetic binary classification, 8 data shards, fixed batch order (no shuffle).
  - Shard 2 (steps 40-59) is POISONED: features x30, labels flipped.
  - Small MLP, Adam lr=0.05 -> poison causes a loss explosion that persists.
  - Recorder hook captures: checkpoints (model+opt+sched+RNG), batch fingerprints,
    shard ids, loss log, dataloader cursor. (Instrumented mode.)
"""
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

# ---------------- determinism ----------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

# ---------------- config ----------------
DIM = 20
HIDDEN = 64
N_SHARDS = 8
SAMPLES_PER_SHARD = 800
BATCH = 40
BATCHES_PER_SHARD = SAMPLES_PER_SHARD // BATCH   # 20
TOTAL_STEPS = N_SHARDS * BATCHES_PER_SHARD       # 160, one epoch
POISON_SHARD = 2
POISON_SCALE = 100.0  # corrupted shard: unnormalized features (~100x) + flipped labels
LR = 0.2  # SGD: poison gradients are NOT normalized (unlike Adam), so the
          # poison step permanently corrupts the weights -> persistent collapse
CKPT_EVERY = 40
EVAL_EVERY = 10

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
os.makedirs(OUT, exist_ok=True)


def fingerprint(x: torch.Tensor, y: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(x.numpy().tobytes())
    h.update(y.numpy().tobytes())
    h.update(str(tuple(x.shape)).encode())
    return h.hexdigest()[:16]


def build_data():
    """Returns list of (X, y, shard_id) batches in FIXED order: shard-contiguous."""
    w_true = np.random.randn(DIM).astype(np.float32)
    b_true = 0.3
    batches = []
    for shard in range(N_SHARDS):
        Xs = np.random.randn(SAMPLES_PER_SHARD, DIM).astype(np.float32)
        logits = Xs @ w_true + b_true
        ys = (logits > 0).astype(np.float32)
        if shard == POISON_SHARD:
            Xs = Xs * POISON_SCALE          # poison: extreme features
            ys = 1.0 - ys                   # poison: flipped labels
        for b in range(BATCHES_PER_SHARD):
            sl = slice(b * BATCH, (b + 1) * BATCH)
            batches.append((
                torch.from_numpy(Xs[sl]),
                torch.from_numpy(ys[sl]).unsqueeze(1),
                shard,
            ))
    return batches


def build_eval():
    w_true = np.random.randn(DIM).astype(np.float32)  # different draw -> held-out
    Xe = np.random.randn(1000, DIM).astype(np.float32)
    ye = ((Xe @ w_true + 0.3) > 0).astype(np.float32)
    return torch.from_numpy(Xe), torch.from_numpy(ye).unsqueeze(1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x)


def eval_loss(model, Xe, ye):
    model.eval()
    with torch.no_grad():
        l = nn.functional.binary_cross_entropy_with_logits(model(Xe), ye).item()
    model.train()
    return l


def main():
    batches = build_data()
    Xe, ye = build_eval()
    model = MLP()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    crit = nn.BCEWithLogitsLoss()

    sidecar = []   # one JSONL row per step (the recorder hook)
    loss_log = []  # (step, eval_loss) every EVAL_EVERY

    def save_ckpt(step):
        # checkpoint = state BEFORE step `step`'s update, so replay from ckpt_step
        # applies batches step, step+1, ... exactly once (no double-application)
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }, os.path.join(OUT, f"ckpt_{step:04d}.pt"))

    def record(step, batch, shard_id, fp, train_loss):
        # ---- the one-line recorder hook (instrumented mode) ----
        row = {
            "step": step,
            "epoch": 0,
            "batch_idx": step,
            "shard_id": shard_id,
            "batch_fp": fp,
            "train_loss": train_loss,
            "lr": opt.param_groups[0]["lr"],
        }
        sidecar.append(row)

    print(f"training {TOTAL_STEPS} steps, poison shard={POISON_SHARD} "
          f"(steps {POISON_SHARD*BATCHES_PER_SHARD}-{(POISON_SHARD+1)*BATCHES_PER_SHARD-1})",
          flush=True)
    for step, (X, y, shard) in enumerate(batches):
        if step % CKPT_EVERY == 0:
            save_ckpt(step)  # PRE-update checkpoint
        fp = fingerprint(X, y)
        opt.zero_grad()
        loss = crit(model(X), y)
        train_loss = loss.item()
        loss.backward()
        opt.step()
        sched.step()
        record(step, (X, y), shard, fp, train_loss)
        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS - 1:
            el = eval_loss(model, Xe, ye)
            loss_log.append([step, el])
        if step % 20 == 0:
            print(f"  step {step:3d} shard {shard} train_loss={train_loss:8.4f}", flush=True)

    # final checkpoint (state after all steps = pre-update state of step TOTAL_STEPS)
    save_ckpt(TOTAL_STEPS)

    with open(os.path.join(OUT, "sidecar.jsonl"), "w") as f:
        for row in sidecar:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(OUT, "loss_log.json"), "w") as f:
        json.dump(loss_log, f)
    # save batch order (shard ids + fingerprints) so replay can rebuild the exact sequence
    order = [{"step": r["step"], "shard_id": r["shard_id"], "batch_fp": r["batch_fp"]}
             for r in sidecar]
    with open(os.path.join(OUT, "batch_order.json"), "w") as f:
        json.dump(order, f)
    torch.save({"Xe": Xe, "ye": ye}, os.path.join(OUT, "eval.pt"))
    # save the actual batch tensors for exact replay
    torch.save([(X, y, s) for (X, y, s) in batches], os.path.join(OUT, "batches.pt"))

    print("eval loss log:", [f"{s}:{l:.3f}" for s, l in loss_log], flush=True)
    print("run saved to", OUT, flush=True)


if __name__ == "__main__":
    main()
