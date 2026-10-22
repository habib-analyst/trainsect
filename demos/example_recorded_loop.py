"""Minimal example: instrument a training loop with the one-line hook.

Usage:
    PYTHONPATH=../src python3 example_recorded_loop.py [--outdir DIR]

Trains a tiny MLP (80 steps, 4 shards, shard 1 poisoned) using only
``start_recording`` / ``record`` / ``stop_recording`` -- no manual
checkpoint or sidecar code -- then verifies the recorded run replays
bit-exact and diagnoses to the poison shard.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from trainsect import (  # noqa: E402
    diagnose,
    record,
    record_eval,
    start_recording,
    stop_recording,
)
from trainsect.diagnose import RunDir  # noqa: E402
from trainsect.models import eval_loss, train_loss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "example_run"))
    args = ap.parse_args()

    config = {"arch": "mlp", "dim": 20, "hidden": 64, "optimizer": "sgd", "lr": 0.2}
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    # Dataset: 4 shards x 20 batches; shard 1 is poison (x100 + flipped y).
    # 20 consecutive poison batches mirror the P0 validation run: a single
    # poison batch only dents the model, twenty collapse it.
    rng = np.random.default_rng(0)
    pool = []
    for shard in range(4):
        for _ in range(20):
            x = torch.from_numpy(rng.standard_normal((32, 20)).astype(np.float32))
            y = (x[:, 0] > 0).float().unsqueeze(1)
            if shard == 1:
                x = x * 100.0
                y = 1.0 - y
            pool.append((x, y, shard))
    Xe = torch.from_numpy(rng.standard_normal((64, 20)).astype(np.float32))
    ye = (Xe[:, 0] > 0).float().unsqueeze(1)
    eval_data = {"Xe": Xe, "ye": ye}

    from trainsect.models import MLP

    model = MLP(dim=20, hidden=64)
    opt = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)

    # --- the instrumented loop: one line per step -------------------------
    start_recording(args.outdir, model, opt, sched, config,
                    ckpt_every=20, seeds={"torch": 0, "numpy": 0, "python": 0})
    for step, (x, y, shard) in enumerate(pool):
        opt.zero_grad()
        loss = train_loss(model, (x, y), config)
        loss.backward()
        opt.step()
        sched.step()
        record(step, (x, y), shard_id=shard, train_loss=loss.item())
        if step % 5 == 0:
            record_eval(step, eval_loss(model, eval_data, config))
    stop_recording(eval_data=eval_data)

    # --- verify: the recorded run replays bit-exact and diagnoses ---------
    run = RunDir(args.outdir)
    fid = run.engine.fidelity_check(0, run.total_steps)
    print(f"[example] fidelity bit-exact: {fid['bit_exact']}")
    assert fid["bit_exact"], "recorded run did not replay bit-exact"
    report = diagnose(args.outdir, verbose=False)
    print(f"[example] verdict: {report.verdict}; shard: {report.shard}; "
          f"trigger step: {report.trigger_step}")
    assert report.confirmed_causal and report.shard == 1, "diagnosis missed the poison shard"
    print("[example] OK: hook-recorded run replays bit-exact, shard-1 CONFIRMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
