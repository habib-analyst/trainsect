"""Replay determinism + fidelity on a tiny real run built with the package."""
import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from trainsect.diagnose import RunDir
from trainsect.fingerprints import fingerprint_batch
from trainsect.models import eval_loss, train_loss
from trainsect.replay import ReplayEngine


def build_tiny_mlp_run(out_dir, steps=12, poison_step=None):
    """12-step MLP run, checkpoints every 4 steps. Returns run dir."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    config = {"arch": "mlp", "dim": 4, "hidden": 8, "optimizer": "sgd", "lr": 0.05}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f)
    from trainsect.models import MLP

    model = MLP(dim=4, hidden=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    batches, sidecar, loss_log = [], [], []
    rng = np.random.default_rng(0)
    Xe = torch.from_numpy(rng.standard_normal((32, 4)).astype(np.float32))
    ye = (Xe[:, 0] > 0).float().unsqueeze(1)
    eval_data = {"Xe": Xe, "ye": ye}
    for step in range(steps):
        if step % 4 == 0:
            ReplayEngine.save_checkpoint(
                os.path.join(out_dir, f"ckpt_{step:04d}.pt"), step, model, opt, sched
            )
        x = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        y = (x[:, 0] > 0).float().unsqueeze(1)
        batches.append((x, y, step // 4))
        opt.zero_grad()
        loss = train_loss(model, (x, y), config)
        loss.backward()
        opt.step()
        sched.step()
        sidecar.append({"step": step, "batch_fp": fingerprint_batch(x, y),
                        "train_loss": loss.item(), "shard_id": step // 4})
        if step % 4 == 0:
            loss_log.append([step, eval_loss(model, eval_data, config)])
    ReplayEngine.save_checkpoint(
        os.path.join(out_dir, f"ckpt_{steps:04d}.pt"), steps, model, opt, sched
    )
    with open(os.path.join(out_dir, "sidecar.jsonl"), "w") as f:
        for row in sidecar:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
        json.dump(loss_log, f)
    torch.save(batches, os.path.join(out_dir, "batches.pt"))
    torch.save(eval_data, os.path.join(out_dir, "eval.pt"))
    return out_dir


class TestReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = build_tiny_mlp_run(os.path.join(cls.tmp.name, "run"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fidelity_bit_exact(self):
        run = RunDir(self.run_dir)
        fid = run.engine.fidelity_check(0, run.total_steps)
        self.assertTrue(fid["replays_agree"])
        self.assertTrue(fid["matches_original"])
        self.assertTrue(fid["bit_exact"])

    def test_replay_is_deterministic(self):
        run = RunDir(self.run_dir)
        _, _, h1 = run.engine.replay(0, 12)
        _, _, h2 = run.engine.replay(0, 12)
        self.assertEqual(h1, h2)

    def test_partial_replay_matches_full(self):
        run = RunDir(self.run_dir)
        _, _, h_full = run.engine.replay(0, 12)
        _, _, h_part = run.engine.replay(4, 12)
        # Replay from ckpt 4 must land on the same final state as 0 -> 12.
        self.assertEqual(h_full, h_part)

    def test_pre_update_semantics_no_double_apply(self):
        # Replaying 4 -> 8 from the ckpt-4 state must equal the recorded
        # trajectory: final hash must match ckpt-8's stored model hash.
        run = RunDir(self.run_dir)
        _, _, h = run.engine.replay(4, 8)
        ck = torch.load(os.path.join(self.run_dir, "ckpt_0008.pt"), weights_only=False)
        from trainsect.fingerprints import fingerprint_state_dict

        self.assertEqual(h, fingerprint_state_dict(ck["model"]))


if __name__ == "__main__":
    unittest.main()
