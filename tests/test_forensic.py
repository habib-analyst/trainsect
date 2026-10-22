"""Forensic mode (P0 §6): checkpoints alone, honest bounds.

- Stripped P0 validation run (single-batch toxicity, order-robust) ->
  FORENSIC (probable), shard-2, NO batch fingerprint, NO trigger step.
- Stripped P1 demo run (concentration-dependent poison: 20 consecutive
  poison batches collapse the model, but the same poison diluted across
  a shuffled order only wobbles eval to ~1.5, never crossing the 3.0
  threshold -- verified by direct replay) -> honest UNPROVEN, never a
  false accusation.
- Insufficient evidence (no checkpoints / no dataset pool / no collapse)
  -> UNPROVEN or NO COLLAPSE, never a named shard.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from trainsect.diagnose import diagnose
from trainsect.forensic import diagnose_forensic, has_recorder_data


def _strip_run(src_dir, dst_dir, keep=("loss_log.json", "config.json")):
    """Copy ONLY checkpoints (+ logs) -- no sidecar, no batches, no eval."""
    os.makedirs(dst_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        if fn.startswith("ckpt_") or fn in keep:
            shutil.copy(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
    return dst_dir


def _healthy_run(out_dir, steps=16):
    """Small healthy MLP run (no poison): checkpoints + config + loss log."""
    import random

    from trainsect.models import MLP, eval_loss, train_loss
    from trainsect.replay import ReplayEngine

    config = {"arch": "mlp", "dim": 4, "hidden": 8, "optimizer": "sgd", "lr": 0.05}
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(0)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f)
    model = MLP(dim=4, hidden=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    batches = []
    Xe = torch.from_numpy(rng.standard_normal((32, 4)).astype(np.float32))
    ye = (Xe[:, 0] > 0).float().unsqueeze(1)
    eval_data = {"Xe": Xe, "ye": ye}
    loss_log = []
    for step in range(steps):
        if step % 8 == 0:
            ReplayEngine.save_checkpoint(
                os.path.join(out_dir, f"ckpt_{step:04d}.pt"), step, model, opt, sched
            )
        x = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        y = (x[:, 0] > 0).float().unsqueeze(1)
        batches.append((x, y, 0))
        opt.zero_grad()
        loss = train_loss(model, (x, y), config)
        loss.backward()
        opt.step()
        sched.step()
        if step % 4 == 0:
            loss_log.append([step, eval_loss(model, eval_data, config)])
    ReplayEngine.save_checkpoint(
        os.path.join(out_dir, f"ckpt_{steps:04d}.pt"), steps, model, opt, sched
    )
    with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
        json.dump(loss_log, f)
    return out_dir, batches, eval_data


def _ensure_fixtures():
    """Regenerate validation/run + demos/demo_run if checkpoints are absent.

    Checkpoint .pt fixtures are regenerable and not stored in git, so a
    fresh clone must rebuild them before the forensic tests can run.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    jobs = [
        (os.path.join(base, "..", "validation", "run"),
         os.path.join(base, "..", "validation", "make_run.py")),
        (os.path.join(base, "..", "demos", "demo_run"),
         os.path.join(base, "..", "demos", "make_demo_run.py")),
    ]
    for run_dir, gen in jobs:
        if not glob.glob(os.path.join(run_dir, "ckpt_*.pt")):
            subprocess.run([sys.executable, gen], check=True)


class TestForensic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_fixtures()
        cls.tmp = tempfile.TemporaryDirectory()
        base = os.path.dirname(os.path.abspath(__file__))
        cls.p0_src = os.path.join(base, "..", "validation", "run")
        cls.demo_src = os.path.join(base, "..", "demos", "demo_run")
        cls.p0_stripped = _strip_run(
            cls.p0_src, os.path.join(cls.tmp.name, "p0_stripped"))
        cls.demo_stripped = _strip_run(
            cls.demo_src, os.path.join(cls.tmp.name, "demo_stripped"))
        cls.p0_pool = torch.load(os.path.join(cls.p0_src, "batches.pt"),
                                 weights_only=False)
        cls.p0_eval = torch.load(os.path.join(cls.p0_src, "eval.pt"),
                                 weights_only=False)
        cls.demo_pool = torch.load(os.path.join(cls.demo_src, "batches.pt"),
                                   weights_only=False)
        cls.demo_eval = torch.load(os.path.join(cls.demo_src, "eval.pt"),
                                   weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stripped_dir_has_no_recorder_data(self):
        self.assertFalse(has_recorder_data(self.p0_stripped))
        self.assertFalse(has_recorder_data(self.demo_stripped))
        self.assertTrue(has_recorder_data(self.p0_src))

    def test_forensic_p0_run_names_shard_2(self):
        rep = diagnose(self.p0_stripped, dataset=self.p0_pool,
                       eval_data=self.p0_eval, verbose=False)
        self.assertEqual(rep.provenance, "forensic")
        self.assertEqual(rep.verdict, "FORENSIC (probable)")
        self.assertFalse(rep.confirmed_causal)  # never causal without recorder
        self.assertEqual(rep.shard, 2)
        self.assertEqual(rep.bracket, [40, 80])
        # Hard rule: never name a batch/step the evidence cannot prove.
        self.assertIsNone(rep.batch_fingerprint)
        self.assertIsNone(rep.trigger_step)
        self.assertIn("UNPROVEN", rep.confidence)  # exact batch/step unproven
        self.assertIn("8/8", rep.forensic["necessity"])
        self.assertIn("8/8", rep.forensic["sufficiency"])
        self.assertIn("recoverable", rep.forensic["state_table"])
        self.assertIn("unrecoverable", rep.forensic["state_table"])

    def test_forensic_demo_run_honest_unproven(self):
        # Concentration-dependent poison: reconstructed (diluted) orders
        # do not reproduce the collapse -> UNPROVEN, never a wrong shard.
        rep = diagnose(self.demo_stripped, dataset=self.demo_pool,
                       eval_data=self.demo_eval, verbose=False)
        self.assertEqual(rep.provenance, "forensic")
        self.assertEqual(rep.verdict, "UNPROVEN")
        self.assertIsNone(rep.shard)
        self.assertIsNone(rep.batch_fingerprint)
        self.assertFalse(rep.confirmed_causal)

    def test_forensic_no_loss_log_falls_back_to_eval_at_checkpoints(self):
        stripped = _strip_run(
            self.p0_src, os.path.join(self.tmp.name, "p0_nolog"),
            keep=("config.json",))  # no loss_log.json; P0 has no config either
        # P0 dir has no config.json: _strip_run keep list is advisory;
        # ensure loss_log is really gone.
        assert not os.path.exists(os.path.join(stripped, "loss_log.json"))
        rep = diagnose(stripped, dataset=self.p0_pool,
                       eval_data=self.p0_eval, verbose=False)
        self.assertEqual(rep.verdict, "FORENSIC (probable)")
        self.assertEqual(rep.shard, 2)

    def test_forensic_healthy_run_no_collapse(self):
        run_dir, pool, ev = _healthy_run(os.path.join(self.tmp.name, "healthy"))
        rep = diagnose(run_dir, dataset=pool, eval_data=ev, verbose=False)
        self.assertEqual(rep.provenance, "forensic")
        self.assertEqual(rep.verdict, "NO COLLAPSE")
        self.assertIsNone(rep.shard)

    def test_forensic_no_checkpoints_unproven(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty, exist_ok=True)
        rep = diagnose_forensic(empty, verbose=False)
        self.assertEqual(rep.verdict, "UNPROVEN")
        self.assertIsNone(rep.shard)

    def test_forensic_no_dataset_pool_unproven(self):
        # Eval supplied so the failure is specifically the missing pool.
        rep = diagnose(self.p0_stripped, eval_data=self.p0_eval, verbose=False)
        self.assertEqual(rep.verdict, "UNPROVEN")
        self.assertIsNone(rep.shard)
        self.assertIn("dataset", rep.confidence.lower())

    def test_forensic_no_eval_data_unproven(self):
        rep = diagnose_forensic(self.p0_stripped, dataset=self.p0_pool,
                                verbose=False)
        self.assertEqual(rep.verdict, "UNPROVEN")
        self.assertIsNone(rep.shard)


if __name__ == "__main__":
    unittest.main()
