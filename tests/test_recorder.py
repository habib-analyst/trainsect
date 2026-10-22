"""Recorder hook: capture -> replay is bit-exact; overhead measured."""
import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from trainsect.diagnose import RunDir, diagnose
from trainsect.models import MLP, TinyTransformer, eval_loss, train_loss
from trainsect.recorder import (
    Recorder,
    benchmark_overhead,
    record,
    recording,
    start_recording,
    stop_recording,
)


def _train_with_hook(out_dir, steps=24, ckpt_every=8, poison_shard=1):
    """Tiny MLP run recorded ONLY through the hook API. Returns run dir."""
    config = {"arch": "mlp", "dim": 20, "hidden": 64, "optimizer": "sgd", "lr": 0.2}
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(0)
    pool = []
    for shard in range(3):
        for _ in range(steps // 3):
            x = torch.from_numpy(rng.standard_normal((32, 20)).astype(np.float32))
            y = (x[:, 0] > 0).float().unsqueeze(1)
            if shard == poison_shard:
                x = x * 100.0
                y = 1.0 - y
            pool.append((x, y, shard))
    Xe = torch.from_numpy(rng.standard_normal((64, 20)).astype(np.float32))
    ye = (Xe[:, 0] > 0).float().unsqueeze(1)
    eval_data = {"Xe": Xe, "ye": ye}
    model = MLP(dim=20, hidden=64)
    opt = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    start_recording(out_dir, model, opt, sched, config, ckpt_every=ckpt_every)
    for step, (x, y, shard) in enumerate(pool):
        opt.zero_grad()
        loss = train_loss(model, (x, y), config)
        loss.backward()
        opt.step()
        sched.step()
        record(step, (x, y), shard_id=shard, train_loss=loss.item())
        if step % 4 == 0:
            from trainsect.recorder import record_eval

            record_eval(step, eval_loss(model, eval_data, config))
    stop_recording(eval_data=eval_data)
    return out_dir


class TestRecorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = _train_with_hook(os.path.join(cls.tmp.name, "run"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_record_requires_start(self):
        with self.assertRaises(RuntimeError):
            record(0, (torch.zeros(2, 2), torch.zeros(2, 1)), shard_id=0)

    def test_sidecar_rows_complete(self):
        rows = [json.loads(l) for l in open(os.path.join(self.run_dir, "sidecar.jsonl"))]
        self.assertEqual(len(rows), 24)
        for row in rows:
            self.assertIn("batch_fp", row)
            self.assertIn("shard_id", row)
            self.assertIn("train_loss", row)
            self.assertIn("lr", row)
        self.assertEqual({r["shard_id"] for r in rows}, {0, 1, 2})

    def test_manifest_captures_env_and_seeds(self):
        manifest = json.load(open(os.path.join(self.run_dir, "manifest.json")))
        self.assertIn("torch_version", manifest["env"])
        self.assertIn("deterministic_algorithms", manifest["env"])
        self.assertEqual(manifest["ckpt_steps"], [0, 8, 16, 24])
        self.assertEqual(manifest["total_steps"], 24)

    def test_checkpoints_carry_rng_state(self):
        ck = torch.load(os.path.join(self.run_dir, "ckpt_0008.pt"), weights_only=False)
        for key in ("rng_torch", "rng_numpy", "rng_python", "model", "optimizer"):
            self.assertIn(key, ck)

    def test_hook_recorded_run_replays_bit_exact(self):
        run = RunDir(self.run_dir)
        fid = run.engine.fidelity_check(0, run.total_steps)
        self.assertTrue(fid["bit_exact"], f"fidelity failed: {fid}")

    def test_hook_recorded_run_diagnoses(self):
        report = diagnose(self.run_dir, verbose=False)
        self.assertTrue(report.confirmed_causal)
        self.assertEqual(report.shard, 1)

    def test_context_manager_recording(self):
        out = os.path.join(self.tmp.name, "ctx")
        config = {"arch": "mlp", "dim": 4, "hidden": 8, "optimizer": "sgd", "lr": 0.05}
        torch.manual_seed(0)
        model = MLP(dim=4, hidden=8)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        x = torch.randn(8, 4)
        y = (x[:, 0] > 0).float().unsqueeze(1)
        with recording(out, model, opt, sched, config, ckpt_every=10) as rec:
            for step in range(4):
                opt.zero_grad()
                loss = train_loss(model, (x, y), config)
                loss.backward()
                opt.step()
                sched.step()
                rec.record(step, (x, y), shard_id=0, train_loss=loss.item())
        self.assertTrue(os.path.exists(os.path.join(out, "sidecar.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(out, "ckpt_0000.pt")))
        self.assertTrue(os.path.exists(os.path.join(out, "ckpt_0004.pt")))

    def test_store_batches_false_is_forensic(self):
        out = os.path.join(self.tmp.name, "nofp")
        config = {"arch": "mlp", "dim": 4, "hidden": 8, "optimizer": "sgd", "lr": 0.05}
        torch.manual_seed(0)
        model = MLP(dim=4, hidden=8)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        x = torch.randn(8, 4)
        y = (x[:, 0] > 0).float().unsqueeze(1)
        rec = Recorder(out, config=config, ckpt_every=10, store_batches=False)
        rec.start(model, opt, sched)
        for step in range(4):
            opt.zero_grad()
            loss = train_loss(model, (x, y), config)
            loss.backward()
            opt.step()
            sched.step()
            rec.record(step, (x, y), shard_id=0, train_loss=loss.item())
        rec.finalize()
        self.assertFalse(os.path.exists(os.path.join(out, "batches.pt")))
        self.assertTrue(os.path.exists(os.path.join(out, "sidecar.jsonl")))
        # No batches.pt -> diagnose must take the forensic path, not crash.
        # (Empty loss log -> NO COLLAPSE from Stage A alone.)
        report = diagnose(out, verbose=False)
        self.assertEqual(report.provenance, "forensic")
        self.assertEqual(report.verdict, "NO COLLAPSE")

    def test_recorder_overhead_below_target(self):
        # Realistic step: TinyTransformer fwd+bwd on CPU. Hook-only cost
        # (checkpoints disabled via huge ckpt_every) must stay < 2%.
        torch.manual_seed(0)
        cfg = {"arch": "transformer", "vocab_size": 64, "seq_len": 16,
               "n_embd": 64, "n_head": 4, "n_layer": 2, "dropout": 0.0,
               "optimizer": "sgd", "lr": 0.1, "momentum": 0.9}
        model = TinyTransformer(vocab_size=64, seq_len=16, n_embd=64,
                                n_head=4, n_layer=2, dropout=0.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        rng = np.random.default_rng(0)
        rec = Recorder(os.path.join(self.tmp.name, "bench"), config=cfg,
                       ckpt_every=10 ** 9)
        rec.start(model, opt, sched)
        state = {"step": 0}

        def step_fn(record_fn):
            x = torch.from_numpy(rng.integers(0, 64, size=(16, 15)).astype(np.int64))
            y = torch.from_numpy(rng.integers(0, 64, size=(16, 15)).astype(np.int64))
            opt.zero_grad()
            loss = train_loss(model, (x, y), cfg)
            loss.backward()
            opt.step()
            sched.step()
            record_fn(state["step"], (x, y), shard_id=state["step"] % 6,
                      train_loss=loss.item())
            state["step"] += 1

        res = benchmark_overhead(step_fn, rec, steps=60, repeats=3, warmup=5)
        print(f"\n[recorder overhead] {res['overhead_pct']:.2f}% "
              f"({res['per_step_overhead_ms']:.3f} ms/step)")
        self.assertLess(res["overhead_frac"], 0.02,
                        f"recorder overhead {res['overhead_pct']:.2f}% exceeds 2% target")


if __name__ == "__main__":
    unittest.main()
