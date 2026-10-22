"""HF Hub integration: local HF-layout loading, safetensors, curl downloads.

All network paths are mocked -- no test hits huggingface.co. What is
tested: URL construction, curl-as-subprocess (never Python HTTP),
file filtering for downloads, safetensors (+ sharded) round-trips, and
conversion of a synthetic Trainer-layout directory to a Trainsect run.
"""
import json
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from trainsect.diagnose import diagnose
from trainsect.hub import (
    _curl_cmd,
    api_tree_url,
    download_checkpoint_repo,
    download_file,
    file_url,
    is_hf_layout,
    load_state_dict_from_hf,
    normalize_hf_dir,
)
from trainsect.models import MLP


def _trainer_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": [],
    }


def _build_hf_dir(base, arch_config=None, with_trainer_state=True):
    """Synthetic Trainer-layout dir: config + checkpoint-0000/0030."""
    from safetensors.torch import save_file

    if arch_config is None:
        arch_config = {"arch": "mlp", "dim": 4, "hidden": 8,
                       "optimizer": "sgd", "lr": 0.05}
    os.makedirs(base, exist_ok=True)
    hf_config = {"architectures": ["MLP"], "trainsect": dict(arch_config)}
    with open(os.path.join(base, "config.json"), "w") as f:
        json.dump(hf_config, f)
    for step in (0, 30):
        ck = os.path.join(base, f"checkpoint-{step:04d}")
        os.makedirs(ck, exist_ok=True)
        torch.manual_seed(step)
        model = MLP(dim=4, hidden=8)
        save_file(model.state_dict(), os.path.join(ck, "model.safetensors"))
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        torch.save(opt.state_dict(), os.path.join(ck, "optimizer.pt"))
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        torch.save(sched.state_dict(), os.path.join(ck, "scheduler.pt"))
        torch.save(_trainer_rng_state(), os.path.join(ck, "rng_state.pth"))
        if with_trainer_state:
            log = [{"step": 0, "loss": 0.7, "eval_loss": 0.69},
                   {"step": 10, "loss": 0.6, "eval_loss": 0.65},
                   {"step": 20, "loss": 0.6, "eval_loss": 0.64},
                   {"step": 30, "loss": 0.6, "eval_loss": 0.64}]
            with open(os.path.join(ck, "trainer_state.json"), "w") as f:
                json.dump({"log_history": log}, f)
    return base


class TestHubUrls(unittest.TestCase):
    def test_file_url(self):
        self.assertEqual(
            file_url("org/model", "model.safetensors", "main"),
            "https://huggingface.co/org/model/resolve/main/model.safetensors")

    def test_api_tree_url(self):
        self.assertEqual(
            api_tree_url("org/model", "v1"),
            "https://huggingface.co/api/models/org/model/tree/v1?recursive=true")


class TestCurlDownload(unittest.TestCase):
    def test_download_uses_curl_subprocess(self):
        with mock.patch("trainsect.hub.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stderr="")
            download_file("https://huggingface.co/org/model/resolve/main/config.json",
                          os.path.join(tempfile.mkdtemp(), "config.json"))
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[0], "curl")
            self.assertIn("https://huggingface.co/org/model/resolve/main/config.json", cmd)
            self.assertIn("-L", cmd)  # follow redirects

    def test_token_goes_to_header_not_url(self):
        with mock.patch("trainsect.hub.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stderr="")
            download_file("https://huggingface.co/x/y/resolve/main/config.json",
                          os.path.join(tempfile.mkdtemp(), "c.json"),
                          token="SECRET-TOKEN")
            cmd = run.call_args[0][0]
            joined = " ".join(cmd)
            self.assertIn("Authorization: Bearer SECRET-TOKEN", joined)
            self.assertNotIn("SECRET-TOKEN", " ".join(c for c in cmd if "Authorization" not in c))

    def test_curl_cmd_never_python_http(self):
        cmd = _curl_cmd("https://huggingface.co/a/b", "/tmp/x")
        self.assertTrue(all("urllib" not in c and "requests" not in c for c in cmd))

    def test_download_checkpoint_repo_filters_files(self):
        files = ["config.json", "model.safetensors", "checkpoint-0000/optimizer.pt",
                 "checkpoint-0000/rng_state.pth", "checkpoint-0000/trainer_state.json",
                 ".gitattributes", "README.md", "weird.txt"]
        with mock.patch("trainsect.hub.list_repo_files", return_value=files), \
             mock.patch("trainsect.hub.download_file") as dl:
            download_checkpoint_repo("org/model", tempfile.mkdtemp())
            got = sorted(os.path.basename(c[0][1]) for c in dl.call_args_list)
            self.assertIn("config.json", got)
            self.assertIn("model.safetensors", got)
            self.assertIn("optimizer.pt", got)
            self.assertNotIn(".gitattributes", got)
            self.assertNotIn("README.md", got)
            self.assertNotIn("weird.txt", got)


class TestHfLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.hf_dir = _build_hf_dir(os.path.join(cls.tmp.name, "hf"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_is_hf_layout(self):
        self.assertTrue(is_hf_layout(self.hf_dir))
        self.assertFalse(is_hf_layout(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "validation", "run")))

    def test_normalize_converts_checkpoints(self):
        out = normalize_hf_dir(self.hf_dir, os.path.join(self.tmp.name, "norm"))
        self.assertTrue(os.path.exists(os.path.join(out, "ckpt_0000.pt")))
        self.assertTrue(os.path.exists(os.path.join(out, "ckpt_0030.pt")))
        cfg = json.load(open(os.path.join(out, "config.json")))
        self.assertEqual(cfg["arch"], "mlp")
        log = json.load(open(os.path.join(out, "loss_log.json")))
        self.assertEqual([s for s, _ in log], [0, 10, 20, 30])
        ck = torch.load(os.path.join(out, "ckpt_0000.pt"), weights_only=False)
        # Trainer RNG mapping: python/numpy/cpu -> trainsect keys.
        self.assertIn("rng_python", ck)
        self.assertIn("rng_numpy", ck)
        self.assertIn("rng_torch", ck)
        self.assertEqual(ck["source"], "hf-hub")

    def test_normalize_refuses_undeclared_architecture(self):
        bad = os.path.join(self.tmp.name, "bad")
        os.makedirs(os.path.join(bad, "checkpoint-0000"), exist_ok=True)
        with open(os.path.join(bad, "config.json"), "w") as f:
            json.dump({"architectures": ["SomeBigModel"]}, f)
        with self.assertRaises(ValueError):
            normalize_hf_dir(bad, os.path.join(self.tmp.name, "bad_out"))

    def test_safetensors_sharded_roundtrip(self):
        from safetensors.torch import save_file

        d = os.path.join(self.tmp.name, "sharded")
        os.makedirs(d, exist_ok=True)
        model = MLP(dim=4, hidden=8)
        sd = model.state_dict()
        keys = list(sd)
        half = len(keys) // 2
        save_file({k: sd[k] for k in keys[:half]},
                  os.path.join(d, "model-00001-of-00002.safetensors"))
        save_file({k: sd[k] for k in keys[half:]},
                  os.path.join(d, "model-00002-of-00002.safetensors"))
        index = {"weight_map": {k: f"model-0000{1 if i < half else 2}-of-00002.safetensors"
                                for i, k in enumerate(keys)}}
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        loaded = load_state_dict_from_hf(d)
        self.assertEqual(set(loaded), set(sd))
        for k in sd:
            self.assertTrue(torch.equal(loaded[k], sd[k]))

    def test_diagnose_accepts_hf_layout(self):
        # Healthy synthetic run (flat eval) -> forensic NO COLLAPSE.
        rng = np.random.default_rng(0)
        Xe = torch.from_numpy(rng.standard_normal((32, 4)).astype(np.float32))
        ye = (Xe[:, 0] > 0).float().unsqueeze(1)
        report = diagnose(self.hf_dir, eval_data={"Xe": Xe, "ye": ye}, verbose=False)
        self.assertEqual(report.provenance, "forensic")
        self.assertTrue(report.run_dir.endswith("_trainsect"))
        self.assertEqual(report.verdict, "NO COLLAPSE")


if __name__ == "__main__":
    unittest.main()
