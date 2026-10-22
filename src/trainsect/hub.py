"""Hugging Face Hub integration (P2) -- READ-ONLY w.r.t. the Hub.

Two entry points:

- :func:`load_hf_checkpoint_dir`: point at a local directory matching the
  HF checkpoint layout (``checkpoint-N/`` subdirs as saved by
  ``Trainer.save_model`` / ``Trainer._save_checkpoint``, or a top-level
  ``model.safetensors`` / ``pytorch_model.bin``) and convert it to a
  Trainsect run directory (``ckpt_*.pt`` + ``config.json`` +
  ``loss_log.json`` when ``trainer_state.json`` has eval history).
- :func:`download_checkpoint_repo`: download a checkpoint repo from the
  Hub (download only, never uploads) and convert it the same way.

Downloads go through ``curl`` as a subprocess: Python HTTP clients hang
on huggingface.co through this VM's egress proxy while curl succeeds.
No Python HTTP is used for Hub traffic.

Honest scope: conversion maps *checkpoint files* (weights, optimizer,
scheduler, RNG states). It does NOT make arbitrary HF architectures
replayable -- Trainsect replays only the architectures in
:mod:`trainsect.models` (``mlp`` / ``transformer``). The HF ``config.json``
must therefore declare a ``"trainsect"`` block (``arch`` + dims +
optimizer + lr, same keys as a Trainsect ``config.json``); otherwise
conversion refuses rather than guessing a model. Converted weights are
loaded with ``map_location="cpu"``: conversion is CPU inspection, and
``diagnose`` still enforces its fidelity / consensus checks on the
converted directory.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

import torch

HF_BASE = "https://huggingface.co"


def api_tree_url(repo_id: str, revision: str = "main") -> str:
    return f"{HF_BASE}/api/models/{repo_id}/tree/{revision}?recursive=true"


def file_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"


def _curl_cmd(url: str, dest: str, token: Optional[str] = None) -> List[str]:
    cmd = ["curl", "-L", "-f", "--retry", "3", "--connect-timeout", "30",
           "-o", dest, url]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    return cmd


def download_file(url: str, dest: str, token: Optional[str] = None) -> str:
    """Download ``url`` to ``dest`` via curl. Raises on failure."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    cmd = _curl_cmd(url, dest, token=token)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curl download failed for {url}: {proc.stderr.strip()[-500:]}"
        )
    return dest


def list_repo_files(repo_id: str, revision: str = "main",
                    token: Optional[str] = None) -> List[str]:
    """File paths in a Hub repo (via the Hub API, over curl)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "tree.json")
        download_file(api_tree_url(repo_id, revision), dest, token=token)
        with open(dest) as f:
            tree = json.load(f)
    return [entry["path"] for entry in tree
            if entry.get("type") == "file"]


_DEFAULT_PATTERNS = (
    "config.json",
    "trainer_state.json",
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)


def _wanted(path: str) -> bool:
    if path in _DEFAULT_PATTERNS:
        return True
    base = os.path.basename(path)
    if base in _DEFAULT_PATTERNS:
        return True
    return base.endswith((".safetensors", ".bin", ".json")) and (
        "model" in base or base.endswith("optimizer.pt")
        or base.endswith("scheduler.pt") or base == "config.json"
        or base == "trainer_state.json"
    )


def download_checkpoint_repo(
    repo_id: str,
    out_dir: str,
    revision: str = "main",
    token: Optional[str] = None,
) -> str:
    """Download a checkpoint repo from the Hub (read-only) to ``out_dir``.

    Downloads checkpoint files (weights incl. safetensors shards, optimizer,
    scheduler, RNG state, trainer state, config). Returns ``out_dir``.
    """
    files = list_repo_files(repo_id, revision=revision, token=token)
    wanted = [f for f in files if _wanted(f)]
    if not wanted:
        raise RuntimeError(f"no checkpoint files found in repo {repo_id!r}")
    for rel in wanted:
        dest = os.path.join(out_dir, rel)
        download_file(file_url(repo_id, rel, revision), dest, token=token)
    return out_dir


# ------------------------------------------------------------ local layout
def is_hf_layout(path: str) -> bool:
    """True iff ``path`` looks like an HF checkpoint directory (and is NOT
    already a Trainsect run directory)."""
    if not os.path.isdir(path):
        return False
    if glob.glob(os.path.join(path, "ckpt_*.pt")):
        return False  # already a Trainsect run dir
    if glob.glob(os.path.join(path, "checkpoint-*")):
        return True
    for name in ("model.safetensors", "pytorch_model.bin",
                 "model.safetensors.index.json"):
        if os.path.exists(os.path.join(path, name)):
            return True
    return False


def load_state_dict_from_hf(path: str) -> Dict[str, torch.Tensor]:
    """Load a model state dict from an HF weights file/dir.

    Supports ``model.safetensors`` (+ sharded ``model-*.safetensors`` via
    the index file) and ``pytorch_model.bin``. ``path`` may be a weights
    file or a directory containing one.
    """
    if os.path.isdir(path):
        index = os.path.join(path, "model.safetensors.index.json")
        single = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(index):
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted(set(weight_map.values()))
            state: Dict[str, torch.Tensor] = {}
            for shard in shards:
                state.update(load_state_dict_from_hf(os.path.join(path, shard)))
            missing = set(weight_map) - set(state)
            if missing:
                raise RuntimeError(f"safetensors index lists missing keys: {sorted(missing)[:5]}")
            return state
        if os.path.exists(single):
            path = single
        elif os.path.exists(bin_path):
            path = bin_path
        else:
            shards = sorted(glob.glob(os.path.join(path, "model-*.safetensors")))
            if not shards:
                raise FileNotFoundError(f"no HF weights found in {path}")
            state = {}
            for shard in shards:
                state.update(load_state_dict_from_hf(shard))
            return state
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"unrecognized HF weights file: {path}")


def _trainer_rng_to_trainsect(rng_state: dict) -> dict:
    """Map ``Trainer``'s ``rng_state.pth`` to Trainsect checkpoint RNG keys."""
    out = {
        "rng_python": rng_state.get("python"),
        "rng_numpy": rng_state.get("numpy"),
        "rng_torch": rng_state.get("cpu"),
        "rng_cuda": rng_state.get("cuda") or [],
    }
    return out


def normalize_hf_dir(hf_dir: str, out_dir: str) -> str:
    """Convert an HF-layout checkpoint directory to a Trainsect run dir.

    - ``checkpoint-N/`` subdirs -> ``ckpt_{N:04d}.pt`` (Trainsect format:
      model + optimizer + scheduler + RNG states).
    - ``trainer_state.json`` log history -> ``loss_log.json`` (eval points).
    - ``config.json`` must carry a ``"trainsect"`` block declaring the
      replayable architecture; otherwise conversion refuses.
    Returns ``out_dir``.
    """
    if not is_hf_layout(hf_dir):
        raise ValueError(f"{hf_dir} does not look like an HF checkpoint directory")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(hf_dir, "config.json")) as f:
        hf_config = json.load(f)
    ts_cfg = hf_config.get("trainsect")
    if not isinstance(ts_cfg, dict) or "arch" not in ts_cfg:
        raise ValueError(
            "HF config.json carries no 'trainsect' block (arch + dims + "
            "optimizer + lr) -- Trainsect cannot replay an undeclared "
            "architecture; refusing to guess"
        )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(ts_cfg, f, indent=2)

    ckpt_dirs = sorted(
        glob.glob(os.path.join(hf_dir, "checkpoint-*")),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
    )
    loss_log: List[List[float]] = []
    if not ckpt_dirs:
        ckpt_dirs = [hf_dir]  # single top-level checkpoint
        top_steps = [int(ts_cfg.get("step", 0))]
    else:
        top_steps = [int(re.search(r"checkpoint-(\d+)", p).group(1)) for p in ckpt_dirs]

    for ckpt_dir, step in zip(ckpt_dirs, top_steps):
        state = load_state_dict_from_hf(ckpt_dir)
        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        optimizer_state = (
            torch.load(opt_path, map_location="cpu", weights_only=False)
            if os.path.exists(opt_path) else {}
        )
        sched_path = os.path.join(ckpt_dir, "scheduler.pt")
        scheduler_state = (
            torch.load(sched_path, map_location="cpu", weights_only=False)
            if os.path.exists(sched_path) else {}
        )
        rng_path = os.path.join(ckpt_dir, "rng_state.pth")
        rng = {}
        if os.path.exists(rng_path):
            rng = _trainer_rng_to_trainsect(
                torch.load(rng_path, map_location="cpu", weights_only=False)
            )
        ckpt = {
            "step": step,
            "model": state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "rng_torch": rng.get("rng_torch"),
            "rng_numpy": rng.get("rng_numpy"),
            "rng_python": rng.get("rng_python"),
            "rng_cuda": rng.get("rng_cuda") or [],
            "deterministic_algorithms": False,
            "source": "hf-hub",
            "source_dir": ckpt_dir,
        }
        torch.save(ckpt, os.path.join(out_dir, f"ckpt_{step:04d}.pt"))
        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path) as f:
                trainer_state = json.load(f)
            for entry in trainer_state.get("log_history", []):
                if "eval_loss" in entry:
                    loss_log.append([int(entry.get("step", step)),
                                     float(entry["eval_loss"])])
    if loss_log:
        loss_log = sorted(set(map(tuple, loss_log)))
        with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
            json.dump([list(x) for x in loss_log], f)
    return out_dir


def load_hf_checkpoint_dir(hf_dir: str, out_dir: Optional[str] = None) -> str:
    """Point at a local HF-layout dir; return a Trainsect run directory.

    Converts in place-adjacent: ``out_dir`` defaults to
    ``<hf_dir>_trainsect``.
    """
    if out_dir is None:
        out_dir = hf_dir.rstrip("/") + "_trainsect"
    return normalize_hf_dir(hf_dir, out_dir)
