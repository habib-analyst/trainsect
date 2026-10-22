"""Recorder hook (instrumented mode, P2).

One-line-per-step capture of everything replay needs:

.. code-block:: python

    from trainsect import start_recording, record, stop_recording

    start_recording("runs/exp1", model, opt, sched, config, ckpt_every=30)
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        record(step, batch, shard_id=shard_of(batch), train_loss=loss.item())
    stop_recording(eval_data=eval_data)

``record()`` is called *after* the step's optimizer update. That keeps
pre-update checkpoint semantics exact (P0 §1, P1 ``ReplayEngine``): the
state right after step ``c``'s update *is* the state before step ``c+1``,
so ``record(step=c)`` saves checkpoint ``ckpt_{c+1}`` when due, and
``start_recording`` captures ``ckpt_0000`` (state before step 0) up front.
A checkpoint saved after the update but labelled with the *same* step
would double-apply that step's batch on replay (P0 bug #1) -- this API
cannot express that mistake.

Captured per step (sidecar JSONL): step, epoch, global batch index,
shard id(s), batch fingerprint, train loss, LR, sampler indices and
worker seeds when supplied. Every ``ckpt_every`` steps (plus the final
step): full pre-update state via :meth:`ReplayEngine.save_checkpoint`
(model + optimizer + scheduler + torch/numpy/python/CUDA RNG states).
Env flags, seeds, torch/Python versions and dataloader info go to
``manifest.json``. Raw batches are accumulated and flushed to
``batches.pt`` at ``finalize`` (``store_batches=False`` records
fingerprints only -- replay then falls back to forensic mode).

Design target: < 2% step-time overhead. The fingerprint hash dominates;
everything else is small state copies. See :func:`benchmark_overhead`
for the measured number on a realistic step.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .fingerprints import fingerprint_batch
from .replay import ReplayEngine

_ACTIVE: Optional["Recorder"] = None


def _env_flags() -> Dict[str, object]:
    flags: Dict[str, object] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
    }
    try:
        flags["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        flags["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    except Exception:  # pragma: no cover - CPU-only builds without cudnn
        pass
    if torch.cuda.is_available():
        try:
            flags["cuda_device_count"] = torch.cuda.device_count()
            flags["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover
            pass
    return flags


class Recorder:
    """Per-step recorder writing a Trainsect instrumented run directory."""

    def __init__(
        self,
        out_dir: str,
        config: Optional[dict] = None,
        ckpt_every: int = 50,
        store_batches: bool = True,
        dataloader_info: Optional[dict] = None,
        seeds: Optional[dict] = None,
        sidecar_flush_every: int = 100,
    ):
        self.out_dir = out_dir
        self.config = dict(config or {})
        self.ckpt_every = int(ckpt_every)
        self.store_batches = store_batches
        self.dataloader_info = dict(dataloader_info or {})
        self.seeds = dict(seeds or {})
        self.sidecar_flush_every = int(sidecar_flush_every)
        self._sidecar: List[dict] = []
        self._loss_log: List[List[float]] = []
        self._batches: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self._model = None
        self._opt = None
        self._sched = None
        self._started = False
        self._finalized = False
        self._last_step: Optional[int] = None
        self._ckpt_steps: List[int] = []
        self._flushed_rows = 0
        self._sidecar_fp = None
        os.makedirs(out_dir, exist_ok=True)

    # -- lifecycle ----------------------------------------------------------
    def start(self, model, optimizer, scheduler) -> "Recorder":
        """Capture ``ckpt_0000`` (state before step 0) + env flags."""
        if self._started:
            raise RuntimeError("Recorder.start() called twice")
        self._model, self._opt, self._sched = model, optimizer, scheduler
        self._started = True
        # A fresh recording owns its sidecar: truncate any stale file so
        # later appends cannot duplicate rows from an earlier run.
        open(os.path.join(self.out_dir, "sidecar.jsonl"), "w").close()
        ReplayEngine.save_checkpoint(
            os.path.join(self.out_dir, "ckpt_0000.pt"), 0, model, optimizer, scheduler
        )
        self._ckpt_steps.append(0)
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        return self

    def record(
        self,
        step: int,
        batch: Tuple[torch.Tensor, torch.Tensor],
        shard_id: int,
        train_loss: Optional[float] = None,
        lr: Optional[float] = None,
        epoch: int = 0,
        batch_idx: Optional[int] = None,
        sampler_indices: Optional[Sequence[int]] = None,
        worker_seeds: Optional[Sequence[int]] = None,
    ) -> dict:
        """Record one optimizer step (call AFTER the step's update).

        Saves the sidecar row for ``step`` and, when ``step + 1`` is a
        checkpoint step, the pre-update checkpoint for step ``step + 1``.
        """
        if not self._started:
            raise RuntimeError(
                "Recorder.record() before start(); call start_recording() first"
            )
        if self._finalized:
            raise RuntimeError("Recorder.record() after finalize()")
        x, y = batch
        fp = fingerprint_batch(x, y)
        if lr is None and self._opt is not None:
            try:
                lr = float(self._opt.param_groups[0]["lr"])
            except Exception:  # pragma: no cover - optimiser without lr
                lr = None
        row: Dict[str, Any] = {
            "step": int(step),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx if batch_idx is not None else step),
            "shard_id": int(shard_id),
            "batch_fp": fp,
            "train_loss": train_loss,
            "lr": lr,
        }
        if sampler_indices is not None:
            row["sampler_indices"] = [int(i) for i in sampler_indices]
        if worker_seeds is not None:
            row["worker_seeds"] = [int(s) for s in worker_seeds]
        self._sidecar.append(row)
        if self.store_batches:
            self._batches.append(
                (x.detach().to("cpu").clone(), y.detach().to("cpu").clone(), int(shard_id))
            )
        self._last_step = int(step)
        if (int(step) + 1) % self.ckpt_every == 0:
            self.checkpoint(int(step) + 1)
        if len(self._sidecar) % self.sidecar_flush_every == 0:
            self._flush_sidecar()
        return row

    def checkpoint(self, step: int) -> str:
        """Save the pre-update checkpoint for ``step`` (state right now).

        Call at a loop top for explicit pre-update capture, or let
        :meth:`record` call it for ``step + 1`` automatically.
        """
        if not self._started:
            raise RuntimeError("Recorder.checkpoint() before start()")
        path = os.path.join(self.out_dir, f"ckpt_{int(step):04d}.pt")
        ReplayEngine.save_checkpoint(
            path, int(step), self._model, self._opt, self._sched
        )
        if int(step) not in self._ckpt_steps:
            self._ckpt_steps.append(int(step))
        return path

    def record_eval(self, step: int, eval_loss_value: float) -> None:
        """Append one eval-loss point to the loss log."""
        self._loss_log.append([int(step), float(eval_loss_value)])

    def callback(self, step, batch, shard_id, **kwargs):
        """Callback-shaped alias for :meth:`record` (loop / trainer hooks)."""
        return self.record(step, batch, shard_id, **kwargs)

    # -- finalise -------------------------------------------------------------
    def finalize(self, eval_data: Optional[dict] = None, total_steps: Optional[int] = None) -> str:
        """Flush sidecar / batches / eval / manifest; save final checkpoint."""
        if self._finalized:
            return self.out_dir
        if not self._started:
            raise RuntimeError("Recorder.finalize() before start()")
        if total_steps is None:
            total_steps = (self._last_step + 1) if self._last_step is not None else 0
        if int(total_steps) not in self._ckpt_steps:
            self.checkpoint(int(total_steps))
        self._flush_sidecar()
        if self._sidecar_fp is not None:
            self._sidecar_fp.close()
            self._sidecar_fp = None
        with open(os.path.join(self.out_dir, "loss_log.json"), "w") as f:
            json.dump(self._loss_log, f)
        if self.store_batches:
            torch.save(self._batches, os.path.join(self.out_dir, "batches.pt"))
        if eval_data is not None:
            torch.save(eval_data, os.path.join(self.out_dir, "eval.pt"))
        manifest = {
            "recorder": "trainsect Recorder (P2)",
            "env": _env_flags(),
            "seeds": self.seeds,
            "dataloader": self.dataloader_info,
            "ckpt_every": self.ckpt_every,
            "ckpt_steps": sorted(self._ckpt_steps),
            "total_steps": int(total_steps),
            "store_batches": self.store_batches,
            "sidecar_rows": len(self._sidecar),
        }
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        self._finalized = True
        return self.out_dir

    def _flush_sidecar(self) -> None:
        # Append-only: write only the rows added since the last flush.
        # (An earlier version rewrote the whole file per flush -- O(n^2)
        # JSON re-serialisation that dominated the hook overhead.)
        if self._flushed_rows >= len(self._sidecar):
            return
        if self._sidecar_fp is None:
            self._sidecar_fp = open(os.path.join(self.out_dir, "sidecar.jsonl"), "a")
        for row in self._sidecar[self._flushed_rows:]:
            self._sidecar_fp.write(json.dumps(row) + "\n")
        self._sidecar_fp.flush()
        self._flushed_rows = len(self._sidecar)

    # -- context manager ------------------------------------------------------
    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        if not self._finalized and self._started:
            self.finalize()


# ---------------------------------------------------------------- module API
def start_recording(
    out_dir: str,
    model,
    optimizer,
    scheduler,
    config: Optional[dict] = None,
    **kwargs,
) -> Recorder:
    """Create the active recorder and capture ``ckpt_0000``. One line."""
    global _ACTIVE
    if _ACTIVE is not None and not _ACTIVE._finalized:
        raise RuntimeError(
            "a recording is already active; call stop_recording() first"
        )
    rec = Recorder(out_dir, config=config, **kwargs)
    rec.start(model, optimizer, scheduler)
    _ACTIVE = rec
    return rec


def record(step, batch, shard_id, **kwargs) -> dict:
    """Record one step on the active recorder (see :class:`Recorder`)."""
    if _ACTIVE is None:
        raise RuntimeError(
            "no active recording; call start_recording() (or use "
            "recording(...)) before record()"
        )
    return _ACTIVE.record(step, batch, shard_id, **kwargs)


def record_eval(step: int, eval_loss_value: float) -> None:
    """Record one eval point on the active recorder."""
    if _ACTIVE is None:
        raise RuntimeError("no active recording; call start_recording() first")
    _ACTIVE.record_eval(step, eval_loss_value)


def stop_recording(eval_data: Optional[dict] = None, total_steps: Optional[int] = None) -> str:
    """Finalize the active recording and clear it. Returns the run dir."""
    global _ACTIVE
    if _ACTIVE is None:
        raise RuntimeError("no active recording to stop")
    out = _ACTIVE.finalize(eval_data=eval_data, total_steps=total_steps)
    _ACTIVE = None
    return out


def get_active_recorder() -> Optional[Recorder]:
    return _ACTIVE


@contextlib.contextmanager
def recording(out_dir, model, optimizer, scheduler, config=None, **kwargs):
    """Context manager: start recording on entry, finalize on exit.

    .. code-block:: python

        with recording("runs/exp1", model, opt, sched, config) as rec:
            for step, batch in enumerate(dataloader):
                loss = train_step(batch)
                record(step, batch, shard_id=..., train_loss=loss.item())
    """
    rec = start_recording(out_dir, model, optimizer, scheduler, config=config, **kwargs)
    try:
        yield rec
    finally:
        if _ACTIVE is rec:
            stop_recording()


# ---------------------------------------------------------------- benchmarking
def benchmark_overhead(
    step_fn: Callable[[Callable], None],
    recorder: Recorder,
    steps: int = 30,
    repeats: int = 3,
    warmup: int = 5,
) -> Dict[str, float]:
    """Measure recorder overhead on a realistic step function.

    ``step_fn(record_fn)`` runs one training step and calls
    ``record_fn(step, batch, shard_id, train_loss=...)``. The hook cost is
    measured DIRECTLY: ``record`` is wrapped in a ``perf_counter`` pair
    and the call times are accumulated over the run, then divided by the
    total loop wall time. This is robust where A/B timing is not: on a
    shared VM, co-tenant load swings end-to-end step times by ~±20%,
    which swamps a sub-2% hook cost in any baseline-vs-recorded
    comparison (verified: A/B trials gave -13%..+12% for a hook that
    costs 0.04 ms/call in isolation). The wrapper itself costs two
    ``perf_counter`` calls (~0.2 us), negligible against the hook.
    Returns the overhead fraction ``record_time / loop_time``.
    """
    fracs, totals = [], []
    for _ in range(repeats):
        acc = {"record_s": 0.0}
        real_record = recorder.record

        def timed_record(*a, **k):
            t0 = time.perf_counter()
            try:
                return real_record(*a, **k)
            finally:
                acc["record_s"] += time.perf_counter() - t0

        for _ in range(warmup):
            step_fn(timed_record)
        acc["record_s"] = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            step_fn(timed_record)
        total = time.perf_counter() - t0
        totals.append(total)
        fracs.append(acc["record_s"] / total if total > 0 else float("nan"))
    fracs_sorted = sorted(fracs)
    med_frac = fracs_sorted[len(fracs_sorted) // 2]
    med_total = sorted(totals)[len(totals) // 2]
    return {
        "baseline_s": med_total * (1.0 - med_frac),
        "recorded_s": med_total,
        "overhead_s": med_total * med_frac,
        "overhead_frac": med_frac,
        "overhead_pct": 100.0 * med_frac,
        "per_step_overhead_ms": 1000.0 * med_total * med_frac / max(steps, 1),
        "steps": float(steps),
        "repeats": float(repeats),
    }
