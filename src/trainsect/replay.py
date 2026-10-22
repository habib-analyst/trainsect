"""Deterministic optimizer-step replay.

A replay loads exact mid-run state (pre-update checkpoint semantics: the
checkpoint at step ``c`` holds the state *before* step ``c``'s optimizer
update, so replay from ``c`` applies batches ``c, c+1, ...`` exactly once)
and advances ``k`` steps under identical inputs.

The fidelity check replays a segment twice and compares parameter hashes;
if the two replays disagree, replay is not deterministic on this stack and
bisection must halt rather than produce a false answer (P0 methodology §1.2).
"""
from __future__ import annotations

import os
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .dataloader import IndexedDataLoader
from .fingerprints import fingerprint_params
from .models import build_model, build_optimizer, eval_loss, train_loss


class ReplayEngine:
    """Loads checkpoints and replays optimizer steps deterministically."""

    def __init__(
        self,
        run_dir: str,
        config: dict,
        dataloader: IndexedDataLoader,
        eval_data: Dict[str, torch.Tensor],
    ):
        self.run_dir = run_dir
        self.config = config
        self.dataloader = dataloader
        self.eval_data = eval_data
        self.replays = 0
        self.replayed_steps = 0

    # -- checkpoint IO ------------------------------------------------------
    def checkpoint_path(self, step: int) -> str:
        return os.path.join(self.run_dir, f"ckpt_{step:04d}.pt")

    def load_state(self, step: int):
        """Load model + optimizer + scheduler + RNG state for step ``step``."""
        ck = torch.load(self.checkpoint_path(step), weights_only=False)
        model = build_model(self.config)
        model.load_state_dict(ck["model"])
        opt = build_optimizer(model, self.config)
        opt.load_state_dict(ck["optimizer"])
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        if "scheduler" in ck and ck["scheduler"]:
            sched.load_state_dict(ck["scheduler"])
        # RNG states are optional: pre-P2 / HF-converted checkpoints may
        # lack them. Missing RNG is NOT silently papered over -- replay
        # without restored RNG will fail the fidelity check (or, in
        # forensic mode, surface as cross-seed disagreement).
        if ck.get("rng_torch") is not None:
            torch.set_rng_state(ck["rng_torch"])
        if ck.get("rng_numpy") is not None:
            np.random.set_state(ck["rng_numpy"])
        if ck.get("rng_python") is not None:
            random.setstate(ck["rng_python"])
        # CUDA RNG (P2): restored only when present in the checkpoint and
        # CUDA is available. Absent on CPU-only runs / pre-P2 checkpoints.
        if ck.get("rng_cuda") and torch.cuda.is_available():
            for i, st in enumerate(ck["rng_cuda"]):
                torch.cuda.set_rng_state(st, i)
        return model, opt, sched

    @staticmethod
    def save_checkpoint(
        path: str,
        step: int,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        sched,
    ) -> None:
        """Save pre-update state (call BEFORE the step's optimizer update)."""
        rng_cuda = []
        if torch.cuda.is_available():
            try:
                rng_cuda = [
                    torch.cuda.get_rng_state(i)
                    for i in range(torch.cuda.device_count())
                ]
            except Exception:  # pragma: no cover - CUDA present but unusable
                rng_cuda = []
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate(),
                "rng_cuda": rng_cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            path,
        )

    def checkpoint_eval_loss(self, step: int) -> float:
        """Eval loss of the checkpoint state itself (the health check)."""
        model, _, _ = self.load_state(step)
        return eval_loss(model, self.eval_data, self.config)

    # -- replay ---------------------------------------------------------------
    def replay(
        self,
        from_step: int,
        to_step: int,
        replace: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
        eval_every: int = 10,
    ) -> Tuple[Dict[int, float], Dict[int, float], str]:
        """Replay optimizer steps ``from_step .. to_step-1``.

        ``replace(step, x, y, shard) -> (x, y)`` optionally substitutes
        batches (counterfactual replay). Returns ``(train_losses,
        eval_losses, end_param_hash)``; eval is sampled every
        ``eval_every`` steps and at the final state.
        """
        self.replays += 1
        model, opt, sched = self.load_state(from_step)
        losses: Dict[int, float] = {}
        evals: Dict[int, float] = {}
        for step in range(from_step, to_step):
            batch = self.dataloader[step]
            x, y = batch.x, batch.y
            if replace is not None:
                x, y = replace(step, x, y, batch.shard_id)
            opt.zero_grad()
            loss = train_loss(model, (x, y), self.config)
            losses[step] = loss.item()
            loss.backward()
            opt.step()
            sched.step()
            self.replayed_steps += 1
            if step % eval_every == 0:
                evals[step] = eval_loss(model, self.eval_data, self.config)
        evals[to_step] = eval_loss(model, self.eval_data, self.config)
        return losses, evals, fingerprint_params(model)

    # -- fidelity ---------------------------------------------------------------
    def fidelity_check(self, from_step: int, to_step: int) -> Dict[str, object]:
        """Replay ``from_step -> to_step`` twice; both must agree bit-exactly
        with each other and with the stored checkpoint at ``to_step`` (if it
        exists and the segment covers the original trajectory)."""
        _, _, h1 = self.replay(from_step, to_step)
        _, _, h2 = self.replay(from_step, to_step)
        result: Dict[str, object] = {
            "replay_hash_1": h1,
            "replay_hash_2": h2,
            "replays_agree": h1 == h2,
        }
        ckpt_path = self.checkpoint_path(to_step)
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, weights_only=False)
            model = build_model(self.config)
            model.load_state_dict(ck["model"])
            h_orig = fingerprint_params(model)
            result["original_hash"] = h_orig
            result["matches_original"] = h1 == h_orig
            result["bit_exact"] = bool(h1 == h2 == h_orig)
        else:
            result["bit_exact"] = bool(h1 == h2)
        return result
