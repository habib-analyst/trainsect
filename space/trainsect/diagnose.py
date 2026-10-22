"""End-to-end diagnosis: Stage A (change point) -> B (checkpoint binary
search) -> C (pinpoint trigger) -> D (shard bisect) -> E (necessity +
sufficiency + offending sample). Mirrors the P0 prototype, on the P1
package's typed modules.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .bisect import bisect_checkpoints, confirm, make_predicate, pinpoint_trigger, shard_bisect
from .collapse import (
    collapse_threshold,
    detect_change_point,
    has_collapse,
    train_spike_threshold,
)
from .dataloader import IndexedDataLoader
from .evidence import EvidenceReport
from .replay import ReplayEngine


class RunDir:
    """An on-disk instrumented training run."""

    def __init__(self, run_dir: str):
        self.dir = run_dir
        self.config = self._load_config()
        self.batches = torch.load(
            os.path.join(run_dir, "batches.pt"), weights_only=False
        )
        self.eval_data = torch.load(
            os.path.join(run_dir, "eval.pt"), weights_only=False
        )
        with open(os.path.join(run_dir, "sidecar.jsonl")) as f:
            self.sidecar = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(run_dir, "loss_log.json")) as f:
            self.loss_log = json.load(f)
        self.ckpts = sorted(
            int(fn[5:9])
            for fn in os.listdir(run_dir)
            if fn.startswith("ckpt_") and fn.endswith(".pt")
        )
        self.loader = IndexedDataLoader.from_sidecar(self.batches, self.sidecar)
        self.engine = ReplayEngine(run_dir, self.config, self.loader, self.eval_data)
        self.total_steps = max(self.ckpts)

    def _load_config(self) -> dict:
        path = os.path.join(self.dir, "config.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        # P0 run has no config.json: P0 MLP defaults.
        return {"arch": "mlp", "dim": 20, "hidden": 64, "optimizer": "sgd", "lr": 0.2}


def offending_sample(engine: ReplayEngine, run: RunDir, trigger_step: int) -> Dict[str, object]:
    """Per-sample loss under frozen weights at the trigger step."""
    model, _, _ = engine.load_state(trigger_step)
    batch = run.loader[trigger_step]
    model.eval()
    with torch.no_grad():
        if run.config.get("arch", "mlp") == "transformer":
            logits = model(batch.x)
            per = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.y.reshape(-1),
                reduction="none",
            ).reshape(batch.y.shape).mean(dim=1).numpy()
        else:
            per = F.binary_cross_entropy_with_logits(
                model(batch.x), batch.y, reduction="none"
            ).squeeze(1).numpy()
    idx = int(np.argmax(per))
    return {
        "sample_index_in_batch": idx,
        "sample_loss": float(per[idx]),
        "batch_mean_loss": float(per.mean()),
    }


def diagnose(
    run_dir: str,
    config_overrides: Optional[dict] = None,
    clean_shard: Optional[int] = None,
    verbose: bool = True,
    dataset=None,
    eval_data=None,
    num_seeds: int = 8,
    search_seeds: int = 3,
) -> EvidenceReport:
    """Run the full Trainsect pipeline on a run directory.

    Routing (P2):
    - HF checkpoint layout (``checkpoint-*/`` or top-level HF weights) is
      normalized first (read-only w.r.t. the Hub) to ``<dir>_trainsect``.
    - A directory with complete recorder data (sidecar + batches + eval)
      takes the instrumented path: bit-exact fidelity check, binary
      search, counterfactual shard bisect, necessity + sufficiency.
    - Anything else takes the forensic path (P0 §6): reconstructed batch
      orders under multiple seeds, honest confidence, never a batch
      fingerprint, never a causal verdict.

    ``dataset`` / ``eval_data`` supply the batch pool and eval split for
    forensic runs (sequences of ``(x, y, shard)`` / dict, or paths to
    ``.pt`` files). ``num_seeds`` (default 8) is the forensic confirmation
    seed count; ``search_seeds`` (default 3) the bracket-search count.
    """
    from .forensic import diagnose_forensic, has_recorder_data
    from .hub import is_hf_layout, load_hf_checkpoint_dir

    if is_hf_layout(run_dir):
        run_dir = load_hf_checkpoint_dir(run_dir)
        if verbose:
            print(f"[trainsect] HF checkpoint layout detected; normalized -> {run_dir}",
                  flush=True)
    if not has_recorder_data(run_dir):
        return diagnose_forensic(
            run_dir,
            dataset=dataset,
            eval_data=eval_data,
            config_overrides=config_overrides,
            num_seeds=num_seeds,
            search_seeds=search_seeds,
            clean_shard=clean_shard,
            verbose=verbose,
        )
    return _diagnose_instrumented(
        run_dir, config_overrides=config_overrides,
        clean_shard=(0 if clean_shard is None else clean_shard),
        verbose=verbose,
    )


def _diagnose_instrumented(
    run_dir: str,
    config_overrides: Optional[dict] = None,
    clean_shard: int = 0,
    verbose: bool = True,
) -> EvidenceReport:
    """Run the full Trainsect pipeline on an instrumented run directory."""
    t0 = time.time()
    run = RunDir(run_dir)
    if config_overrides:
        run.config.update(config_overrides)
        run.engine.config = run.config
    engine = run.engine

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # Fidelity check (P0 §1.2): replay 0 -> total twice + vs stored final ckpt.
    fidelity = engine.fidelity_check(0, run.total_steps)
    bit_exact = bool(fidelity["bit_exact"])
    log(
        f"[fidelity] replay {fidelity['replay_hash_1']} vs original "
        f"{fidelity.get('original_hash')} -> "
        f"{'BIT-EXACT' if bit_exact else 'NON-DETERMINISTIC'}"
    )
    if not bit_exact:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="ABORTED",
            confirmed_causal=False,
            fidelity_bit_exact=False,
            replays=engine.replays,
            replayed_steps=engine.replayed_steps,
            wall_time_s=time.time() - t0,
            notes=["Replay is non-deterministic on this stack; bisection halted rather than producing a false answer."],
        )

    # Stage A: change point on the eval-loss log.
    stage_a = detect_change_point(run.loss_log)
    log(f"[Stage A] change point: {stage_a}")
    eval_thresh = float(stage_a["threshold"])
    if stage_a["change_point_step"] is None:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="NO COLLAPSE",
            confirmed_causal=False,
            fidelity_bit_exact=True,
            change_point=stage_a,
            replays=engine.replays,
            replayed_steps=engine.replayed_steps,
            wall_time_s=time.time() - t0,
            confidence="no eval-loss excursion above threshold in the log",
            notes=["No collapse detected in the eval-loss log; nothing to bisect."],
        )

    # Stage B: checkpoint binary search with the eval-based predicate.
    spike_thresh = train_spike_threshold(run.sidecar)
    log(f"[thresholds] train-spike={spike_thresh:.3f} eval-health={eval_thresh:.3f}")
    predicate = make_predicate(
        engine,
        run.total_steps,
        eval_thresh,
        loss_log=run.loss_log,
        baseline=float(stage_a["baseline"]),
    )
    stage_b = bisect_checkpoints(run.ckpts, predicate, run.total_steps)
    log(f"[Stage B] checkpoint binary search -> {stage_b}")
    if not stage_b["ok"]:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="ABORTED",
            confirmed_causal=False,
            fidelity_bit_exact=True,
            change_point=stage_a,
            replays=engine.replays,
            replayed_steps=engine.replayed_steps,
            wall_time_s=time.time() - t0,
            notes=[str(stage_b["reason"])],
        )

    lo, hi = (int(stage_b["bracket"][0]), int(stage_b["bracket"][1]))

    # Stage C: pinpoint trigger step.
    stage_c = pinpoint_trigger(engine, lo, hi, spike_thresh, eval_thresh)
    s_star = stage_c["trigger_step"]
    log(
        f"[Stage C] pinpoint: first train spike at step {s_star} "
        f"(loss={stage_c['train_loss_at_trigger']}); first eval crossing at "
        f"{stage_c['eval_crossing_step']} inside bracket [{lo}, {hi})"
    )

    # Stage D: counterfactual shard bisect.
    shard = shard_bisect(engine, run.loader, lo, hi, eval_thresh, clean_shard)
    log(f"[Stage D] offending shard = shard-{shard}")

    # Stage E: necessity + sufficiency + batch fingerprint + offending sample.
    conf = confirm(engine, run.loader, lo, hi, shard, eval_thresh, clean_shard)
    confirmed = bool(conf["necessity"] and conf["sufficiency"])
    fp = run.loader[s_star].fingerprint if s_star is not None else None
    sample = offending_sample(engine, run, s_star) if s_star is not None else {}
    log(f"[Stage E] necessity={conf['necessity']} sufficiency={conf['sufficiency']}")
    if not confirmed:
        log("[Stage E] VERDICT UNCONFIRMED - not reported as causal")
    log(f"[Stage E] trigger batch: step {s_star}, shard {shard}, fingerprint {fp}")
    log(f"[Stage E] offending sample: {sample}")

    wall = time.time() - t0
    return EvidenceReport(
        run_dir=run_dir,
        verdict="CONFIRMED (causal)" if confirmed else "UNCONFIRMED",
        confirmed_causal=confirmed,
        fidelity_bit_exact=True,
        change_point=stage_a,
        bracket=[lo, hi],
        trigger_step=s_star,
        shard=shard,
        batch_fingerprint=fp,
        offending_sample=sample,
        confirmation=conf,
        replays=engine.replays,
        replayed_steps=engine.replayed_steps,
        wall_time_s=wall,
        confidence=(
            "causal by counterfactual replay (necessity + sufficiency)"
            if confirmed
            else "shard named by bisection but necessity/sufficiency failed -- do not treat as causal"
        ),
        provenance="instrumented",
    )
