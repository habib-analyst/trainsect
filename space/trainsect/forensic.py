"""Forensic mode: diagnosis from a checkpoints-only directory (P0 §6).

When no recorder data is present (no ``sidecar.jsonl`` / ``batches.pt`` /
``eval.pt``), Trainsect recovers what it can and states the rest as
explicit uncertainty:

===========================  ============================================
Recoverable from checkpoints  Unrecoverable (without recorder)
===========================  ============================================
model weights theta_t          exact batch order (sampler seed unknown)
optimizer moments, step count  per-step RNG draws (dropout, augmentation)
LR scheduler state (if saved)  per-batch fingerprints
loss history (if logged)       DataLoader worker seeds
eval loss at each checkpoint   exact trigger step / batch
===========================  ============================================

Procedure: Stage A (change point on the logged losses, or on eval-at-
checkpoints) still works. Replay then uses a *reconstructed* dataloader:
the known batch pool in a candidate order, under ``search_seeds``
candidate seeds for the bracket search and ``num_seeds`` seeds for the
confirmation. The bracket is found by consensus binary search over the
checkpoints. A shard is named only if ALL of these hold:

1. the collapse reproduces from the bracket start under the search seeds;
2. excluding the shard kills the collapse under (nearly) all confirm
   seeds (necessity) and keeping only it reproduces it (sufficiency);
3. excluding every *other* shard does NOT kill it (specificity);
4. frozen-weight per-batch loss ranking independently ranks it top.

Otherwise the report says **UNPROVEN** with a confidence level and what
additional evidence would settle it. Forensic mode NEVER sets a batch
fingerprint or a trigger step: the exact batch is unprovable without
the recorder sidecar. A forensic verdict is never ``CONFIRMED (causal)``.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .collapse import detect_change_point, has_collapse, is_healthy
from .dataloader import IndexedDataLoader
from .evidence import EvidenceReport
from .models import train_loss
from .replay import ReplayEngine

INSTRUMENTED_FILES = ("sidecar.jsonl", "batches.pt", "eval.pt")

RECOVERABLE = [
    "model weights (theta_t)",
    "optimizer moments + step count",
    "LR scheduler state (if saved in checkpoint)",
    "loss history (if loss_log.json present)",
    "eval loss at each checkpoint (recomputable from weights)",
]
UNRECOVERABLE = [
    "exact batch order (sampler seed / epoch unknown)",
    "per-step RNG draws (dropout masks, augmentation)",
    "per-batch fingerprints (content re-derivable only if order known)",
    "DataLoader worker seeds",
    "exact trigger step / batch (unprovable without recorder sidecar)",
]

SETTLING_EVIDENCE = (
    "recorder sidecar (exact batch order + per-batch fingerprints), "
    "RNG states, and eval data from an instrumented run"
)


def has_recorder_data(run_dir: str) -> bool:
    """True iff the directory holds a complete instrumented run."""
    return all(os.path.exists(os.path.join(run_dir, f)) for f in INSTRUMENTED_FILES)


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _load_pool(run_dir: str, dataset) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]]:
    """The batch pool: explicit ``dataset`` arg, ``dataset.pt``, or the
    stored ``batches.pt`` (whose *order* is unknown without a sidecar, so
    it is only a pool, never an order)."""
    if dataset is not None:
        if isinstance(dataset, str):
            return list(torch.load(dataset, weights_only=False))
        return list(dataset)
    for name in ("dataset.pt", "batches.pt"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return list(torch.load(path, weights_only=False))
    return None


def _load_eval(run_dir: str, eval_data):
    if eval_data is not None:
        if isinstance(eval_data, str):
            return torch.load(eval_data, weights_only=False)
        return eval_data
    path = os.path.join(run_dir, "eval.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    return None


def _load_config(run_dir: str, config_overrides: Optional[dict]) -> Optional[dict]:
    cfg: Optional[dict] = None
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        cfg = _load_json(path)
    if cfg is None and not config_overrides:
        # P0-compat fallback (mirrors RunDir): the P0 validation run ships
        # no config.json; its MLP defaults are the only sane rebuild.
        cfg = {"arch": "mlp", "dim": 20, "hidden": 64, "optimizer": "sgd", "lr": 0.2}
    if config_overrides:
        cfg = {**(cfg or {}), **config_overrides}
    return cfg


def _loss_log_from_checkpoints(
    run_dir: str, ckpts: List[int], config: dict, eval_data: dict
) -> List[List[float]]:
    """Eval loss at each checkpoint, from the weights themselves.

    Used when no ``loss_log.json`` exists: checkpoints alone still give
    an eval-loss curve (at checkpoint granularity) for Stage A.
    """
    from .models import eval_loss

    log: List[List[float]] = []
    for step in ckpts:
        ck = torch.load(os.path.join(run_dir, f"ckpt_{step:04d}.pt"), weights_only=False)
        from .models import build_model

        model = build_model(config)
        model.load_state_dict(ck["model"])
        log.append([step, eval_loss(model, eval_data, config)])
    return log


def _candidate_loader(
    pool: Sequence[Tuple[torch.Tensor, torch.Tensor, int]], seed: int
) -> IndexedDataLoader:
    """A reconstructed batch order: the pool permuted under ``seed``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    return IndexedDataLoader.from_raw([pool[i] for i in order])


def _already_collapsed(ckpt_step: int, health: float, eval_threshold: float,
                       loss_log, baseline: Optional[float]) -> bool:
    """Mirror of the instrumented health clause: above threshold counts as
    already-collapsed only if the model had previously reached
    near-baseline health (an untrained checkpoint is not 'collapsed')."""
    if is_healthy(health, eval_threshold):
        return False
    if loss_log is None or baseline is None:
        return True
    near_baseline = max(2 * baseline, baseline + 0.5)
    for step, loss in loss_log:
        if int(step) < ckpt_step and loss <= near_baseline:
            return True
    return False


def _forensic_predicate(engine: ReplayEngine, loader: IndexedDataLoader,
                        end_step: int, eval_threshold: float,
                        loss_log, baseline: Optional[float]):
    """P(c) for one candidate order: healthy + replay reproduces collapse.

    Collapse is a healthy -> unhealthy TRANSITION during replay, not a
    persistently high loss: a replay that never reaches near-baseline
    health has not collapsed, it never learned (common under
    reconstructed orders, which dilute a concentration-dependent
    trigger). Without this, a never-learning replay misreads as
    'collapse' and the bracket search converges on a wrong bracket.
    """
    near_baseline = max(2 * (baseline or 0.0), (baseline or 0.0) + 0.5)

    def predicate(ckpt_step: int) -> bool:
        health = engine.checkpoint_eval_loss(ckpt_step)
        if _already_collapsed(ckpt_step, health, eval_threshold, loss_log, baseline):
            return False
        _, evals, _ = engine.replay(ckpt_step, end_step)
        healthy_reached = is_healthy(health, eval_threshold)
        for step in sorted(evals):
            loss = evals[step]
            if math.isfinite(loss) and loss <= near_baseline:
                healthy_reached = True
            elif healthy_reached and ((not math.isfinite(loss)) or loss > eval_threshold):
                return True
        return False
    return predicate


def _consensus_bisect(ckpts, predicates, end_step) -> Dict[str, object]:
    """Binary search on the consensus predicate (majority over seeds).

    Returns ``{"ok": True, "bracket": (lo, hi), "votes": {...}}`` or
    ``{"ok": False, ...}`` when the collapse does not reproduce from the
    earliest checkpoint under the reconstructed orders.
    """
    def consensus(step: int) -> Tuple[bool, List[bool]]:
        votes = [p(step) for p in predicates]
        return sum(votes) * 2 > len(votes), votes

    ok0, votes0 = consensus(ckpts[0])
    votes = {int(ckpts[0]): [bool(v) for v in votes0]}
    if not ok0:
        return {"ok": False, "votes": votes,
                "reason": "collapse does NOT reproduce from earliest checkpoint "
                          "under reconstructed batch orders"}
    lo, hi = 0, len(ckpts) - 1
    ok_hi, votes_hi = consensus(ckpts[hi])
    votes[int(ckpts[hi])] = [bool(v) for v in votes_hi]
    if ok_hi:
        return {"ok": True, "bracket": (ckpts[hi], end_step), "votes": votes,
                "note": "collapse reproduces even from the last checkpoint"}
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok_mid, votes_mid = consensus(ckpts[mid])
        votes[int(ckpts[mid])] = [bool(v) for v in votes_mid]
        if ok_mid:
            lo = mid
        else:
            hi = mid
    return {"ok": True, "bracket": (ckpts[lo], ckpts[hi]), "votes": votes}


def frozen_ranking(engine: ReplayEngine, loader: IndexedDataLoader,
                   ckpt_step: int) -> List[Tuple[int, float, int]]:
    """Per-shard mean per-batch loss under frozen weights at ``ckpt_step``.

    Forward passes only (no optimizer steps). An independent, approximate
    signal -- explicitly NOT causal proof -- used to cross-check the
    counterfactual result.
    """
    model, _, _ = engine.load_state(ckpt_step)
    model.eval()
    per_shard: Dict[int, List[float]] = {}
    with torch.no_grad():
        for batch in loader.batches:
            loss = train_loss(model, (batch.x, batch.y), engine.config)
            per_shard.setdefault(batch.shard_id, []).append(float(loss.item()))
    ranked = sorted(
        ((shard, sum(ls) / len(ls), len(ls)) for shard, ls in per_shard.items()),
        key=lambda t: -t[1],
    )
    return ranked


def diagnose_forensic(
    run_dir: str,
    dataset=None,
    eval_data=None,
    config_overrides: Optional[dict] = None,
    num_seeds: int = 8,
    search_seeds: int = 3,
    clean_shard: Optional[int] = None,
    verbose: bool = True,
) -> EvidenceReport:
    """Forensic diagnosis of a checkpoints-only directory.

    ``dataset`` is the batch pool (sequence of ``(x, y, shard)`` or a path
    to a ``.pt`` file holding one); ``eval_data`` likewise. When omitted,
    ``dataset.pt`` / ``batches.pt`` / ``eval.pt`` in the directory are
    used if present. ``num_seeds`` (default 8, P0 §6) is the confirmation
    seed count; ``search_seeds`` (default 3) is the bracket-search count.
    """
    t0 = time.time()
    notes = [
        "FORENSIC MODE: no recorder data (sidecar/batches/eval) -- "
        "batch order, per-step RNG, and fingerprints are unrecoverable.",
        "recoverable: " + "; ".join(RECOVERABLE),
        "unrecoverable: " + "; ".join(UNRECOVERABLE),
    ]

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def _unproven(reason: str, extra_notes=()) -> EvidenceReport:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="UNPROVEN",
            confirmed_causal=False,
            fidelity_bit_exact=False,
            replays=0,
            replayed_steps=0,
            wall_time_s=time.time() - t0,
            confidence=f"unproven: {reason}",
            provenance="forensic",
            notes=list(notes) + list(extra_notes) + [
                f"to settle this, provide: {SETTLING_EVIDENCE}"
            ],
            forensic={"state_table": {"recoverable": RECOVERABLE,
                                      "unrecoverable": UNRECOVERABLE}},
        )

    ckpts = sorted(
        int(fn[5:9])
        for fn in os.listdir(run_dir)
        if fn.startswith("ckpt_") and fn.endswith(".pt")
    )
    if not ckpts:
        return _unproven("no checkpoints found in directory")
    config = _load_config(run_dir, config_overrides)
    if config is None:
        return _unproven("no config.json and no config supplied -- model cannot be rebuilt")
    total_steps = max(ckpts)

    # Stage A first: it needs only the loss log. A flat log answers
    # NO COLLAPSE without any eval data or dataset pool. Only when the
    # log is missing do we need eval data (to rebuild it from the
    # checkpoints), and only when a collapse IS in the log do we need
    # eval data + pool for the replay stages.
    loss_log = None
    ll_path = os.path.join(run_dir, "loss_log.json")
    if os.path.exists(ll_path):
        loss_log = _load_json(ll_path)
    else:
        eval_d_early = _load_eval(run_dir, eval_data)
        if eval_d_early is None:
            return _unproven("no loss_log.json and no eval data (eval.pt / eval_data) "
                             "-- cannot build an eval-loss curve")
        loss_log = _loss_log_from_checkpoints(run_dir, ckpts, config, eval_d_early)
        notes.append("no loss_log.json -- Stage A used eval loss recomputed at each checkpoint")
    stage_a = detect_change_point(loss_log)
    log(f"[forensic Stage A] change point: {stage_a}")
    eval_thresh = float(stage_a["threshold"])
    baseline = float(stage_a.get("baseline", 0.0))
    if stage_a["change_point_step"] is None:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="NO COLLAPSE",
            confirmed_causal=False,
            fidelity_bit_exact=False,
            change_point=stage_a,
            wall_time_s=time.time() - t0,
            confidence="no eval-loss excursion above threshold in the log",
            provenance="forensic",
            notes=list(notes) + ["no collapse detected; nothing to bisect."],
            forensic={"state_table": {"recoverable": RECOVERABLE,
                                      "unrecoverable": UNRECOVERABLE}},
        )

    eval_d = _load_eval(run_dir, eval_data)
    if eval_d is None:
        return _unproven("no eval data (eval.pt / eval_data) -- eval-based predicate cannot run")
    pool = _load_pool(run_dir, dataset)
    if not pool:
        return _unproven("no dataset pool (dataset / dataset.pt / batches.pt) -- cannot reconstruct a dataloader")
    if len(pool) < total_steps:
        return _unproven(
            f"batch pool ({len(pool)}) shorter than checkpoint range ({total_steps})"
        )

    # Search end: the observed collapse must reproduce by the change point
    # plus two eval intervals -- replaying to the run end is unnecessary
    # cost and tests a different (later) collapse, if any.
    cp = int(stage_a["change_point_step"])
    logged_steps = sorted(int(s) for s, _ in loss_log)
    gaps = [b - a for a, b in zip(logged_steps, logged_steps[1:]) if b > a]
    gap = float(np.median(gaps)) if gaps else 10.0
    end_search = min(total_steps, int(cp + 2 * gap))
    # The bracket's hi endpoint must be <= end_search for the predicate to
    # be meaningful; if the last checkpoint is beyond it, clamp probes.
    log(f"[forensic] search replays run probe -> {end_search} (change point {cp} + 2 gaps)")

    # Consensus binary search under search_seeds reconstructed orders.
    search_loaders = [_candidate_loader(pool, s) for s in range(search_seeds)]
    search_engines = [
        ReplayEngine(run_dir, config, loader, eval_d) for loader in search_loaders
    ]
    predicates = [
        _forensic_predicate(eng, loader, end_search, eval_thresh, loss_log, baseline)
        for eng, loader in zip(search_engines, search_loaders)
    ]
    stage_b = _consensus_bisect(ckpts, predicates, total_steps)
    replays = sum(e.replays for e in search_engines)
    replayed = sum(e.replayed_steps for e in search_engines)
    log(f"[forensic Stage B] consensus binary search -> {stage_b}")
    if not stage_b["ok"]:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="UNPROVEN",
            confirmed_causal=False,
            fidelity_bit_exact=False,
            change_point=stage_a,
            replays=replays,
            replayed_steps=replayed,
            wall_time_s=time.time() - t0,
            confidence=(
                "unproven: collapse did not reproduce from the earliest "
                f"checkpoint under {search_seeds}/{search_seeds} reconstructed orders"
            ),
            provenance="forensic",
            notes=list(notes) + [str(stage_b.get("reason", "")),
                                 f"to settle this, provide: {SETTLING_EVIDENCE}"],
            forensic={"state_table": {"recoverable": RECOVERABLE,
                                      "unrecoverable": UNRECOVERABLE},
                      "search_votes": stage_b.get("votes", {})},
        )
    lo, hi = (int(stage_b["bracket"][0]), int(stage_b["bracket"][1]))

    # Frozen-weight ranking (cheap, independent signal) picks the naming
    # candidate: the top-ranked shard. If the counterfactuals do not back
    # it up, the answer is UNPROVEN -- the ranking alone never names a shard.
    ranked = frozen_ranking(search_engines[0], search_loaders[0], lo)
    candidate = int(ranked[0][0])
    if len(ranked) > 1:
        margin = ranked[0][1] / max(ranked[1][1], 1e-12)
    else:
        margin = float("inf")
    log(f"[forensic] frozen-weight ranking: {ranked} -> candidate shard-{candidate}")
    if clean_shard is None:
        clean_shard = int(ranked[-1][0])
    log(f"[forensic] benign-substitute shard: shard-{clean_shard}")

    # Confirmation under num_seeds fresh candidate orders, bracket only.
    confirm_loaders = [_candidate_loader(pool, 1000 + s) for s in range(num_seeds)]
    confirm_engines = [
        ReplayEngine(run_dir, config, loader, eval_d) for loader in confirm_loaders
    ]

    def _frac(fn) -> Tuple[int, int]:
        hits = sum(1 for eng, loader in zip(confirm_engines, confirm_loaders)
                   if fn(eng, loader))
        return hits, num_seeds

    # Substitute-clean check: replaying the bracket with EVERYTHING
    # replaced by substitutes from the cleanest shard must NOT collapse.
    # If it does, the substitutes are not benign and no counterfactual
    # conclusion is valid.
    def _sub_clean(eng, loader) -> bool:
        benign = loader.benign_batches(clean_shard)
        def replace(step, x, y, shard):
            sub = benign[step % len(benign)]
            return sub.x, sub.y
        _, evals, _ = eng.replay(lo, hi, replace=replace)
        collapse, _ = has_collapse(evals, eval_thresh)
        return not collapse
    clean_hits, _ = _frac(_sub_clean)
    log(f"[forensic] substitute-clean check: {clean_hits}/{num_seeds} seeds clean")
    if clean_hits * 2 <= num_seeds:
        return EvidenceReport(
            run_dir=run_dir,
            verdict="UNPROVEN",
            confirmed_causal=False,
            fidelity_bit_exact=False,
            change_point=stage_a,
            bracket=[lo, hi],
            replays=replays + sum(e.replays for e in confirm_engines),
            replayed_steps=replayed + sum(e.replayed_steps for e in confirm_engines),
            wall_time_s=time.time() - t0,
            confidence=(
                f"unproven: benign-substitute shard {clean_shard} is not "
                f"collapse-free ({clean_hits}/{num_seeds} clean)"
            ),
            provenance="forensic",
            notes=list(notes) + [
                "counterfactual substitution is invalid without a clean substitute shard",
                f"to settle this, provide: {SETTLING_EVIDENCE}",
            ],
            forensic={"state_table": {"recoverable": RECOVERABLE,
                                      "unrecoverable": UNRECOVERABLE},
                      "bracket": [lo, hi],
                      "frozen_ranking": ranked,
                      "substitute_clean": f"{clean_hits}/{num_seeds}"},
        )

    # Necessity (exclude candidate) + sufficiency (only candidate).
    def _necessity(eng, loader) -> bool:
        benign = loader.benign_batches(clean_shard)
        def replace(step, x, y, shard):
            if shard == candidate:
                sub = benign[step % len(benign)]
                return sub.x, sub.y
            return x, y
        _, evals, _ = eng.replay(lo, hi, replace=replace)
        collapse, _ = has_collapse(evals, eval_thresh)
        return not collapse
    def _sufficiency(eng, loader) -> bool:
        benign = loader.benign_batches(clean_shard)
        def replace(step, x, y, shard):
            if shard != candidate:
                sub = benign[step % len(benign)]
                return sub.x, sub.y
            return x, y
        _, evals, _ = eng.replay(lo, hi, replace=replace)
        collapse, _ = has_collapse(evals, eval_thresh)
        return collapse
    nec_hits, _ = _frac(_necessity)
    suf_hits, _ = _frac(_sufficiency)
    log(f"[forensic] candidate shard-{candidate}: necessity {nec_hits}/{num_seeds}, "
        f"sufficiency {suf_hits}/{num_seeds}")

    # Specificity: excluding any OTHER shard must NOT kill the collapse.
    # Checked under the search seeds to bound cost; a failure on any other
    # shard means the evidence does not single out the candidate.
    other_shards = [s for s, _, _ in ranked if s != candidate]
    specificity_ok = True
    specificity_detail = {}
    for other in other_shards:
        kills = 0
        for eng, loader in zip(search_engines, search_loaders):
            benign = loader.benign_batches(clean_shard)
            def replace(step, x, y, shard, _o=other):
                if shard == _o:
                    sub = benign[step % len(benign)]
                    return sub.x, sub.y
                return x, y
            _, evals, _ = eng.replay(lo, hi, replace=replace)
            collapse, _ = has_collapse(evals, eval_thresh)
            if not collapse:
                kills += 1
        specificity_detail[int(other)] = f"excluding kills collapse on {kills}/{search_seeds} seeds"
        if kills * 2 > search_seeds:
            specificity_ok = False
    log(f"[forensic] specificity: {specificity_detail} -> {'OK' if specificity_ok else 'FAIL'}")

    # Replay counters: search engines also ran the specificity replays, so
    # sum both engine groups (each engine counted its own replays).
    total_replays = sum(e.replays for e in search_engines) + sum(
        e.replays for e in confirm_engines
    )
    total_replayed = sum(e.replayed_steps for e in search_engines) + sum(
        e.replayed_steps for e in confirm_engines
    )

    need = max(1, num_seeds - 1)  # allow one dissenting seed
    names_shard = (
        nec_hits >= need and suf_hits >= need and specificity_ok
    )
    forensic_info = {
        "state_table": {"recoverable": RECOVERABLE, "unrecoverable": UNRECOVERABLE},
        "bracket": [lo, hi],
        "frozen_ranking": [(int(s), float(m), int(n)) for s, m, n in ranked],
        "frozen_margin_x": float(margin),
        "candidate_shard": candidate,
        "necessity": f"{nec_hits}/{num_seeds}",
        "sufficiency": f"{suf_hits}/{num_seeds}",
        "specificity": specificity_detail,
        "substitute_clean": f"{clean_hits}/{num_seeds}",
        "search_votes": stage_b.get("votes", {}),
        "search_seeds": search_seeds,
        "num_seeds": num_seeds,
    }
    if names_shard:
        confidence = (
            f"forensic (probable): bracket [{lo}, {hi}); shard-{candidate} "
            f"necessary {nec_hits}/{num_seeds} seeds, sufficient {suf_hits}/{num_seeds} "
            f"seeds, specificity holds, frozen-weight ranking agrees "
            f"(margin {margin:.1f}x). Exact batch/step UNPROVEN -- "
            f"reconstructed orders, not the true order."
        )
        return EvidenceReport(
            run_dir=run_dir,
            verdict="FORENSIC (probable)",
            confirmed_causal=False,
            fidelity_bit_exact=False,
            change_point=stage_a,
            bracket=[lo, hi],
            trigger_step=None,  # exact step unprovable without recorder
            shard=candidate,
            batch_fingerprint=None,  # exact batch NEVER named in forensic mode
            confirmation={"necessity": True, "sufficiency": True},
            replays=total_replays,
            replayed_steps=total_replayed,
            wall_time_s=time.time() - t0,
            confidence=confidence,
            provenance="forensic",
            notes=list(notes) + [
                "shard named on counterfactual + ranking agreement across "
                "reconstructed orders; this is probable, not causal proof.",
                f"to prove the exact batch, provide: {SETTLING_EVIDENCE}",
            ],
            forensic=forensic_info,
        )
    reasons = []
    if nec_hits < need:
        reasons.append(f"necessity only {nec_hits}/{num_seeds}")
    if suf_hits < need:
        reasons.append(f"sufficiency only {suf_hits}/{num_seeds}")
    if not specificity_ok:
        reasons.append("specificity failed (excluding another shard also kills the collapse)")
    return EvidenceReport(
        run_dir=run_dir,
        verdict="UNPROVEN",
        confirmed_causal=False,
        fidelity_bit_exact=False,
        change_point=stage_a,
        bracket=[lo, hi],
        replays=total_replays,
        replayed_steps=total_replayed,
        wall_time_s=time.time() - t0,
        confidence="unproven: " + "; ".join(reasons),
        provenance="forensic",
        notes=list(notes) + [
            "evidence does not single out one shard; no shard named.",
            f"to settle this, provide: {SETTLING_EVIDENCE}",
        ],
        forensic=forensic_info,
    )
