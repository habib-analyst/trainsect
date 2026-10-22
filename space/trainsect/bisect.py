"""Bisection: checkpoint binary search + counterfactual shard bisect.

Predicate (P0 methodology §3.1):

    P(c) = checkpoint c is healthy AND replaying c -> B reproduces the
           collapse (eval-based).

P is monotone decreasing in checkpoint index: ``T, T, ..., T, F, ..., F``.
Binary search finds the first ``F``, bracketing the trigger to
``[c_j, c_{j+1})`` in O(log n) replays.

Shard bisect (P0 §4.2) then splits the candidate shards in the bracket:
replay the bracket with one half's batches replaced by benign substitutes;
if the collapse persists, the trigger is in the kept half, else in the
replaced half. The causal evidence is the counterfactual pair
(:func:`confirm`): excluding the named shard kills the collapse
(necessity) and keeping only it reproduces the collapse (sufficiency).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .collapse import has_collapse, has_train_spike, is_healthy
from .dataloader import IndexedDataLoader
from .replay import ReplayEngine


# ---------------------------------------------------------------- predicate
def make_predicate(
    engine: ReplayEngine,
    end_step: int,
    eval_threshold: float,
    loss_log=None,
    baseline: Optional[float] = None,
) -> Callable[[int], Tuple[bool, Optional[int]]]:
    """Build P(c): checkpoint c is NOT already collapsed, AND replaying
    c -> end reproduces the collapse (eval-based).

    "Already collapsed" means the checkpoint's own eval is above threshold
    AND the model had previously reached near-baseline health (from the
    loss log). The second clause is load-bearing: an *untrained*
    checkpoint (eval at chance, e.g. 4.3 for a 64-vocab LM) is not
    "already collapsed" -- it never was healthy. Without it, the binary
    search aborts on any run whose initial eval exceeds the threshold.
    """

    def already_collapsed(ckpt_step: int, health: float) -> bool:
        if is_healthy(health, eval_threshold):
            return False
        if loss_log is None or baseline is None:
            return True  # legacy behaviour: above threshold == collapsed
        near_baseline = max(2 * baseline, baseline + 0.5)
        for step, loss in loss_log:
            if int(step) < ckpt_step and loss <= near_baseline:
                return True  # was healthy before, now above threshold
        return False  # never healthy before this checkpoint: untrained

    def predicate(ckpt_step: int) -> Tuple[bool, Optional[int]]:
        health = engine.checkpoint_eval_loss(ckpt_step)
        if already_collapsed(ckpt_step, health):
            return False, None
        _, evals, _ = engine.replay(ckpt_step, end_step)
        return has_collapse(evals, eval_threshold)

    return predicate


def bisect_checkpoints(
    ckpts: Sequence[int],
    predicate: Callable[[int], Tuple[bool, Optional[int]]],
    end_step: int,
) -> Dict[str, object]:
    """Binary-search the first checkpoint with P(c) == False.

    Returns ``{"ok": True, "bracket": (lo, hi)}`` or
    ``{"ok": False, "reason": ...}`` when the collapse does not reproduce
    from the earliest checkpoint (trigger predates the range or is
    non-reproducible) -- reported honestly, never papered over.
    """
    ckpts = list(ckpts)
    p0, _ = predicate(ckpts[0])
    if not p0:
        return {
            "ok": False,
            "reason": "collapse does NOT reproduce from earliest checkpoint "
            "- trigger predates range or is non-reproducible",
        }
    lo, hi = 0, len(ckpts) - 1
    p_hi, _ = predicate(ckpts[hi])
    if p_hi:
        return {
            "ok": True,
            "bracket": (ckpts[hi], end_step),
            "note": "collapse reproduces even from the last checkpoint - "
            "trigger inside the final interval",
        }
    while hi - lo > 1:
        mid = (lo + hi) // 2
        p, _ = predicate(ckpts[mid])
        if p:
            lo = mid
        else:
            hi = mid
    return {"ok": True, "bracket": (ckpts[lo], ckpts[hi])}


def pinpoint_trigger(
    engine: ReplayEngine,
    lo: int,
    hi: int,
    spike_threshold: float,
    eval_threshold: float,
) -> Dict[str, object]:
    """One replay from the bracket start: first train spike pinpoints the
    trigger batch; the first eval crossing confirms it is the collapse."""
    losses, evals, _ = engine.replay(lo, hi)
    spike, s_star = has_train_spike(losses, spike_threshold)
    collapse, s_eval = has_collapse(evals, eval_threshold)
    return {
        "trigger_step": s_star,
        "train_spike_found": spike,
        "train_loss_at_trigger": losses.get(s_star) if s_star is not None else None,
        "eval_crossing_step": s_eval,
        "collapse_in_bracket": collapse,
    }


def shard_bisect(
    engine: ReplayEngine,
    loader: IndexedDataLoader,
    lo: int,
    hi: int,
    eval_threshold: float,
    clean_shard: int = 0,
) -> int:
    """Counterfactual shard bisection over the trigger bracket."""
    candidates = loader.shards_in_range(lo, hi)
    benign = loader.benign_batches(clean_shard)
    while len(candidates) > 1:
        half = len(candidates) // 2
        h2 = set(candidates[half:])

        def replace(step, x, y, shard, _h2=h2):
            if shard in _h2:
                sub = benign[step % len(benign)]
                return sub.x, sub.y
            return x, y

        _, evals, _ = engine.replay(lo, hi, replace=replace)
        collapse, _ = has_collapse(evals, eval_threshold)
        if collapse:
            candidates = candidates[:half]  # trigger in first half
        else:
            candidates = candidates[half:]  # trigger in second half
    return candidates[0]


def confirm(
    engine: ReplayEngine,
    loader: IndexedDataLoader,
    lo: int,
    hi: int,
    shard: int,
    eval_threshold: float,
    clean_shard: int = 0,
) -> Dict[str, bool]:
    """Necessity + sufficiency by counterfactual replay.

    - necessity: replay with ONLY this shard's batches replaced -> collapse
      must vanish (shard is necessary for the collapse).
    - sufficiency: replay with every OTHER shard's batches replaced ->
      collapse must reproduce (shard alone suffices).

    If either fails, the verdict is UNCONFIRMED and must not be reported
    as causal.
    """
    benign = loader.benign_batches(clean_shard)

    def excl(step, x, y, s):
        if s == shard:
            sub = benign[step % len(benign)]
            return sub.x, sub.y
        return x, y

    def only(step, x, y, s):
        if s != shard:
            sub = benign[step % len(benign)]
            return sub.x, sub.y
        return x, y

    _, evals_ex, _ = engine.replay(lo, hi, replace=excl)
    collapse_ex, _ = has_collapse(evals_ex, eval_threshold)
    _, evals_on, _ = engine.replay(lo, hi, replace=only)
    collapse_on, _ = has_collapse(evals_on, eval_threshold)
    return {"necessity": not collapse_ex, "sufficiency": collapse_on}
