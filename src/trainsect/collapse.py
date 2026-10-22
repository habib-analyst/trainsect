"""Collapse detection: eval-based, never train-spike-based.

The collapse predicate is **eval loss crossing a health threshold** during
replay. This distinction is load-bearing (P0 methodology §2.2, validated by
the P0 bug where a benign transient train-loss spike at step 120 on a
healthy model was mistaken for a collapse): transient train-loss spikes are
not collapses. Train loss is used only to pinpoint the trigger batch *inside*
an already-established collapse bracket.

Threshold calibration (P0 §9, bug #3): ``max(10 * baseline,
baseline + 10 * MAD, floor)``. The 5x-MAD rule misfired because healthy
training itself drifts; the 10x-baseline margin separates real collapses
(34x above threshold in P0 run 1) from healthy drift (0.45x).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def collapse_threshold(
    loss_log: Sequence[Sequence[float]],
    floor: float = 3.0,
    baseline_n: int = 3,
) -> Tuple[float, float, float]:
    """Derive the eval-collapse threshold from the healthy part of the log.

    Baseline = median of the *lowest quartile* of finite eval losses (the
    model's healthy level), not the first few points: a model still in
    warmup (demo: eval 4.3 at step 0, healthy 0.002 by step 40) would
    otherwise inflate the baseline and mask a real collapse. MAD is
    computed within that same lowest quartile.
    """
    finite = sorted(float(l) for _, l in loss_log if math.isfinite(l))
    if not finite:
        return floor, 0.0, 0.0
    k = max(1, len(finite) // 4) if len(finite) >= 4 else len(finite)
    low = finite[:k]
    baseline = float(np.median(low))
    mad = float(np.median([abs(l - baseline) for l in low])) if low else 0.0
    thresh = max(10 * baseline, baseline + 10 * mad, floor)
    return thresh, baseline, mad


def detect_change_point(
    loss_log: Sequence[Sequence[float]],
    threshold: Optional[float] = None,
    sustained: int = 2,
) -> Dict[str, object]:
    """Stage A: first step where eval loss exceeds threshold ``sustained``
    times in a row (or goes non-finite). Symptom locator, not a cause.

    Warmup skip: steps before the model first reaches near-baseline eval
    (``max(2*baseline, baseline + 0.5)``) are warmup, not collapse -- an
    untrained model's initial loss is high by definition.
    """
    if threshold is None:
        threshold, baseline, mad = collapse_threshold(loss_log)
    else:
        _t, baseline, mad = collapse_threshold(loss_log)
    near_baseline = max(2 * baseline, baseline + 0.5)
    warmup_end = None
    for step, loss in loss_log:
        if math.isfinite(loss) and loss <= near_baseline:
            warmup_end = int(step)
            break
    flagged = 0
    for step, loss in loss_log:
        if warmup_end is not None and int(step) < warmup_end:
            continue  # warmup transient, not a collapse
        if (not math.isfinite(loss)) or loss > threshold:
            flagged += 1
            if flagged >= sustained:
                return {
                    "change_point_step": int(step),
                    "threshold": threshold,
                    "baseline": baseline,
                    "eval_loss_at_cp": float(loss),
                }
        else:
            flagged = 0
    return {
        "change_point_step": None,
        "threshold": threshold,
        "baseline": baseline,
    }


def is_healthy(eval_loss_value: float, threshold: float) -> bool:
    """A checkpoint state is healthy iff its own eval loss is finite and
    below threshold. An already-collapsed checkpoint must NOT count as
    'trigger ahead of it' (that would break predicate monotonicity)."""
    return math.isfinite(eval_loss_value) and eval_loss_value <= threshold


def has_collapse(
    evals: Dict[int, float], threshold: float
) -> Tuple[bool, Optional[int]]:
    """Collapse = eval loss crosses the threshold at ANY point during replay
    (or goes non-finite). Train-loss spikes are deliberately NOT consulted.
    """
    for step, loss in evals.items():
        if (not math.isfinite(loss)) or loss > threshold:
            return True, step
    return False, None


def has_train_spike(
    losses: Dict[int, float], spike_threshold: float
) -> Tuple[bool, Optional[int]]:
    """First train-loss spike (non-finite or above threshold). Used ONLY to
    pinpoint the trigger batch inside an already-established collapse
    bracket -- never as the collapse predicate itself."""
    for step, loss in losses.items():
        if (not math.isfinite(loss)) or loss > spike_threshold:
            return True, step
    return False, None


def train_spike_threshold(sidecar: Sequence[dict], n: int = 20) -> float:
    """Spike threshold from the first ``n`` recorded train losses.

    Uses the lowest quartile of those early losses as the healthy level:
    the first steps include warmup (demo: train loss 4.38 at step 0), and a
    plain median would set the threshold above the real trigger spike.
    """
    early = sorted(float(r["train_loss"]) for r in sidecar[:n] if r.get("train_loss") is not None)
    if not early:
        return float("inf")
    k = max(1, len(early) // 4)
    low = early[:k]
    return max(10 * float(np.median(low)), 2.0)
