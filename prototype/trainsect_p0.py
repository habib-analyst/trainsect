"""Trainsect P0 prototype: checkpoint binary search + shard bisect via replay.

Implements the methodology in docs/P0-methodology.md at minimal scale:
  Stage A: change-point detection on the eval-loss log (no replay).
  Stage B: binary search over checkpoints with the causal predicate
           P(c) = "a loss spike reproduces on replay c -> B".
  Stage C: pinpoint the exact trigger step with one replay from the bracket start.
  Stage D: counterfactual shard bisect (replace shard halves with benign batches).
  Stage E: necessity + sufficiency confirmation, batch fingerprint, offending sample.

Usage: python trainsect_p0.py <run_dir>
"""
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

DIM = 20
HIDDEN = 64
LR = 0.05


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x)


class Run:
    def __init__(self, run_dir):
        self.dir = run_dir
        self.batches = torch.load(os.path.join(run_dir, "batches.pt"), weights_only=False)
        self.eval = torch.load(os.path.join(run_dir, "eval.pt"), weights_only=False)
        with open(os.path.join(run_dir, "sidecar.jsonl")) as f:
            self.sidecar = [json.loads(l) for l in f]
        with open(os.path.join(run_dir, "loss_log.json")) as f:
            self.loss_log = json.load(f)
        self.ckpts = sorted(
            int(fn[5:9]) for fn in os.listdir(run_dir)
            if fn.startswith("ckpt_") and fn.endswith(".pt")
        )
        self.total_steps = max(self.ckpts)
        # benign substitute batches: shard 0 (known-clean) batches, cycled
        self.benign = [b for b in self.batches if b[2] == 0]
        self.replays = 0
        self.replayed_steps = 0

    def load_state(self, step):
        ck = torch.load(os.path.join(self.dir, f"ckpt_{step:04d}.pt"), weights_only=False)
        model = MLP()
        model.load_state_dict(ck["model"])
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        opt.load_state_dict(ck["optimizer"])
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng_torch"])
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        return model, opt, sched

    def param_hash(self, model):
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.detach().numpy().tobytes())
        return h.hexdigest()[:16]

    def replay(self, from_step, to_step, replace=None):
        """Replay optimizer steps from_step..to_step-1. `replace(step,X,y,shard)->(X,y)`.

        Returns (per-step train losses, eval losses sampled every 10 steps + at end,
        end param hash). Deterministic (CPU, seeded).
        """
        self.replays += 1
        model, opt, sched = self.load_state(from_step)
        crit = nn.BCEWithLogitsLoss()
        losses = {}
        evals = {}
        Xe, ye = self.eval["Xe"], self.eval["ye"]
        for step in range(from_step, to_step):
            X, y, shard = self.batches[step]
            if replace is not None:
                X, y = replace(step, X, y, shard)
            opt.zero_grad()
            loss = crit(model(X), y)
            losses[step] = loss.item()
            loss.backward()
            opt.step()
            sched.step()
            self.replayed_steps += 1
            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    evals[step] = nn.functional.binary_cross_entropy_with_logits(
                        model(Xe), ye).item()
                model.train()
        # eval at the final replayed state
        model.eval()
        with torch.no_grad():
            evals[to_step] = nn.functional.binary_cross_entropy_with_logits(
                model(Xe), ye).item()
        return losses, evals, self.param_hash(model)

    # ---------------- Stage A: change-point detection ----------------
    def detect_collapse(self):
        losses = [l for _, l in self.loss_log]
        baseline = float(np.median(losses[:3]))
        mad = float(np.median([abs(l - baseline) for l in losses[:3]])) or 1e-6
        # Collapse = ORDER-OF-MAGNITUDE explosion, not healthy drift. A 5x-MAD rule
        # misfires when clean training itself drifts (LR=0.1 run: healthy eval
        # drifted to ~3.0 on clean data, collapse hit 234). 10x baseline separates them.
        thresh = max(10 * baseline, baseline + 10 * mad, 3.0)
        flagged = 0
        for step, l in self.loss_log:
            if l > thresh or not np.isfinite(l):
                flagged += 1
                if flagged >= 2:
                    return {"change_point_step": step, "threshold": thresh,
                            "baseline": baseline, "eval_loss_at_cp": l}
            else:
                flagged = 0
        return {"change_point_step": None, "threshold": thresh, "baseline": baseline}

    # ---------------- Stage B: checkpoint binary search ----------------
    def spike_threshold(self):
        early = [r["train_loss"] for r in self.sidecar[:20]]
        return max(10 * float(np.median(early)), 2.0)

    def has_spike(self, losses, thresh):
        for s, l in losses.items():
            if (not np.isfinite(l)) or l > thresh:
                return True, s
        return False, None

    def checkpoint_eval_loss(self, step):
        """Health of the checkpoint state itself: eval loss of the loaded weights."""
        model, _, _ = self.load_state(step)
        Xe, ye = self.eval["Xe"], self.eval["ye"]
        model.eval()
        with torch.no_grad():
            return nn.functional.binary_cross_entropy_with_logits(
                model(Xe), ye).item()

    def has_collapse(self, evals, eval_thresh):
        """Collapse = eval loss crosses the health threshold at ANY point during
        replay (or goes non-finite). Eval-based, NOT train-spike-based: transient
        train-loss spikes on a healthy model are not collapses (validated: LR=0.1
        run had a benign train spike at step 120 with healthy eval)."""
        for s, l in evals.items():
            if (not np.isfinite(l)) or l > eval_thresh:
                return True, s
        return False, None

    def predicate(self, ckpt_step, thresh, eval_thresh):
        """Causal predicate: checkpoint state is HEALTHY, but replaying forward
        from it reproduces the COLLAPSE (eval-based). (Health check is what keeps
        the predicate monotone for persistent collapses: an already-NaN checkpoint
        must not count as 'trigger ahead of it'.)"""
        el = self.checkpoint_eval_loss(ckpt_step)
        if (not np.isfinite(el)) or el > eval_thresh:
            return False, None  # already collapsed at this checkpoint
        _, evals, _ = self.replay(ckpt_step, self.total_steps)
        return self.has_collapse(evals, eval_thresh)

    def bisect_checkpoints(self, thresh, eval_thresh):
        ckpts = self.ckpts
        # endpoint probes first (honesty: collapse must reproduce from earliest ckpt)
        p0, _ = self.predicate(ckpts[0], thresh, eval_thresh)
        if not p0:
            return {"ok": False, "reason": "collapse does NOT reproduce from earliest checkpoint "
                                           "- trigger predates range or is non-reproducible"}
        lo, hi = 0, len(ckpts) - 1  # invariant: P(ckpts[lo])=True; find first False
        # ensure P(ckpts[hi]) known; if True for all, bracket is (last, B]
        p_hi, _ = self.predicate(ckpts[hi], thresh, eval_thresh)
        if p_hi:
            return {"ok": True, "bracket": (ckpts[hi], self.total_steps),
                    "note": "spike reproduces even from the last checkpoint - "
                            "trigger inside the final interval"}
        while hi - lo > 1:
            mid = (lo + hi) // 2
            p, _ = self.predicate(ckpts[mid], thresh, eval_thresh)
            if p:
                lo = mid
            else:
                hi = mid
        return {"ok": True, "bracket": (ckpts[lo], ckpts[hi])}

    # ---------------- Stage D: shard bisect ----------------
    def shard_bisect(self, lo, hi, thresh, eval_thresh):
        steps = list(range(lo, hi))
        shards = sorted({self.batches[s][2] for s in steps})
        candidates = shards[:]

        def benign_cycle(step):
            # deterministic benign substitute: shard-0 batch at same position
            return self.benign[step % len(self.benign)]

        while len(candidates) > 1:
            half = len(candidates) // 2
            h2 = set(candidates[half:])  # replace second half with benign batches

            def replace(step, X, y, shard, _h2=h2):
                if shard in _h2:
                    bx, by, _ = benign_cycle(step)
                    return bx, by
                return X, y

            _, evals, _ = self.replay(lo, hi, replace=replace)
            collapse, _ = self.has_collapse(evals, eval_thresh)
            if collapse:
                candidates = candidates[:half]  # trigger in first half
            else:
                candidates = candidates[half:]  # trigger in second half
        return candidates[0]

    def confirm(self, lo, hi, shard, thresh, eval_thresh):
        """Necessity (exclude shard -> no collapse) + sufficiency (only shard -> collapse).
        Both use the eval-based collapse predicate. If necessity fails, the verdict
        is UNCONFIRMED and must not be reported as causal (validated: the LR=0.1
        run's spurious step-120 spike failed necessity, correctly)."""
        def excl(step, X, y, s):
            if s == shard:
                bx, by, _ = self.benign[step % len(self.benign)]
                return bx, by
            return X, y

        def only(step, X, y, s):
            if s != shard:
                bx, by, _ = self.benign[step % len(self.benign)]
                return bx, by
            return X, y

        _, evals_ex, _ = self.replay(lo, hi, replace=excl)
        collapse_ex, _ = self.has_collapse(evals_ex, eval_thresh)
        _, evals_on, _ = self.replay(lo, hi, replace=only)
        collapse_on, _ = self.has_collapse(evals_on, eval_thresh)
        return {"necessity": not collapse_ex, "sufficiency": collapse_on}

    def offending_sample(self, trigger_step):
        """Within the trigger batch, rank per-sample loss under frozen weights at trigger step."""
        model, _, _ = self.load_state(trigger_step)
        X, y, shard = self.batches[trigger_step]
        model.eval()
        with torch.no_grad():
            per = nn.functional.binary_cross_entropy_with_logits(
                model(X), y, reduction="none").squeeze(1).numpy()
        idx = int(np.argmax(per))
        return {"sample_index_in_batch": idx, "sample_loss": float(per[idx]),
                "batch_mean_loss": float(per.mean())}


def main():
    run_dir = sys.argv[1]
    t0 = time.time()
    run = Run(run_dir)

    # Fidelity check: replay(0 -> total) must reproduce the ORIGINAL final state
    # exactly (hash vs ckpt_total), and two replays must agree with each other.
    _, _, h_replay = run.replay(0, run.total_steps)
    _, _, h_replay2 = run.replay(0, run.total_steps)
    ck_final = torch.load(os.path.join(run_dir, f"ckpt_{run.total_steps:04d}.pt"),
                          weights_only=False)
    m_final = MLP()
    m_final.load_state_dict(ck_final["model"])
    h_orig = run.param_hash(m_final)
    fidelity = (h_replay == h_orig) and (h_replay == h_replay2)
    print(f"[fidelity] replay hash {h_replay} vs original-final {h_orig} "
          f"vs replay-again {h_replay2} -> {'BIT-EXACT' if fidelity else 'NON-DETERMINISTIC'}")

    stage_a = run.detect_collapse()
    print(f"[Stage A] change point: {stage_a}")

    thresh = run.spike_threshold()
    eval_thresh = stage_a["threshold"]
    print(f"[spike threshold] train-loss spike threshold = {thresh:.3f}, "
          f"eval-loss health threshold = {eval_thresh:.3f}")

    stage_b = run.bisect_checkpoints(thresh, eval_thresh)
    print(f"[Stage B] checkpoint binary search -> {stage_b}")
    if not stage_b["ok"]:
        print("ABORT:", stage_b["reason"])
        return

    lo, hi = stage_b["bracket"]
    # Stage C: pinpoint exact trigger step with ONE replay from bracket start.
    # Train-loss spike pinpoints the batch; eval crossing confirms it is the collapse.
    losses, evals, _ = run.replay(lo, hi)
    spike, s_star = run.has_spike(losses, thresh)
    collapse, s_eval = run.has_collapse(evals, eval_thresh)
    print(f"[Stage C] pinpoint: first train spike at step {s_star} "
          f"(loss={losses[s_star]:.3f}); first eval crossing at {s_eval} "
          f"inside bracket [{lo}, {hi})")

    # Stage D: shard bisect (eval-based collapse predicate)
    shard = run.shard_bisect(lo, hi, thresh, eval_thresh)
    print(f"[Stage D] offending shard = shard-{shard}")

    # Stage E: confirm + name batch + sample
    conf = run.confirm(lo, hi, shard, thresh, eval_thresh)
    fp = run.sidecar[s_star]["batch_fp"]
    sample = run.offending_sample(s_star)
    print(f"[Stage E] necessity={conf['necessity']} sufficiency={conf['sufficiency']}")
    if not (conf["necessity"] and conf["sufficiency"]):
        print("[Stage E] VERDICT UNCONFIRMED - not reported as causal")
    print(f"[Stage E] trigger batch: step {s_star}, shard {shard}, fingerprint {fp}")
    print(f"[Stage E] offending sample: {sample}")

    wall = time.time() - t0
    confirmed = bool(conf["necessity"] and conf["sufficiency"])
    print(f"\n=== TRAINSECT P0 RESULT ===")
    print(f"verdict:               {'CONFIRMED (causal)' if confirmed else 'UNCONFIRMED'}")
    print(f"steps searched:        {run.total_steps}")
    print(f"replays run:           {run.replays}")
    print(f"replayed steps total:  {run.replayed_steps}")
    print(f"shard identified:      shard-{shard}")
    print(f"trigger step:          {s_star}")
    print(f"batch fingerprint:     {fp}")
    print(f"wall time:             {wall:.1f}s")
    print(f"minimal repro:         trainsect replay --ckpt ckpt_{lo:04d} "
          f"--steps {hi-lo} --exclude-shard {shard}   # collapse vanishes")
    print(f"                       trainsect replay --ckpt ckpt_{lo:04d} "
          f"--steps {hi-lo} --only-shard {shard}     # collapse reproduces")

    with open(os.path.join(run_dir, "trainsect_result.json"), "w") as f:
        json.dump({
            "confirmed_causal": confirmed,
            "fidelity_bit_exact": fidelity,
            "change_point": stage_a,
            "spike_threshold": thresh,
            "bracket": [lo, hi],
            "trigger_step": s_star,
            "shard": shard,
            "batch_fp": fp,
            "offending_sample": sample,
            "confirmation": conf,
            "replays": run.replays,
            "replayed_steps": run.replayed_steps,
            "wall_time_s": wall,
        }, f, indent=2)


if __name__ == "__main__":
    main()
