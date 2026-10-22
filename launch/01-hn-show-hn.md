# DRAFT ONLY, DO NOT POST

## Show HN post

**Title:** Show HN: Trainsect, git bisect for training runs (causal proof, not attribution scores)

**Body:**

Your training run collapsed. Which exact optimizer step, data shard, and batch caused it?

Trainsect answers with a causal proof, not an attribution score. It reloads optimizer state at a checkpoint, replays forward deterministically, binary-searches the checkpoint timeline for the trigger step-range, then counterfactually replays with each data shard excluded until only the guilty shard remains. The output is a falsifiable minimal repro you can re-run:

```
collapse: eval-loss change point at step 80
trigger step-range: [60, 90)
offending shard:   shard-3 / batch fp 639a024b373ce19a
repro: trainsect replay --ckpt ckpt_0060 --steps 30 --exclude-shard 3  # collapse vanishes
```

The demo is the magic trick. A nanoGPT-style tiny transformer trains for 120 steps on CPU. Shard 3 (steps 60-79) is poisoned with uniformly random tokens. Trainsect knows nothing about the injection. It diagnoses it live:

- Named shard: shard-3 (the injected poison)
- Trigger step: 60 (the first poison batch)
- Batch fingerprint: 639a024b373ce19a
- Offending sample: index 3 in the batch (loss 13.74 vs batch mean 12.46)
- Necessity + sufficiency: both true
- Verdict: CONFIRMED (causal)
- Fidelity: BIT-EXACT replay
- Cost: 8 replays, 540 replayed steps, 52.2 s to diagnose

The manual alternative is hand-bisecting checkpoints and re-running training by hand. That is an estimate, not a measurement, but the shape of the job is days of engineer time: re-run, squint at logs, guess, repeat. Trainsect replaces the guessing with two replays that prove it: exclude the shard and the collapse vanishes (necessity), replay with only the shard and the collapse reproduces (sufficiency).

How it works:

1. Deterministic replay. A training step is a pure function of (model, optimizer, scheduler, batch, RNG). Checkpoints store all of it with pre-update semantics. Before trusting any bisection, Trainsect replays a segment twice and compares SHA-256 parameter hashes. If they disagree, bisection halts.
2. Collapse predicate, eval-based, never train-spike. A checkpoint counts as collapsed only by its eval loss. An early spike-based version misfired on a benign transient spike and named an innocent shard, so this was fixed and is covered by tests.
3. Checkpoint binary search. Predicate P(c) = "checkpoint c is healthy AND replaying forward reproduces the collapse" is monotone in c, so binary search brackets the trigger to one checkpoint interval in O(log n) replays.
4. Counterfactual shard bisection. Replay the trigger interval with each candidate shard's batches replaced by benign substitutes. The surviving shard must pass necessity AND sufficiency. That pair is the causal proof.
5. Evidence report. JSON + human-readable report with minimal-repro commands, so anyone can re-run the proof.

What it is not: TRAK / influence functions / TracIn assign approximate scores via linearization or Taylor approximations. git bisect finds the bad commit. Trainsect bisects the training-time axis itself, optimizer steps within one run, and proves the culprit by re-running it twice.

Honest scope and limitations:

- CPU, deterministic, single-process stacks only. GPU replay is expected to fail the bit-exact fidelity check on nondeterministic kernels, and the tool aborts honestly rather than guessing.
- Forensic mode (checkpoints alone, no recorder data) never names unproven batches. On one validation run it returned FORENSIC (probable); on the demo run stripped of recorder data it returned UNPROVEN, because the collapse is concentration-dependent and reconstructed orders could not reproduce it. An honest bracket beats a precise-sounding guess.
- Distributed training (DDP/FSDP) is not supported.
- Checkpoint storage at scale is an open cost: every-K-steps full optimizer state is heavy at LLM scale.
- The collapse threshold floor (3.0) is thin for bounded cross-entropy losses. The demo collapse (eval 3.87 vs threshold 3.0, 1.29x) is proven by necessity/sufficiency, not by the margin. The report shows the numbers.

Measured numbers (CPU, this repo): 74 tests, all passing. Recorder overhead 0.7-1.0% step time, under the under-2% design target. Two P0 validation runs (MLP, 160 steps) both reproduced: trigger step 40, shard 2, necessity + sufficiency true, CONFIRMED, bit-exact.

Repo: [LINK — goes live after GitHub push]

Try the demo when it lands, and break it. I want the failure cases: stacks where replay is not bit-exact, collapses where the predicate misfires, forensic runs where the verdict should be UNPROVEN but is not.

---

## Terminal-recording demo plan (for the Show HN post)

Record one continuous terminal session, no cuts, no audio, no editing of output. Suggested tool: asciinema, then embed the player or a GIF export. Target length: 60-90 seconds.

Step by step, what the recording shows:

1. A clean terminal. The prompt shows the repo directory. Nothing pre-staged, no hidden state.
2. Type: `trainsect demo --outdir demos` and press Enter.
3. Training scrolls: 120 steps on CPU. Visible lines include step 0 (shard 0, train loss 4.38), step 40 (shard 2, train loss 0.0027, healthy), step 60 (shard 3, train loss 12.46, the first poison batch), and the eval loss log showing eval flat near 0.00 through step 60, then 3.87 at step 70.
4. The diagnose stage runs live. Visible stage lines, in order:
   - `[fidelity] replay 7622e7196398a345 vs original 7622e7196398a345 -> BIT-EXACT`
   - `[Stage A] change point: ... change_point_step 80, threshold 3.0`
   - `[Stage B] checkpoint binary search -> bracket (60, 90)`
   - `[Stage C] pinpoint: first train spike at step 60`
   - `[Stage D] offending shard = shard-3`
   - `[Stage E] necessity=True sufficiency=True`
5. The evidence report prints: verdict CONFIRMED (causal), trigger step 60, shard-3, batch fingerprint 639a024b373ce19a, offending sample index 3, 8 replays, 540 replayed steps, 52.2 s wall time.
6. Then, still in the same uncut recording, type the two minimal-repro commands from the report:
   - `trainsect replay ... --from-step 60 --steps 30 --exclude-shard 3` and show the collapse vanishing (necessity).
   - `trainsect replay ... --from-step 60 --steps 30 --only-shard 3` and show the collapse reproducing (sufficiency).
7. End on the report's verdict line. No voiceover, no captions added in post. If a run is re-recorded for timing, say so in the post text; never splice a clean output onto a different run.

What the recording must NOT show: a pre-computed output file catted to look live, a sped-up timer that hides the real 52.2 s wall time, or any claim beyond what the report prints. If the live run takes longer on the recording machine, show the real time.
