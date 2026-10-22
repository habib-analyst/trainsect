# DRAFTS ONLY, DO NOT POST
# LinkedIn cost-math cards: 4 posts. No em-dashes anywhere in this file (commas and periods only). Each post stands alone. Post in order, spaced out, only after the repo link goes live.

---

## Card 1: The cost of not knowing which step broke the run

Training failures are not rare. They are the norm at scale.

Meta's Llama 3 report (The Llama 3 Herd of Models, 2024) records 466 job interruptions across a 54-day pre-training snapshot, 419 of them unexpected. Most were hardware, and automation recovered most of them. But every interruption still costs the same thing: lost compute, lost time, and an engineer asking, what just broke?

Hardware failures have tooling: restart from checkpoint, move on. Data-caused collapses do not. When a run collapses because of a bad shard, there is no alert that names the step, the shard, or the batch. There is a loss curve, and a week of guessing.

Trainsect is built for that gap. It points at a checkpoint directory and finds exactly where and why training collapsed, with a causal proof, not a score. The demo: 8 replays, 540 replayed steps, 52.2 seconds to diagnose, verdict CONFIRMED (causal).

The manual alternative is hand-bisecting checkpoints and re-running training by hand. That is an estimate, not a measurement, but it is the job every team without this tool is doing today.

Repo: [LINK, goes live after GitHub push]

---

## Card 2: What one bad shard costs

The BigScience BLOOM-176B training report, as cited in later failure-analysis work, records that each hardware failure cost an average of 1.5 hours of lost training time, plus 5 to 10 hours of recovery process per failure.

That is the hardware side, and it is the cheap side, because hardware failures are legible. You know what failed. You restart.

A data-caused collapse is the expensive kind. The run does not crash, it silently goes wrong. Eval loss jumps, and nothing in the logs names the cause. The team re-runs segments, re-reads the data pipeline, and argues about whether it was the data, the LR schedule, or a bad seed. The compute cost is the re-runs. The real cost is the engineer-days, and that part is an estimate, not a measurement.

Trainsect answers it with two replays. Exclude the suspect shard, and the collapse vanishes (necessity). Replay with only that shard, and the collapse reproduces (sufficiency). That pair is the proof, and on the demo run it took 8 replays and 52.2 seconds.

Repo: [LINK, goes live after GitHub push]

---

## Card 3: The magic trick, with the receipt

A tiny transformer trains 120 steps on CPU. Shard 3, steps 60 to 79, is secretly poisoned with random tokens. Trainsect knows nothing about the injection.

It diagnoses it live:

- Trigger step: 60, the first poison batch
- Offending shard: shard 3
- Batch fingerprint: 639a024b373ce19a
- Offending sample: index 3 in the batch, loss 13.74 vs batch mean 12.46
- Necessity: true. Sufficiency: true.
- Verdict: CONFIRMED (causal)
- Fidelity: BIT-EXACT replay
- Cost: 8 replays, 540 replayed steps, 52.2 seconds

No attribution score. No ranked guess. A minimal repro you can re-run yourself: replay from step 60, exclude shard 3, watch the collapse vanish.

And the part I am proudest of: when Trainsect cannot prove it, it says so. On the demo run stripped of recorder data, forensic mode returned UNPROVEN and named no shard, because reconstructed data orders could not reproduce the collapse. An honest bracket beats a precise-sounding guess.

Repo: [LINK, goes live after GitHub push]

---

## Card 4: What it does not do

Honest scope, because a debugging tool that overclaims is worse than no tool.

Trainsect works on CPU, deterministic, single-process stacks. GPU replay is expected to fail the bit-exact fidelity check, because GPU kernels are nondeterministic. When that happens, Trainsect aborts honestly instead of guessing.

Forensic mode, checkpoints alone with no recorder data, never names unproven batches. Its verdict is never CONFIRMED (causal).

Distributed training (DDP/FSDP) is not supported. Checkpoint storage at scale is an open cost.

What it does, on the stacks it supports: 74 tests passing, recorder overhead 0.7 to 1.0 percent of step time, and a causal answer to the question every training team asks after a collapse, which step, which shard, which batch, proven by replay.

If you run training jobs and you have a collapse this tool gets wrong, I want that run. Failure cases are the roadmap.

Repo: [LINK, goes live after GitHub push]
