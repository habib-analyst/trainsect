# DRAFT ONLY, DO NOT POST

## X thread (10 posts, each within 280 characters)

Post numbering is 1/10 through 10/10. Post each as written, in order, as a reply chain. Character counts were checked programmatically (see note at end).

---

**1/10**

Your training run collapsed. Which exact step, shard, and batch did it? Trainsect answers with a causal proof, not an attribution score. It is git bisect for training runs. Thread:

---

**2/10**

The demo: a tiny transformer trains 120 steps on CPU. Shard 3 (steps 60-79) is secretly poisoned with random tokens. Trainsect knows nothing about the injection. It diagnoses it live.

---

**3/10**

Step 1: deterministic replay. A training step is a pure function of model, optimizer, scheduler, batch, RNG. Trainsect replays a segment twice and compares SHA-256 hashes. If they disagree, it aborts. No guessing.

---

**4/10**

Step 2: checkpoint binary search. Predicate: checkpoint is healthy AND replaying forward reproduces the collapse. Monotone, so binary search brackets the trigger to one interval. O(log n) replays.

---

**5/10**

Step 3: counterfactual shard bisect. Replay the trigger interval with each shard excluded. The guilty shard passes two tests: necessity (collapse vanishes without it) and sufficiency (collapse reproduces with only it).

---

**6/10**

Result on the demo: trigger step 60, shard 3, batch fingerprint 639a024b373ce19a, offending sample index 3. Necessity true, sufficiency true. Verdict: CONFIRMED (causal). Fidelity: BIT-EXACT.

---

**7/10**

Cost: 8 replays, 540 replayed steps, 52.2 seconds to diagnose. The manual alternative is hand-bisecting checkpoints and re-running training by hand. Estimate: days of engineer time. No measurement, just the shape of the job.

---

**8/10**

Honest limits: CPU deterministic stacks only. GPU replay is expected to fail the bit-exact check, and Trainsect aborts honestly instead of guessing. Forensic mode (checkpoints alone) never names unproven batches.

---

**9/10**

No TRAK-style scores. No log-scanning guesses. The output is a falsifiable minimal repro you can re-run: replay from step 60, exclude shard 3, watch the collapse vanish.

---

**10/10**

74 tests pass. Recorder overhead 0.7-1.0% step time. Repo: [LINK — goes live after GitHub push]. Try the demo when it lands, and break it. I want the failure cases.

---

## Bisection-animation concept (for the thread's attached video/GIF)

A single 15-20 second looping animation, no audio, no voiceover. Four phases, in order:

1. **The loss curve (0-4 s).** A line chart of eval loss vs step, 0 to 120. The curve sits flat near 0.00 through step 60, then jumps to ~3.87 at step 70 and stays near 4.0. A vertical marker drops at the visible change point, step 80, labeled "collapse seen here". Then a second label appears: "cause is earlier".
2. **Bracket narrowing (4-11 s).** The chart zooms to the step axis. A bracket spans the full checkpoint range, then binary search narrows it in visible jumps: each probe is a dot on the timeline that flashes green (healthy, collapse reproduces ahead) or red (already collapsed). The bracket tightens to [60, 90), then the pinpoint lands on step 60 with a marker labeled "trigger step 60".
3. **Shard named (11-16 s).** The view switches to six shard blocks, shard 0 through shard 5, laid out in step order. Shards 0, 1, 2, 4, 5 fade to grey one by one as counterfactual replays exclude them and the collapse persists. Shard 3 stays lit. Two checkmarks appear beside it: "without shard 3: collapse vanishes" and "only shard 3: collapse reproduces".
4. **The proof (16-20 s).** Two mini loss curves side by side. Left: "exclude shard 3", curve stays flat. Right: "only shard 3", curve jumps at step 70. Final frame: "CONFIRMED (causal). 8 replays, 540 steps, 52.2 s." Hold 2 seconds, loop.

Numbers shown in the animation must match the demo report exactly: change point 80, bracket [60, 90), trigger step 60, shard 3, fingerprint 639a024b373ce19a, 8 replays, 540 replayed steps, 52.2 s. Do not round 52.2 s to 50 s in the animation.
