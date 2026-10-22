# DRAFT ONLY, DO NOT POST

## r/MachineLearning technique write-up

**Title:** [P] Trainsect: git bisect for training runs, causal proof by deterministic replay + counterfactual shard bisection

**Body:**

Training runs collapse, and the standard debugging loop is: stare at the loss curve, guess which data or step did it, re-run, repeat. Influence functions and TRAK give approximate per-sample scores. This tool takes a different route: it replays the run deterministically and proves the culprit by counterfactual replay.

### Method

**1. Deterministic replay as a forensic primitive.** A training step is treated as a pure function of (model, optimizer, scheduler, batch, RNG) state. Checkpoints store all of it with pre-update semantics: the checkpoint at step c holds the state before step c's update, so replay from c applies batches c, c+1, ... exactly once. Before trusting any bisection, Trainsect replays a segment twice and compares SHA-256 parameter hashes. If the two replays disagree, bisection halts and reports the divergence.

**2. Collapse predicate, eval-based.** A checkpoint counts as collapsed only by its eval loss, with the threshold from the lowest quartile of the loss log. This is load-bearing. An early version bisected on train-loss spikes and converged on a benign transient spike (train loss 9.18 at step 120 on a healthy model) and named an innocent shard. The necessity check caught it, and the predicate was fixed to eval-based. Per-step train loss is used only to pinpoint the trigger batch inside an already-established collapse bracket.

**3. Checkpoint binary search.** Define P(c) = "checkpoint c is healthy, AND replaying forward from c reproduces the collapse." P is monotone decreasing in c, so binary search brackets the trigger to one checkpoint interval in O(log n) replays. The health clause matters: without it, replaying from an already-NaN checkpoint would "reproduce" the collapse (NaN in, NaN out) and break monotonicity.

**4. Counterfactual shard bisection.** Replay the trigger interval with each candidate shard's batches replaced by benign substitutes (from a known-clean shard, default shard 0). The surviving shard must pass two tests: necessity (collapse vanishes without it) and sufficiency (collapse reproduces with only it). That pair is the causal claim. Within the offending batch, weights are frozen at the bracket start and per-sample losses are ranked to name the offending sample.

**5. Evidence report.** JSON + human-readable report with minimal-repro commands, so anyone can re-run the proof.

### Validation

Two P0 validation runs (MLP 20-64-64-1, 160 steps, 8 shards, poison shard 2 = unnormalized features x100 + flipped labels, steps 40-59, CPU):

- Run 1 (SGD lr=0.2, persistent NaN collapse): trigger step 40, shard 2, necessity + sufficiency true, CONFIRMED, bit-exact. 8 replays, 760 replayed steps, 22.1 s.
- Run 2 (SGD lr=0.1, recoverable collapse + benign transient spike): trigger step 40, shard 2, necessity + sufficiency true, CONFIRMED, bit-exact. 10 replays, 840 replayed steps, 20.2 s.

Magic-trick demo (nanoGPT-style tiny transformer, 120 steps, shard 3 poisoned with random tokens, steps 60-79, diagnosed with no knowledge of the injection):

- Change point 80, bracket [60, 90), trigger step 60, shard-3, batch fingerprint 639a024b373ce19a, offending sample index 3 (loss 13.74 vs batch mean 12.46), necessity + sufficiency true, CONFIRMED (causal), BIT-EXACT fidelity. 8 replays, 540 replayed steps, 52.2 s.

Forensic mode (checkpoints alone, no recorder data; batch order reconstructed under multiple seeds, shard named only if necessity + sufficiency + specificity all hold and frozen-weight loss ranking agrees):

- P0 validation run stripped of recorder data: FORENSIC (probable), shard-2, bracket [40, 80). Collapse reproduces under 8/8 reconstructed orders.
- Demo run stripped of recorder data: UNPROVEN, no shard named. The collapse is concentration-dependent: 20 consecutive poison batches collapse the model, but the same poison diluted across a shuffled order only wobbles eval to ~1.5 and never crosses 3.0, verified by direct replay. Reconstructed orders cannot reproduce it, so no claim is made.

Test suite: 74 tests, all passing. Recorder overhead: 0.7-1.0% step time (median of repeats), under the under-2% design target. Checkpoint saves are separate and amortized (~10.9 ms/save on the tiny demo model).

### Relation to existing work

git bisect / rldk bisect search the code axis (commits). culprit bisects checkpoint releases across runs in robotics inference stacks. merriv bisects precomputed PASS/BLOCK labels, no replay. NeuralDBG produces ranked causal hypotheses without replay. TRAK / influence functions / TracIn give approximate scores via linearization, projection, or first-order Taylor approximations. Trainsect bisects the training-time axis itself (optimizer steps within one run) and then the data-shard axis, with necessity + sufficiency by replay. The composition is the claim: deterministic optimizer-step replay + checkpoint binary search + counterfactual shard bisection.

### Scope and limitations, stated plainly

- CPU, deterministic, single-process stacks only. GPU replay is expected to fail the bit-exact fidelity check on nondeterministic kernels. The tool aborts honestly rather than guessing. Tolerance-based (non-bit-exact) replay verdicts are future work, not implemented.
- Forensic mode never names unproven batches, never sets a batch fingerprint or trigger step, and its verdict is never CONFIRMED (causal). Without recorder data, exact batch/step identity is unrecoverable.
- Counterfactual substitution uses benign batches from a known-clean shard; the substitute shard must itself be collapse-free, which the necessity check verifies.
- The collapse threshold floor (3.0) is thin for bounded cross-entropy losses. The demo collapse (eval 3.87 vs threshold 3.0, 1.29x) is proven by necessity/sufficiency, not by the margin. The report shows the numbers.
- Distributed training (DDP/FSDP) is not supported: replay across ranks would need per-rank RNG + collective-order capture.
- Checkpoint storage at scale is an open cost: every-K-steps full optimizer state is heavy at LLM scale. Delta-checkpoints are not yet implemented.
- The method proves causation for one trajectory. It does not claim the same shard would collapse a different initialization, seed, or data order. The demo's own forensic result is the evidence: same poison, shuffled order, no collapse.

Repo: [LINK — goes live after GitHub push]

Happy to be challenged on: the monotonicity assumption for self-healing collapses, whether the benign-substitution counterfactual is principled enough, and what a tolerance-based GPU replay verdict should look like.
