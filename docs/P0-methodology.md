# Trainsect — P0 Methodology

**Status:** P0 research + prototype. Read-only. No publishing, no posting, no papers.
**Date:** 2026-10-14
**Question answered:** "Which exact optimizer step — and which exact data batch — caused this training collapse? Prove it by re-running it."

---

## 0. The novel claim (crisp)

Every existing training-debugger bisects the **wrong axis**:

- **Code axis** — `git bisect` (and rldk's `bisect`) finds the bad *commit* by re-running training per commit.
- **Release axis** — culprit bisects *checkpoint releases* across runs (robotics inference stacks).
- **Approximate axis** — TRAK / influence functions / TracIn assign *scores* to samples via linearization, projection, or first-order Taylor approximations.

Trainsect bisects the **training-time axis itself**: it treats the optimizer trajectory as a deterministic, replayable computation, reloads optimizer state at step *t*, replays forward, and binary-searches the exact step-range where the collapse triggers. Then it bisects the data-shard axis with **counterfactual replay** (collapse disappears iff shard S is excluded) to name the exact shard, batch, and sample.

The output is not a ranked hypothesis. It is a falsifiable, re-runnable minimal repro:

```
collapse: eval-loss change point at step ~4,132
trigger step-range: [4,100, 4,200)      # proven by replay, not by log-scanning
offending shard:   shard-7 / batch fp 9f3c…
repro: trainsect replay --ckpt ckpt-4100 --exclude-shard 7   # collapse vanishes
```

No existing tool combines (a) deterministic optimizer-step replay across checkpoints, (b) binary search over the checkpoint timeline, and (c) causal data-shard bisection. See §8 for the gap-verification evidence.

---

## 1. Deterministic replay

### 1.1 The training step as a pure function

A training step is deterministic given complete state:

```
θ_{t+1}, opt_{t+1}, rng_{t+1} = STEP(θ_t, opt_t, batch_t, rng_t)
```

To reproduce the trajectory from checkpoint at step *t* to step *t+k*, the following state must be captured at *t*:

| State | Source | Why it matters |
|---|---|---|
| `model.state_dict()` | checkpoint | θ_t |
| `optimizer.state_dict()` | checkpoint | momentum buffers, Adam m/v, step counts |
| LR scheduler state | checkpoint | lr at step t (non-restartable schedules diverge fast) |
| Dataloader cursor | recorder | epoch, global batch index, sampler generator state |
| Batch sequence *t..t+k−1* | recorder / dataset | the exact batches, in order |
| RNG states | recorder | `torch.get_rng_state()`, `numpy`, `random`, CUDA RNG if GPU |
| DataLoader worker seeds | recorder | per-worker `torch.Generator` seeds (num_workers > 0) |
| Env determinism flags | recorder | `torch.use_deterministic_algorithms`, `cudnn.deterministic`, `cudnn.benchmark` |

### 1.2 Replay fidelity check

Before trusting any bisection, Trainsect replays *t → t+k* **twice** and compares parameter hashes (SHA-256 over the flattened state dict). If the two replays disagree, replay is not deterministic on this stack — bisection halts and reports the divergence instead of a false answer. On CPU with deterministic flags this check passes bit-exactly; on GPU with nondeterministic kernels it is expected to fail, and the failure itself is reported (cf. the well-documented GPU fp-nondeterminism problem — same seeds, divergent runs by step 1 on parallel hardware).

### 1.3 What replay is NOT

- It is not re-running the whole training job (rldk's `replay`).
- It is not resuming training to continue (HF Accelerate `load_state`).
- It is a **forensic primitive**: load exact mid-run state, advance *k* steps under identical inputs, observe. Cheap because *k* is small (the binary search keeps it that way).

---

## 2. Collapse detection

Two stages. Stage A is cheap and runs on logs; Stage B is causal and runs on replay.

### 2.1 Stage A — change-point detection on the eval-loss curve

Input: logged eval loss series L(s) at steps s.

1. **Metric predicate** (fast, interpretable): collapse flag C(s) = true if any of:
   - L(s) is NaN / ±Inf;
   - L(s) > median(L) + 5·MAD sustained for ≥ m eval intervals;
   - L(s)/L(s−1) > 3 (single-interval explosion).
2. **Statistical change-point**: binary segmentation (CUSUM-style) on L(s) to find the first index τ where the segment mean shifts by more than δ = 3·σ_baseline. This yields a coarse collapse window [τ_lo, τ_hi] and seeds the binary search with endpoints, without any replay.

The change point is a *symptom locator*, not a cause. Stage B finds the cause.

### 2.2 Stage B — causal confirmation by replay

The predicate used in bisection is not "loss was high in the log" but:

> **P(c)**: checkpoint *c* is healthy (its own eval loss is below threshold), AND replaying forward c → B reproduces the collapse — eval loss crosses the collapse threshold at some point during the replay.

The predicate is **eval-based, not train-loss-spike-based**. This distinction was forced by validation (§10): a healthy model can show transient train-loss spikes (LR=0.1 run: benign spike at step 120, eval untouched) that a spike-based predicate mistakes for collapses. Eval loss is the collapse signal; per-step train loss is used only for pinpointing the trigger batch *inside* an already-established collapse bracket.

This is what makes the answer causal: the collapse is re-created on demand from state, not inferred from a curve.

---

## 3. Checkpoint binary search

### 3.1 Setup

Checkpoints at steps {c_0 < c_1 < … < c_n}, known-bad point B (the change point from §2.1, B > c_n or B = c_n).

Checkpoints use **pre-update semantics**: ckpt at step c holds the state *before* step c's optimizer update, so replay from c applies batches c, c+1, … exactly once.

Define P(c_i) = "checkpoint c_i is healthy, and collapse reproduces on replay c_i → B". The trigger step τ lies in some interval [c_j, c_{j+1}).

**Monotonicity:** replay from any c_i ≤ τ passes through the trigger, so P(c_i) = true for c_i ≤ τ and false for c_i > τ. P is monotone decreasing in i: `T, T, …, T, F, …, F`. Binary search finds the transition — the first F at index j+1 — so the trigger is bracketed to **[c_j, c_{j+1})**. Cost: **O(log n) replays** instead of O(n).

The health clause in P is load-bearing for persistent collapses: without it, replaying from an already-NaN checkpoint would "reproduce" the collapse (NaN in, NaN out) and break monotonicity. A checkpoint whose own eval loss is already above threshold counts as *already collapsed*, not as *trigger ahead of it*.

### 3.2 Algorithm

```
1. Endpoint probes (honesty first):
     if P(c_0) is False:  collapse predates the checkpoint range or is non-reproducible → STOP, report honestly.
     if P(c_n) is True but trigger not in range: extend range or report.
2. Binary search over i in [0, n] for first index with P(c_i) = False → j+1.
3. If P is non-monotone at any probe (flaky replay), mark verdict "unreliable at c_i",
   retry with a second replay; persistent non-monotonicity → report bracket as approximate.
4. Output: step range [c_j, c_{j+1}), with the replay log as evidence.
```

### 3.3 Cost model

Each replay covers at most (B − c_i) steps, but the search naturally probes midpoints, so total replayed steps ≈ (B − c_0)·(1 + small constant) in the worst case, versus full re-run cost for naive approaches. In practice, replay segments are short because the bracket tightens exponentially.

---

## 4. Data-shard bisect

Once the step range [c_j, c_{j+1}) is known, the trigger is one of the K batches in that range.

### 4.1 Indexed dataloader + per-batch fingerprints

- Every batch the model ever sees gets an identity: `(epoch, global_batch_idx, shard_id)` and a **content fingerprint** `fp = SHA256(x_bytes ‖ y_bytes ‖ shape ‖ dtype)`.
- The recorder logs `step → fp` for every step (instrumented mode). Fingerprint hashing is the main overhead; batch bytes are hashed once per step.
- Shards: the dataset is partitioned into S named shards (files, directories, or index ranges). Each batch maps to the shard(s) it draws from.

### 4.2 Counterfactual shard bisection

To find the offending shard, replay the trigger range with data **surgically replaced**:

```
BISECT-SHARDS(c_j, c_{j+1}):
    batches = [b_1 … b_K] in [c_j, c_{j+1}) with fingerprints
    candidate set = all shards present in batches
    while |candidates| > 1:
        split candidates into halves H1, H2
        replay [c_j → c_{j+1}) with every batch from H2 replaced by a benign
            substitute (e.g. replay of the same batch from a clean epoch, or zeros with matching stats)
        if collapse persists → trigger ∈ H1, discard H2
        else                   → trigger ∈ H2, discard H1
    output the surviving shard
```

The **causal evidence** is the counterfactual pair:
- *Include* shard S → collapse reproduces.
- *Exclude* shard S (only S) → collapse vanishes.

That is a proof of causation for this trajectory, not an attribution score.

### 4.3 Batch → sample

Within the offending batch, no optimizer replay is needed: freeze weights at c_j, run a forward pass per sample (or per sample-half, bisecting again), and rank per-sample loss/gradient-norm contribution. The sample(s) whose removal kills the batch's loss spike are named, with their dataset indices and fingerprints.

### 4.4 Minimal-repro command

The final artifact:

```
trainsect replay --ckpt ckpt-4100 --steps 100 --exclude-shard shard-7
# → loss stays flat: shard-7 is necessary for the collapse
trainsect replay --ckpt ckpt-4100 --steps 100 --only-shard shard-7 --only-batch fp:9f3c…
# → collapse reproduces: batch is sufficient
```

Necessity + sufficiency, both demonstrated by replay. This is the falsifiable core of the invention.

---

## 5. Instrumented mode — the one-line recorder hook

```python
from trainsect import record

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    record(step, model, optimizer, dataloader, batch)   # one line
```

`record()` captures per call (cheap path) and per-K-steps (checkpoint path):

**Every step (sidecar JSONL, ~bytes per step):**
- step, epoch, global batch index
- batch fingerprint fp (SHA-256 of content + order)
- shard id(s) the batch was drawn from
- loss value (for the Stage-A change-point detector)
- LR (for schedule forensics)

**Every K steps (alongside the model checkpoint):**
- `model.state_dict()`, `optimizer.state_dict()`, scheduler state (deep copy)
- RNG states: torch CPU, torch CUDA (all devices), numpy, python `random`
- DataLoader sampler state: the `torch.Generator` state + per-worker seeds
- env flags: `torch.are_deterministic_algorithms_enabled()`, `cudnn.deterministic`, `cudnn.benchmark`

**Design target:** < 2% step-time overhead. The fingerprint hash dominates; everything else is small state copies.

With this state, replay from any recorded checkpoint is **bit-exact** on deterministic stacks (verified by the §1.2 fidelity check), and the binary search + shard bisect run at full causal strength.

---

## 6. Forensic mode — checkpoints alone, honest bounds

When only a checkpoint directory exists (no recorder), Trainsect recovers what it can and **states the rest as explicit uncertainty**:

| Recoverable from checkpoints | Unrecoverable (without recorder) |
|---|---|
| θ_t, optimizer moments, step count | exact batch order (sampler seed/epoch unknown) |
| LR scheduler state (if saved) | per-step RNG draws (dropout masks, augmentation) |
| epoch / global step (if logged) | per-batch fingerprints (content re-derivable only if dataset + order known) |
| loss history (if logged) | DataLoader worker seeds |

**Forensic procedure:**
1. Stage A (change-point on logged losses) still works — it needs no replay.
2. Replay from c_j uses a *reconstructed* dataloader (same dataset, same batch size, best-guess or user-supplied seed). Run the replay under **M candidate seeds** (default M=8).
3. Verdict is reported with confidence: *"trigger in [c_j, c_{j+1}) — reproduced under 6/8 seeds; exact batch unidentifiable; uncertainty ±1 epoch of steps."*
4. Shard naming degrades gracefully: without fingerprints, per-batch causal proof is impossible, so forensic mode falls back to **per-batch loss ranking under frozen weights** — explicitly labeled *approximate, not causal*.

**Hard rule:** forensic mode never names a batch it cannot prove. An honest bracket with a confidence interval beats a precise-sounding guess.

---

## 7. Why this is genuinely novel (vs. the adjacent tools)

| Tool | Axis it searches | Causal? |
|---|---|---|
| `git bisect` / rldk `bisect` | **code commits** (re-runs training per commit) | yes, but for code regressions |
| culprit | **model releases** across runs (robotics stacks) | counterfactual, but inference-time |
| merriv `checkpoint_bisect` | **precomputed PASS/BLOCK labels** in a JSONL | no replay at all |
| NeuralDBG | in-loop symptoms (gradients, activations) | hypotheses, no replay |
| TRAK / influence fns / TracIn | **approximate per-sample scores** | no — linearization/projection/Taylor approximations |
| Manual "data bisection" (blog folklore) | a single crashing batch, by hand | yes, but manual, single-batch, no tool |
| **Trainsect** | **optimizer steps within one run** → **data shards** via replay | **yes — necessity + sufficiency by replay** |

The invention is the composition: *deterministic optimizer-step replay* as a forensic primitive + *binary search over the checkpoint timeline* + *counterfactual shard bisection* — yielding a machine-checkable minimal repro instead of a ranked guess.

---

## 8. Gap-verification evidence (2026-10-14)

Searches run: "checkpoint bisect" / "training replay debugger" / "loss spike bisection" across GitHub, PyPI, arXiv, HN, and general web; plus targeted checks of rldk, culprit, NeuralDBG, TRAK/influence-functions.

- **rldk** (github.com/adityachallapally/rldk): RLHF training-health toolkit. `rldk bisect` bisects **git commits** (`--good origin/main~15 --bad HEAD --cmd "python train.py …"`); `rldk replay` replays a **whole run** with the original seed; `rldk diff-ckpt` compares two checkpoints; `first_divergence` compares two runs. All adjacent — none replays optimizer steps across checkpoints within one run or bisects data shards.
- **culprit** (github.com/aneesh-pothuru/culprit, `docs/BRIEF.md`): closest conceptually — "git bisect for a stack of models — down to the component, the checkpoint, and the training data", with counterfactual replay. But: (a) robotics domain (MCAP/ROS 2 + agent traces, multi-model inference stacks), (b) bisects **checkpoint releases across runs** by replaying failing *inference inputs*, not optimizer steps within one training run, (c) data audit is influence-style attribution. Different axis, different domain; also appears to be a design brief, not an implemented tool.
- **merriv** (github.com/niansia/merriv, `examples/checkpoint_bisect/`): `merriv bisect` finds the first failing revision in a JSONL of **already-computed PASS/BLOCK statuses** — no checkpoint files are loaded, no replay, no data. Label bisection, not training bisection.
- **NeuralDBG** (github.com/LambdaSection/NeuralDBG, PyPI `neuraldbg`): in-loop semantic event extraction + ranked causal hypotheses ("gradient vanishing originated in layer X at step 234"). No checkpoint replay, no binary search, no shard bisection. Diagnostic hypotheses, not causal proof.
- **TRAK / influence functions / TracIn**: approximate attribution. TRAK's own analysis literature documents the three approximations (linearization, ALO, projection); TracIn is a first-order Taylor approximation that happens to use 3–20 saved checkpoints for influence sums — not optimizer-step replay for bisection.
- **Folklore**: Jiyang Kang's "How to Debug Loss Spikes and NaNs" (Medium, Jan 2026) describes manual *data bisection* (split the crashing batch, run fwd/bwd per half) as practitioner advice — confirming the technique's value and the absence of a tool that automates it. A dev.to "Git Bisect for Your Data" piece covers time-series query bisection (pg-timemachine), not training.
- **PyPI / HN sweeps**: no package or Show-HN implements deterministic training-step replay + checkpoint bisection + shard bisection.

**Verdict: GAP CONFIRMED.** Nothing found is Trainsect. The nearest misses differ on the searched axis (commits / releases / labels / scores) rather than the training-time axis.

---

## 9. Validation notes (what P0 actually proved — and what broke)

Two validation runs, same setup (MLP 20→64→64→1, 160 steps, 8 shards, poison shard 2 = unnormalized features ×100 + flipped labels, steps 40–59, CPU, torch 2.14.1+cpu). Full numbers in §10.

**Run 1 (SGD lr=0.2): persistent NaN collapse.** Poison → train loss 648 at step 40 → NaN from step 50, never recovers. Pipeline: change point 50 → bracket [40,80) → trigger step 40 → shard 2, necessity + sufficiency both true → CONFIRMED. 8 replays, 760 replayed steps, 22.1 s.

**Run 2 (SGD lr=0.1): recoverable collapse + benign transient spike.** Poison → eval 231 at step 50, then model RECOVERS to eval 0.69 by step 80; plus a benign train-loss spike (9.18) at step 120 on a healthy model. Pipeline with the corrected eval-based predicate: bracket [40,80) → trigger step 40 → shard 2, necessity + sufficiency true → CONFIRMED. 10 replays, 840 replayed steps, 20.2 s.

**Three real bugs found and fixed during validation (not hidden):**
1. *Checkpoint off-by-one.* First version saved checkpoints POST-update, so replay double-applied the checkpoint's own batch. Fixed to pre-update checkpoint semantics (ckpt at step c = state before step c's update).
2. *Train-spike predicate misfires.* First version bisected on "a train-loss spike occurs during replay". On Run 2 it converged on the benign step-120 spike and named the innocent shard 6 — and the necessity check correctly returned False. Fixed: bisection + bisect + confirm all use the eval-based collapse predicate; train spikes only pinpoint within the bracket.
3. *Collapse-threshold calibration.* A 5×MAD eval threshold misfired: clean training at lr=0.1 drifted to eval ~3.0 on its own, and the exclude-shard-2 counterfactual replay sat at 3.097 — a marginal crossing that failed necessity. Fixed: collapse threshold = max(10×baseline, baseline + 10×MAD). Collapse signal (231) is 34× above it; healthy drift (3.1) is 0.45×. The margin, not just the boolean, is the evidence.

**What P0 did NOT validate:** forensic mode (no recorder) — the uncertainty bounds in §6 are designed but untested; distributed/multi-rank replay; GPU bit-exactness (expected to fail the fidelity check on nondeterministic kernels); checkpoint storage at scale.

## 10. P0 validation results (reproducible)

```
Setup:  ~/workspace/trainsect/validation/make_run.py   (generates run/)
        ~/workspace/trainsect/prototype/trainsect_p0.py (the prototype)
Run 1 (lr=0.2, persistent NaN collapse):
  fidelity: replay reproduces original final state BIT-EXACT (hash dcdda46f7dc65e42, 3 ways)
  change point 50 | bracket [40,80) | trigger step 40 | shard-2 ✓ | necessity ✓ sufficiency ✓
  8 replays, 760 replayed steps, 22.1 s wall
Run 2 (lr=0.1, recoverable collapse + benign spike):
  fidelity: BIT-EXACT (hash 6d5de9cc257bddf1, 3 ways)
  change point 50 | bracket [40,80) | trigger step 40 | shard-2 ✓ | necessity ✓ sufficiency ✓
  10 replays, 840 replayed steps, 20.2 s wall
```

## 11. Open questions for P1 (not blocking P0)

1. **Distributed training:** replay across DDP/FSDP ranks requires per-rank RNG + collective ordering capture. P0 is single-process; P1 must define the multi-rank recorder contract.
2. **Checkpoint granularity vs. storage:** every-K-steps full optimizer state is heavy at LLM scale; P1 should evaluate delta-checkpoints or copy-on-write state capture.
3. **Non-monotone collapse predicates:** loss spikes that self-heal break binary-search monotonicity; P1 needs the multi-transition detector (sample-then-bisect, cf. §3.2 retry logic).
4. **Benign batch substitution** (§4.2): the counterfactual "replace with benign batch" needs a principled substitute that doesn't itself perturb optimization (clean-epoch replay of the same indices is the leading candidate).
5. **GPU bit-exactness:** expected to fail the §1.2 fidelity check on nondeterministic kernels; P1 must quantify tolerance-based (not bit-exact) replay verdicts.
