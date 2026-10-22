# DRAFT ONLY, DO NOT POST

## DEV tutorial: Find the exact step that collapsed your training run with Trainsect

Trainsect is git bisect for training runs. You point it at a checkpoint directory, and it tells you which optimizer step, data shard, and batch caused the collapse, with a causal proof, not an attribution score. This tutorial runs the built-in magic-trick demo end to end: install, run, read the evidence report, and re-run the proof yourself.

### What you need

- Python 3.10 or newer
- `torch` and `numpy`
- A CPU machine. The demo runs on CPU. GPU replay is expected to fail the bit-exact fidelity check, and Trainsect aborts honestly rather than guessing, so run this tutorial on CPU.

### Step 1: Install

The repo link goes live after the GitHub push. Until then, this command is a placeholder:

```bash
pip install git+https://[LINK — goes live after GitHub push]
```

For development (once the repo is live):

```bash
git clone https://[LINK — goes live after GitHub push]
cd trainsect
pip install --user -e .
```

### Step 2: Run the magic-trick demo

```bash
trainsect demo --outdir demos
```

What this does: it trains a nanoGPT-style tiny transformer for 120 steps on CPU, with shard 3 (steps 60-79) poisoned by uniformly random tokens. Then it diagnoses the run live, with no knowledge of the injection.

You will see training lines scroll by. Watch three of them:

```
step   0 shard 0 train_loss=  4.3811
step  40 shard 2 train_loss=  0.0027
step  60 shard 3 train_loss= 12.4577
```

Step 40 is a healthy model. Step 60 is the first poison batch. The eval loss log tells the same story: eval sits near 0.00 through step 60, then reads 3.871 at step 70 and stays near 4.0.

### Step 3: Watch the diagnose stages

After training, the diagnose pipeline runs in five stages. You will see one line per stage:

```
[fidelity] replay 7622e7196398a345 vs original 7622e7196398a345 -> BIT-EXACT
[Stage A] change point: {'change_point_step': 80, 'threshold': 3.0, ...}
[Stage B] checkpoint binary search -> {'ok': True, 'bracket': (60, 90)}
[Stage C] pinpoint: first train spike at step 60 (loss=12.457655906677246); ...
[Stage D] offending shard = shard-3
[Stage E] necessity=True sufficiency=True
```

What each stage means:

- **Fidelity.** The segment is replayed twice and the SHA-256 parameter hashes are compared. They match, so replay is bit-exact on this stack. If they disagreed, bisection would halt here.
- **Stage A, change point.** Binary segmentation on the eval-loss log finds the first sustained shift: step 80, eval 4.08, threshold 3.0. This is a symptom locator, not a cause.
- **Stage B, checkpoint binary search.** The predicate "checkpoint is healthy AND replaying forward reproduces the collapse" is monotone, so binary search brackets the trigger to one checkpoint interval: [60, 90).
- **Stage C, pinpoint.** Inside the bracket, the first train-loss spike is at step 60. Train loss is used only for pinpointing inside an established bracket, never for the collapse verdict itself.
- **Stage D, shard bisect.** Counterfactual replay names shard-3.
- **Stage E, confirmation.** Necessity + sufficiency, both true.

### Step 4: Read the evidence report

The demo writes `demos/demo_report.json` and a human-readable `demos/demo_report.txt`. The key fields:

| Field | Demo value | Meaning |
|---|---|---|
| `verdict` | CONFIRMED (causal) | Necessity + sufficiency both demonstrated by replay |
| `provenance` | instrumented | Recorder data was present, so exact batch identity is recoverable |
| `fidelity_bit_exact` | true | Replay reproduced the original trajectory bit-exactly |
| `change_point` | 80 | First sustained eval-loss shift (symptom, not cause) |
| `bracket` | [60, 90) | Trigger step-range proven by replay |
| `trigger_step` | 60 | First poison batch |
| `shard` | 3 | Offending data shard |
| `batch_fingerprint` | 639a024b373ce19a | SHA-256 identity of the trigger batch |
| `offending_sample` | index 3, loss 13.74 vs batch mean 12.46 | Worst sample in the trigger batch, found by frozen-weight forward passes |
| `confirmation` | necessity true, sufficiency true | The causal proof |
| `replays` / `replayed_steps` | 8 / 540 | Total cost of the diagnosis |
| `wall_time_s` | 52.2 | Wall-clock diagnose time on the demo machine |

### Step 5: Re-run the proof yourself

The report ends with minimal-repro commands. Run them. They are the whole point: the claim is falsifiable, so falsify it.

```bash
# Necessity: collapse vanishes without shard 3
trainsect replay --checkpoints demos/demo_run --from-step 60 --steps 30 --exclude-shard 3

# Sufficiency: collapse reproduces with only shard 3
trainsect replay --checkpoints demos/demo_run --from-step 60 --steps 30 --only-shard 3

# Trigger batch alone
trainsect replay --checkpoints demos/demo_run --from-step 60 --steps 30 --only-batch 639a024b373ce19a
```

If the first command does not flatten the loss, or the second does not reproduce the collapse, the proof fails, and that failure is the bug report.

### Step 6: Diagnose a run you recorded

The one-line recorder hook, added to your own training loop:

```python
from trainsect import start_recording, record, stop_recording

start_recording("runs/exp1", model, opt, sched, config, ckpt_every=30)
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    record(step, batch, shard_id=shard_of(batch), train_loss=loss.item())
stop_recording(eval_data=eval_data)
```

Note: `record()` is called after the step's update, which keeps pre-update checkpoint semantics exact. Then diagnose:

```bash
trainsect diagnose --checkpoints runs/exp1 --output report.json --text-output report.txt
```

Recorder overhead is measured at 0.7-1.0% of step time on the demo stack, under the under-2% design target. Checkpoint saves are separate and amortized.

### Step 7: Understand forensic mode

If you point `diagnose` at a directory with checkpoints but no recorder data, it takes the forensic path automatically. It reconstructs batch orders from the known pool under multiple seeds (3 for the bracket search, 8 for confirmation) and names a shard only if necessity + sufficiency + specificity all hold and frozen-weight loss ranking agrees. Otherwise it reports UNPROVEN with what evidence would settle it.

Forensic mode never sets a batch fingerprint or trigger step, and its verdict is never CONFIRMED (causal). On the demo run stripped of recorder data, it correctly returned UNPROVEN: the collapse needs 20 consecutive poison batches, and the same poison diluted across a shuffled order only wobbles eval to about 1.5, never crossing 3.0. That is the tool refusing to guess, and it is a feature.

### Limitations, before you reach for it

- CPU, deterministic, single-process stacks only.
- Distributed training (DDP/FSDP) is not supported.
- The collapse threshold floor (3.0) is thin for bounded cross-entropy losses. The demo collapse (eval 3.87 vs threshold 3.0, 1.29x) is proven by necessity/sufficiency, not by the margin.
- Checkpoint storage at scale is an open cost. Every-K-steps full optimizer state is heavy at LLM scale.

### Try it, then break it

The repo link goes live after the GitHub push: [LINK — goes live after GitHub push]. The most useful thing you can do with this tool is find a collapse it gets wrong: a stack where replay is not bit-exact, a collapse where the predicate misfires, a forensic run where the verdict should be UNPROVEN but is not. Those failure cases are the roadmap.
