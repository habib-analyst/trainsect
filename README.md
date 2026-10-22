# Trainsect — git bisect for training runs

**Your training run collapsed. Which exact optimizer step, data shard, and batch caused it?**

Trainsect answers with a **causal proof, not an attribution score**: it reloads optimizer state at a checkpoint, replays forward deterministically, binary-searches the checkpoint timeline for the trigger step-range, then counterfactually replays with each data shard excluded until only the guilty shard remains. The output is a falsifiable minimal repro:

```
collapse: eval-loss change point at step 80
trigger step-range: [60, 90)          # proven by replay, not log-scanning
offending shard:   shard-3 / batch fp 639a024b373ce19a
repro: trainsect replay --ckpt ckpt_0060 --steps 30 --exclude-shard 3  # collapse vanishes
```

Live demo: <https://huggingface.co/spaces/habib-analyst/trainsect-demo>

## 30-second value proposition

Existing debuggers bisect the **wrong axis**: `git bisect` finds the bad *commit*, release-bisect tools find the bad *checkpoint release*, and TRAK / influence functions assign *approximate scores* to samples. Trainsect bisects the **training-time axis itself** — optimizer steps within one run — and proves the culprit by re-running it twice (necessity: collapse vanishes without the shard; sufficiency: collapse reproduces with only the shard). If replay is not bit-exact, Trainsect aborts honestly instead of guessing.

## Install

```bash
pip install git+https://github.com/habib-analyst/trainsect.git
```

Requires Python ≥ 3.10, `torch`, `numpy`. For development:

```bash
git clone https://github.com/habib-analyst/trainsect.git
cd trainsect
pip install --user -e .
```

## Quickstart

**Diagnose a run you recorded** (checkpoints + batch fingerprints + eval log):

```bash
trainsect diagnose --checkpoints runs/exp1 --output report.json --text-output report.txt
```

**See the magic trick** (tiny transformer + injected poison shard, diagnosed live):

```bash
trainsect demo --outdir demos
```

**Record your own runs** with the one-line hook:

```python
from trainsect import start_recording, record, stop_recording

start_recording("runs/exp1", model, opt, sched, config, ckpt_every=30)
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    record(step, batch, shard_id=shard_of(batch), train_loss=loss.item())
stop_recording(eval_data=eval_data)
```

`record()` is called *after* the step's update, which keeps pre-update checkpoint semantics exact. See `demos/example_recorded_loop.py` for a complete minimal example (hook-recorded run replays bit-exact and diagnoses to the poison shard).

## The magic-trick demo (what it proves)

`trainsect demo` trains a nanoGPT-style tiny transformer for 120 steps on CPU, with shard 3 (steps 60–79) poisoned by uniformly random tokens, then diagnoses it live with no knowledge of the injection:

| Result | Value |
|---|---|
| Named shard | **shard-3** (the injected poison) |
| Trigger step | **60** (first poison batch) |
| Batch fingerprint | `639a024b373ce19a` |
| Offending sample | index 3 in the batch (loss 13.74 vs batch mean 12.46) |
| Necessity + sufficiency | both **true** |
| Verdict | **CONFIRMED (causal)** |
| Fidelity | **BIT-EXACT** replay |
| Cost | 8 replays, 540 replayed steps, ~52 s diagnose |

Transcript: `demos/demo_transcript.txt`; report: `demos/demo_report.json` / `demos/demo_report.txt`. Regenerate from scratch: `PYTHONPATH=src python3 demos/make_demo_run.py`.

## Methodology summary — causal replay-bisection

Full methodology: [`docs/P0-methodology.md`](docs/P0-methodology.md).

1. **Deterministic replay.** A training step is a pure function of (model, optimizer, scheduler, batch, RNG) state. Checkpoints store all of it with **pre-update semantics** (ckpt at step *c* = state *before* step *c*'s update). Before trusting any bisection, Trainsect replays a segment **twice** and compares SHA-256 parameter hashes — if they disagree, bisection halts.
2. **Collapse predicate (eval-based, never train-spike).** A checkpoint counts as collapsed only by its *eval* loss (threshold from the lowest quartile of the loss log). Transient train-loss spikes on a healthy model never count — an early spike-based version misfired on exactly this case and was fixed.
3. **Checkpoint binary search.** Predicate P(c) = "checkpoint *c* is healthy AND replaying forward reproduces the collapse" is monotone in *c*, so binary search brackets the trigger to one checkpoint interval in O(log n) replays.
4. **Counterfactual shard bisection.** Replay the trigger interval with each candidate shard's batches replaced by benign substitutes. The surviving shard must pass **necessity** (collapse vanishes without it) **and sufficiency** (collapse reproduces with only it) — that pair is the causal proof.
5. **Evidence report.** JSON + human-readable report with the minimal-repro commands, so anyone can re-run the proof.

### Forensic mode — checkpoints alone

`diagnose` takes the forensic path automatically when a directory lacks recorder data. It reconstructs batch orders from the known pool under multiple seeds (3 for the bracket search, 8 for confirmation) and names a shard only if necessity + sufficiency + specificity all hold *and* frozen-weight loss ranking agrees. Otherwise it reports **UNPROVEN** with what evidence would settle it. Forensic mode **never** sets a batch fingerprint or trigger step, and its verdict is never CONFIRMED (causal).

Measured on the two demo runs, stripped of recorder data:

| Run | Forensic verdict | Why |
|---|---|---|
| P0 validation (MLP, ×100-feature poison) | **FORENSIC (probable), shard-2**, bracket [40, 80) | Poison is single-batch toxic: collapse reproduces under 8/8 reconstructed orders; necessity 8/8, sufficiency 8/8, frozen margin 1969× |
| Demo (transformer, random-token poison) | **UNPROVEN**, no shard named | Collapse is concentration-dependent: 20 consecutive poison batches collapse the model, but the same poison diluted across a shuffled order only wobbles eval to ~1.5 (never crosses 3.0) — verified by direct replay. Reconstructed orders cannot reproduce it, so no claim is made. |

### HF Hub integration (read-only)

- `trainsect diagnose --checkpoints <hf-dir>` auto-detects HF checkpoint layout (`checkpoint-N/` subdirs or top-level HF weights) and normalizes it. The HF `config.json` must declare a `"trainsect"` block (arch + dims + optimizer + lr) — Trainsect replays only its own architectures and refuses to guess an undeclared one.
- `trainsect download --repo <id> --out <dir>` downloads a checkpoint repo (weights, optimizer, scheduler, RNG state, trainer state, config). Download only, never uploads. Downloads go through `curl` as a subprocess; `HF_TOKEN` / `--token` goes to an `Authorization` header, never the URL.

## Measured numbers (CPU, this repo, 2026-10-14)

| Measurement | Value |
|---|---|
| Test suite | **74 tests, all passing** (`PYTHONPATH=src python3 -m unittest discover -s tests -t .`) |
| Recorder overhead | **0.7–1.0%** step time (~0.08–0.11 ms/record on a ~12 ms step; median of repeats). Under the <2% design target. Checkpoint saves are separate and amortized: ~10.9 ms/save on the tiny demo model (~+3%/step at `ckpt_every=30`, ~+0.9% at `ckpt_every=100`) |
| P0 validation runs (MLP, 160 steps) | Both reproduce: trigger step 40, shard 2, necessity + sufficiency true, **CONFIRMED**, bit-exact fidelity. Regenerate: `python3 validation/make_run.py` |
| Demo run | See table above |
| Forensic mode | See table above (3.6–4.8 s on the demo runs) |

## Honest scope + limitations

- **CPU, deterministic, single-process stacks only.** GPU replay is expected to fail the bit-exact fidelity check (nondeterministic kernels) — the tool **aborts honestly** rather than guessing.
- **Forensic mode never names unproven batches.** Without recorder data, exact batch/step identity is unrecoverable; an honest bracket with a confidence interval beats a precise-sounding guess.
- **Counterfactual substitution** uses benign batches from a known-clean shard (default shard 0); the substitute shard must itself be collapse-free, which necessity checks.
- **Threshold floor (3.0)** is thin for bounded cross-entropy losses: the demo's collapse (eval 3.87 vs threshold 3.0, 1.29×) is proven by necessity/sufficiency, not by the margin — the report shows the numbers.
- **Distributed training** (DDP/FSDP) is not supported: replay across ranks would need per-rank RNG + collective-order capture.
- **Checkpoint storage at scale** is an open cost: every-K-steps full optimizer state is heavy at LLM scale (delta-checkpoints not yet implemented).

## Package layout (`src/trainsect/`)

| Module | Role |
|---|---|
| `fingerprints.py` | Per-batch SHA-256 fingerprints, parameter hashes |
| `dataloader.py` | `IndexedDataLoader`: deterministic iteration, shard→batch map, fingerprint reproduction, benign substitution |
| `models.py` | `MLP` + `TinyTransformer` (nanoGPT-style), loss fns |
| `replay.py` | `ReplayEngine`: pre-update checkpoint load, deterministic replay, bit-exact fidelity check |
| `collapse.py` | Eval-based collapse predicate, lowest-quartile threshold + warmup skip |
| `bisect.py` | Checkpoint binary search, counterfactual shard bisect, necessity + sufficiency |
| `evidence.py` | `EvidenceReport`: JSON + human-readable, minimal-repro commands |
| `diagnose.py` | Stage A→E pipeline over a run directory; routes instrumented / forensic / HF-layout |
| `demo.py` | Magic-trick demo: tiny transformer + injected poison shard |
| `recorder.py` | Recorder hook (`start_recording` / `record` / `stop_recording`, `recording(...)`), overhead benchmark |
| `forensic.py` | Forensic mode: checkpoints alone, multi-seed reconstruction, honest confidence |
| `hub.py` | HF Hub integration: local HF-layout loading, curl downloads (read-only) |
| `cli.py` | `diagnose` / `demo` / `replay` / `download` |

CLI: `trainsect diagnose --checkpoints <run_dir>`, `trainsect demo`, `trainsect replay --checkpoints <run_dir> --from-step 60 --steps 30 --exclude-shard 3`, `trainsect download --repo <id> --out <dir>`.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

74 tests: fingerprint stability, dataloader/shard mapping, eval predicate (no train-spike misfires), bisect correctness, necessity/sufficiency logic, replay determinism + fidelity, evidence reports, a P0-reproduction integration test, recorder capture → bit-exact replay + overhead, forensic mode on stripped runs (correct-or-UNPROVEN, never a false accusation), and HF layout loading / curl downloads (network mocked).

Note: the checkpoint `.pt` fixtures (`validation/run/`, `demos/demo_run/`) are regenerable and not stored in git — tests regenerate them automatically if missing (`python3 validation/make_run.py`, `python3 demos/make_demo_run.py`).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Bug reports that include a minimal failing run directory (or a `make_*` generator reproducing one) are the most useful kind.

## License

MIT — see [`LICENSE`](LICENSE).
