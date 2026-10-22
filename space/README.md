---
title: Trainsect Demo
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Trainsect Demo

**git bisect for training runs.** Name the exact optimizer step, data shard, and batch that caused a training collapse — proven by deterministic replay (necessity + sufficiency), not by attribution scores.

- **Preloaded poisoned run** tab: a real, previously computed diagnosis — the search narrowing to shard-3, the evidence report, and the verbatim terminal transcript.
- **Run live bisection** tab: re-runs the whole magic trick from scratch on this Space's CPU (~1–2 min).
- **How it works** tab: methodology summary + honest scope and limitations.

Canonical source: [github.com/habib-analyst/trainsect](https://github.com/habib-analyst/trainsect). The `trainsect/` directory in this Space is a vendored snapshot of that package so the demo runs standalone.

Method: causal replay-bisection. Scope: CPU, deterministic, single-process stacks only — GPU replay aborts honestly instead of guessing. Forensic mode (checkpoints alone) never names unproven batches.
