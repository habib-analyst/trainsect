---
title: Trainsect Demo
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Trainsect Demo

**git bisect for training runs.** Name the exact optimizer step, data shard, and batch that caused a training collapse — proven by deterministic replay (necessity + sufficiency), not by attribution scores.

Preloaded demo (Static Space): the search narrowing to shard-3, the evidence report, the verbatim terminal transcript, plus how-it-works and methodology tabs. No live run on this page — run the magic trick yourself with `pip install git+https://github.com/habib-analyst/trainsect.git` then `trainsect demo`.

Canonical source: [github.com/habib-analyst/trainsect](https://github.com/habib-analyst/trainsect).

Method: causal replay-bisection. Scope: CPU, deterministic, single-process stacks only — GPU replay aborts honestly instead of guessing. Forensic mode (checkpoints alone) never names unproven batches.
