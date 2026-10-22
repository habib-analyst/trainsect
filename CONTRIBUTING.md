# Contributing to Trainsect

Thanks for considering a contribution. Trainsect's core promise is **honest causal evidence**: a verdict the tool cannot prove is a verdict it must not state. Keep that bar in every change.

## Setup

```bash
git clone https://github.com/habib-analyst/trainsect.git
cd trainsect
pip install --user -e .
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

Python ≥ 3.10, `torch`, `numpy`. The checkpoint `.pt` fixtures (`validation/run/`, `demos/demo_run/`) are regenerable and not stored in git — tests regenerate them automatically if missing.

## What makes a good contribution

- **Bug reports**: a minimal failing run directory, or a `make_*`-style generator script that reproduces one. Include the full `trainsect diagnose` output and the evidence report.
- **Bug fixes**: a regression test that fails before and passes after, plus the fix. If the fix changes a verdict, say so explicitly in the PR description.
- **New architectures / dataloader support**: only through the `trainsect` config block / explicit recorder contract — never by guessing an undeclared architecture (see `hub.py`).
- **Docs**: corrections to claimed numbers are always welcome. Every measured number in the README must be reproducible from the repo.

## Hard rules (do not break these)

1. **Never name a batch the evidence cannot prove.** Forensic mode never sets a batch fingerprint or trigger step; its verdict is never CONFIRMED (causal).
2. **Eval-based collapse predicate only.** Train-loss spikes never count as collapse.
3. **Fidelity check stays.** If two replays disagree bit-exactly, bisection halts — no tolerance-based verdicts without an explicit, tested design.
4. **Pre-update checkpoint semantics.** ckpt at step *c* = state *before* step *c*'s update. Do not "fix" the off-by-one back.
5. **No secrets in the repo.** Tokens go to `HF_TOKEN` / `--token` (header, never URL) at runtime, never into code, tests, or fixtures. Test doubles must use obviously-fake values.
6. **Free-only, CPU-first.** No paid services, no GPU-only paths, no new heavy dependencies without discussion.

## PR process

- Keep PRs small and focused; one behavior change per PR.
- Run the full test suite before submitting. Note wall-clock time if a test is slow — the suite should stay runnable in a couple of minutes on CPU.
- Update the README's measured-numbers table if your change moves any number.
- By submitting, you agree your contribution is licensed under the repo's MIT License.
