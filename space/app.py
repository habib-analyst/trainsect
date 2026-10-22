"""Trainsect demo Space: preloaded poisoned run + live bisection.

The preloaded view shows a real, previously computed diagnosis (bundled
``demo_report.json`` / ``demo_transcript.txt``). The live button re-runs the
whole magic trick from scratch on this Space's CPU: it trains a tiny
transformer with an injected poison shard, then diagnoses it with no
knowledge of the injection.

The ``trainsect/`` package directory next to this file is a vendored
snapshot of the canonical source at https://github.com/habib-analyst/trainsect
so the demo runs standalone.
"""
import contextlib
import io
import json
import os
import tempfile
import traceback

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_preloaded():
    with open(os.path.join(HERE, "demo_report.json")) as f:
        report = json.load(f)
    with open(os.path.join(HERE, "demo_transcript.txt")) as f:
        transcript = f.read()
    # The bundled report carries the local path it was generated at; show a
    # neutral path instead so the repro commands read cleanly.
    local = "/home/hatch/workspace/trainsect/demos/demo_run"
    report = json.loads(json.dumps(report).replace(local, "demos/demo_run (preloaded)"))
    transcript = transcript.replace(local, "demos/demo_run (preloaded)")
    return report, transcript


def _funnel_markdown(report):
    cp = report["change_point"]
    return f"""## How the search narrows to the exact shard

```
full run:            120 steps, 6 shards x 20 batches
Stage A (symptom):   eval-loss change point at step {cp["change_point_step"]}
                     (threshold {cp["threshold"]}, baseline {cp["baseline"]:.4f},
                      eval at change point {cp["eval_loss_at_cp"]:.3f})
Stage B (bisect):    checkpoint binary search -> bracket {tuple(report["bracket"])}
Stage C (pinpoint):  trigger step {report["trigger_step"]} (first train spike inside bracket)
Stage D (shard):     offending shard = shard-{report["shard"]}
Stage E (proof):     necessity = {str(report["confirmation"]["necessity"]).lower()},
                     sufficiency = {str(report["confirmation"]["sufficiency"]).lower()}
VERDICT:             {report["verdict"]}  |  fidelity: {"BIT-EXACT" if report["fidelity_bit_exact"] else "NOT BIT-EXACT"}
```

**Necessity** = replay the trigger interval *without* shard-3 and the collapse vanishes.
**Sufficiency** = replay it with *only* shard-3 and the collapse reproduces.
That pair is a causal proof for this trajectory — not an attribution score.

- Batch fingerprint: `{report["batch_fingerprint"]}`
- Offending sample: index {report["offending_sample"]["sample_index_in_batch"]} in the batch
  (loss {report["offending_sample"]["sample_loss"]:.2f} vs batch mean {report["offending_sample"]["batch_mean_loss"]:.2f})
- Cost: {report["replays"]} replays, {report["replayed_steps"]} replayed steps, {report["wall_time_s"]:.1f} s diagnose
"""


def _report_markdown(report):
    lines = [
        "### Evidence report (preloaded)",
        "",
        f"- **Verdict:** {report['verdict']}",
        f"- **Provenance:** {report['provenance']}",
        f"- **Confidence:** {report['confidence']}",
        f"- **Fidelity:** {'BIT-EXACT' if report['fidelity_bit_exact'] else 'NOT BIT-EXACT'}",
        f"- **Change point:** step {report['change_point']['change_point_step']} "
        f"(threshold {report['change_point']['threshold']})",
        f"- **Step range:** {report['bracket']}",
        f"- **Trigger step:** {report['trigger_step']}",
        f"- **Offending shard:** shard-{report['shard']}",
        f"- **Batch fingerprint:** `{report['batch_fingerprint']}`",
        f"- **Necessity:** {report['confirmation']['necessity']} | "
        f"**Sufficiency:** {report['confirmation']['sufficiency']}",
        "",
        "**Minimal repro:**",
        "",
    ]
    for cmd in report["minimal_repro"]:
        lines.append(f"```\n{cmd}\n```")
    return "\n".join(lines)


PRELOADED_REPORT, PRELOADED_TRANSCRIPT = _load_preloaded()


def run_live():
    """Re-run the magic trick from scratch; return (transcript, report_json)."""
    try:
        from trainsect.demo import run_demo
    except Exception:
        return ("Import failed:\n" + traceback.format_exc(), {})
    tmp = tempfile.mkdtemp(prefix="trainsect_live_")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            report = run_demo(tmp, None)
        return buf.getvalue(), report
    except Exception:
        return (
            buf.getvalue() + "\n[LIVE RUN FAILED]\n" + traceback.format_exc(),
            {},
        )


HEADER = """# 🔬 Trainsect — git bisect for training runs

**Your training run collapsed. Which exact optimizer step, data shard, and batch caused it?**
Trainsect answers with a **causal proof, not an attribution score**: it reloads optimizer state at a checkpoint,
replays forward deterministically, binary-searches the checkpoint timeline, then counterfactually replays with each
data shard excluded until only the guilty shard remains.

Source: [github.com/habib-analyst/trainsect](https://github.com/habib-analyst/trainsect) · Methodology: `docs/P0-methodology.md` in the repo
"""

HOW_IT_WORKS = """### How it works (causal replay-bisection)

1. **Deterministic replay.** A training step is a pure function of (model, optimizer, scheduler, batch, RNG) state.
   Checkpoints store all of it with pre-update semantics. Before trusting any bisection, Trainsect replays a segment
   **twice** and compares SHA-256 parameter hashes — if they disagree, bisection halts.
2. **Collapse predicate (eval-based, never train-spike).** A checkpoint counts as collapsed only by its *eval* loss.
   Transient train-loss spikes on a healthy model never count.
3. **Checkpoint binary search.** Predicate P(c) = "checkpoint c is healthy AND replaying forward reproduces the
   collapse" is monotone, so binary search brackets the trigger to one checkpoint interval in O(log n) replays.
4. **Counterfactual shard bisection.** Replay the trigger interval with each candidate shard's batches replaced by
   benign substitutes. The survivor must pass **necessity** (collapse vanishes without it) **and sufficiency**
   (collapse reproduces with only it).
5. **Evidence report.** JSON + human-readable report with minimal-repro commands, so anyone can re-run the proof.

### Honest scope + limitations

- **CPU, deterministic, single-process stacks only.** GPU replay is expected to fail the bit-exact fidelity check
  (nondeterministic kernels) — the tool **aborts honestly** rather than guessing.
- **Forensic mode never names unproven batches.** Without recorder data, exact batch/step identity is unrecoverable;
  an honest bracket with a confidence interval beats a precise-sounding guess.
- **Distributed training** (DDP/FSDP) is not supported.
"""

with gr.Blocks(title="Trainsect — git bisect for training runs") as demo:
    gr.Markdown(HEADER)
    with gr.Tab("Preloaded poisoned run"):
        gr.Markdown(_funnel_markdown(PRELOADED_REPORT))
        gr.Markdown(_report_markdown(PRELOADED_REPORT))
        gr.Markdown("### Full terminal transcript (preloaded, verbatim)")
        gr.Textbox(PRELOADED_TRANSCRIPT, lines=30, max_lines=40, autoscroll=True, label="demo_transcript.txt")
        gr.Markdown("### Evidence report JSON (preloaded)")
        gr.JSON(PRELOADED_REPORT, label="demo_report.json")
    with gr.Tab("Run live bisection"):
        gr.Markdown(
            "Re-runs the whole magic trick from scratch on this Space's CPU: trains a tiny transformer "
            "(120 steps, poison shard 3 injected), then diagnoses it with no knowledge of the injection. "
            "Takes roughly **1–2 minutes** on the free CPU tier."
        )
        run_btn = gr.Button("▶ Re-run live bisection", variant="primary")
        live_log = gr.Textbox(label="Live terminal output", lines=30, max_lines=40, autoscroll=True)
        live_report = gr.JSON(label="Live evidence report")
        run_btn.click(fn=run_live, inputs=[], outputs=[live_log, live_report])
    with gr.Tab("How it works"):
        gr.Markdown(HOW_IT_WORKS)

demo.queue()
if __name__ == "__main__":
    demo.launch()
