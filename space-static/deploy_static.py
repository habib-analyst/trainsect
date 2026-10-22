#!/usr/bin/env python3
"""Build + deploy the Trainsect STATIC HF Space (habib-analyst/trainsect-demo).

Static = preloaded demo only (funnel visualization, evidence report,
verbatim transcript, methodology). No live run — Static Spaces cannot
execute Python. Free tier; no PRO needed.

Auth: surrogate credential via dynamic_credentials helper, sent as
``Authorization: Bearer hsurr:...`` ONLY to huggingface.co.
Transport: curl subprocess (Python HTTP clients hang on huggingface.co
through this VM's egress proxy). Never prints, logs, or persists the
real credential.
"""
import base64
import html
import json
import os
import subprocess
import sys
import urllib.parse

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import dynamic_credential_entry  # noqa: E402

CRED = "custom.huggingface"
SPACE = "habib-analyst/trainsect-demo"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE}"
CURL = ["curl", "-s", "--max-time", "300"]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOCAL_PATH = "/home/hatch/workspace/trainsect/demos/demo_run"
NEUTRAL_PATH = "demos/demo_run (preloaded)"


def _surrogate() -> str:
    return dynamic_credential_entry(CRED)["surrogate"]


def _curl(args):
    return subprocess.run(CURL + args, capture_output=True, text=True)


def _load_artifacts():
    with open(os.path.join(REPO, "space", "demo_report.json")) as f:
        report = json.load(f)
    with open(os.path.join(REPO, "space", "demo_transcript.txt")) as f:
        transcript = f.read()
    # Neutralize the local path so repro commands read cleanly.
    report = json.loads(json.dumps(report).replace(LOCAL_PATH, NEUTRAL_PATH))
    transcript = transcript.replace(LOCAL_PATH, NEUTRAL_PATH)
    return report, transcript


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px;color:#1a1a1a;line-height:1.6}
h1{font-size:1.9em;margin-bottom:0}
h2{margin-top:1.6em}
.tagline{font-size:1.1em;color:#444}
.banner{background:#fff8e1;border:1px solid #f0c929;border-radius:8px;padding:12px 16px;margin:16px 0}
.banner b{color:#7a5b00}
.tabs{display:flex;gap:4px;border-bottom:2px solid #ddd;margin-top:20px}
.tab-btn{background:none;border:none;padding:10px 18px;font-size:1em;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px}
.tab-btn.active{border-bottom-color:#2563eb;font-weight:700;color:#2563eb}
.tab{display:none}
.tab.active{display:block}
.funnel{margin:20px 0}
.frow{display:flex;align-items:center;gap:12px;margin:6px 0}
.fbar{height:34px;border-radius:6px;color:#fff;font-weight:700;display:flex;align-items:center;padding:0 12px;white-space:nowrap;overflow:hidden}
.flabel{font-size:.95em}
code{background:#f1f3f5;border-radius:4px;padding:2px 6px;font-size:.9em}
pre{background:#111827;color:#e5e7eb;border-radius:8px;padding:14px;overflow-x:auto;font-size:.85em;line-height:1.5}
pre.cmd{background:#f8fafc;color:#111827;border:1px solid #e2e8f0}
table{border-collapse:collapse;width:100%;margin:12px 0}
td,th{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f8fafc}
.verdict{color:#15803d;font-weight:700}
ol li,ul li{margin:6px 0}
footer{margin-top:32px;color:#666;font-size:.9em;border-top:1px solid #eee;padding-top:12px}
"""

HOW_IT_WORKS = """
<h2>How it works — causal replay-bisection</h2>
<ol>
<li><b>Deterministic replay.</b> A training step is a pure function of (model, optimizer, scheduler, batch, RNG) state.
Checkpoints store all of it with pre-update semantics. Before trusting any bisection, Trainsect replays a segment
<b>twice</b> and compares SHA-256 parameter hashes — if they disagree, bisection halts.</li>
<li><b>Collapse predicate (eval-based, never train-spike).</b> A checkpoint counts as collapsed only by its <i>eval</i> loss.
Transient train-loss spikes on a healthy model never count.</li>
<li><b>Checkpoint binary search.</b> Predicate P(c) = "checkpoint c is healthy AND replaying forward reproduces the
collapse" is monotone, so binary search brackets the trigger to one checkpoint interval in O(log n) replays.</li>
<li><b>Counterfactual shard bisection.</b> Replay the trigger interval with each candidate shard's batches replaced by
benign substitutes. The survivor must pass <b>necessity</b> (collapse vanishes without it) <b>and sufficiency</b>
(collapse reproduces with only it).</li>
<li><b>Evidence report.</b> JSON + human-readable report with minimal-repro commands, so anyone can re-run the proof.</li>
</ol>
<h2>Honest scope + limitations</h2>
<ul>
<li><b>CPU, deterministic, single-process stacks only.</b> GPU replay is expected to fail the bit-exact fidelity check
(nondeterministic kernels) — the tool <b>aborts honestly</b> rather than guessing.</li>
<li><b>Forensic mode never names unproven batches.</b> Without recorder data, exact batch/step identity is unrecoverable;
an honest bracket with a confidence interval beats a precise-sounding guess.</li>
<li><b>Distributed training</b> (DDP/FSDP) is not supported.</li>
</ul>
<h2>Forensic mode — checkpoints alone</h2>
<p><code>diagnose</code> takes the forensic path automatically when a directory lacks recorder data. It reconstructs
batch orders from the known pool under multiple seeds (3 for the bracket search, 8 for confirmation) and names a shard
only if necessity + sufficiency + specificity all hold <i>and</i> frozen-weight loss ranking agrees. Otherwise it reports
<b>UNPROVEN</b> with what evidence would settle it. Forensic mode <b>never</b> sets a batch fingerprint or trigger step,
and its verdict is never CONFIRMED (causal).</p>
<p>Measured on the two demo runs, stripped of recorder data: the P0 validation run (MLP) diagnoses
<b>FORENSIC (probable), shard-2</b>, bracket [40, 80) — collapse reproduces under 8/8 reconstructed orders.
The transformer demo diagnoses <b>UNPROVEN</b>, no shard named — its collapse is concentration-dependent and
reconstructed orders cannot reproduce it, so no claim is made.</p>
"""

METHODOLOGY = """
<h2>Methodology — the novel claim</h2>
<p>Every existing training-debugger bisects the <b>wrong axis</b>:</p>
<ul>
<li><b>Code axis</b> — <code>git bisect</code> (and rldk's <code>bisect</code>) finds the bad <i>commit</i> by re-running training per commit.</li>
<li><b>Release axis</b> — culprit bisects <i>checkpoint releases</i> across runs (robotics inference stacks).</li>
<li><b>Approximate axis</b> — TRAK / influence functions / TracIn assign <i>scores</i> to samples via linearization, projection, or first-order Taylor approximations.</li>
</ul>
<p>Trainsect bisects the <b>training-time axis itself</b>: it treats the optimizer trajectory as a deterministic,
replayable computation, reloads optimizer state at step <i>t</i>, replays forward, and binary-searches the exact
step-range where the collapse triggers. Then it bisects the data-shard axis with <b>counterfactual replay</b>
(collapse disappears iff shard S is excluded) to name the exact shard, batch, and sample.</p>
<p>The output is not a ranked hypothesis. It is a falsifiable, re-runnable minimal repro. No existing tool combines
(a) deterministic optimizer-step replay across checkpoints, (b) binary search over the checkpoint timeline, and
(c) causal data-shard bisection.</p>
<h2>Deterministic replay</h2>
<p>A training step is deterministic given complete state: θ<sub>t+1</sub>, opt<sub>t+1</sub>, rng<sub>t+1</sub> =
STEP(θ<sub>t</sub>, opt<sub>t</sub>, batch<sub>t</sub>, rng<sub>t</sub>). Checkpoints capture model + optimizer +
scheduler state; the one-line recorder hook captures dataloader cursor, batch sequence, RNG states, worker seeds, and
determinism flags. Before trusting any bisection, Trainsect replays t → t+k <b>twice</b> and compares SHA-256
parameter hashes — if they disagree, bisection halts and reports the divergence.</p>
<h2>Collapse detection</h2>
<p>Two stages. Stage A is cheap and runs on logs: change-point detection on the eval-loss curve (metric predicate +
binary segmentation). Stage B is causal and runs on replay: P(c) = checkpoint <i>c</i> is healthy (its own eval loss
below threshold) AND replaying forward c → B reproduces the collapse. The predicate is <b>eval-based, not
train-loss-spike-based</b> — a healthy model can show transient train-loss spikes that a spike-based predicate mistakes
for collapses. Eval loss is the collapse signal; per-step train loss pinpoints the trigger batch <i>inside</i> an
already-established collapse bracket.</p>
<h2>Checkpoint binary search + shard bisect</h2>
<p>Checkpoints at steps {c<sub>0</sub> &lt; c<sub>1</sub> &lt; … &lt; c<sub>n</sub>}, known-bad point B. P(c) is
monotone in <i>c</i>, so binary search brackets the trigger to one checkpoint interval in O(log n) replays. Then
counterfactual shard bisection: replay the trigger interval with each candidate shard's batches replaced by benign
substitutes from a known-clean shard. The survivor must pass <b>necessity</b> and <b>sufficiency</b>. Batch → sample:
within the trigger batch, the offending sample is the one whose per-sample loss dominates the batch.
The evidence report ships minimal-repro commands for all three counterfactuals.</p>
<h2>Instrumented mode — the one-line recorder hook</h2>
<p><code>start_recording(...) / record(step, batch, ...) / stop_recording(...)</code> — <code>record()</code> is called
<i>after</i> the step's update, preserving pre-update checkpoint semantics. Measured overhead 0.7–1.0% of step time.
A hook-recorded run replays bit-exact.</p>
<h2>Gap verification (2026-10-14)</h2>
<p>Checked against rldk (bisects code commits), culprit (bisects checkpoint releases across runs, robotics),
merriv (bisects precomputed labels), NeuralDBG (in-loop hypotheses without replay), and TRAK/influence functions
(approximate scores, not causal replay). Manual data bisection exists only as practitioner folklore. Nothing does
deterministic optimizer-step replay + checkpoint binary search + data-shard bisect.</p>
<p>Full methodology: <code>docs/P0-methodology.md</code> in the repo.</p>
"""

JS_TABS = """
<script>
function showTab(id){
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});
  document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById(id).classList.add('active');
  document.getElementById('btn-'+id).classList.add('active');
}
</script>
"""


def build_index(report, transcript):
    cp = report["change_point"]
    conf = report["confirmation"]
    esc_transcript = html.escape(transcript)
    funnel = f"""
<h2>How the search narrows to the exact shard</h2>
<div class="funnel">
  <div class="frow"><div class="fbar" style="width:100%;background:#1e3a8a">full run — 120 steps, 6 shards × 20 batches</div></div>
  <div class="frow"><div class="fbar" style="width:82%;background:#1d4ed8">Stage A — symptom: eval-loss change point at step {cp["change_point_step"]}</div></div>
  <div class="frow"><div class="fbar" style="width:64%;background:#2563eb">Stage B — bisect: checkpoint binary search → bracket {tuple(report["bracket"])}</div></div>
  <div class="frow"><div class="fbar" style="width:46%;background:#3b82f6">Stage C — pinpoint: trigger step {report["trigger_step"]} (first train spike inside bracket)</div></div>
  <div class="frow"><div class="fbar" style="width:28%;background:#60a5fa">Stage D — shard: offending shard = shard-{report["shard"]}</div></div>
  <div class="frow"><div class="fbar" style="width:14%;background:#16a34a">Stage E — proof: necessity + sufficiency ✓</div></div>
</div>
<p><b>Necessity</b> = replay the trigger interval <i>without</i> shard-3 and the collapse vanishes.
<b>Sufficiency</b> = replay it with <i>only</i> shard-3 and the collapse reproduces.
That pair is a causal proof for this trajectory — not an attribution score.</p>
<ul>
<li>Batch fingerprint: <code>{report["batch_fingerprint"]}</code></li>
<li>Offending sample: index {report["offending_sample"]["sample_index_in_batch"]} in the batch
(loss {report["offending_sample"]["sample_loss"]:.2f} vs batch mean {report["offending_sample"]["batch_mean_loss"]:.2f})</li>
<li>Cost: {report["replays"]} replays, {report["replayed_steps"]} replayed steps, {report["wall_time_s"]:.1f} s diagnose</li>
</ul>
"""
    evidence = f"""
<h2>Evidence report (preloaded)</h2>
<table>
<tr><th>Verdict</th><td><span class="verdict">{html.escape(report["verdict"])}</span></td></tr>
<tr><th>Provenance</th><td>{html.escape(report["provenance"])}</td></tr>
<tr><th>Confidence</th><td>{html.escape(report["confidence"])}</td></tr>
<tr><th>Fidelity</th><td>{"BIT-EXACT" if report["fidelity_bit_exact"] else "NOT BIT-EXACT"}</td></tr>
<tr><th>Change point</th><td>step {cp["change_point_step"]} (threshold {cp["threshold"]}, baseline {cp["baseline"]:.4f}, eval at change point {cp["eval_loss_at_cp"]:.3f})</td></tr>
<tr><th>Step range</th><td>{report["bracket"]}</td></tr>
<tr><th>Trigger step</th><td>{report["trigger_step"]}</td></tr>
<tr><th>Offending shard</th><td>shard-{report["shard"]}</td></tr>
<tr><th>Batch fingerprint</th><td><code>{report["batch_fingerprint"]}</code></td></tr>
<tr><th>Necessity</th><td>{conf["necessity"]}</td></tr>
<tr><th>Sufficiency</th><td>{conf["sufficiency"]}</td></tr>
</table>
<h3>Minimal repro</h3>
"""
    for cmd in report["minimal_repro"]:
        evidence += f'<pre class="cmd">{html.escape(cmd)}</pre>\n'
    header = """<h1>🔬 Trainsect — git bisect for training runs</h1>
<p class="tagline"><b>Your training run collapsed. Which exact optimizer step, data shard, and batch caused it?</b>
Trainsect answers with a <b>causal proof, not an attribution score</b>: it reloads optimizer state at a checkpoint,
replays forward deterministically, binary-searches the checkpoint timeline, then counterfactually replays with each
data shard excluded until only the guilty shard remains.</p>
<div class="banner"><b>Preloaded demo.</b> This Static Space shows a real, previously computed diagnosis — the poisoned run below
was trained and diagnosed for real. To run the magic trick live on your machine:
<code>pip install git+https://github.com/habib-analyst/trainsect.git</code> then <code>trainsect demo</code>.</div>
"""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trainsect — git bisect for training runs</title>
<style>{CSS}</style></head>
<body>
{header}
<div class="tabs">
<button class="tab-btn active" id="btn-tab-run" onclick="showTab('tab-run')">Preloaded poisoned run</button>
<button class="tab-btn" id="btn-tab-how" onclick="showTab('tab-how')">How it works</button>
<button class="tab-btn" id="btn-tab-method" onclick="showTab('tab-method')">Methodology</button>
</div>
<div class="tab active" id="tab-run">
{funnel}
{evidence}
<h3>Full terminal transcript (preloaded, verbatim)</h3>
<pre>{esc_transcript}</pre>
</div>
<div class="tab" id="tab-how">{HOW_IT_WORKS}</div>
<div class="tab" id="tab-method">{METHODOLOGY}</div>
<footer>Source: <a href="https://github.com/habib-analyst/trainsect">github.com/habib-analyst/trainsect</a> · MIT license ·
Methodology: <code>docs/P0-methodology.md</code> in the repo</footer>
{JS_TABS}
</body></html>
"""


SPACE_README = """---
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
"""


def space_exists() -> bool:
    out = _curl(["-o", "/dev/null", "-w", "%{http_code}",
                 f"https://huggingface.co/api/spaces/{SPACE}"])
    return out.stdout.strip() == "200"


def create_space() -> None:
    payload = json.dumps(
        {"type": "space", "name": "trainsect-demo",
         "sdk": "static", "private": False}
    )
    out = _curl(["-X", "POST", "https://huggingface.co/api/repos/create",
                 "-H", f"Authorization: Bearer {_surrogate()}",
                 "-H", "Content-Type: application/json",
                 "-d", payload])
    print("create_space ->", out.stdout[:500])
    if out.returncode != 0 or '"error"' in out.stdout:
        raise RuntimeError(f"space creation failed: {out.stdout[:500]}")


def upload_files(file_pairs, message: str) -> None:
    """Upload multiple files in ONE commit via the commit endpoint.

    The legacy /upload endpoint is retired. The commit endpoint takes
    NDJSON (Content-Type: application/x-ndjson): a header line
    ``{"key": "header", "value": {"summary": ..., "description": ...}}``
    followed by one JSON line per file op:
    ``{"key": "file", "value": {"content": <base64>, "path": ..., "encoding": "base64"}}``.
    (Format verified against huggingface_hub's _commit_api.py.)
    """
    lines = [json.dumps({"key": "header",
                         "value": {"summary": message, "description": ""}})]
    for local_path, repo_path in file_pairs:
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        lines.append(json.dumps({"key": "file", "value": {
            "content": content_b64, "path": repo_path, "encoding": "base64"}}))
    out = _curl([
        "-X", "POST",
        f"https://huggingface.co/api/spaces/{SPACE}/commit/main",
        "-H", f"Authorization: Bearer {_surrogate()}",
        "-H", "Content-Type: application/x-ndjson",
        "--data-binary", "\n".join(lines),
    ])
    body = out.stdout.strip()
    ok = out.returncode == 0 and "error" not in body
    print(f"commit ({len(file_pairs)} files) -> {'OK' if ok else 'CHECK: ' + body[:500]}")
    if not ok:
        raise RuntimeError(f"commit failed: {body[:500]}")


def main() -> None:
    report, transcript = _load_artifacts()
    index_html = build_index(report, transcript)
    index_path = os.path.join(HERE, "index.html")
    with open(index_path, "w") as f:
        f.write(index_html)
    readme_path = os.path.join(HERE, "README.md")
    with open(readme_path, "w") as f:
        f.write(SPACE_README)
    print(f"built index.html ({len(index_html)} bytes)")
    for needle in ["shard-3", "639a024b373ce19a", "CONFIRMED (causal)",
                   "BIT-EXACT", "necessity", "Methodology"]:
        assert needle in index_html, f"missing in index.html: {needle}"
    print("index.html content checks OK")
    if space_exists():
        print(f"Space already exists: {SPACE_URL}")
    else:
        create_space()
        print(f"Space created: {SPACE_URL}")
    upload_files([(index_path, "index.html"), (readme_path, "README.md")],
                 "Static demo: preloaded poisoned run + methodology")
    # Verify the public page serves the static content.
    out = _curl([SPACE_URL])
    served = "Trainsect" in out.stdout and "shard-3" in out.stdout
    print(f"verify public page -> {'OK' if served else 'CHECK (may need a moment to build)'}")
    print(f"Deployed: {SPACE_URL}")


if __name__ == "__main__":
    main()
