"""CLI: ``trainsect diagnose``, ``trainsect demo``, ``trainsect replay``,
``trainsect download``.

``diagnose`` runs the full pipeline on an instrumented run directory
(checkpoints + batches + sidecar + eval data) and writes a JSON + text
evidence report. With no recorder data it takes the forensic path
(checkpoints alone, multi-seed reconstructed orders, honest confidence).
A directory in HF checkpoint layout is normalized first. ``demo`` runs
the magic trick end to end. ``replay`` is the minimal-repro primitive:
one counterfactual replay from a checkpoint, used by the repro commands
in evidence reports. ``download`` fetches a checkpoint repo from the
HF Hub (read-only) into a local directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .collapse import collapse_threshold, has_collapse
from .diagnose import RunDir, diagnose
from .demo import DEFAULT_CONFIG, run_demo
from .hub import download_checkpoint_repo


def _add_diagnose_args(p):
    p.add_argument("--checkpoints", required=True,
                   help="instrumented run directory, checkpoints-only directory, "
                        "or HF-layout checkpoint directory")
    p.add_argument("--config", default=None, help="JSON file with config overrides")
    p.add_argument("--output", default=None, help="write JSON report to this path")
    p.add_argument("--text-output", default=None, help="write text report to this path")
    p.add_argument("--clean-shard", type=int, default=None,
                   help="shard known to be clean, used for benign substitutes "
                        "(default: shard 0 instrumented / auto forensic)")
    p.add_argument("--dataset", default=None,
                   help="forensic only: .pt file with the batch pool [(x, y, shard), ...]")
    p.add_argument("--eval-data", default=None,
                   help="forensic only: .pt file with the eval split")
    p.add_argument("--num-seeds", type=int, default=8,
                   help="forensic only: confirmation seed count (default 8)")
    p.add_argument("--search-seeds", type=int, default=3,
                   help="forensic only: bracket-search seed count (default 3)")


def cmd_diagnose(args) -> int:
    overrides = None
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
    report = diagnose(args.checkpoints, config_overrides=overrides,
                      clean_shard=args.clean_shard,
                      dataset=args.dataset, eval_data=args.eval_data,
                      num_seeds=args.num_seeds, search_seeds=args.search_seeds)
    print()
    print(report.to_text())
    if args.output:
        with open(args.output, "w") as f:
            f.write(report.to_json())
        print(f"[trainsect] JSON report -> {args.output}")
    if args.text_output:
        with open(args.text_output, "w") as f:
            f.write(report.to_text())
        print(f"[trainsect] text report -> {args.text_output}")
    return 0 if report.confirmed_causal else 1


def cmd_download(args) -> int:
    token = args.token or os.environ.get("HF_TOKEN")
    download_checkpoint_repo(args.repo, args.out, revision=args.revision, token=token)
    print(f"[trainsect] downloaded {args.repo}@{args.revision} -> {args.out}")
    print("[trainsect] diagnose it with: trainsect diagnose --checkpoints "
          f"{args.out} --dataset <pool.pt> --eval-data <eval.pt>")
    return 0


def cmd_demo(args) -> int:
    config = dict(DEFAULT_CONFIG)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.poison_shard is not None:
        config["poison_shard"] = args.poison_shard
    report = run_demo(args.outdir, config)
    return 0 if report["confirmed_causal"] else 1


def cmd_replay(args) -> int:
    run = RunDir(args.checkpoints)
    engine = run.engine
    loader = run.loader
    to_step = args.from_step + args.steps
    thresh, _baseline, _mad = collapse_threshold(run.loss_log)
    if args.exclude_shard is not None and args.only_shard is not None:
        print("error: --exclude-shard and --only-shard are mutually exclusive",
              file=sys.stderr)
        return 2
    benign = loader.benign_batches(0)
    replace = None
    if args.exclude_shard is not None:
        def replace(step, x, y, s, _shard=args.exclude_shard):
            if s == _shard:
                sub = benign[step % len(benign)]
                return sub.x, sub.y
            return x, y
    elif args.only_shard is not None:
        def replace(step, x, y, s, _shard=args.only_shard):
            if s != _shard:
                sub = benign[step % len(benign)]
                return sub.x, sub.y
            return x, y
    elif args.only_batch is not None:
        target = loader.reproduce(args.only_batch)
        if target is None:
            print(f"error: no batch with fingerprint {args.only_batch}",
                  file=sys.stderr)
            return 2
        def replace(step, x, y, s, _target=target):
            if step == _target.step:
                return x, y
            sub = benign[step % len(benign)]
            return sub.x, sub.y
    losses, evals, _ = engine.replay(args.from_step, to_step, replace=replace)
    collapse, at = has_collapse(evals, thresh)
    print(f"[replay] {args.from_step} -> {to_step} eval_threshold={thresh:.3f}")
    print(f"[replay] eval trajectory: { {s: round(v, 3) for s, v in evals.items()} }")
    print(f"[replay] collapse: {collapse} (first crossing at step {at})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trainsect",
        description="git bisect for training runs: name the exact step/shard/batch that caused a collapse, proven by replay.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="diagnose an instrumented run directory")
    _add_diagnose_args(d)
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("demo", help="run the magic trick: tiny transformer + poison shard")
    m.add_argument("--outdir", default=os.path.join(os.getcwd(), "demos"),
                   help="directory for demo run + report + transcript")
    m.add_argument("--seed", type=int, default=None)
    m.add_argument("--poison-shard", type=int, default=None)
    m.set_defaults(func=cmd_demo)

    r = sub.add_parser("replay", help="one (counterfactual) replay from a checkpoint")
    r.add_argument("--checkpoints", required=True, help="instrumented run directory")
    r.add_argument("--from-step", type=int, required=True)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--exclude-shard", type=int, default=None)
    r.add_argument("--only-shard", type=int, default=None)
    r.add_argument("--only-batch", default=None, help="batch fingerprint")
    r.set_defaults(func=cmd_replay)

    dl = sub.add_parser("download", help="download a checkpoint repo from the HF Hub (read-only)")
    dl.add_argument("--repo", required=True, help="Hub repo id, e.g. org/model")
    dl.add_argument("--out", required=True, help="local directory to download into")
    dl.add_argument("--revision", default="main")
    dl.add_argument("--token", default=None, help="HF token (or HF_TOKEN env); public repos need none")
    dl.set_defaults(func=cmd_download)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
