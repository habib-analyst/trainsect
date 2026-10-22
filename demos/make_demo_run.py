"""Reproduce the Trainsect P1 magic trick from scratch.

Regenerates the demo run (nanoGPT-style tiny transformer + injected poison
shard 3), diagnoses it live, and saves the evidence report + terminal
transcript next to this script.

Usage:
    PYTHONPATH=../src python3 make_demo_run.py [--seed N] [--poison-shard N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from trainsect.demo import DEFAULT_CONFIG, run_demo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--poison-shard", type=int, default=None)
    args = ap.parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.poison_shard is not None:
        config["poison_shard"] = args.poison_shard
    demos_dir = os.path.dirname(os.path.abspath(__file__))
    report = run_demo(demos_dir, config)
    assert report["confirmed_causal"], "demo did NOT confirm -- see transcript"
    assert report["shard"] == config["poison_shard"], (
        f"named shard {report['shard']} != injected poison shard "
        f"{config['poison_shard']}"
    )
    print(f"[make_demo_run] OK: named shard-{report['shard']} (poison shard), "
          f"trigger step {report['trigger_step']}")


if __name__ == "__main__":
    main()
