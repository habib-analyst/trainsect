"""Trainsect: git bisect for training runs.

Name the exact optimizer step, data shard, and batch that caused a training
collapse -- proven by deterministic replay (necessity + sufficiency), not
by attribution scores.
"""
from .bisect import bisect_checkpoints, confirm, shard_bisect
from .collapse import detect_change_point, has_collapse
from .dataloader import IndexedBatch, IndexedDataLoader
from .demo import generate_demo_run, run_demo
from .diagnose import RunDir, diagnose
from .evidence import EvidenceReport
from .fingerprints import fingerprint_batch, fingerprint_params
from .forensic import diagnose_forensic, has_recorder_data
from .hub import (
    download_checkpoint_repo,
    is_hf_layout,
    load_hf_checkpoint_dir,
    load_state_dict_from_hf,
    normalize_hf_dir,
)
from .models import MLP, TinyTransformer, build_model
from .recorder import (
    Recorder,
    benchmark_overhead,
    get_active_recorder,
    record,
    record_eval,
    recording,
    start_recording,
    stop_recording,
)
from .replay import ReplayEngine

__version__ = "0.2.0"

__all__ = [
    "IndexedBatch",
    "IndexedDataLoader",
    "EvidenceReport",
    "ReplayEngine",
    "RunDir",
    "MLP",
    "TinyTransformer",
    "build_model",
    "bisect_checkpoints",
    "confirm",
    "shard_bisect",
    "detect_change_point",
    "has_collapse",
    "fingerprint_batch",
    "fingerprint_params",
    "diagnose",
    "diagnose_forensic",
    "has_recorder_data",
    "generate_demo_run",
    "run_demo",
    "Recorder",
    "recording",
    "start_recording",
    "stop_recording",
    "record",
    "record_eval",
    "get_active_recorder",
    "benchmark_overhead",
    "is_hf_layout",
    "normalize_hf_dir",
    "load_hf_checkpoint_dir",
    "load_state_dict_from_hf",
    "download_checkpoint_repo",
    "__version__",
]
