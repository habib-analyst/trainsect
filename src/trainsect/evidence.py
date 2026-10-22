"""Evidence reports: JSON + human-readable output.

A report must be actionable on its own: step range, shard id, batch
fingerprint, minimal-repro command, necessity/sufficiency results, and
confidence. Forensic (no-recorder) runs must never name a batch they cannot
prove -- see ``confidence`` / ``provenance``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvidenceReport:
    run_dir: str
    verdict: str  # "CONFIRMED (causal)" | "UNCONFIRMED" | "NO COLLAPSE" | "ABORTED" | "FORENSIC (probable)" | "UNPROVEN"
    confirmed_causal: bool
    fidelity_bit_exact: bool
    change_point: Dict[str, object] = field(default_factory=dict)
    bracket: List[int] = field(default_factory=list)
    trigger_step: Optional[int] = None
    shard: Optional[int] = None
    batch_fingerprint: Optional[str] = None
    offending_sample: Dict[str, object] = field(default_factory=dict)
    confirmation: Dict[str, bool] = field(default_factory=dict)
    replays: int = 0
    replayed_steps: int = 0
    wall_time_s: float = 0.0
    confidence: str = ""
    provenance: str = "instrumented"  # instrumented | forensic
    notes: List[str] = field(default_factory=list)
    # Forensic-only detail: state table, per-seed reproduction fractions,
    # frozen-weight ranking, specificity. Empty on instrumented runs.
    forensic: Dict[str, object] = field(default_factory=dict)

    # -- minimal repro --------------------------------------------------------
    def repro_commands(self) -> List[str]:
        if not self.bracket or self.shard is None:
            return []
        lo, hi = self.bracket
        cmds = [
            f"trainsect replay --checkpoints {self.run_dir} "
            f"--from-step {lo} --steps {hi - lo} --exclude-shard {self.shard} "
            f"# collapse vanishes (necessity)",
            f"trainsect replay --checkpoints {self.run_dir} "
            f"--from-step {lo} --steps {hi - lo} --only-shard {self.shard} "
            f"# collapse reproduces (sufficiency)",
        ]
        if self.batch_fingerprint:
            cmds.append(
                f"trainsect replay --checkpoints {self.run_dir} "
                f"--from-step {lo} --steps {hi - lo} --only-batch {self.batch_fingerprint} "
                f"# trigger batch alone"
            )
        return cmds

    # -- serialisation ----------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["minimal_repro"] = self.repro_commands()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        lines = [
            "=== TRAINSECT EVIDENCE REPORT ===",
            f"run:                 {self.run_dir}",
            f"verdict:             {self.verdict}",
            f"provenance:          {self.provenance}",
            f"confidence:          {self.confidence or 'n/a'}",
            f"fidelity:            {'BIT-EXACT' if self.fidelity_bit_exact else 'NON-DETERMINISTIC (do not trust bisection)'}",
            f"change point:        {self.change_point.get('change_point_step')} "
            f"(threshold={self.change_point.get('threshold')})",
            f"step range:          {self.bracket}",
            f"trigger step:        {self.trigger_step}",
            f"offending shard:     {('shard-' + str(self.shard)) if self.shard is not None else 'n/a'}",
            f"batch fingerprint:   {self.batch_fingerprint or 'n/a'}",
            f"offending sample:    {self.offending_sample or 'n/a'}",
            f"necessity:           {self.confirmation.get('necessity')}",
            f"sufficiency:         {self.confirmation.get('sufficiency')}",
            f"replays:             {self.replays} ({self.replayed_steps} replayed steps)",
            f"wall time:           {self.wall_time_s:.1f}s",
            "",
            "minimal repro:",
        ]
        for cmd in self.repro_commands():
            lines.append(f"  {cmd}")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines) + "\n"
