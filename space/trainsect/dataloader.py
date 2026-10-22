"""Indexed dataloader: deterministic batch iteration with fingerprints.

Every batch the model ever sees gets an identity ``(step, shard_id,
fingerprint)``. The :class:`IndexedDataLoader` owns the full batch sequence
and provides:

- deterministic iteration in recorded order (no re-shuffling),
- ``shard_id -> [steps]`` mapping,
- replay of any shard subset (with benign substitution for the rest),
- exact reproduction of any batch from its fingerprint.

Fingerprints are computed once at construction and are stable across runs
(see :mod:`trainsect.fingerprints`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .fingerprints import fingerprint_batch


@dataclass(frozen=True)
class IndexedBatch:
    """One batch in the recorded training sequence."""

    step: int
    shard_id: int
    x: torch.Tensor
    y: torch.Tensor
    fingerprint: str

    def matches(self, fingerprint: str) -> bool:
        return self.fingerprint == fingerprint


@dataclass
class IndexedDataLoader:
    """The recorded batch sequence, indexed by step, shard, and fingerprint."""

    batches: List[IndexedBatch] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_raw(
        cls, raw: Sequence[Tuple[torch.Tensor, torch.Tensor, int]]
    ) -> "IndexedDataLoader":
        """Build from ``[(x, y, shard_id), ...]`` in recorded order."""
        batches = [
            IndexedBatch(
                step=i,
                shard_id=int(shard),
                x=x,
                y=y,
                fingerprint=fingerprint_batch(x, y),
            )
            for i, (x, y, shard) in enumerate(raw)
        ]
        return cls(batches=batches)

    @classmethod
    def from_sidecar(
        cls,
        raw: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
        sidecar: Sequence[dict],
    ) -> "IndexedDataLoader":
        """Build from raw batches + a recorder sidecar (fingerprints verified).

        Raises ``ValueError`` if any recorded fingerprint disagrees with the
        recomputed fingerprint of the stored batch -- a tampered or
        mis-ordered batch sequence must never silently pass.
        """
        loader = cls.from_raw(raw)
        for row in sidecar:
            step = int(row["step"])
            recorded = row["batch_fp"]
            actual = loader.batches[step].fingerprint
            if recorded != actual:
                raise ValueError(
                    f"fingerprint mismatch at step {step}: sidecar "
                    f"{recorded} != recomputed {actual}"
                )
        return loader

    # -- basic access -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, step: int) -> IndexedBatch:
        return self.batches[step]

    def shard_ids(self) -> List[int]:
        """Sorted unique shard ids present in the sequence."""
        return sorted({b.shard_id for b in self.batches})

    def steps_for_shard(self, shard_id: int) -> List[int]:
        return [b.step for b in self.batches if b.shard_id == shard_id]

    def batches_in_range(self, lo: int, hi: int) -> List[IndexedBatch]:
        """Batches with ``lo <= step < hi`` (pre-update step semantics)."""
        return [b for b in self.batches if lo <= b.step < hi]

    def shards_in_range(self, lo: int, hi: int) -> List[int]:
        return sorted({b.shard_id for b in self.batches_in_range(lo, hi)})

    def shard_of(self, step: int) -> int:
        return self.batches[step].shard_id

    # -- fingerprint reproduction ------------------------------------------
    def reproduce(self, fingerprint: str) -> Optional[IndexedBatch]:
        """Return the exact batch with this fingerprint, or ``None``."""
        for b in self.batches:
            if b.fingerprint == fingerprint:
                return b
        return None

    def verify_fingerprint(self, fingerprint: str) -> bool:
        """True iff a batch with this fingerprint exists in the sequence."""
        return self.reproduce(fingerprint) is not None

    # -- counterfactual replay support -------------------------------------
    def benign_batches(self, clean_shard: int = 0) -> List[IndexedBatch]:
        """Batches from the clean shard, used as benign substitutes.

        Defaults to shard 0. Callers that know a different shard is clean
        should pass it explicitly; substitution is only valid if the
        substitute shard is itself collapse-free (checked by necessity).
        """
        benign = [b for b in self.batches if b.shard_id == clean_shard]
        if not benign:
            raise ValueError(f"no batches in clean shard {clean_shard}")
        return benign

    def substitute(
        self,
        step: int,
        keep: Callable[[int, int], bool],
        benign: Sequence[IndexedBatch],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` for ``step``: original if ``keep(step, shard)``,
        else a deterministic benign substitute (cycled by step)."""
        batch = self.batches[step]
        if keep(step, batch.shard_id):
            return batch.x, batch.y
        sub = benign[step % len(benign)]
        return sub.x, sub.y

    def iter_range(
        self, lo: int, hi: int
    ) -> Iterable[IndexedBatch]:  # pragma: no cover - trivial
        yield from self.batches_in_range(lo, hi)
