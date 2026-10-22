"""Per-batch content fingerprints.

A batch fingerprint is SHA-256 over the batch's raw bytes plus its shape,
in a fixed order (P0-compatible format). It is stable across runs and
batch content always yields the same fingerprint, and any change to content,
order, shape, or dtype changes it. Fingerprints are the identity Trainsect
uses to name a batch in evidence reports and minimal-repro commands.
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import torch


def fingerprint_tensor(t: torch.Tensor) -> bytes:
    """Hash one tensor: dtype + shape + raw bytes (CPU, contiguous)."""
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(str(tuple(t.shape)).encode("utf-8"))
    h.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return h.digest()


def fingerprint_batch(x: torch.Tensor, y: torch.Tensor, length: int = 16) -> str:
    """SHA-256 hex fingerprint of a batch ``(x, y)``.

    Format is P0-compatible (so P0 sidecar fingerprints verify and
    reproduce exactly): ``SHA256(x_bytes ‖ y_bytes ‖ str(x_shape))``.
    Swapping ``x``/``y``, permuting rows, changing a value, or changing
    ``x``'s shape all change the fingerprint; dtype changes change the
    raw bytes themselves.
    """
    h = hashlib.sha256()
    h.update(x.detach().to("cpu").contiguous().numpy().tobytes())
    h.update(y.detach().to("cpu").contiguous().numpy().tobytes())
    h.update(str(tuple(x.shape)).encode("utf-8"))
    return h.hexdigest()[:length]


def fingerprint_state_dict(state_dict: dict) -> str:
    """SHA-256 hex (16 chars) over a model's flattened parameters, in order."""
    h = hashlib.sha256()
    for _name, p in state_dict.items():
        h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def fingerprint_params(model: torch.nn.Module) -> str:
    """Parameter hash of a model (replay-fidelity check input)."""
    return fingerprint_state_dict(model.state_dict())


# Backwards-compatible alias used by the P0 prototype's naming.
batch_fingerprint = fingerprint_batch
param_hash = fingerprint_params
