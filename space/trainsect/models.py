"""Model architectures Trainsect can replay.

Two architectures are supported in P1:

- ``mlp``: the P0 binary-classification MLP (20 -> 64 -> 64 -> 1), BCE loss.
  Kept for backwards compatibility with the P0 validation run.
- ``transformer``: a nanoGPT-style tiny causal transformer (token + position
  embeddings, N decoder blocks, cross-entropy next-token loss). Used by the
  P1 magic-trick demo.

Both expose the same interface: ``build_model(config)``,
``train_loss(model, batch)`` and ``eval_loss(model, eval_data)``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- MLP (P0)
class MLP(nn.Module):
    def __init__(self, dim: int = 20, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------- Tiny transformer (P1 demo)
class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """nanoGPT-style causal LM: token emb + pos emb + decoder blocks."""

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 16,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList(
            [_Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Causal mask: -inf above the diagonal, 0 elsewhere. Built once.
        mask = torch.full((seq_len, seq_len), float("-inf"))
        self.register_buffer("causal_mask", torch.triu(mask, diagonal=1))

    def forward(self, idx):
        _, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t))
        for block in self.blocks:
            x = block(x, self.causal_mask[:t, :t])
        return self.head(self.ln_f(x))


# ------------------------------------------------------------------ factory
def build_model(config: Dict[str, Any]) -> nn.Module:
    arch = config.get("arch", "mlp")
    if arch == "mlp":
        return MLP(dim=config.get("dim", 20), hidden=config.get("hidden", 64))
    if arch == "transformer":
        return TinyTransformer(
            vocab_size=config.get("vocab_size", 64),
            seq_len=config.get("seq_len", 16),
            n_embd=config.get("n_embd", 64),
            n_head=config.get("n_head", 4),
            n_layer=config.get("n_layer", 2),
            dropout=config.get("dropout", 0.0),
        )
    raise ValueError(f"unknown arch {arch!r} (expected 'mlp' or 'transformer')")


def build_optimizer(
    model: nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    kind = config.get("optimizer", "sgd")
    lr = float(config.get("lr", 0.05))
    if kind == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=config.get("momentum", 0.9)
        )
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"unknown optimizer {kind!r}")


def train_loss(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], config: Dict[str, Any]
) -> torch.Tensor:
    """Per-batch training loss (differentiable)."""
    x, y = batch
    if config.get("arch", "mlp") == "transformer":
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return F.binary_cross_entropy_with_logits(model(x), y)


def eval_loss(
    model: nn.Module, eval_data: Dict[str, torch.Tensor], config: Dict[str, Any]
) -> float:
    """Eval loss of the current weights (no gradients)."""
    model.eval()
    with torch.no_grad():
        if config.get("arch", "mlp") == "transformer":
            logits = model(eval_data["Xe"])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                eval_data["ye"].reshape(-1),
            ).item()
        else:
            loss = F.binary_cross_entropy_with_logits(
                model(eval_data["Xe"]), eval_data["ye"]
            ).item()
    model.train()
    return loss
