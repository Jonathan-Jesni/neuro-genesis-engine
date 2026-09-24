"""Tiny GPT with Mixture-of-Experts MLP layers built on the live-expandable gate.

Every transformer block's MLP is an :class:`MoELayer`: a
:class:`~core.moe.dynamic_gating.DynamicNoisyTopKGate` over an ``nn.ModuleList``
of expert MLPs, wired to an :class:`~core.moe.dynamic_gating.ExpertFoundry`.
Growing the model (``MoEGPT.grow``) registers one new expert in EVERY MoE layer
through the foundry's transactional path, so gate size, expert list and
optimizer state stay consistent (see CLAUDE.md hard invariants).

Initial experts are plain :class:`ExpertMLP` instances; grown experts come from
:func:`expert_source` via the foundry. Both have identical architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.moe.dynamic_gating import DynamicNoisyTopKGate, ExpertFoundry


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    seq_len: int = 256
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 1024           # hidden width of EACH expert
    n_experts: int = 4         # initial experts per MoE layer
    top_k: int = 2
    gate_loss_coef: float = 1e-2
    dropout: float = 0.0


class ExpertMLP(nn.Module):
    """2-layer GELU MLP. Same architecture as :func:`expert_source` emits."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def expert_source(d_model: int, d_ff: int, class_name: str) -> str:
    """Foundry-compatible source for a new expert (no imports, zero-arg ctor)."""
    return f"""
class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear({d_model}, {d_ff})
        self.fc2 = nn.Linear({d_ff}, {d_model})

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
"""


class MoELayer(nn.Module):
    """Sparse top-k MoE over token vectors, with live expansion via the foundry.

    After each forward, ``last_stats`` holds detached routing statistics for the
    batch (per-expert token fraction and load entropy) for logging/triggers.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.gate = DynamicNoisyTopKGate(
            cfg.d_model, cfg.n_experts, k=cfg.top_k, loss_coef=cfg.gate_loss_coef
        )
        self.experts = nn.ModuleList(
            ExpertMLP(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)
        )
        # Not an nn.Module attribute: the foundry holds references to the gate
        # and expert list, it owns no parameters itself.
        self.foundry = ExpertFoundry(self.gate, self.experts, cfg.d_model, cfg.d_model)
        self.last_stats: dict[str, object] = {}
        self._grown = 0

    @property
    def num_experts(self) -> int:
        return self.gate.num_experts

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])                          # [N, d]
        g = self.gate(flat)
        out = torch.zeros_like(flat)
        idx, w = g.top_k_indices, g.top_k_gates                  # [N, k]
        n_exp = g.gates.shape[1]
        for e in range(n_exp):
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = self.experts[e](flat[rows])
            out.index_add_(0, rows, (y * w[rows, slot].unsqueeze(-1)).to(out.dtype))

        with torch.no_grad():
            counts = torch.bincount(idx.reshape(-1), minlength=n_exp).float()
            frac = counts / counts.sum().clamp_min(1.0)
            nz = frac[frac > 0]
            entropy = float(-(nz * nz.log()).sum())
            self.last_stats = {
                "load": frac.tolist(),
                "entropy": entropy,
                # normalised to [0, 1] so it is comparable across expert counts
                "entropy_norm": entropy / math.log(n_exp) if n_exp > 1 else 1.0,
            }
        return out.reshape(shape), g.aux_loss

    def grow(self, optimizer: Optional[torch.optim.Optimizer], tag: str) -> int:
        """Register one new expert through the foundry. Returns its index."""
        self._grown += 1
        name = f"GrownExpertL{self.layer_idx}N{self._grown}{tag}"
        reg = self.foundry.register_expert_from_source(
            expert_source(self.cfg.d_model, self.cfg.d_ff, name), optimizer=optimizer
        )
        return reg.index


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoELayer(cfg, layer_idx)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        m, aux = self.moe(self.ln2(x))
        return x + m, aux


class MoEGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def moe_layers(self) -> list[MoELayer]:
        return [b.moe for b in self.blocks]

    @property
    def num_experts(self) -> int:
        """Experts per MoE layer (all layers grow together)."""
        return self.moe_layers[0].num_experts

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        aux_total = x.new_zeros((), dtype=torch.float32)
        for blk in self.blocks:
            x, aux = blk(x)
            aux_total = aux_total + aux
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total

    def grow(self, optimizer: Optional[torch.optim.Optimizer], tag: str = "") -> None:
        """Add one expert to every MoE layer (transactional per layer).

        Must NOT be called while any gate's ``expand_lock`` is held (the foundry
        re-acquires it). Call between train steps only.
        """
        for layer in self.moe_layers:
            layer.grow(optimizer, tag)

    def routing_stats(self) -> list[dict]:
        return [layer.last_stats for layer in self.moe_layers]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
