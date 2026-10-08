"""Sequential multi-domain token stream + fixed held-out eval batches.

Reads the uint16 token arrays written by ``experiments.prepare_data``. The
training stream presents domains one after another (a phase per domain);
within a phase it samples random windows from that domain. The true domain id
travels with every batch for LOGGING/EVALUATION ONLY — only the oracle arm may
act on it, and nothing feeds it to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch import Tensor


def load_tokens(data_dir: Path, domain: str, split: str) -> np.ndarray:
    path = Path(data_dir) / f"{domain}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m experiments.prepare_data` first"
        )
    return np.load(path, mmap_mode="r")


def _windows(arr: np.ndarray, starts: np.ndarray, seq_len: int) -> tuple[Tensor, Tensor]:
    chunk = np.stack([np.asarray(arr[s:s + seq_len + 1]) for s in starts]).astype(np.int64)
    t = torch.from_numpy(chunk)
    return t[:, :-1], t[:, 1:]


def parse_ramp(spec: str) -> Optional[tuple[str, str, float]]:
    """``"ramp:stories:code:0.5"`` -> ("stories", "code", 0.5): a gradual-drift
    phase whose share of ``code`` rises linearly from 0 at the phase's first
    step to 0.5 at its last. Returns None for any other spec."""
    if not spec.startswith("ramp:"):
        return None
    _, old, new, q = spec.split(":")
    return old, new, float(q)


def parse_phase(spec: str) -> list[tuple[str, float]]:
    """``"stories"`` -> [("stories", 1.0)];
    ``"stories:0.9+code:0.1"`` -> [("stories", 0.9), ("code", 0.1)] (a mixture
    phase: each sequence is drawn from one domain with these probabilities).
    A ramp phase (see :func:`parse_ramp`) reports its phase-average mix."""
    ramp = parse_ramp(spec)
    if ramp is not None:
        old, new, q = ramp
        return [(old, 1.0 - q / 2), (new, q / 2)]
    parts = []
    for item in spec.split("+"):
        name, _, w = item.partition(":")
        parts.append((name.strip(), float(w) if w else 1.0))
    total = sum(w for _, w in parts)
    return [(n, w / total) for n, w in parts]


def base_domains(phases: list[str]) -> list[str]:
    """Distinct underlying domains across all phase specs, in first-seen order."""
    seen: list[str] = []
    for spec in phases:
        for name, _ in parse_phase(spec):
            if name not in seen:
                seen.append(name)
    return seen


@dataclass
class Batch:
    x: Tensor
    y: Tensor
    domain_id: int      # evaluation/oracle only
    phase_step: int     # step index within the current phase
    step: int           # global step index


class DomainStream:
    """Yields ``steps_per_phase`` batches from each phase in order.

    A phase is a domain name or a mixture spec (see :func:`parse_phase`). Pure
    phases use exactly the original sampling path (one ``integers`` call per
    batch), so runs without mixtures reproduce earlier results bit-for-bit.
    """

    def __init__(
        self,
        data_dir: Path,
        domains: list[str],
        steps_per_phase: int,
        batch_size: int,
        seq_len: int,
        seed: int,
    ) -> None:
        self.domains = domains
        self.phases = [parse_phase(s) for s in domains]
        self.data = {d: load_tokens(data_dir, d, "train") for d in base_domains(domains)}
        self.steps_per_phase = steps_per_phase
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    @property
    def total_steps(self) -> int:
        return self.steps_per_phase * len(self.domains)

    def boundaries(self) -> list[int]:
        """Global steps at which a new domain begins (excluding step 0)."""
        return [self.steps_per_phase * i for i in range(1, len(self.domains))]

    def _rows(self, names: list[str], pick) -> tuple[Tensor, Tensor]:
        rows = []
        for k in pick:
            arr = self.data[names[k]]
            start = self.rng.integers(0, len(arr) - self.seq_len - 1)
            rows.append(np.asarray(arr[start:start + self.seq_len + 1]))
        t = torch.from_numpy(np.stack(rows).astype(np.int64))
        return t[:, :-1], t[:, 1:]

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        for d_id, mix in enumerate(self.phases):
            names = [n for n, _ in mix]
            probs = np.array([w for _, w in mix])
            ramp = parse_ramp(self.domains[d_id])
            for ps in range(self.steps_per_phase):
                if ramp is not None:
                    # Gradual drift: P(new domain) grows linearly over the phase.
                    q_t = ramp[2] * ps / max(1, self.steps_per_phase - 1)
                    pick = (self.rng.random(self.batch_size) < q_t).astype(int)
                    x, y = self._rows([ramp[0], ramp[1]], pick)
                elif len(mix) == 1:
                    arr = self.data[names[0]]
                    hi = len(arr) - self.seq_len - 1
                    starts = self.rng.integers(0, hi, size=self.batch_size)
                    x, y = _windows(arr, starts, self.seq_len)
                else:
                    pick = self.rng.choice(len(mix), size=self.batch_size, p=probs)
                    x, y = self._rows(names, pick)
                yield Batch(x, y, d_id, ps, step)
                step += 1


class HeldOut:
    """Fixed, deterministic eval windows per domain (same across arms/seeds)."""

    def __init__(
        self,
        data_dir: Path,
        domains: list[str],
        n_batches: int,
        batch_size: int,
        seq_len: int,
        seed: int = 1234,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.batches: dict[str, list[tuple[Tensor, Tensor]]] = {}
        for d in domains:
            arr = load_tokens(data_dir, d, "val")
            hi = len(arr) - seq_len - 1
            self.batches[d] = [
                _windows(arr, rng.integers(0, hi, size=batch_size), seq_len)
                for _ in range(n_batches)
            ]
