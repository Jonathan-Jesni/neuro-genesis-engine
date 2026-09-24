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
from typing import Iterator

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


@dataclass
class Batch:
    x: Tensor
    y: Tensor
    domain_id: int      # evaluation/oracle only
    phase_step: int     # step index within the current phase
    step: int           # global step index


class DomainStream:
    """Yields ``steps_per_phase`` batches from each domain in order."""

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
        self.arrays = [load_tokens(data_dir, d, "train") for d in domains]
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

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        for d_id, arr in enumerate(self.arrays):
            hi = len(arr) - self.seq_len - 1
            for ps in range(self.steps_per_phase):
                starts = self.rng.integers(0, hi, size=self.batch_size)
                x, y = _windows(arr, starts, self.seq_len)
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
