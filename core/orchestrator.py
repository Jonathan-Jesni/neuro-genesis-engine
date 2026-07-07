"""Autonomous training-loop orchestrator for the Neuro-Genesis Engine.

Ties the finished pieces of ``core.moe.dynamic_gating`` into a self-expanding
training loop:

    train step -> rolling loss-spike detection -> package failure context ->
    pluggable expert-source generator -> ExpertFoundry registration ->
    continue training with the new expert live.

The module is deliberately standalone-testable: no Antigravity, no external
API, no GPU required. The concrete ``template_mlp_expert_source`` generator is
a placeholder for a future LLM call, but produces working expert source today
so the whole loop runs end-to-end locally (``python core/orchestrator.py``).

Spike-detection choice (documented per spec): **z-score over a rolling
window**, not a fixed ratio. A plain ratio threshold ("loss > 2x average")
misfires during early training, when loss is large and falling fast and even
healthy steps jump around; a z-score adapts to the observed noise level of the
recent window. Guards applied: no detection until ``min_history`` samples
exist, and the standard deviation is floored (absolute ``eps`` plus
``rel_floor * |mean|``) so a near-flat loss curve cannot fire on numerically
trivial jitter. A detected spike is NOT added to the rolling history -- the
outlier must not inflate the baseline it was measured against.

Hard constraints inherited from the gate/foundry contracts (do not weaken):
  * The full autograd step (forward -> backward -> optimizer.step) runs under
    ``gate.expand_lock``; expansion is therefore never concurrent with a
    backward pass.
  * Registration (which internally calls ``gate.expand``) happens strictly
    AFTER the training step completes and the lock is RELEASED --
    ``expand_lock`` is a plain (non-reentrant) ``threading.Lock``, so calling
    the foundry while holding it would deadlock.
  * The orchestrator never appends to the expert list or calls
    ``gate.expand`` directly; ``ExpertFoundry.register_expert_from_source``
    owns that ordering and its transactional rollback.
  * The live expert count is always read from ``gate.num_experts`` /
    ``gates.shape[1]``, never cached across an expansion.

Logging: every step, spike, candidate generation, registration outcome, and
expert-count change is appended to a JSON-lines file with a wall-clock
timestamp. The log alone is sufficient to reconstruct "what happened and
when" for a run (see ``tests/test_orchestrator.py``).
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.moe.dynamic_gating import (
    DynamicNoisyTopKGate,
    ExpertFoundry,
    ExpertValidationError,
)

__all__ = [
    "SpikeInfo",
    "RollingSpikeDetector",
    "FailureContext",
    "template_mlp_expert_source",
    "JsonlLogger",
    "StepResult",
    "TrainingOrchestrator",
]


# =============================================================================
# Spike detection
# =============================================================================
@dataclass(frozen=True)
class SpikeInfo:
    """Detector output for a loss value judged to be a spike."""

    loss: float
    rolling_mean: float
    rolling_std: float
    z_score: float


class RollingSpikeDetector:
    """Rolling z-score loss-spike detector.

    Flags a spike when ``(loss - mean) / std_floored > z_threshold`` over the
    last ``window`` non-spike losses. See the module docstring for why z-score
    was chosen over a fixed ratio, and for the std-floor / warmup guards.
    """

    def __init__(
        self,
        window: int = 50,
        z_threshold: float = 4.0,
        min_history: int = 10,
        eps: float = 1e-8,
        rel_floor: float = 0.01,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if min_history < 2:
            raise ValueError(f"min_history must be >= 2, got {min_history}")
        self.window = window
        self.z_threshold = z_threshold
        self.min_history = min_history
        self.eps = eps
        self.rel_floor = rel_floor
        self._history: collections.deque[float] = collections.deque(maxlen=window)

    def update(self, loss: float) -> Optional[SpikeInfo]:
        """Feed one loss; return ``SpikeInfo`` if it is a spike, else None.

        Non-spike losses enter the rolling history; spikes do not (an outlier
        must not inflate the baseline it was measured against). Non-finite
        losses are treated as spikes with ``z_score=inf`` and are likewise
        kept out of the history.
        """
        if not math.isfinite(loss):
            if len(self._history) >= self.min_history:
                mean = sum(self._history) / len(self._history)
                return SpikeInfo(loss=loss, rolling_mean=mean,
                                 rolling_std=0.0, z_score=math.inf)
            # Not enough history to judge; swallow it into the warmup? No --
            # a NaN in warmup is still never baseline material. Just ignore.
            return None

        if len(self._history) >= self.min_history:
            n = len(self._history)
            mean = sum(self._history) / n
            var = sum((v - mean) ** 2 for v in self._history) / n
            std = math.sqrt(var)
            # Floor the std: absolute eps + a fraction of the mean, so a flat
            # loss curve needs a genuinely proportional jump to fire.
            std_floored = max(std, self.eps, self.rel_floor * abs(mean))
            z = (loss - mean) / std_floored
            if z > self.z_threshold:
                return SpikeInfo(loss=loss, rolling_mean=mean,
                                 rolling_std=std, z_score=z)

        self._history.append(loss)
        return None


# =============================================================================
# Failure context + placeholder generator
# =============================================================================
@dataclass(frozen=True)
class FailureContext:
    """Everything a source generator gets to know about a detected spike."""

    step: int
    loss: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    num_experts: int
    input_dim: int
    output_dim: int
    batch: Tensor  # detached clone of the input batch that triggered the spike


def template_mlp_expert_source(ctx: FailureContext) -> str:
    """Placeholder expert-source generator (future LLM call goes here).

    Emits a 2-layer GELU MLP sized to the failure context's dims. Obeys the
    foundry sandbox contract: no imports, zero-arg constructor, only the
    pre-injected ``nn`` / ``F`` names. The class name embeds the triggering
    step so registrations are traceable in logs and audit records.
    """
    hidden = max(4, 2 * ctx.input_dim)
    return f"""
class GeneratedExpertStep{ctx.step}(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear({ctx.input_dim}, {hidden})
        self.fc2 = nn.Linear({hidden}, {ctx.output_dim})

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
"""


# =============================================================================
# Structured logging
# =============================================================================
class JsonlLogger:
    """Append-only JSON-lines event log, flushed per write.

    Each line is one JSON object: ``{"ts": <unix float>, "event": <str>, ...}``.
    Flushing per event means a crashed run still leaves a complete, parseable
    prefix on disk.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _batch_stats(x: Tensor) -> dict[str, Any]:
    """Compact, JSON-safe summary of a batch (never the raw tensor)."""
    xf = x.detach().float()
    return {
        "shape": list(x.shape),
        "mean": xf.mean().item(),
        "std": xf.std().item(),
        "abs_max": xf.abs().max().item(),
        "l2_norm": xf.norm().item(),
    }


# =============================================================================
# The orchestrator
# =============================================================================
@dataclass(frozen=True)
class StepResult:
    """What one ``train_step`` did, for callers that drive the loop manually."""

    step: int
    loss: float
    aux_loss: float
    num_experts: int
    spiked: bool
    registered: bool


class TrainingOrchestrator:
    """Standard MoE training loop instrumented with autonomous expert growth.

    One ``train_step`` = gate forward -> dense top-k expert combine -> task
    loss + aux loss -> backward -> optimizer step, all under
    ``gate.expand_lock``; then (lock released) spike detection and, on a
    spike, candidate generation + transactional foundry registration.

    A rejected candidate (``ExpertValidationError``) or a crashing generator
    is logged and training continues -- a bad candidate must never kill the
    loop. The foundry's rollback guarantees state is unchanged on rejection.
    """

    def __init__(
        self,
        gate: DynamicNoisyTopKGate,
        experts: nn.ModuleList,
        foundry: ExpertFoundry,
        optimizer: torch.optim.Optimizer,
        *,
        log_path: Union[str, Path],
        expert_source_generator: Callable[[FailureContext], str] = template_mlp_expert_source,
        detector: Optional[RollingSpikeDetector] = None,
        task_loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
        max_experts: Optional[int] = None,
        spike_cooldown_steps: int = 10,
    ) -> None:
        self.gate = gate
        self.experts = experts
        self.foundry = foundry
        self.optimizer = optimizer
        self.generator = expert_source_generator
        self.detector = detector if detector is not None else RollingSpikeDetector()
        self.task_loss_fn = task_loss_fn
        self.max_experts = max_experts
        self.spike_cooldown_steps = spike_cooldown_steps
        self.logger = JsonlLogger(log_path)

        self.step_count = 0
        self._last_registration_step: Optional[int] = None
        # Run counters, surfaced in run_end and the __main__ summary.
        self.spikes_seen = 0
        self.registrations = 0
        self.rejections = 0

    # -- forward ---------------------------------------------------------
    def _moe_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Route ``x`` and combine expert outputs. Returns (y_hat, aux_loss).

        Dense combine: iterate exactly the ``gates.shape[1]`` columns of the
        snapshot the gate returned. By the append-only contract those columns
        are always a valid prefix of ``self.experts``, even if a registration
        landed between the gate call and this loop. Rows not routed to an
        expert have a zero gate weight, so skipping the compute for all-zero
        columns is just an optimisation, not a correctness requirement.
        """
        out = self.gate(x)
        gates = out.gates
        y_hat: Optional[Tensor] = None
        for i in range(gates.shape[1]):
            col = gates[:, i : i + 1]
            if not bool((col != 0).any()):
                continue  # no token routed here this batch
            contrib = col * self.experts[i](x)
            y_hat = contrib if y_hat is None else y_hat + contrib
        if y_hat is None:  # k>=1 guarantees this never triggers; belt & braces
            y_hat = torch.zeros(x.shape[0], self.foundry.expert_output_dim,
                                dtype=gates.dtype, device=x.device)
        return y_hat, out.aux_loss

    # -- one step ----------------------------------------------------------
    def train_step(self, x: Tensor, y: Tensor) -> StepResult:
        """Run one training step; detect spikes; maybe grow an expert."""
        self.step_count += 1
        step = self.step_count

        # --- the autograd step, serialized against expansion --------------
        # Contract: expand() must never run mid-backward. Holding expand_lock
        # across forward+backward+step guarantees that; expand() takes the
        # same lock internally.
        with self.gate.expand_lock:
            self.optimizer.zero_grad()
            y_hat, aux_loss = self._moe_forward(x)
            task_loss = self.task_loss_fn(y_hat, y)
            total = task_loss + aux_loss
            total.backward()
            self.optimizer.step()

        loss_val = float(task_loss.detach())
        aux_val = float(aux_loss.detach())
        self.logger.log(
            "step",
            step=step,
            loss=loss_val,
            aux_loss=aux_val,
            num_experts=self.gate.num_experts,
        )

        # --- spike handling: strictly post-step, lock RELEASED -------------
        # (expand_lock is non-reentrant; registration acquires it internally.)
        spike = self.detector.update(loss_val)
        registered = False
        if spike is not None:
            self.spikes_seen += 1
            self.logger.log(
                "spike",
                step=step,
                loss=spike.loss,
                rolling_mean=spike.rolling_mean,
                rolling_std=spike.rolling_std,
                z_score=spike.z_score,
                num_experts=self.gate.num_experts,
                batch=_batch_stats(x),
            )
            registered = self._handle_spike(step, spike, x)

        return StepResult(
            step=step,
            loss=loss_val,
            aux_loss=aux_val,
            num_experts=self.gate.num_experts,
            spiked=spike is not None,
            registered=registered,
        )

    # -- spike -> candidate -> registration ---------------------------------
    def _handle_spike(self, step: int, spike: SpikeInfo, x: Tensor) -> bool:
        """Generate + register a candidate expert. Returns True on success."""
        # Cap and cooldown gates first -- both are logged so the demo log can
        # explain why a spike produced no expert.
        if self.max_experts is not None and self.gate.num_experts >= self.max_experts:
            self.logger.log(
                "registration_skipped", step=step, reason="max_experts_reached",
                num_experts=self.gate.num_experts, max_experts=self.max_experts,
            )
            return False
        if (
            self._last_registration_step is not None
            and step - self._last_registration_step < self.spike_cooldown_steps
        ):
            self.logger.log(
                "registration_skipped", step=step, reason="cooldown",
                last_registration_step=self._last_registration_step,
                cooldown_steps=self.spike_cooldown_steps,
            )
            return False

        ctx = FailureContext(
            step=step,
            loss=spike.loss,
            rolling_mean=spike.rolling_mean,
            rolling_std=spike.rolling_std,
            z_score=spike.z_score,
            num_experts=self.gate.num_experts,
            input_dim=self.foundry.expert_input_dim,
            output_dim=self.foundry.expert_output_dim,
            batch=x.detach().clone(),
        )

        # A crashing generator must not kill the training loop either.
        try:
            source = self.generator(ctx)
        except Exception as exc:  # noqa: BLE001 - deliberate loop shield
            self.rejections += 1
            self.logger.log(
                "registration_rejected", step=step, stage="generation",
                reason=repr(exc), num_experts=self.gate.num_experts,
            )
            return False

        self.logger.log(
            "candidate_generated",
            step=step,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source_chars=len(source),
        )

        old_count = self.gate.num_experts
        try:
            reg = self.foundry.register_expert_from_source(
                source, optimizer=self.optimizer
            )
        except ExpertValidationError as exc:
            # Foundry rollback guarantees gate/experts/optimizer are unchanged.
            self.rejections += 1
            self.logger.log(
                "registration_rejected", step=step, stage="validation",
                reason=str(exc), num_experts=self.gate.num_experts,
            )
            return False

        self.registrations += 1
        self._last_registration_step = step
        self.logger.log(
            "registration_success",
            step=step,
            index=reg.index,
            class_name=reg.class_name,
            source_sha256=reg.source_sha256,
            build_seconds=reg.build_seconds,
        )
        self.logger.log(
            "expert_count_change",
            step=step,
            old=old_count,
            new=self.gate.num_experts,
        )
        return True

    # -- convenience loop -----------------------------------------------------
    def run(
        self,
        steps: int,
        data_fn: Callable[[int], tuple[Tensor, Tensor]],
    ) -> list[StepResult]:
        """Run ``steps`` training steps, pulling batches from ``data_fn(step)``."""
        self.logger.log(
            "run_start",
            steps=steps,
            num_experts=self.gate.num_experts,
            k=self.gate.k,
            in_features=self.gate.in_features,
            expert_output_dim=self.foundry.expert_output_dim,
            max_experts=self.max_experts,
            spike_cooldown_steps=self.spike_cooldown_steps,
            detector={
                "window": getattr(self.detector, "window", None),
                "z_threshold": getattr(self.detector, "z_threshold", None),
                "min_history": getattr(self.detector, "min_history", None),
            },
        )
        results = [self.train_step(*data_fn(self.step_count + 1)) for _ in range(steps)]
        self.logger.log(
            "run_end",
            steps=len(results),
            num_experts=self.gate.num_experts,
            spikes_seen=self.spikes_seen,
            registrations=self.registrations,
            rejections=self.rejections,
        )
        return results


# =============================================================================
# Standalone demo: watch spike -> generate -> register -> continue, locally.
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    IN_DIM = 16
    OUT_DIM = 16
    N_STEPS = 300
    OOD_EVERY = 40  # inject an out-of-distribution batch every N steps

    gate = DynamicNoisyTopKGate(IN_DIM, num_experts=4, k=2).train()
    experts = nn.ModuleList(nn.Linear(IN_DIM, OUT_DIM) for _ in range(4))
    foundry = ExpertFoundry(gate, experts, expert_input_dim=IN_DIM,
                            expert_output_dim=OUT_DIM)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + list(experts.parameters()), lr=1e-3
    )

    # Synthetic regression task: y = x @ W_true + noise.
    W_true = torch.randn(IN_DIM, OUT_DIM) * 0.5

    def data_fn(step: int) -> tuple[Tensor, Tensor]:
        x = torch.randn(32, IN_DIM)
        if step % OOD_EVERY == 0:
            x = x * 8.0  # OOD scale blow-up -> engineered loss spike
        y = x @ W_true + 0.01 * torch.randn(32, OUT_DIM)
        return x, y

    orch = TrainingOrchestrator(
        gate, experts, foundry, optimizer,
        log_path="orchestrator_run.jsonl",
        detector=RollingSpikeDetector(window=30, z_threshold=4.0, min_history=10),
        spike_cooldown_steps=20,
        max_experts=12,
    )

    print(f"running {N_STEPS} steps (OOD batch every {OOD_EVERY})...")
    results = orch.run(N_STEPS, data_fn)

    print("\n=== run summary ===")
    print(f"steps          : {len(results)}")
    print(f"spikes seen    : {orch.spikes_seen}")
    print(f"registrations  : {orch.registrations}")
    print(f"rejections     : {orch.rejections}")
    print(f"experts        : 4 -> {gate.num_experts} (len(experts)={len(experts)})")
    print(f"final task loss: {results[-1].loss:.4f}")
    print(f"log written to : {orch.logger.path}")
    orch.logger.close()
