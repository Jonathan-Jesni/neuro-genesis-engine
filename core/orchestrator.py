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

Checkpoint/resume: built for time-boxed shared compute (hard quota kills at
any moment, persistent storage survives). ``save_checkpoint`` writes an
atomic, fully-consistent snapshot -- only ever between train steps -- and
``TrainingOrchestrator.from_checkpoint`` reconstructs everything for exact
resumption: the grown gate, every registered expert (rebuilt from its stored
SOURCE, since a generated class is not reconstructible from a state_dict
alone), Adam momentum, counters, and the spike detector's rolling baseline.
See ``from_checkpoint``'s docstring for the rebuild-before-load ordering and
the optimizer group-structure guard.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
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
    "GenerationAttempt",
    "template_mlp_expert_source",
    "MockLLMGenerator",
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

    # -- checkpoint support ---------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Config + rolling-window contents, JSON/torch.save-safe.

        The window contents matter: resuming with an empty window would leave
        the detector unable to fire until ``min_history`` refills -- i.e. it
        would MISS genuine spikes right after a resume.
        """
        return {
            "window": self.window,
            "z_threshold": self.z_threshold,
            "min_history": self.min_history,
            "eps": self.eps,
            "rel_floor": self.rel_floor,
            "history": list(self._history),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RollingSpikeDetector":
        """Reconstruct a detector with its rolling baseline intact."""
        det = cls(
            window=state["window"],
            z_threshold=state["z_threshold"],
            min_history=state["min_history"],
            eps=state["eps"],
            rel_floor=state["rel_floor"],
        )
        det._history.extend(state["history"])
        return det


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


@dataclass(frozen=True)
class GenerationAttempt:
    """Record of one failed generation attempt, fed back to the generator.

    Carries the exact source string that was tried and the exact
    ``ExpertValidationError`` message it triggered, so a self-correcting
    generator can fix its own mistake instead of blindly re-rolling.
    """

    attempt_number: int
    source: str
    rejection_reason: str


def template_mlp_expert_source(
    ctx: FailureContext, prior_attempt: Optional[GenerationAttempt] = None
) -> str:
    """Placeholder expert-source generator (future LLM call goes here).

    Emits a 2-layer GELU MLP sized to the failure context's dims. Obeys the
    foundry sandbox contract: no imports, zero-arg constructor, only the
    pre-injected ``nn`` / ``F`` names. The class name embeds the triggering
    step so registrations are traceable in logs and audit records.

    Retry contract demonstration: when ``prior_attempt`` is given, every
    dimension below is re-derived from ``ctx`` alone -- NOTHING is recovered
    or parsed from the failed attempt's source. The failed source exists only
    for diagnosis (its rejection_reason); the failure context is the single
    source of truth for what a correct expert must look like. A fixed-shape
    template rarely needs correcting, but any real generator must follow the
    same rule: correct FROM the contract, not from the broken artifact.
    """
    hidden = max(4, 2 * ctx.input_dim)
    suffix = f"Retry{prior_attempt.attempt_number}" if prior_attempt is not None else ""
    return f"""
class GeneratedExpertStep{ctx.step}{suffix}(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear({ctx.input_dim}, {hidden})
        self.fc2 = nn.Linear({hidden}, {ctx.output_dim})

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
"""


class MockLLMGenerator:
    """Deterministic stand-in for a future Fireworks-backed LLM generator.

    Simulates an LLM's REAL failure modes rather than always succeeding: on a
    first attempt it can emit syntactically broken source, source with the
    wrong output dimension, or source containing a forbidden import. On retry
    it "reads" ``prior_attempt.rejection_reason`` and emits corrected source
    (derived entirely from the failure context's dims) that passes validation.
    The correction branch taken is encoded in the generated class name
    (``FixedSyntax`` / ``FixedDim`` / ``FixedImport``) so tests and logs can
    verify the rejection reason genuinely drove the correction.

    .. warning:: FRAGILITY -- do not copy this error-handling pattern into a
        real generator. The retry logic below string-matches substrings of the
        foundry's human-readable ``ExpertValidationError`` prose ("syntax
        error", "output shape", "imports are not allowed"). That text is NOT a
        stable interface -- it can be reworded at any time without notice.
        This is acceptable for a mock/test tool only. A real Fireworks-backed
        generator should consume structured error information (e.g. an
        error-code field added to ``ExpertValidationError``), or simply pass
        the raw reason string to the LLM as correction context -- never
        branch programmatically on exact exception wording.

    Args:
        failure_rate: Probability that a FIRST attempt emits broken source.
        seed: Seed for the internal ``random.Random`` (deterministic tests).
        always_fail: Every attempt (first and retries) emits broken source;
            for exercising retry exhaustion.
        force_mode: Pin the first-attempt failure mode instead of sampling:
            one of ``"syntax"``, ``"wrong_dim"``, ``"forbidden_import"``.
    """

    FAILURE_MODES = ("syntax", "wrong_dim", "forbidden_import")

    def __init__(
        self,
        *,
        failure_rate: float = 1.0,
        seed: int = 0,
        always_fail: bool = False,
        force_mode: Optional[str] = None,
    ) -> None:
        if force_mode is not None and force_mode not in self.FAILURE_MODES:
            raise ValueError(
                f"force_mode must be one of {self.FAILURE_MODES}, got {force_mode!r}"
            )
        self._rng = random.Random(seed)
        self.failure_rate = failure_rate
        self.always_fail = always_fail
        self.force_mode = force_mode

    def __call__(
        self, ctx: FailureContext, prior_attempt: Optional[GenerationAttempt] = None
    ) -> str:
        if self.always_fail:
            return self._broken_source("wrong_dim", ctx)
        if prior_attempt is None:
            if self._rng.random() < self.failure_rate:
                mode = self.force_mode or self._rng.choice(self.FAILURE_MODES)
                return self._broken_source(mode, ctx)
            return self._good_source(ctx, "FirstTry")
        # Retry: pick the correction branch from the rejection reason.
        # (String-matching on exception prose -- MOCK-ONLY, see class docstring.)
        reason = prior_attempt.rejection_reason.lower()
        if "syntax error" in reason:
            label = "FixedSyntax"
        elif "output shape" in reason:
            label = "FixedDim"
        elif "imports are not allowed" in reason or "forbidden pattern" in reason:
            label = "FixedImport"
        else:
            label = "FixedOther"
        return self._good_source(ctx, label)

    def _good_source(self, ctx: FailureContext, label: str) -> str:
        # Correct-by-reconstruction: dims come from the failure context ONLY,
        # never from anything in a failed attempt's source.
        hidden = max(4, 2 * ctx.input_dim)
        return f"""
class MockExpertStep{ctx.step}{label}(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear({ctx.input_dim}, {hidden})
        self.fc2 = nn.Linear({hidden}, {ctx.output_dim})

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
"""

    def _broken_source(self, mode: str, ctx: FailureContext) -> str:
        if mode == "syntax":
            # Unterminated def -> foundry's "syntax error in source: ..."
            return "class Broken(nn.Module):\n    def __init__(self"
        if mode == "wrong_dim":
            # Off-by-one output dim -> "forward() output shape ... != expected"
            return f"""
class MockExpertStep{ctx.step}WrongDim(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear({ctx.input_dim}, {ctx.output_dim + 1})

    def forward(self, x):
        return self.lin(x)
"""
        # forbidden_import -> "imports are not allowed in generated expert source"
        return f"""
import os

class MockExpertStep{ctx.step}Evil(nn.Module):
    def forward(self, x):
        return x
"""


def _rebuild_generated_expert(source: str, class_name: str) -> nn.Module:
    """Re-instantiate a checkpointed generated expert from its source string.

    Sources stored in a checkpoint already passed the FULL foundry validation
    pipeline (static screen, sandboxed exec, interface contract, smoke test)
    at original registration time, so resume only needs to re-exec and
    instantiate. The exec still runs in the same restricted namespace the
    foundry used (reusing its builtin whitelist and filename stamp) so a
    tampered checkpoint gets no wider capabilities than a fresh registration
    -- but note the checkpoint file itself must come from trusted storage;
    this is consistency protection, not an authentication mechanism.
    """
    from core.moe import dynamic_gating as _dg

    compiled = compile(source, _dg._GENERATED_FILENAME, "exec")
    namespace: dict[str, Any] = {
        "__builtins__": dict(_dg._SAFE_BUILTINS),
        "__name__": _dg._GENERATED_FILENAME,
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Tensor": Tensor,
    }
    exec(compiled, namespace)  # noqa: S102 - same sandbox as the foundry
    obj = namespace.get(class_name)
    if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
        raise ValueError(
            f"checkpointed source does not define nn.Module subclass {class_name!r}"
        )
    return obj()


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
    spike, candidate generation + transactional foundry registration with up
    to ``max_generation_attempts`` self-correcting retries (each retry gets a
    :class:`GenerationAttempt` carrying the prior source + rejection reason).

    A rejected candidate (``ExpertValidationError``), a crashing generator,
    or a fully exhausted retry sequence is logged and training continues --
    a bad candidate must never kill the loop. The foundry's rollback
    guarantees state is unchanged on every rejection.
    """

    def __init__(
        self,
        gate: DynamicNoisyTopKGate,
        experts: nn.ModuleList,
        foundry: ExpertFoundry,
        optimizer: torch.optim.Optimizer,
        *,
        log_path: Union[str, Path],
        expert_source_generator: Callable[
            [FailureContext, Optional[GenerationAttempt]], str
        ] = template_mlp_expert_source,
        detector: Optional[RollingSpikeDetector] = None,
        task_loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
        max_experts: Optional[int] = None,
        spike_cooldown_steps: int = 10,
        max_generation_attempts: int = 3,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_every_n_steps: Optional[int] = None,
    ) -> None:
        if max_generation_attempts < 1:
            raise ValueError(
                f"max_generation_attempts must be >= 1, got {max_generation_attempts}"
            )
        self.gate = gate
        self.experts = experts
        self.foundry = foundry
        self.optimizer = optimizer
        self.generator = expert_source_generator
        self.detector = detector if detector is not None else RollingSpikeDetector()
        self.task_loss_fn = task_loss_fn
        self.max_experts = max_experts
        self.spike_cooldown_steps = spike_cooldown_steps
        self.max_generation_attempts = max_generation_attempts
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.logger = JsonlLogger(log_path)

        self.step_count = 0
        # Cooldown marker: the last step at which a generation SEQUENCE ran to
        # completion -- set on registration success AND on retry exhaustion.
        # Exhaustion must arm the cooldown too: under a persistent spike
        # condition (broken generator + sustained OOD data), every subsequent
        # spiking step would otherwise fire a full max_generation_attempts
        # retry sequence with zero throttling. Repeatedly-failing spikes are
        # throttled exactly like repeatedly-succeeding ones.
        self._last_generation_step: Optional[int] = None
        # Run counters, surfaced in run_end and the __main__ summary.
        self.spikes_seen = 0
        self.registrations = 0
        self.rejections = 0

        # Checkpoint bookkeeping. Base experts (constructed by the caller,
        # arbitrary architecture) vs generated experts (registered through the
        # foundry, reconstructible from their stored SOURCE): a state_dict
        # alone cannot rebuild a generated class, so we retain every
        # successfully registered source for from_checkpoint.
        self._n_base_experts = len(experts)
        self._registered_sources: list[dict[str, str]] = []

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

    # -- spike -> candidate -> registration (with self-correcting retries) ---
    def _handle_spike(self, step: int, spike: SpikeInfo, x: Tensor) -> bool:
        """Generate + register a candidate expert, retrying with feedback.

        The generator gets up to ``max_generation_attempts`` tries; each retry
        receives a :class:`GenerationAttempt` carrying the previous source and
        the exact foundry rejection reason, so it can correct its own mistake.
        Returns True on the first successful registration; False if the spike
        was skipped (cap/cooldown), the generator crashed, or all attempts
        were rejected -- none of which may ever crash the training loop.
        """
        # Cap and cooldown gates first -- both are logged so the demo log can
        # explain why a spike produced no expert.
        if self.max_experts is not None and self.gate.num_experts >= self.max_experts:
            self.logger.log(
                "registration_skipped", step=step, reason="max_experts_reached",
                num_experts=self.gate.num_experts, max_experts=self.max_experts,
            )
            return False
        if (
            self._last_generation_step is not None
            and step - self._last_generation_step < self.spike_cooldown_steps
        ):
            self.logger.log(
                "registration_skipped", step=step, reason="cooldown",
                last_generation_step=self._last_generation_step,
                cooldown_steps=self.spike_cooldown_steps,
            )
            return False

        # Built ONCE for the whole retry sequence: nothing registers between
        # attempts, so the context (expert count included) stays accurate.
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

        prior: Optional[GenerationAttempt] = None
        last_reason = ""
        for attempt in range(1, self.max_generation_attempts + 1):
            # A crashing generator must not kill the training loop. It also
            # aborts the retry sequence: with no source produced, there is
            # nothing to build correction feedback from.
            try:
                source = self.generator(ctx, prior)
            except Exception as exc:  # noqa: BLE001 - deliberate loop shield
                self.rejections += 1
                # A crash-abort arms the cooldown just like exhaustion does: a
                # generator broken enough to raise (vs. merely emit bad source)
                # must not get a fresh sequence on every subsequent spike.
                self._last_generation_step = step
                self.logger.log(
                    "registration_rejected", step=step, stage="generation",
                    attempt=attempt, reason=repr(exc),
                    num_experts=self.gate.num_experts,
                )
                return False

            self.logger.log(
                "candidate_generated",
                step=step,
                attempt=attempt,
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                source_chars=len(source),
            )

            old_count = self.gate.num_experts
            try:
                reg = self.foundry.register_expert_from_source(
                    source, optimizer=self.optimizer
                )
            except ExpertValidationError as exc:
                # Foundry rollback guarantees gate/experts/optimizer unchanged.
                self.rejections += 1
                last_reason = str(exc)
                self.logger.log(
                    "registration_rejected", step=step, stage="validation",
                    attempt=attempt, reason=last_reason,
                    num_experts=self.gate.num_experts,
                )
                # Feed the REAL rejection back for a self-corrected retry.
                prior = GenerationAttempt(
                    attempt_number=attempt,
                    source=source,
                    rejection_reason=last_reason,
                )
                continue

            # --- success: the ONLY path that records a source. A rejected
            # attempt can never end up in _registered_sources, so a checkpoint
            # can never carry a failed candidate as if it were a live expert.
            # The append is the FIRST statement after the foundry commits, to
            # minimise the (uncloseable-in-pure-Python) interrupt window in
            # which the gate has grown but the source is not yet recorded --
            # a checkpoint saved from that window fails from_checkpoint's
            # count guard LOUDLY rather than resuming silently wrong.
            self._registered_sources.append(
                {"class_name": reg.class_name, "source": source}
            )
            self.registrations += 1
            self._last_generation_step = step
            self.logger.log(
                "registration_success",
                step=step,
                attempt=attempt,
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

        # All attempts rejected: the spike goes unhandled and training simply
        # continues. Exhaustion arms the cooldown exactly like a success does
        # (see __init__): a persistently-failing generator must not burn a
        # full retry sequence on every subsequent spiking step.
        self._last_generation_step = step
        self.logger.log(
            "registration_exhausted",
            step=step,
            attempts=self.max_generation_attempts,
            last_reason=last_reason,
            num_experts=self.gate.num_experts,
        )
        return False

    # -- checkpoint / resume ----------------------------------------------------
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Atomically write a full-resumption checkpoint.

        Call this ONLY between ``train_step`` calls, never from inside one --
        that is what guarantees a checkpoint never captures a state with
        gradients half-applied. ``run()`` honours this by checkpointing at
        loop level only.

        The state capture runs under ``gate.expand_lock`` so no expansion can
        tear the gate/optimizer pairing mid-snapshot. The write is atomic
        (temp file + ``os.replace``): an interrupt mid-write can never
        corrupt the previous good checkpoint on shared storage.
        """
        target = Path(path) if path is not None else self.checkpoint_path
        if target is None:
            raise ValueError(
                "no checkpoint path: pass `path` or set checkpoint_path on the "
                "orchestrator"
            )

        with self.gate.expand_lock:
            detector_state = None
            get_state = getattr(self.detector, "get_state", None)
            if callable(get_state):
                detector_state = get_state()
            payload: dict[str, Any] = {
                "version": 1,
                "step_count": self.step_count,
                "num_experts": self.gate.num_experts,
                "gate_config": {
                    "in_features": self.gate.in_features,
                    "num_experts": self.gate.num_experts,
                    "k": self.gate.k,
                    "loss_coef": self.gate.loss_coef,
                    "noise_eps": self.gate.noise_eps,
                    "w_gate_init_std": self.gate.w_gate_init_std,
                },
                "gate_state": self.gate.state_dict(),
                "n_base_experts": self._n_base_experts,
                "generated_experts": [dict(e) for e in self._registered_sources],
                "experts_state": self.experts.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "detector_state": detector_state,
                "counters": {
                    "spikes_seen": self.spikes_seen,
                    "registrations": self.registrations,
                    "rejections": self.rejections,
                    # Key name kept for checkpoint-format stability; it maps
                    # to _last_generation_step (set on success AND exhaustion).
                    "last_registration_step": self._last_generation_step,
                },
                "orch_config": {
                    "max_experts": self.max_experts,
                    "spike_cooldown_steps": self.spike_cooldown_steps,
                    "max_generation_attempts": self.max_generation_attempts,
                },
                "foundry_config": {
                    "expert_input_dim": self.foundry.expert_input_dim,
                    "expert_output_dim": self.foundry.expert_output_dim,
                    "timeout_s": self.foundry.timeout_s,
                    "smoke_batch": self.foundry.smoke_batch,
                },
            }

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)
        self.logger.log("checkpoint_saved", step=self.step_count, path=str(target))
        return target

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        *,
        base_expert_factory: Callable[[], nn.Module],
        log_path: Union[str, Path],
        expert_source_generator: Callable[
            [FailureContext, Optional[GenerationAttempt]], str
        ] = template_mlp_expert_source,
        optimizer_factory: Optional[Callable[[list], torch.optim.Optimizer]] = None,
        detector: Optional[Any] = None,
        task_loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
        map_location: Any = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_every_n_steps: Optional[int] = None,
    ) -> "TrainingOrchestrator":
        """Reconstruct an orchestrator for exact resumption from a checkpoint.

        REBUILD-BEFORE-LOAD ORDERING (read this before touching the method --
        it is the part most likely to be gotten wrong):

        A ``state_dict`` can only be loaded into a module whose structure
        already matches it. But the checkpoint describes a gate that has been
        EXPANDED beyond its construction size, and an expert ``ModuleList``
        that has GROWN with dynamically generated classes. Both must therefore
        be rebuilt to their checkpointed shape FIRST, and only then loaded:

          1. Construct the gate at the *checkpointed* ``num_experts`` (stored
             gate config), not the original construction size.
          2. Rebuild the experts list to the exact checkpointed length:
             ``n_base_experts`` modules from ``base_expert_factory`` (the
             caller must supply the same architecture used at original
             construction -- the orchestrator cannot know it), then one
             re-exec'd module per stored generated-expert source, in
             registration order (order defines state_dict keys AND optimizer
             param-group positions).
          3. Only now ``load_state_dict`` the gate and the experts -- the
             keys and shapes finally match.
          4. Construct the foundry (its constructor validates that gate and
             expert counts agree).
          5. Rebuild the optimizer with the exact save-time group structure:
             group 0 = ``list(gate.parameters()) + base expert params`` (the
             documented construction pattern), then one ``add_param_group``
             per parameterized generated expert, in order. Then verify the
             structure (see guard below) and ``load_state_dict`` -- which
             restores Adam moments AND per-group hyperparams (lr, betas).
          6. Restore counters, cooldown marker, sources, detector state.

        SILENT-CORRUPTION GUARD: ``optimizer.load_state_dict`` matches state
        to parameters BY POSITION, not by name. If the rebuilt group
        structure deviates from the save-time structure it will not raise --
        it will silently attach momentum buffers to the wrong parameters.
        This method therefore verifies, before loading, that (a) the rebuilt
        group count and per-group param counts match what the reconstructed
        gate/experts imply, and (b) they match the checkpoint's stored
        ``param_groups`` -- and raises ``ValueError`` on any disagreement.

        Args:
            path: Checkpoint file written by :meth:`save_checkpoint`.
            base_expert_factory: Zero-arg callable returning ONE base expert
                with the same architecture used at original construction.
            log_path: JSONL log for the resumed run (appends if it exists).
            expert_source_generator: Generator for FUTURE spikes (not needed
                to rebuild past experts -- their sources are in the checkpoint).
            optimizer_factory: ``params -> Optimizer``; defaults to Adam.
                Hyperparameters are overwritten by the checkpoint on load.
            detector: Overrides the checkpointed detector state if given.
            map_location: Forwarded to ``torch.load`` (e.g. "cpu").
            checkpoint_path: Where the resumed run saves its own checkpoints;
                defaults to ``path`` (resume-in-place).
            checkpoint_every_n_steps: Periodic-save cadence for the resumed run.

        Raises:
            ValueError: On any consistency violation -- expert-count
                disagreement or optimizer group-structure mismatch. Never
                proceeds silently.
        """
        # weights_only=True: the checkpoint contains only tensors/containers,
        # so loading it is never a pickle-code-execution vector. (The expert
        # SOURCES inside are re-exec'd, but only through the same restricted
        # namespace the foundry used -- see _rebuild_generated_expert.)
        ckpt = torch.load(path, map_location=map_location, weights_only=True)

        # -- 1. gate at checkpointed size --------------------------------------
        gc = ckpt["gate_config"]
        gate = DynamicNoisyTopKGate(
            gc["in_features"],
            gc["num_experts"],
            k=gc["k"],
            loss_coef=gc["loss_coef"],
            noise_eps=gc["noise_eps"],
            w_gate_init_std=gc["w_gate_init_std"],
        )

        # -- 2. experts rebuilt to checkpointed length, BEFORE any load --------
        n_base = ckpt["n_base_experts"]
        generated = ckpt["generated_experts"]
        experts = nn.ModuleList(base_expert_factory() for _ in range(n_base))
        for entry in generated:
            experts.append(
                _rebuild_generated_expert(entry["source"], entry["class_name"])
            )

        # Loud consistency check: checkpointed count vs what we rebuilt.
        n_ckpt = ckpt["num_experts"]
        if not (gate.num_experts == len(experts) == n_ckpt):
            raise ValueError(
                f"checkpoint inconsistency: checkpoint claims num_experts="
                f"{n_ckpt}, but the rebuilt gate has {gate.num_experts} and the "
                f"rebuilt expert list has {len(experts)} "
                f"({n_base} base + {len(generated)} generated). Refusing to "
                f"load state into mismatched structures."
            )

        # -- 3. load module state (shapes/keys now match) -----------------------
        gate.load_state_dict(ckpt["gate_state"])
        experts.load_state_dict(ckpt["experts_state"])

        # -- 4. foundry (validates gate/expert count agreement itself) ----------
        fc = ckpt["foundry_config"]
        foundry = ExpertFoundry(
            gate,
            experts,
            expert_input_dim=fc["expert_input_dim"],
            expert_output_dim=fc["expert_output_dim"],
            timeout_s=fc["timeout_s"],
            smoke_batch=fc["smoke_batch"],
        )

        # -- 5. optimizer: rebuild save-time group structure, verify, load ------
        if optimizer_factory is None:
            optimizer_factory = torch.optim.Adam
        base_params = list(gate.parameters()) + [
            p for e in experts[:n_base] for p in e.parameters()
        ]
        optimizer = optimizer_factory(base_params)
        # Mirror the foundry's registration behavior exactly: one fresh group
        # per generated expert, skipped when the expert has no parameters.
        parameterized_gen = []
        for e in experts[n_base:]:
            ps = list(e.parameters())
            if ps:
                optimizer.add_param_group({"params": ps})
                parameterized_gen.append(len(ps))

        # Silent-corruption guard (see docstring): verify group structure
        # against the LIVE rebuilt objects and against the checkpoint, param
        # counts included, before the positional load.
        expected_counts = [len(base_params)] + parameterized_gen
        live_counts = [len(g["params"]) for g in optimizer.param_groups]
        ckpt_counts = [
            len(g["params"]) for g in ckpt["optimizer_state"]["param_groups"]
        ]
        if live_counts != expected_counts:
            raise ValueError(
                f"optimizer group structure mismatch: rebuilt optimizer has "
                f"per-group param counts {live_counts}, but the reconstructed "
                f"gate/experts imply {expected_counts} (1 base group + one per "
                f"parameterized generated expert). Did optimizer_factory split "
                f"params into multiple groups?"
            )
        if len(live_counts) != len(ckpt_counts):
            raise ValueError(
                f"optimizer group count mismatch vs checkpoint: rebuilt "
                f"{len(live_counts)} groups {live_counts}, checkpoint stores "
                f"{len(ckpt_counts)} groups {ckpt_counts}. load_state_dict "
                f"matches by position; refusing to load mismatched structure."
            )
        for i, (lc, cc) in enumerate(zip(live_counts, ckpt_counts)):
            if lc != cc:
                raise ValueError(
                    f"optimizer param count mismatch in group {i}: rebuilt "
                    f"group has {lc} params, checkpoint group has {cc}. "
                    f"load_state_dict matches by position; refusing to load "
                    f"positionally-mismatched momentum state."
                )
        optimizer.load_state_dict(ckpt["optimizer_state"])

        # -- 6. detector + orchestrator + counters ------------------------------
        if detector is None:
            ds = ckpt.get("detector_state")
            detector = (
                RollingSpikeDetector.from_state(ds) if ds else RollingSpikeDetector()
            )

        orch = cls(
            gate,
            experts,
            foundry,
            optimizer,
            log_path=log_path,
            expert_source_generator=expert_source_generator,
            detector=detector,
            task_loss_fn=task_loss_fn,
            max_experts=ckpt["orch_config"]["max_experts"],
            spike_cooldown_steps=ckpt["orch_config"]["spike_cooldown_steps"],
            # .get(): checkpoints written before the retry feature keep loading.
            max_generation_attempts=ckpt["orch_config"].get(
                "max_generation_attempts", 3
            ),
            checkpoint_path=checkpoint_path if checkpoint_path else path,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
        )
        orch.step_count = ckpt["step_count"]
        counters = ckpt["counters"]
        orch.spikes_seen = counters["spikes_seen"]
        orch.registrations = counters["registrations"]
        orch.rejections = counters["rejections"]
        orch._last_generation_step = counters["last_registration_step"]
        # The constructor recorded len(experts) as the base count; correct it
        # to the checkpointed split so future checkpoints stay reconstructible.
        orch._n_base_experts = n_base
        orch._registered_sources = [dict(e) for e in generated]

        orch.logger.log(
            "resumed_from_checkpoint",
            path=str(path),
            step_count=orch.step_count,
            num_experts=gate.num_experts,
            registrations=orch.registrations,
        )
        return orch

    # -- convenience loop -----------------------------------------------------
    def run(
        self,
        steps: int,
        data_fn: Callable[[int], tuple[Tensor, Tensor]],
    ) -> list[StepResult]:
        """Run ``steps`` training steps, pulling batches from ``data_fn(step)``.

        Checkpointing (when ``checkpoint_path`` is set): every
        ``checkpoint_every_n_steps`` completed steps, once at the end of the
        run, and on interruption. All saves happen at loop level -- between
        ``train_step`` calls -- so a checkpoint never captures a mid-step
        state. On interruption (quota kill delivered as KeyboardInterrupt, or
        any error) the in-flight step is abandoned and a checkpoint of the
        last consistent state is written before the exception re-raises.
        """
        self.logger.log(
            "run_start",
            steps=steps,
            num_experts=self.gate.num_experts,
            k=self.gate.k,
            in_features=self.gate.in_features,
            expert_output_dim=self.foundry.expert_output_dim,
            max_experts=self.max_experts,
            spike_cooldown_steps=self.spike_cooldown_steps,
            max_generation_attempts=self.max_generation_attempts,
            checkpoint_every_n_steps=self.checkpoint_every_n_steps,
            detector={
                "window": getattr(self.detector, "window", None),
                "z_threshold": getattr(self.detector, "z_threshold", None),
                "min_history": getattr(self.detector, "min_history", None),
            },
        )
        results: list[StepResult] = []
        try:
            for _ in range(steps):
                x, y = data_fn(self.step_count + 1)
                results.append(self.train_step(x, y))
                if (
                    self.checkpoint_path is not None
                    and self.checkpoint_every_n_steps
                    and self.step_count % self.checkpoint_every_n_steps == 0
                ):
                    self.save_checkpoint()
        except BaseException:
            # Interruption path. We are at loop level (the raising frame has
            # unwound), so the save below never captures a mid-step state --
            # it reflects the last completed step; the interrupted one is
            # abandoned and will be re-run after resume.
            self.logger.log(
                "run_interrupted", step=self.step_count, completed_steps=len(results)
            )
            if self.checkpoint_path is not None:
                try:
                    self.save_checkpoint()
                except Exception as save_exc:  # never mask the original error
                    self.logger.log(
                        "checkpoint_failed", step=self.step_count, reason=repr(save_exc)
                    )
            raise

        if self.checkpoint_path is not None:
            self.save_checkpoint()
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
#
# The demo's building blocks live at MODULE level (not inside __main__) so
# that resume_demo.py imports the exact same expert architecture and data
# distribution instead of redefining them -- any drift between the original
# run and a resumed run would silently invalidate the resumed weights.
# =============================================================================
DEMO_IN_DIM = 16
DEMO_OUT_DIM = 16
DEMO_STEPS = 300
DEMO_OOD_EVERY = 40  # inject an out-of-distribution batch every N steps
DEMO_LOG = "orchestrator_run.jsonl"
DEMO_CHECKPOINT = "orchestrator_demo.ckpt"
DEMO_CHECKPOINT_EVERY = 25
DEMO_MAX_EXPERTS = 12
DEMO_COOLDOWN = 20


def demo_base_expert_factory() -> nn.Module:
    """The demo's base-expert architecture -- ALSO used by from_checkpoint."""
    return nn.Linear(DEMO_IN_DIM, DEMO_OUT_DIM)


def make_demo_data_fn() -> Callable[[int], tuple[Tensor, Tensor]]:
    """Build the demo's synthetic-regression data function.

    ``W_true`` comes from a DEDICATED seeded generator, not the global RNG, so
    the original run and any resumed run reconstruct the IDENTICAL regression
    task no matter what the global RNG state is at call time. Step numbering
    drives the OOD injection, so a resumed run keeps the same spike cadence.
    """
    gen = torch.Generator().manual_seed(0)
    w_true = torch.randn(DEMO_IN_DIM, DEMO_OUT_DIM, generator=gen) * 0.5

    def data_fn(step: int) -> tuple[Tensor, Tensor]:
        x = torch.randn(32, DEMO_IN_DIM)
        if step % DEMO_OOD_EVERY == 0:
            x = x * 8.0  # OOD scale blow-up -> engineered loss spike
        y = x @ w_true + 0.01 * torch.randn(32, DEMO_OUT_DIM)
        return x, y

    return data_fn


def print_demo_summary(orch: TrainingOrchestrator, results: list[StepResult],
                       *, start_experts: int, start_counts: tuple[int, int, int]) -> None:
    """Shared summary printer so demo and resume output are comparable."""
    s0, r0, j0 = start_counts
    print("\n=== run summary ===")
    print(f"steps          : {len(results)}")
    print(f"spikes seen    : {orch.spikes_seen - s0}")
    print(f"registrations  : {orch.registrations - r0}")
    print(f"rejections     : {orch.rejections - j0}")
    print(f"experts        : {start_experts} -> {orch.gate.num_experts} "
          f"(len(experts)={len(orch.experts)})")
    print(f"final task loss: {results[-1].loss:.4f}")
    print(f"log written to : {orch.logger.path}")


if __name__ == "__main__":
    torch.manual_seed(0)

    gate = DynamicNoisyTopKGate(DEMO_IN_DIM, num_experts=4, k=2).train()
    experts = nn.ModuleList(demo_base_expert_factory() for _ in range(4))
    foundry = ExpertFoundry(gate, experts, expert_input_dim=DEMO_IN_DIM,
                            expert_output_dim=DEMO_OUT_DIM)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + list(experts.parameters()), lr=1e-3
    )

    orch = TrainingOrchestrator(
        gate, experts, foundry, optimizer,
        log_path=DEMO_LOG,
        detector=RollingSpikeDetector(window=30, z_threshold=4.0, min_history=10),
        spike_cooldown_steps=DEMO_COOLDOWN,
        max_experts=DEMO_MAX_EXPERTS,
        # Checkpointing on: periodic saves plus a save from run()'s interrupt
        # handler, so a Ctrl+C at ANY point leaves a resumable checkpoint for
        # resume_demo.py. (A real-interrupt test previously showed the handler
        # firing correctly but having nothing configured to save to.)
        checkpoint_path=DEMO_CHECKPOINT,
        checkpoint_every_n_steps=DEMO_CHECKPOINT_EVERY,
    )

    print(f"running {DEMO_STEPS} steps (OOD batch every {DEMO_OOD_EVERY}, "
          f"checkpoint every {DEMO_CHECKPOINT_EVERY})...")
    results = orch.run(DEMO_STEPS, make_demo_data_fn())
    print_demo_summary(orch, results, start_experts=4, start_counts=(0, 0, 0))
    orch.logger.close()
