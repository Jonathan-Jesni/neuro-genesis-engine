"""Verification suite for core.orchestrator.

Run with:   python -m pytest tests/ -v
Or plainly: python tests/test_orchestrator.py   (minimal runner at the bottom,
            same convention as test_dynamic_gating.py).
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import torch

# Make the repo root importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.moe.dynamic_gating import DynamicNoisyTopKGate, ExpertFoundry  # noqa: E402
from core.orchestrator import (  # noqa: E402
    GenerationAttempt,
    MockLLMGenerator,
    RollingSpikeDetector,
    SpikeInfo,
    TrainingOrchestrator,
    template_mlp_expert_source,
)

IN_DIM = 8
OUT_DIM = 8


class StubDetector:
    """Deterministic detector: fires a spike on chosen update() call numbers.

    The real detector is unit-tested separately; orchestration tests use this
    stub so spike timing is exact regardless of loss values.
    """

    def __init__(self, fire_on_calls):
        self.calls = 0
        self.fire_on = set(fire_on_calls)

    def update(self, loss):
        self.calls += 1
        if self.calls in self.fire_on:
            return SpikeInfo(loss=loss, rolling_mean=0.0, rolling_std=1.0, z_score=99.0)
        return None


def _make_orchestrator(log_path, *, n_start=4, generator=None, detector=None, **kwargs):
    torch.manual_seed(0)
    gate = DynamicNoisyTopKGate(IN_DIM, n_start, k=2).train()
    experts = torch.nn.ModuleList(
        torch.nn.Linear(IN_DIM, OUT_DIM) for _ in range(n_start)
    )
    foundry = ExpertFoundry(
        gate, experts, expert_input_dim=IN_DIM, expert_output_dim=OUT_DIM
    )
    opt = torch.optim.Adam(list(gate.parameters()) + list(experts.parameters()))
    kwargs.setdefault("spike_cooldown_steps", 0)
    orch = TrainingOrchestrator(
        gate, experts, foundry, opt,
        log_path=log_path,
        expert_source_generator=generator or template_mlp_expert_source,
        detector=detector or StubDetector(fire_on_calls=()),
        **kwargs,
    )
    return gate, experts, opt, orch


def _data_fn(step):
    x = torch.randn(16, IN_DIM)
    y = torch.randn(16, OUT_DIM)
    return x, y


def _read_events(log_path):
    with open(log_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------
# 1. Spike detector (real one, unit-tested without any training)
# ---------------------------------------------------------------------------
def test_spike_detector_triggers_on_outlier():
    det = RollingSpikeDetector(window=20, z_threshold=4.0, min_history=5)
    # Warmup + steady phase: no spike may fire.
    for _ in range(15):
        assert det.update(1.0) is None
    # A 10x outlier must fire, with sane metadata.
    spike = det.update(10.0)
    assert spike is not None
    assert spike.z_score > 4.0
    assert abs(spike.rolling_mean - 1.0) < 1e-6
    # The outlier must NOT have entered the rolling history (it would inflate
    # the baseline it was judged against): baseline is still ~1.0, so both a
    # steady value passes and a repeat outlier fires again identically.
    assert max(det._history) < 2.0
    assert det.update(1.0) is None
    spike2 = det.update(10.0)
    assert spike2 is not None
    assert abs(spike2.rolling_mean - 1.0) < 1e-6


def test_spike_detector_ignores_gradual_drift():
    # A slow upward ramp is regime change, not a spike: the rolling window
    # adapts and the z-score never clears the threshold.
    det = RollingSpikeDetector(window=10, z_threshold=4.0, min_history=5)
    for i in range(60):
        assert det.update(1.0 + 0.05 * i) is None, f"false spike at i={i}"


def test_spike_detector_flags_nonfinite_loss():
    det = RollingSpikeDetector(window=10, z_threshold=4.0, min_history=3)
    for _ in range(5):
        det.update(1.0)
    spike = det.update(float("nan"))
    assert spike is not None
    # NaN never becomes baseline material.
    assert all(v == 1.0 for v in det._history)


# ---------------------------------------------------------------------------
# 2. Loop resilience: bad candidates must never crash training
# ---------------------------------------------------------------------------
def test_rejected_candidate_does_not_crash_loop():
    bad_generator = lambda ctx, prior_attempt=None: f"""
class BadDim{ctx.step}(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear({ctx.input_dim}, {ctx.output_dim + 3})  # wrong out dim
    def forward(self, x):
        return self.lin(x)
"""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        # max_generation_attempts=1 pins the original single-shot semantics
        # this test was written for (exactly one rejection per spike).
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=bad_generator, detector=StubDetector(fire_on_calls=(2,)),
            max_generation_attempts=1,
        )
        try:
            results = orch.run(4, _data_fn)  # must complete despite rejection
        finally:
            orch.logger.close()
        assert len(results) == 4
        # Foundry rollback: counts unchanged and still in lockstep.
        assert gate.num_experts == len(experts) == 4
        events = _read_events(log)
        rejected = [e for e in events if e["event"] == "registration_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["stage"] == "validation"
        assert rejected[0]["reason"]  # non-empty, actionable reason string
        assert not any(e["event"] == "registration_success" for e in events)


def test_generator_exception_does_not_crash_loop():
    def crashing_generator(ctx, prior_attempt=None):
        raise RuntimeError("LLM endpoint on fire")

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        # Spikes at 2 AND 3 with cooldown 5: the crash-abort at step 2 must arm
        # the cooldown (a crashing generator is throttled like a failing one).
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=crashing_generator,
            detector=StubDetector(fire_on_calls=(2, 3)), spike_cooldown_steps=5,
        )
        try:
            results = orch.run(4, _data_fn)
        finally:
            orch.logger.close()
        assert len(results) == 4
        assert gate.num_experts == len(experts) == 4
        events = _read_events(log)
        rejected = [e for e in events if e["event"] == "registration_rejected"]
        assert len(rejected) == 1  # crash ABORTS the sequence: no retries
        assert rejected[0]["stage"] == "generation"
        assert "on fire" in rejected[0]["reason"]
        skipped = [e for e in events if e["event"] == "registration_skipped"]
        assert len(skipped) == 1 and skipped[0]["reason"] == "cooldown"


# ---------------------------------------------------------------------------
# 2b. Self-correcting retry loop
# ---------------------------------------------------------------------------
def test_retry_succeeds_on_second_attempt():
    # Mock LLM deterministically emits broken source on attempt 1 and corrects
    # itself on attempt 2 using the foundry's rejection reason.
    gen = MockLLMGenerator(failure_rate=1.0, force_mode="wrong_dim")
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=gen, detector=StubDetector(fire_on_calls=(2,))
        )
        try:
            orch.run(3, _data_fn)
            orch.save_checkpoint(ck)
        finally:
            orch.logger.close()

        assert gate.num_experts == len(experts) == 5
        events = _read_events(log)
        cand = [e for e in events if e["event"] == "candidate_generated"]
        assert [e["attempt"] for e in cand] == [1, 2], "exactly two attempts"
        succ = [e for e in events if e["event"] == "registration_success"]
        assert len(succ) == 1 and succ[0]["attempt"] == 2
        assert not any(e["event"] == "registration_exhausted" for e in events)

        # The checkpoint carries ONLY the successful (second) source -- never
        # the rejected first attempt.
        import hashlib
        ckpt = torch.load(ck, weights_only=True)
        assert len(ckpt["generated_experts"]) == 1
        stored_sha = hashlib.sha256(
            ckpt["generated_experts"][0]["source"].encode("utf-8")
        ).hexdigest()
        assert stored_sha == cand[1]["source_sha256"]
        assert stored_sha != cand[0]["source_sha256"]


def test_retry_exhaustion_never_crashes():
    # Failures DIFFER per attempt (escalating wrong dim), so the last_reason
    # assertion below is not vacuous: if the code stored the FIRST rejection
    # reason instead of the last, the test would fail. (An always-identical
    # failure -- e.g. MockLLMGenerator(always_fail=True) -- could not tell
    # first from last.)
    def escalating_bad_generator(ctx, prior_attempt=None):
        n = 1 if prior_attempt is None else prior_attempt.attempt_number + 1
        return f"""
class BadDimAttempt{n}(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear({ctx.input_dim}, {ctx.output_dim + n})
    def forward(self, x):
        return self.lin(x)
"""

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        # Spikes at steps 2 AND 3; cooldown 5 -> the exhaustion at step 2 must
        # throttle the step-3 spike (exhaustion arms the cooldown like success).
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=escalating_bad_generator,
            detector=StubDetector(fire_on_calls=(2, 3)),
            max_generation_attempts=3, spike_cooldown_steps=5,
        )
        try:
            results = orch.run(4, _data_fn)  # must complete despite exhaustion
            orch.save_checkpoint(ck)
        finally:
            orch.logger.close()

        assert len(results) == 4
        assert gate.num_experts == len(experts) == 4  # nothing registered
        events = _read_events(log)
        cand = [e for e in events if e["event"] == "candidate_generated"]
        assert [e["attempt"] for e in cand] == [1, 2, 3]
        rejected = [e for e in events if e["event"] == "registration_rejected"]
        assert len(rejected) == 3
        # Three genuinely different reasons (out dims +1, +2, +3)...
        assert len({e["reason"] for e in rejected}) == 3
        exhausted = [e for e in events if e["event"] == "registration_exhausted"]
        assert len(exhausted) == 1
        assert exhausted[0]["attempts"] == 3
        # ...and last_reason is specifically the THIRD attempt's failure
        # (the foundry smoke-tests with its smoke_batch of 4, so the offending
        # shape it reports is (4, OUT_DIM + 3)).
        assert exhausted[0]["last_reason"] == rejected[2]["reason"]
        assert f"(4, {OUT_DIM + 3})" in exhausted[0]["last_reason"]
        # The step-3 spike hit the cooldown armed by the exhaustion.
        skipped = [e for e in events if e["event"] == "registration_skipped"]
        assert len(skipped) == 1 and skipped[0]["reason"] == "cooldown"

        # Checkpoint after exhaustion carries no phantom expert and resumes.
        orch2 = TrainingOrchestrator.from_checkpoint(
            ck, base_expert_factory=_base_expert_factory,
            log_path=os.path.join(tmp, "run2.jsonl"),
        )
        try:
            assert orch2.gate.num_experts == len(orch2.experts) == 4
            assert orch2._registered_sources == []
        finally:
            orch2.logger.close()

    # Smoke the mock's always_fail knob through the same exhaustion path.
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=MockLLMGenerator(always_fail=True),
            detector=StubDetector(fire_on_calls=(2,)), max_generation_attempts=3,
        )
        try:
            orch.run(3, _data_fn)
        finally:
            orch.logger.close()
        assert gate.num_experts == len(experts) == 4
        events = _read_events(log)
        assert sum(e["event"] == "registration_exhausted" for e in events) == 1


def test_prior_attempt_carries_real_rejection_reason():
    calls = []       # the prior_attempt received on each call
    returned = []    # the source returned by each call

    def recording_generator(ctx, prior_attempt=None):
        calls.append(prior_attempt)
        if prior_attempt is None:
            src = f"""
class WrongDim{ctx.step}(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear({ctx.input_dim}, {ctx.output_dim + 2})
    def forward(self, x):
        return self.lin(x)
"""
        else:
            src = template_mlp_expert_source(ctx, prior_attempt)
        returned.append(src)
        return src

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=recording_generator, detector=StubDetector(fire_on_calls=(2,))
        )
        try:
            orch.run(3, _data_fn)
        finally:
            orch.logger.close()

        assert gate.num_experts == 5  # retry succeeded
        assert len(calls) == 2
        assert calls[0] is None, "first attempt must receive prior_attempt=None"
        ga = calls[1]
        assert isinstance(ga, GenerationAttempt)
        assert ga.attempt_number == 1
        # The prior attempt carries the EXACT source that failed...
        assert ga.source == returned[0]
        # ...and the REAL rejection reason the foundry produced for it -- the
        # same string that was logged, mentioning the actual shape mismatch.
        events = _read_events(log)
        rejected = [e for e in events if e["event"] == "registration_rejected"]
        assert len(rejected) == 1
        assert ga.rejection_reason == rejected[0]["reason"]
        assert "output shape" in ga.rejection_reason


def test_mock_llm_corrects_each_failure_mode():
    # Every simulated LLM failure mode must be corrected on attempt 2, with
    # the correction branch chosen from the rejection reason (visible in the
    # generated class name).
    for mode, label in (
        ("syntax", "FixedSyntax"),
        ("wrong_dim", "FixedDim"),
        ("forbidden_import", "FixedImport"),
    ):
        gen = MockLLMGenerator(failure_rate=1.0, force_mode=mode)
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "run.jsonl")
            gate, experts, opt, orch = _make_orchestrator(
                log, generator=gen, detector=StubDetector(fire_on_calls=(1,))
            )
            try:
                orch.run(2, _data_fn)
            finally:
                orch.logger.close()
            assert gate.num_experts == len(experts) == 5, f"mode {mode} failed"
            events = _read_events(log)
            succ = [e for e in events if e["event"] == "registration_success"]
            assert len(succ) == 1 and succ[0]["attempt"] == 2, f"mode {mode}"
            assert label in succ[0]["class_name"], (
                f"mode {mode}: correction branch not driven by rejection "
                f"reason (got {succ[0]['class_name']})"
            )


# ---------------------------------------------------------------------------
# 3. Successful growth
# ---------------------------------------------------------------------------
def test_successful_registration_grows_in_lockstep():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, detector=StubDetector(fire_on_calls=(2,))
        )
        groups_before = len(opt.param_groups)
        try:
            results = orch.run(5, _data_fn)  # spike at step 2, then 3 more steps
        finally:
            orch.logger.close()
        # Gate and expert list grew together, exactly once.
        assert gate.num_experts == len(experts) == 5
        assert results[1].registered and results[1].spiked
        # The new expert's params joined the optimizer as a fresh group.
        assert len(opt.param_groups) == groups_before + 1
        # Training genuinely continued post-registration (steps 3-5 ran with
        # the grown gate and produced finite losses).
        assert all(r.num_experts == 5 for r in results[2:])
        assert all(torch.isfinite(torch.tensor(r.loss)) for r in results)
        # The registered expert is live and callable.
        y = experts[4](torch.randn(3, IN_DIM))
        assert y.shape == (3, OUT_DIM)


def test_respects_max_experts_cap():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, detector=StubDetector(fire_on_calls=(2, 3)), max_experts=4
        )
        try:
            orch.run(4, _data_fn)
        finally:
            orch.logger.close()
        assert gate.num_experts == len(experts) == 4  # never grew
        events = _read_events(log)
        skipped = [e for e in events if e["event"] == "registration_skipped"]
        assert len(skipped) == 2
        assert all(e["reason"] == "max_experts_reached" for e in skipped)


# ---------------------------------------------------------------------------
# 4. The log is a complete, parseable record of the run
# ---------------------------------------------------------------------------
def test_log_is_complete_and_parseable():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, detector=StubDetector(fire_on_calls=(3,))
        )
        try:
            orch.run(6, _data_fn)
        finally:
            orch.logger.close()

        events = _read_events(log)  # every line must parse as JSON
        # Every record carries a timestamp and an event type, in time order.
        assert all("ts" in e and "event" in e for e in events)
        ts = [e["ts"] for e in events]
        assert ts == sorted(ts)

        # The full lifecycle is present.
        kinds = [e["event"] for e in events]
        assert kinds[0] == "run_start"
        assert kinds[-1] == "run_end"
        for required in ("spike", "candidate_generated",
                         "registration_success", "expert_count_change"):
            assert required in kinds, f"missing {required} event"

        # One step record per training step, correctly numbered.
        steps = [e for e in events if e["event"] == "step"]
        assert [e["step"] for e in steps] == [1, 2, 3, 4, 5, 6]

        # Reconstruct the expert-count timeline FROM THE LOG ALONE and check
        # it matches the live final state -- this is the "demo raw material"
        # guarantee.
        count = next(e for e in events if e["event"] == "run_start")["num_experts"]
        for e in events:
            if e["event"] == "expert_count_change":
                assert e["old"] == count
                count = e["new"]
        assert count == gate.num_experts == len(experts) == 5

        # run_end totals agree with the events in the body.
        end = events[-1]
        assert end["registrations"] == kinds.count("registration_success") == 1
        assert end["spikes_seen"] == kinds.count("spike") == 1


# ---------------------------------------------------------------------------
# 5. End-to-end with the REAL detector: engineered OOD batches cause growth
# ---------------------------------------------------------------------------
def test_end_to_end_ood_spike_grows_expert():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log,
            detector=RollingSpikeDetector(window=20, z_threshold=4.0, min_history=8),
        )
        W = torch.randn(IN_DIM, OUT_DIM)

        def data_fn(step):
            x = torch.randn(32, IN_DIM)
            if step == 20:
                x = x * 10.0  # engineered OOD batch
            return x, x @ W

        try:
            orch.run(25, data_fn)
        finally:
            orch.logger.close()
        assert orch.spikes_seen >= 1, "OOD batch did not register as a spike"
        assert gate.num_experts == len(experts) == 4 + orch.registrations
        assert orch.registrations >= 1


# ---------------------------------------------------------------------------
# 6. Checkpoint / resume
# ---------------------------------------------------------------------------
def _base_expert_factory():
    return torch.nn.Linear(IN_DIM, OUT_DIM)


def test_checkpoint_roundtrip_resume():
    with tempfile.TemporaryDirectory() as tmp:
        log1 = os.path.join(tmp, "run1.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        gate, experts, opt, orch = _make_orchestrator(
            log1, detector=StubDetector(fire_on_calls=(2,))
        )
        try:
            orch.run(5, _data_fn)  # registration at step 2: 4 -> 5 experts
            assert gate.num_experts == len(experts) == 5
            orch.save_checkpoint(ck)
        finally:
            orch.logger.close()

        base_w = experts[0].weight.detach().clone()
        gen_w = experts[4].fc1.weight.detach().clone()  # GeneratedExpertStep2
        exp_avg = opt.state[gate.w_gate]["exp_avg"].detach().clone()
        step_before = orch.step_count

        log2 = os.path.join(tmp, "run2.jsonl")
        orch2 = TrainingOrchestrator.from_checkpoint(
            ck,
            base_expert_factory=_base_expert_factory,
            log_path=log2,
            detector=StubDetector(fire_on_calls=()),
        )
        try:
            g2, e2 = orch2.gate, orch2.experts
            # Counts in lockstep, step counter restored.
            assert g2.num_experts == len(e2) == 5
            assert orch2.step_count == step_before == 5
            assert orch2.registrations == 1
            # Specific weights match EXACTLY -- base expert AND the expert
            # that was dynamically generated before the checkpoint.
            assert torch.equal(e2[0].weight.detach(), base_w)
            assert torch.equal(e2[4].fc1.weight.detach(), gen_w)
            # Adam momentum survived the resume (same reasoning as the
            # original optimizer-remap work: cold moments = lost history).
            assert torch.equal(opt2_state := orch2.optimizer.state[g2.w_gate]["exp_avg"], exp_avg)
            assert opt2_state is not exp_avg  # genuinely reloaded, not aliased
            # Resume training: step numbering continues, losses stay finite.
            results = orch2.run(5, _data_fn)
            assert [r.step for r in results] == [6, 7, 8, 9, 10]
            assert all(math.isfinite(r.loss) for r in results)
        finally:
            orch2.logger.close()
        steps2 = [e["step"] for e in _read_events(log2) if e["event"] == "step"]
        assert steps2 == [6, 7, 8, 9, 10]


def test_resumed_detector_keeps_rolling_baseline():
    from core.orchestrator import RollingSpikeDetector as RSD

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        det = RSD(window=20, z_threshold=4.0, min_history=5)
        # Establish a real rolling baseline (well past min_history).
        for i in range(12):
            assert det.update(1.0 + 0.01 * (i % 3)) is None
        history_before = list(det._history)

        gate, experts, opt, orch = _make_orchestrator(log, detector=det)
        try:
            orch.save_checkpoint(ck)
        finally:
            orch.logger.close()

        log2 = os.path.join(tmp, "run2.jsonl")
        orch2 = TrainingOrchestrator.from_checkpoint(
            ck, base_expert_factory=_base_expert_factory, log_path=log2
        )
        try:
            det2 = orch2.detector
            # The rolling window came back exactly -- not a fresh detector.
            assert list(det2._history) == history_before
            # No misfire on a normal loss right after resume...
            assert det2.update(1.01) is None
            # ...and a genuine outlier fires IMMEDIATELY. A reset (empty)
            # window could not do this until min_history refilled -- that is
            # the missed-spike regression this test guards against.
            spike = det2.update(50.0)
            assert spike is not None and spike.z_score > 4.0
        finally:
            orch2.logger.close()


def test_periodic_checkpointing_during_run():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        gate, experts, opt, orch = _make_orchestrator(
            log,
            detector=StubDetector(fire_on_calls=()),
            checkpoint_path=ck,
            checkpoint_every_n_steps=2,
        )
        try:
            orch.run(5, _data_fn)
        finally:
            orch.logger.close()
        events = _read_events(log)
        saves = [e for e in events if e["event"] == "checkpoint_saved"]
        # Periodic at steps 2 and 4, final save at run end (step 5).
        assert [e["step"] for e in saves] == [2, 4, 5]
        # Saves only ever reflect a COMPLETED step (loop-level checkpointing).
        completed = {e["step"] for e in events if e["event"] == "step"}
        assert all(e["step"] in completed for e in saves)
        # Atomic write: final file present, no temp file left behind.
        assert os.path.exists(ck)
        assert not os.path.exists(ck + ".tmp")
        ckpt = torch.load(ck, weights_only=True)
        assert ckpt["step_count"] == 5


def test_checkpoint_count_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        gate, experts, opt, orch = _make_orchestrator(
            log, detector=StubDetector(fire_on_calls=(2,))
        )
        try:
            orch.run(3, _data_fn)  # one registration -> 5 experts
            orch.save_checkpoint(ck)
        finally:
            orch.logger.close()

        ckpt = torch.load(ck, weights_only=True)
        ckpt["num_experts"] = 7  # corrupt the claimed count
        torch.save(ckpt, ck)

        try:
            TrainingOrchestrator.from_checkpoint(
                ck,
                base_expert_factory=_base_expert_factory,
                log_path=os.path.join(tmp, "run2.jsonl"),
            )
            assert False, "expected ValueError on count mismatch"
        except ValueError as exc:
            msg = str(exc)
            assert "7" in msg and "5" in msg  # names both disagreeing counts


def test_interrupted_run_saves_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        gate, experts, opt, orch = _make_orchestrator(
            log, detector=StubDetector(fire_on_calls=()), checkpoint_path=ck
        )

        def interrupting_data_fn(step):
            if step == 3:  # quota kill mid-run, delivered between steps
                raise KeyboardInterrupt
            return _data_fn(step)

        try:
            try:
                orch.run(5, interrupting_data_fn)
                assert False, "expected KeyboardInterrupt to propagate"
            except KeyboardInterrupt:
                pass
        finally:
            orch.logger.close()

        events = _read_events(log)
        assert any(e["event"] == "run_interrupted" for e in events)
        assert os.path.exists(ck)
        # The checkpoint reflects the last COMPLETED step and resumes cleanly.
        orch2 = TrainingOrchestrator.from_checkpoint(
            ck,
            base_expert_factory=_base_expert_factory,
            log_path=os.path.join(tmp, "run2.jsonl"),
            detector=StubDetector(fire_on_calls=()),
        )
        try:
            assert orch2.step_count == 2
            results = orch2.run(2, _data_fn)
            assert [r.step for r in results] == [3, 4]
        finally:
            orch2.logger.close()


def test_optimizer_group_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        ck = os.path.join(tmp, "ck.pt")
        gate, experts, opt, orch = _make_orchestrator(
            log, detector=StubDetector(fire_on_calls=(2,))
        )
        try:
            orch.run(3, _data_fn)  # registration -> optimizer has 2 groups
            assert len(opt.param_groups) == 2
            orch.save_checkpoint(ck)
        finally:
            orch.logger.close()

        # Corrupt the stored optimizer structure: drop the generated expert's
        # param group and its state entries. Without the guard,
        # load_state_dict would NOT raise -- it matches by position and would
        # silently attach the wrong momentum to the wrong params.
        ckpt = torch.load(ck, weights_only=True)
        dropped = ckpt["optimizer_state"]["param_groups"].pop()
        for pid in dropped["params"]:
            ckpt["optimizer_state"]["state"].pop(pid, None)
        torch.save(ckpt, ck)

        try:
            TrainingOrchestrator.from_checkpoint(
                ck,
                base_expert_factory=_base_expert_factory,
                log_path=os.path.join(tmp, "run2.jsonl"),
            )
            assert False, "expected ValueError on optimizer group mismatch"
        except ValueError as exc:
            assert "group" in str(exc).lower()


# ---------------------------------------------------------------------------
# Minimal runner for environments without pytest.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
