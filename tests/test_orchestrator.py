"""Verification suite for core.orchestrator.

Run with:   python -m pytest tests/ -v
Or plainly: python tests/test_orchestrator.py   (minimal runner at the bottom,
            same convention as test_dynamic_gating.py).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

# Make the repo root importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.moe.dynamic_gating import DynamicNoisyTopKGate, ExpertFoundry  # noqa: E402
from core.orchestrator import (  # noqa: E402
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
    orch = TrainingOrchestrator(
        gate, experts, foundry, opt,
        log_path=log_path,
        expert_source_generator=generator or template_mlp_expert_source,
        detector=detector or StubDetector(fire_on_calls=()),
        spike_cooldown_steps=0,
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
    bad_generator = lambda ctx: f"""
class BadDim{ctx.step}(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear({ctx.input_dim}, {ctx.output_dim + 3})  # wrong out dim
    def forward(self, x):
        return self.lin(x)
"""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=bad_generator, detector=StubDetector(fire_on_calls=(2,))
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
    def crashing_generator(ctx):
        raise RuntimeError("LLM endpoint on fire")

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "run.jsonl")
        gate, experts, opt, orch = _make_orchestrator(
            log, generator=crashing_generator, detector=StubDetector(fire_on_calls=(2,))
        )
        try:
            results = orch.run(4, _data_fn)
        finally:
            orch.logger.close()
        assert len(results) == 4
        assert gate.num_experts == len(experts) == 4
        events = _read_events(log)
        rejected = [e for e in events if e["event"] == "registration_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["stage"] == "generation"
        assert "on fire" in rejected[0]["reason"]


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
