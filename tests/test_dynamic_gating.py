"""Verification suite for core.moe.dynamic_gating.

Run with:   python -m pytest tests/ -v
Or plainly: python tests/test_dynamic_gating.py   (a minimal runner is provided
            at the bottom for environments without pytest).
"""

from __future__ import annotations

import os
import sys
import threading
import time

import torch

# Make the repo root importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.moe.dynamic_gating import (  # noqa: E402
    DynamicNoisyTopKGate,
    ExpertFoundry,
    ExpertValidationError,
    remap_optimizer_for_expansion,
)

CUDA = torch.cuda.is_available()


def _make_experts(n: int, in_dim: int, out_dim: int) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(torch.nn.Linear(in_dim, out_dim) for _ in range(n))


# ---------------------------------------------------------------------------
# 1. Gating math
# ---------------------------------------------------------------------------
def test_train_mode_adds_noise():
    torch.manual_seed(0)
    gate = DynamicNoisyTopKGate(8, 6, k=2).train()
    x = torch.randn(32, 8)
    a = gate(x).gates
    b = gate(x).gates
    assert not torch.allclose(a, b), "training-mode routing should be stochastic"


def test_eval_mode_is_deterministic():
    gate = DynamicNoisyTopKGate(8, 6, k=2).eval()
    x = torch.randn(32, 8)
    a = gate(x).gates
    b = gate(x).gates
    assert torch.equal(a, b), "eval-mode routing must be deterministic"


def test_rows_sum_to_one_and_k_nonzeros():
    gate = DynamicNoisyTopKGate(8, 6, k=2).eval()
    x = torch.randn(16, 8)
    out = gate(x)
    row_sums = out.gates.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    nnz = (out.gates > 0).sum(dim=1)
    assert torch.all(nnz == 2), f"expected exactly 2 nonzeros per row, got {nnz}"


def test_k_greater_than_num_experts_degrades():
    gate = DynamicNoisyTopKGate(8, 1, k=2).eval()  # 1 expert, k=2
    x = torch.randn(4, 8)
    out = gate(x)
    assert out.gates.shape == (4, 1)
    assert torch.allclose(out.gates, torch.ones_like(out.gates), atol=1e-5)
    assert torch.isfinite(out.aux_loss)


# ---------------------------------------------------------------------------
# 2. Auxiliary loss
# ---------------------------------------------------------------------------
def test_aux_loss_present_finite_both_modes():
    gate = DynamicNoisyTopKGate(8, 6, k=2)
    x = torch.randn(16, 8)
    for mode in (gate.train, gate.eval):
        mode()
        out = gate(x)
        assert out.aux_loss.dim() == 0
        assert torch.isfinite(out.aux_loss)


def test_aux_loss_differentiable_in_train():
    gate = DynamicNoisyTopKGate(8, 6, k=2).train()
    x = torch.randn(16, 8)
    out = gate(x)
    out.aux_loss.backward()
    assert gate.w_gate.grad is not None
    assert torch.isfinite(gate.w_gate.grad).all()
    # w_noise also participates via the load estimator's noise_std.
    assert gate.w_noise.grad is not None


def test_skewed_router_has_higher_loss():
    gate = DynamicNoisyTopKGate(4, 4, k=1).eval()
    with torch.no_grad():
        # Balanced-ish random gate vs a gate that sends everything to expert 0.
        balanced = gate.w_gate.clone()
        gate.w_gate.copy_(balanced)
    x = torch.randn(64, 4)
    balanced_loss = gate(x).aux_loss.item()
    with torch.no_grad():
        gate.w_gate.zero_()
        gate.w_gate[:, 0] = 10.0  # force collapse onto expert 0
    skewed_loss = gate(x).aux_loss.item()
    assert skewed_loss > balanced_loss


# ---------------------------------------------------------------------------
# 3. Expansion exactness
# ---------------------------------------------------------------------------
def test_expand_preserves_old_columns_exactly():
    gate = DynamicNoisyTopKGate(8, 4, k=2)
    old_gate = gate.w_gate.detach().clone()
    old_noise = gate.w_noise.detach().clone()
    gate.expand(3)
    assert gate.num_experts == 7
    assert torch.equal(gate.w_gate.detach()[:, :4], old_gate)
    assert torch.equal(gate.w_noise.detach()[:, :4], old_noise)
    # Still routable.
    out = gate(torch.randn(5, 8))
    assert out.gates.shape == (5, 7)
    assert torch.isfinite(out.aux_loss)


def test_expand_preserves_routing_where_new_experts_unselected():
    # The real invariant of copy-on-write expansion: old columns are untouched,
    # so wherever the new experts are not selected, routing is bit-identical.
    gate = DynamicNoisyTopKGate(8, 4, k=2).eval()
    x = torch.randn(64, 8)
    before = gate(x).gates
    gate.expand(2)
    after = gate(x).gates

    # Rows where no new expert (index >= 4) was selected. Old columns are
    # byte-identical and eval routing is deterministic, so these rows must be
    # BIT-identical -- torch.equal, not allclose (a tolerance would mask the
    # exactness guarantee this test exists to prove).
    unchanged_rows = (after[:, 4:] == 0).all(dim=1)
    assert unchanged_rows.any(), "expected at least some rows to keep old routing"
    assert torch.equal(before[unchanged_rows], after[unchanged_rows][:, :4])


def test_router_pair_tracks_direct_param_reassignment():
    # nn.Module.__setattr__ routes Parameter assignment through
    # register_parameter; the gate's override must re-publish the reader
    # snapshot or forward() would silently keep routing with the old weights.
    gate = DynamicNoisyTopKGate(8, 4, k=2).eval()
    new_w = torch.nn.Parameter(torch.randn(8, 4))
    gate.w_gate = new_w
    assert gate._router_params[0] is new_w
    assert gate._router_params[1] is gate.w_noise


def test_router_pair_tracks_assign_load():
    # load_state_dict(assign=True) rebinds the Parameter OBJECTS (via setattr,
    # not an in-place copy) -- the one load path that can strand the snapshot
    # on the pre-load tensors while state_dict()/optimizers see the new ones.
    src = DynamicNoisyTopKGate(8, 4, k=2)
    dst = DynamicNoisyTopKGate(8, 4, k=2).eval()
    stale_gate, stale_noise = dst._router_params
    dst.load_state_dict(src.state_dict(), assign=True)
    # Snapshot must follow the rebound objects, not the originals.
    assert dst._router_params[0] is dst.w_gate
    assert dst._router_params[1] is dst.w_noise
    assert dst._router_params[0] is not stale_gate
    assert dst._router_params[1] is not stale_noise
    # And forward must route with the LOADED weights: identical inputs through
    # src and dst produce bit-identical eval routing.
    x = torch.randn(8, 8)
    assert torch.equal(dst(x).gates, src.eval()(x).gates)


# ---------------------------------------------------------------------------
# 4. Optimizer remap
# ---------------------------------------------------------------------------
def test_optimizer_state_preserved_across_expansion():
    gate = DynamicNoisyTopKGate(8, 4, k=2).train()
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    x = torch.randn(16, 8)
    for _ in range(5):
        opt.zero_grad()
        out = gate(x)
        (out.gates.sum() + out.aux_loss).backward()
        opt.step()

    old_gate_param = gate.w_gate
    exp_avg_before = opt.state[old_gate_param]["exp_avg"].clone()
    noise_exp_avg_before = opt.state[gate.w_noise]["exp_avg"].clone()
    # Clone the step: the remap carries it over BY REFERENCE, so comparing the
    # un-cloned original against the migrated value would compare an object
    # with itself and pass vacuously.
    raw_step = opt.state[old_gate_param]["step"]
    step_before = raw_step.clone() if torch.is_tensor(raw_step) else raw_step

    gate.expand(2, optimizer=opt)

    new_state = opt.state[gate.w_gate]
    # Old slice preserved exactly; new columns zero. Same for w_noise.
    assert torch.equal(new_state["exp_avg"][:, :4], exp_avg_before)
    assert torch.all(new_state["exp_avg"][:, 4:] == 0)
    assert torch.equal(opt.state[gate.w_noise]["exp_avg"][:, :4], noise_exp_avg_before)
    # Step counter carried over.
    step_after = new_state["step"]
    if torch.is_tensor(step_after):
        assert torch.equal(step_after, step_before)
    else:
        assert step_after == step_before
    # Optimizer keeps stepping without error and updates new columns.
    opt.zero_grad()
    out = gate(x)
    (out.gates.sum() + out.aux_loss).backward()
    opt.step()
    assert torch.isfinite(gate.w_gate).all()


def test_remap_raises_for_unregistered_param():
    gate = DynamicNoisyTopKGate(8, 4, k=2)
    opt = torch.optim.Adam(gate.parameters())
    stray_old = torch.nn.Parameter(torch.zeros(8, 4))
    stray_new = torch.nn.Parameter(torch.zeros(8, 6))
    try:
        remap_optimizer_for_expansion(opt, [stray_old], [stray_new])
        assert False, "expected ValueError for unregistered param"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 5. Autocast
# ---------------------------------------------------------------------------
def test_autocast_returns_input_dtype_and_finite():
    if not CUDA:
        # CPU autocast (bf16) path: still must be finite and dtype-correct.
        gate = DynamicNoisyTopKGate(8, 6, k=2).eval()
        x = torch.randn(16, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            xh = x.to(torch.bfloat16)
            out = gate(xh)
        assert out.gates.dtype == torch.bfloat16
        assert torch.isfinite(out.gates.float()).all()
        return
    gate = DynamicNoisyTopKGate(8, 6, k=2).cuda().eval()
    x = torch.randn(16, 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        xh = x.half()
        out = gate(xh)
    assert out.gates.dtype == torch.float16
    assert torch.isfinite(out.gates.float()).all()


# ---------------------------------------------------------------------------
# 6. Foundry
# ---------------------------------------------------------------------------
GOOD_SRC = """
class GeneratedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)
    def forward(self, x):
        return F.relu(self.lin(x))
"""


def _make_foundry(n_start=4):
    gate = DynamicNoisyTopKGate(16, n_start, k=2)
    experts = _make_experts(n_start, 16, 16)
    foundry = ExpertFoundry(gate, experts, expert_input_dim=16, expert_output_dim=16)
    return gate, experts, foundry


def test_foundry_registers_and_routes():
    gate, experts, foundry = _make_foundry()
    opt = torch.optim.Adam(list(gate.parameters()) + list(experts.parameters()))
    reg = foundry.register_expert_from_source(GOOD_SRC, optimizer=opt)
    assert reg.index == 4
    assert gate.num_experts == 5
    assert len(experts) == 5
    # End-to-end: route and dispatch through the (now 5) experts.
    x = torch.randn(8, 16)
    out = gate(x)
    assert out.gates.shape == (8, 5)
    # New expert is actually callable in the live list.
    y = experts[4](x)
    assert y.shape == (8, 16)


def test_foundry_syntax_error_no_state_change():
    gate, experts, foundry = _make_foundry()
    try:
        foundry.register_expert_from_source("class Broken(nn.Module):\n    def __init__(self)")
        assert False, "expected failure"
    except ExpertValidationError:
        pass
    assert gate.num_experts == 4 and len(experts) == 4


def test_foundry_wrong_output_dim_rejected_before_append():
    # NOTE: this failure happens at the smoke test, BEFORE the expert is ever
    # appended -- it does not exercise the rollback branch. That branch is
    # covered by test_foundry_rolls_back_on_post_append_failure below.
    gate, experts, foundry = _make_foundry()
    bad = """
class WrongDim(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 8)   # wrong output dim
    def forward(self, x):
        return self.lin(x)
"""
    try:
        foundry.register_expert_from_source(bad)
        assert False, "expected failure"
    except ExpertValidationError:
        pass
    assert gate.num_experts == 4 and len(experts) == 4


def test_foundry_rolls_back_on_post_append_failure():
    # Force a failure AFTER experts.append() and add_param_group() succeeded:
    # the optimizer owns the experts' params but NOT the gate's, so the remap
    # inside gate.expand() raises. Registration must restore the expert list,
    # the gate, and the optimizer to their exact pre-registration state.
    gate, experts, foundry = _make_foundry()
    opt = torch.optim.Adam(experts.parameters())
    w_gate_before = gate.w_gate.detach().clone()
    n_groups_before = len(opt.param_groups)
    try:
        foundry.register_expert_from_source(GOOD_SRC, optimizer=opt)
        assert False, "expected failure"
    except ExpertValidationError:
        pass
    assert gate.num_experts == 4 and len(experts) == 4
    assert torch.equal(gate.w_gate.detach(), w_gate_before)
    # The param group added for the rolled-back expert was removed again, and
    # no optimizer state is keyed on params that are no longer in any group.
    assert len(opt.param_groups) == n_groups_before
    live = {p for g in opt.param_groups for p in g["params"]}
    assert all(p in live for p in opt.state)


def test_foundry_nan_expert_rejected():
    gate, experts, foundry = _make_foundry()
    nan_src = """
class NanExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)
    def forward(self, x):
        return self.lin(x) * float('nan')
"""
    try:
        foundry.register_expert_from_source(nan_src)
        assert False, "expected failure"
    except ExpertValidationError:
        pass
    assert gate.num_experts == 4 and len(experts) == 4


def test_foundry_blocks_imports_and_open():
    gate, experts, foundry = _make_foundry()
    for evil in (
        "import os\nclass E(nn.Module):\n    def forward(self, x):\n        return x",
        "class E(nn.Module):\n    def forward(self, x):\n        open('/etc/passwd')\n        return x",
        "class E(nn.Module):\n    def forward(self, x):\n        return type(x).__class__\n",
    ):
        try:
            foundry.register_expert_from_source(evil)
            assert False, f"expected rejection of: {evil!r}"
        except ExpertValidationError:
            pass
    assert gate.num_experts == 4 and len(experts) == 4


def test_foundry_timeout_on_infinite_init():
    gate, experts, foundry = _make_foundry()
    foundry.timeout_s = 1.0
    loop_src = """
class Looper(nn.Module):
    def __init__(self):
        super().__init__()
        n = 0
        while True:
            n = n + 1
    def forward(self, x):
        return x
"""
    try:
        foundry.register_expert_from_source(loop_src)
        assert False, "expected timeout failure"
    except ExpertValidationError as exc:
        assert "timeout" in str(exc).lower()
    assert gate.num_experts == 4 and len(experts) == 4


# ---------------------------------------------------------------------------
# 7. Concurrency smoke test
# ---------------------------------------------------------------------------
def test_concurrent_read_during_expansion():
    # TRAIN mode is the hard case: forward consumes BOTH w_gate and w_noise,
    # so a torn parameter pair (grown gate + ungrown noise) crashes with a
    # shape mismatch. Eval mode only touches w_gate and cannot see that race.
    #
    # Contention is calibrated against a negative control: with the atomic
    # pair swap reverted to two independent writes, this configuration
    # (4 readers x 300 handshaked expansions, 10us GIL switch interval)
    # caught the torn pair in 5/5 runs, in under a second per run.
    gate = DynamicNoisyTopKGate(8, 4, k=2).train()
    x = torch.randn(16, 8)
    errors: list[BaseException] = []
    stop = threading.Event()
    iterations = [0]

    def reader():
        try:
            while not stop.is_set():
                out = gate(x)
                e = out.gates.shape[1]
                assert e >= 4, f"gate shrank to {e} experts"
                assert torch.isfinite(out.gates).all()
                # Not just "didn't crash": the snapshot must be internally
                # coherent -- normalised rows with exactly k nonzeros.
                row_sums = out.gates.sum(dim=1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
                nnz = (out.gates > 0).sum(dim=1)
                assert torch.all(nnz == 2), f"expected 2 nonzeros per row, got {nnz}"
                iterations[0] += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    n_expansions = 300
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # maximise thread interleaving in the swap window
    readers = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for r in readers:
        r.start()
    try:
        for _ in range(n_expansions):
            # Handshake: require at least one FRESH forward between expansions,
            # so every swap overlaps live readers instead of racing past idle
            # ones (a few back-to-back expansions can finish before a reader
            # completes a single iteration -- a guaranteed false-negative).
            target = iterations[0] + 1
            deadline = time.monotonic() + 10.0
            while iterations[0] < target and not errors:
                assert time.monotonic() < deadline, "readers made no progress in 10s"
                time.sleep(0)  # yield the GIL
            if errors:
                break
            gate.expand(1)
    finally:
        stop.set()
        for r in readers:
            r.join(timeout=10)
        sys.setswitchinterval(old_interval)
    assert not errors, f"reader thread errored: {errors[:1]}"
    assert gate.num_experts == 4 + n_expansions


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
