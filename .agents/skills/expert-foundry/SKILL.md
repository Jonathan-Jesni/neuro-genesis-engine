---
name: expert-foundry
description: >
  Use when turning a raw PyTorch source string into a validated, live expert
  module and registering it into a DynamicNoisyTopKGate + expert ModuleList at
  runtime (core.moe.dynamic_gating.ExpertFoundry). Covers the registration
  contract, the transactional rollback guarantee, and the hard rule that the
  built-in sandbox is accident-prevention, not a security boundary against
  untrusted source.
---

# ExpertFoundry

Compiles a source string defining an `nn.Module` subclass, validates it against
a fixed interface contract, smoke-tests it under a timeout, and registers it as
a new live expert — appending to the expert list and growing the gate by one, in
one transaction. Import:

```python
from core.moe.dynamic_gating import (
    ExpertFoundry, ExpertRegistration, ExpertValidationError,
)
```

Pairs with [dynamic-noisy-topk-gate](../dynamic-noisy-topk-gate/SKILL.md).

## When to use
- You have generated (or received) Python source for a new expert and want it
  added to a running MoE without rebuilding the model.
- You need the registration to be all-or-nothing: on any validation failure the
  gate size, expert list, and optimizer must be left exactly as they were.

## Construct

```python
foundry = ExpertFoundry(
    gate,                    # a DynamicNoisyTopKGate
    experts,                 # nn.ModuleList, append-only, len == gate.num_experts
    expert_input_dim,        # int, feature dim fed to each expert.forward
    expert_output_dim,       # int, required output feature dim
    *,
    timeout_s=10.0,          # wall-clock budget for build + smoke test
    smoke_batch=4,           # batch size for the dummy forward
)
```

Precondition, enforced: `gate.num_experts == len(experts)` at construction. The
expert list MUST stay append-only for the lifetime of the foundry — never
reorder or truncate it; the gate's copy-on-write expansion relies on old gate
columns always mapping to a still-valid prefix of the expert list.

## register_expert_from_source

```python
reg = foundry.register_expert_from_source(
    source,                  # str: defines exactly one (or a named) nn.Module
                             #   subclass, zero-arg constructor
    *,
    class_name=None,         # str: pick a specific class if source defines several
    optimizer=None,          # migrate gate state + add the expert's params
)                            # -> ExpertRegistration
```

`source` contract (all enforced; violations raise `ExpertValidationError`):
- Defines one `nn.Module` subclass with a zero-argument `__init__`.
- `forward(x)` returns a Tensor of shape `[smoke_batch, expert_output_dim]`,
  floating dtype, all-finite.
- No `import` statements. `torch`, `nn`, `F`, `math`, `Tensor` are pre-injected;
  use them directly.

Returns `ExpertRegistration(index, class_name, source_sha256, build_seconds)`.
On success the gate has grown by one and `experts[index]` is the new module.

## HARD CONSTRAINTS

1. **The sandbox is accident-prevention, NOT a security boundary.** The restricted
   builtins and forbidden-substring/import screen stop honest mistakes, not
   adversaries: `torch.save`/`torch.load` (file I/O), `torch.hub` (network), and
   `getattr` with string-concatenated attribute names are all reachable. If the
   `source` originates from ANY untrusted party (not this agent's own
   LLM-generated code), you MUST escalate to human review before registering.
   Do not treat a passing validation as proof the source is safe. For genuinely
   untrusted code, registration must happen inside an external sandbox
   (subprocess/container with seccomp, no network, read-only FS, CPU/mem limits).

2. **A hung build is abandoned, not killed.** On `timeout_s` expiry the worker
   thread keeps running as a daemon (Python cannot preempt it) and registration
   fails cleanly. It does not reclaim the CPU. Do not lower `timeout_s` to
   "kill" runaway code — it only abandons faster.

3. **Registration is transactional; the expert list is append-only.** Do not
   manually append to `experts` or call `gate.expand` yourself around a foundry
   registration — the foundry owns that ordering (optimizer group first, then
   append, then grow the gate) and its rollback. Bypassing it can leave the gate
   routing to an expert column with no backing module.

## Failure modes and retry
- `ExpertValidationError` from any stage (syntax, forbidden pattern, wrong output
  dim, NaN/Inf output, timeout, post-append failure) → the foundry state (expert
  list, gate size, optimizer groups AND state) is rolled back to exactly the
  pre-call state. **Safe to retry with corrected source.** This is the common
  loop: generate → register → on `ExpertValidationError`, fix the source from the
  error message → register again.
- Untrusted-source origin → do NOT retry into the in-process foundry at all;
  escalate per constraint 1.

## Minimal correct sequence

```python
gate = DynamicNoisyTopKGate(16, 4, k=2)
experts = torch.nn.ModuleList(torch.nn.Linear(16, 16) for _ in range(4))
foundry = ExpertFoundry(gate, experts, expert_input_dim=16, expert_output_dim=16)
opt = torch.optim.Adam(list(gate.parameters()) + list(experts.parameters()))

src = """
class GeneratedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)
    def forward(self, x):
        return F.relu(self.lin(x))
"""

try:
    reg = foundry.register_expert_from_source(src, optimizer=opt)
    assert gate.num_experts == len(experts) == 5   # counts stay in lockstep
except ExpertValidationError as e:
    ...  # inspect e, correct src, retry — state is unchanged
```
