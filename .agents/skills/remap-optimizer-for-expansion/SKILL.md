---
name: remap-optimizer-for-expansion
description: >
  Reference for how optimizer state (Adam/AdamW/RMSprop momentum buffers)
  survives a DynamicNoisyTopKGate expansion. Read this when growing a gate that
  is being trained by a stateful optimizer, to understand why the migration must
  go through gate.expand(n, optimizer=opt) and must never be called standalone
  in a way that could desync gate size from optimizer state.
---

# remap_optimizer_for_expansion

Migrates per-parameter optimizer state from the pre-expansion gate parameters
onto the post-expansion (larger) ones. Stateful optimizers key their state on
parameter object *identity*; `expand()` replaces those objects, so without this
migration the existing experts' momentum buffers are orphaned and their training
history is silently lost.

Import (rarely needed directly — see the constraint below):

```python
from core.moe.dynamic_gating import remap_optimizer_for_expansion
```

## When to use
You almost never call this directly. It exists to be driven by
[dynamic-noisy-topk-gate](../dynamic-noisy-topk-gate/SKILL.md)'s `expand`.

- Correct path: `gate.expand(n, optimizer=opt)`. This calls the remap for you,
  under `gate.expand_lock`, at the correct moment (before the parameters are
  rebound), with both old and new parameters in hand.
- Direct call: only if you are implementing a custom expansion of tensors that
  are NOT the gate's `w_gate`/`w_noise`, and you fully control the swap.

## Signature (for reference)

```python
remap_optimizer_for_expansion(
    optimizer,                 # torch.optim.Optimizer being trained
    old_params,                # Sequence[nn.Parameter] just replaced
    new_params,                # Sequence[nn.Parameter] replacements, same order/rank,
)                              #   trailing dims larger-or-equal. Returns None.
```

Preserves matching moment buffers into the leading slice, zero-fills new
columns, carries the `step` counter over verbatim.

Raises `ValueError` on length mismatch, or if any `old` param is not found in
any of the optimizer's param groups (checked for ALL params before anything is
mutated).

## HARD CONSTRAINTS

1. **Prefer `gate.expand(n, optimizer=opt)` over a standalone call.** A standalone
   remap that succeeds while the corresponding parameter swap does NOT (or vice
   versa) desyncs `gate.num_experts` from what the optimizer trains — leading to
   the optimizer updating a dangling tensor while forward routes through a
   different one. `expand` orders these correctly and atomically.

2. **The migration is two-phase — do not interrupt or partially replicate it.**
   - Phase 1 (validate + build): locate every `old` param in its param group and
     build every resized buffer. Any failure (missing param, allocation error)
     raises HERE, before the optimizer is touched at all.
   - Phase 2 (commit): pop old state, install new state, swap the param object in
     its group — plain dict/list writes that cannot fail.
   Consequence: if the function raises, the optimizer is byte-for-byte unchanged.
   There is no such thing as a "half-migrated" optimizer from a single call.

3. **Multiple param groups are supported; identity keying is exact.** The remap
   searches all param groups and swaps the param in its owning group.
   `optimizer.state` is global (not per-group), so state re-keying is correct
   regardless of grouping. Do not attempt to "help" by moving params between
   groups around an expansion.

4. **Calling before the first `optimizer.step()` is safe.** If no state exists
   yet for a param, migration is skipped and only the group swap happens. Do not
   guard the call with a "has the optimizer stepped yet?" check.

## Failure modes and retry
- `ValueError` from a direct call → nothing mutated (guaranteed by phase 1).
  Fix the `old_params`/`new_params` you passed (wrong length, or a param that was
  never registered with this optimizer) and retry.
- Failure surfacing through `gate.expand(optimizer=...)` → the expand aborted
  before rebinding, so gate and optimizer are both intact. **Do not retry the
  expansion blindly.** First inspect: confirm `gate.num_experts` equals the
  pre-expand value and that `w_gate`/`w_noise` are still in exactly one param
  group each. Only retry once you have confirmed no partial state.

## Minimal correct sequence

```python
# Do NOT do this manually. This IS the intended usage:
gate.expand(2, optimizer=opt)   # remap runs inside, transactionally, under lock

# Old experts keep their Adam moments; new columns start at zero; step continues.
new_state = opt.state[gate.w_gate]
assert torch.equal(new_state["exp_avg"][:, :old_n], preserved_moments)
```
