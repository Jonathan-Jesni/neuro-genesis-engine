---
name: dynamic-noisy-topk-gate
description: >
  Use when routing inputs through the Neuro-Genesis Mixture-of-Experts layer
  (core.moe.dynamic_gating.DynamicNoisyTopKGate): constructing the gate, running
  a forward pass to obtain sparse routing weights plus the load-balancing aux
  loss, or growing the expert count at runtime. Invoke this before touching the
  gate in any way — it defines concurrency, dtype, and expansion constraints
  that silently corrupt training or crash forward if violated.
---

# DynamicNoisyTopKGate

Shazeer-style noisy top-k MoE gate whose expert dimension can grow while the
network is live. Import:

```python
from core.moe.dynamic_gating import DynamicNoisyTopKGate, GateOutput
```

Do not read or re-derive `dynamic_gating.py`. This file is the contract.

## When to use
- You need sparse `[batch, num_experts]` routing weights for an MoE layer.
- You need the load-balancing auxiliary loss that prevents expert collapse.
- You need to add experts at runtime without rebuilding the model.

## Construct

```python
gate = DynamicNoisyTopKGate(
    in_features,        # int >= 1, router input width
    num_experts,        # int >= 1, initial expert count
    k=2,                # int >= 1, experts routed per token
    *,
    loss_coef=1e-2,     # multiplier on the aux loss
    noise_eps=1e-2,     # floor on exploration-noise std
    w_gate_init_std=0.01,
    device=None, dtype=None,
)
```

`num_experts` is never hardcoded downstream; read it live via the read-only
property `gate.num_experts` (it returns `w_gate.shape[1]`). Never cache it across
an `expand()`.

## forward

```python
out = gate(x)   # x: 2-D Tensor [batch, in_features]. Non-2-D raises ValueError.
```

Returns a `GateOutput` NamedTuple:
- `out.gates` — dense `[batch, num_experts]`, ≤`k` nonzeros per row summing to 1,
  cast back to `x.dtype`.
- `out.top_k_indices` — `[batch, k_eff]` long, selected expert ids
  (`k_eff = min(k, num_experts)`).
- `out.top_k_gates` — `[batch, k_eff]` weights aligned with the indices.
- `out.aux_loss` — fp32 scalar, ALWAYS present (train and eval).

Add the aux loss to your task loss on EVERY training step:
`(task_loss + out.aux_loss).backward()`. Omitting it causes expert collapse.

Autocast: the router forces fp32 internally and casts `gates` back to `x.dtype`.
Pass fp16/bf16 activations directly; do not upcast at the call site.

## expand

```python
gate.expand(n_new_experts, *, optimizer=None)   # n_new_experts >= 1
```

Grows the expert dimension by `n_new_experts` via copy-on-write. Existing expert
columns are preserved byte-for-byte; new columns start small. Pass `optimizer`
whenever a stateful optimizer (Adam/AdamW/etc.) is training the gate — see
[remap-optimizer-for-expansion](../remap-optimizer-for-expansion/SKILL.md).

## HARD CONSTRAINTS (violating any of these is a bug, not a style choice)

1. **Never read `gate.w_gate` and `gate.w_noise` as two separate accesses.**
   `forward` reads the fused snapshot `gate._router_params` in ONE atomic load
   per call. Reading the two parameters separately can observe a torn pair
   (a grown `w_gate` next to an ungrown `w_noise`) and crash on a shape
   mismatch. If you need both matrices, take `wg, wn = gate._router_params` in a
   single statement — never `gate.w_gate` then `gate.w_noise`.

2. **Serialize `expand()` against training with `gate.expand_lock`.**
   If any training step (forward+backward+step) can run concurrently with an
   expansion, acquire the lock around the training step:
   ```python
   with gate.expand_lock:
       (task_loss + out.aux_loss).backward()
       optimizer.step()
   ```
   `expand()` itself takes this lock. Copy-on-write makes concurrent *inference*
   safe lock-free, but a concurrent *backward* is NOT protected: gradients would
   land on an orphaned tensor.

3. **Never call `expand()` mid-backward.** The swap replaces the parameter
   object; a backward already in flight writes gradients to the old tensor,
   which is no longer the module's parameter. Complete backward + `optimizer.step()`
   before expanding.

4. **Never use `torch.func.functional_call` with this gate. Full stop.**
   It rebinds parameters through a raw `_parameters` dict write that bypasses
   the gate's staleness tracking, so `forward` keeps reading the module's real
   (non-substituted) parameters. This is not "use with caution" — it is
   incompatible. If a task requires functional-call-style parameter injection,
   escalate rather than work around it.

## Failure modes and retry
- `forward` raising `ValueError` (wrong rank / feature dim) → caller bug in the
  input shape. Fix the input; safe to retry.
- `expand` raising `ValueError` (`n_new_experts < 1`) → caller bug; nothing
  mutated; safe to retry with a valid count.
- `expand(optimizer=...)` raising → the optimizer remap is transactional and
  aborts before any parameter is rebound, so the gate is unchanged. Do NOT
  blindly retry: first confirm `gate.num_experts` and the optimizer param set
  are consistent (see the remap skill), then retry.

## Minimal correct sequence

```python
gate = DynamicNoisyTopKGate(in_features=16, num_experts=4, k=2).train()
opt = torch.optim.Adam(gate.parameters(), lr=1e-3)

with gate.expand_lock:              # serialize step vs. any concurrent expand
    out = gate(x)                   # x: [batch, 16]
    (out.gates.sum() + out.aux_loss).backward()
    opt.step(); opt.zero_grad()

gate.expand(2, optimizer=opt)       # 4 -> 6 experts, Adam moments migrated
assert gate.num_experts == 6        # read live, never cached
```
