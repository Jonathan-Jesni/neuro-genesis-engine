"""Dynamic Mixture-of-Experts gating + live expert code-generation framework.

This module is the foundational routing layer of the Neuro-Genesis Engine. It
provides two cooperating pieces plus one free helper:

    * ``DynamicNoisyTopKGate`` -- a Shazeer-style noisy top-k MoE gate whose
      expert count can grow *while the network is live*. Router math runs in
      fp32 under autocast, a load-balancing auxiliary loss is always returned,
      and expansion is a copy-on-write swap that is safe to read concurrently.

    * ``remap_optimizer_for_expansion`` -- a free function that migrates an
      optimizer's per-parameter state (e.g. Adam ``exp_avg`` / ``exp_avg_sq``)
      from the pre-expansion parameter tensors onto the post-expansion tensors,
      preserving the moments of existing experts *exactly* and zero-initialising
      the new columns. Without this, swapping the gate parameter silently
      orphans Adam's momentum buffers and existing experts lose their history.

    * ``ExpertFoundry`` -- a dynamic code-generation framework that turns a raw
      string of PyTorch source into a validated, live expert module. It exec's
      the source in a restricted namespace, enforces a fixed interface contract
      (``forward(x) -> Tensor`` with a fixed output dim), runs a NaN/shape smoke
      test under a wall-clock timeout, and registers the expert transactionally
      with rollback on any failure.

--------------------------------------------------------------------------------
Concurrency strategy (READ THIS before touching ``expand`` or ``forward``)
--------------------------------------------------------------------------------
The design goal is: **inference reads must be safe while an expansion write is
in progress, without readers taking a lock.** We achieve this with copy-on-write
plus reliance on the CPython GIL for the reference swap.

  * The gate publishes its two router matrices as ONE object: a plain tuple
    ``self._router_params = (w_gate, w_noise)``. ``forward`` reads that single
    attribute exactly once and derives the expert count from the captured
    locals. It never re-reads ``self`` afterwards. Because the old parameter
    tensors are never mutated in place, a forward pass that began before an
    expansion runs to completion coherently against the *old* expert set.

  * ``expand`` never mutates existing tensors. It builds brand-new, larger
    tensors off to the side, copies the old columns in, and then publishes the
    new ``(w_gate, w_noise)`` pair with a single attribute assignment to
    ``_router_params`` (the subsequent ``register_parameter`` calls are module
    bookkeeping that readers never consult). Both the tuple read in ``forward``
    and the tuple write in ``expand`` are individual bytecode operations, hence
    atomic under the GIL: a reader sees either the whole old pair or the whole
    new pair -- never a torn state where ``w_gate`` has grown but ``w_noise``
    has not. (Swapping the two parameters independently WOULD tear: a training
    forward could then pair a grown gate with an ungrown noise matrix and crash
    on the shape mismatch. That is exactly why the pair is fused.)

  * The caller's expert collection MUST be append-only (never reorder or
    truncate), so that an "old" gate matrix always refers to a still-valid
    prefix of the expert list. ``ExpertFoundry`` enforces this (it only ever
    appends to its ``nn.ModuleList``).

  * ``expand`` and ``remap_optimizer_for_expansion`` take ``expand_lock`` so two
    writers cannot interleave.

Explicit NON-goal: this makes concurrent *inference* safe. It does NOT make a
concurrent *backward* safe -- if a training step captured the old parameter and
an expansion swaps it mid-step, the backward writes gradients onto a tensor that
is no longer the module's parameter. Training steps and expansions must be
externally serialised; hold ``gate.expand_lock`` around your optimizer step if
you expand from another thread.

--------------------------------------------------------------------------------
Optimizer-remapping strategy (READ THIS before wiring up training)
--------------------------------------------------------------------------------
Stateful optimizers (Adam/AdamW/RMSprop/etc.) key their per-parameter state on
the *identity* of the ``Parameter`` object. ``expand`` replaces the parameter
object, so unless we intervene, the optimizer's ``state[old_param]`` (holding
``exp_avg``, ``exp_avg_sq``, ``step``) becomes dead weight and the new parameter
starts from a cold state -- existing experts lose all momentum. That is the
exact "silently break Adam momentum buffers" failure the spec forbids.

``remap_optimizer_for_expansion`` fixes this by, for each (old, new) pair:
  1. popping ``optimizer.state[old]``;
  2. for every *tensor* buffer whose shape matches the old parameter, allocating
     a zero tensor of the new (larger) shape and copying the old values into the
     leading slice (``new_buf[..., :E_old] = old_buf``) -- so existing experts
     keep their moments bit-for-bit and new columns start at zero;
  3. leaving scalar entries (like the ``step`` counter) untouched;
  4. installing the migrated state under ``optimizer.state[new]`` and swapping
     ``old`` -> ``new`` inside the owning ``param_group["params"]``.

Why zero-init the new columns rather than ``add_param_group``? The old and new
experts share ONE fused matrix (columns of ``w_gate``); you cannot put half a
tensor in a separate param group. So the whole matrix must live in one group and
we migrate state in place. The only cost is that the preserved global ``step``
means Adam's bias-correction under-inflates the very first few updates of the
new columns (it assumes they have ``step`` gradients of history when they have
zero). This is benign and identical to how LoRA / net2net style parameter
injection is handled in practice; new experts converge fine.

The remap is two-phase: it first validates every precondition and builds the
resized buffers WITHOUT touching the optimizer, then commits with plain dict
and list writes that cannot fail. If it raises, the optimizer is untouched --
a partial remap can never leave state popped for one parameter and stale for
another.

The convenience path is ``gate.expand(n, optimizer=opt)``, which performs the
swap and the remap together under the lock, transactionally: if anything
raises, the gate, its parameters, and the optimizer are all left exactly as
they were.

--------------------------------------------------------------------------------
Worked example
--------------------------------------------------------------------------------
>>> import torch
>>> from core.moe.dynamic_gating import DynamicNoisyTopKGate, ExpertFoundry
>>> gate = DynamicNoisyTopKGate(in_features=16, num_experts=4, k=2)
>>> experts = torch.nn.ModuleList(torch.nn.Linear(16, 16) for _ in range(4))
>>> foundry = ExpertFoundry(gate=gate, experts=experts,
...                         expert_input_dim=16, expert_output_dim=16)
>>> opt = torch.optim.Adam(list(gate.parameters()) + list(experts.parameters()))
>>> x = torch.randn(8, 16)
>>> out = gate(x)                       # GateOutput(gates, indices, values, aux_loss)
>>> task_loss = out.gates.sum()         # stand-in for a real downstream loss
>>> (task_loss + out.aux_loss).backward()
>>> opt.step(); opt.zero_grad()
>>> src = '''
... class GeneratedExpert(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.lin = nn.Linear(16, 16)
...     def forward(self, x):
...         return F.relu(self.lin(x))
... '''
>>> reg = foundry.register_expert_from_source(src, optimizer=opt)  # doctest: +SKIP
>>> gate.num_experts                                               # doctest: +SKIP
5
"""

from __future__ import annotations

import ast
import builtins as _builtins
import hashlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "GateOutput",
    "DynamicNoisyTopKGate",
    "remap_optimizer_for_expansion",
    "ExpertFoundry",
    "ExpertRegistration",
    "ExpertValidationError",
]


# =============================================================================
# Gate output container
# =============================================================================
class GateOutput(NamedTuple):
    """Structured return value of :meth:`DynamicNoisyTopKGate.forward`.

    Attributes:
        gates: Dense ``[batch, num_experts]`` tensor of routing weights. Each
            row has at most ``k`` non-zero entries that sum to 1. Cast back to
            the caller's input dtype so it can multiply autocast activations.
        top_k_indices: ``[batch, k_eff]`` long tensor of selected expert ids.
        top_k_gates: ``[batch, k_eff]`` tensor of the (normalised) weights for
            the selected experts, in the same order as ``top_k_indices``.
        aux_loss: Scalar load-balancing loss (importance + load). ALWAYS
            present -- add it to your task loss every step to prevent expert
            collapse. It is returned in eval mode too (you typically ignore it
            there).
    """

    gates: Tensor
    top_k_indices: Tensor
    top_k_gates: Tensor
    aux_loss: Tensor


def _squared_cv(x: Tensor, eps: float = 1e-10) -> Tensor:
    """Squared coefficient of variation ``(std / mean)**2`` of a 1-D tensor.

    This is Shazeer's dispersion measure: 0 when perfectly balanced, growing as
    load concentrates on a few experts. Defined as 0 for a single element (a
    lone expert cannot be "imbalanced"), which also avoids a 0/0.
    """
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    mean = x.mean()
    # Population variance (unbiased=False): matches the reference implementation
    # and is well-defined for n >= 1.
    var = x.var(unbiased=False)
    return var / (mean * mean + eps)


# =============================================================================
# The gate
# =============================================================================
class DynamicNoisyTopKGate(nn.Module):
    """Noisy top-k MoE gate with a dynamically expandable expert dimension.

    The gate maps an input of shape ``[batch, in_features]`` to sparse routing
    weights of shape ``[batch, num_experts]`` using the noisy top-k mechanism of
    Shazeer et al., "Outrageously Large Neural Networks" (2017)::

        clean_logits = x @ W_gate
        noise_std    = softplus(x @ W_noise) + noise_eps
        noisy_logits = clean_logits + N(0, 1) * noise_std     (training only)
        keep top-k logits, softmax over them, scatter back to a dense matrix.

    ``num_experts`` is NEVER hardcoded: it is always ``W_gate.shape[1]``. Call
    :meth:`expand` to grow it at runtime.

    Autocast: the router math is forced to fp32 (see :meth:`forward`) because
    top-k selection and the softmax over close logits are precision-sensitive;
    the returned ``gates`` are cast back to the input dtype so the module drops
    into a mixed-precision model transparently.

    See the module docstring for the full concurrency and optimizer-remapping
    contracts. The short version:
        * ``forward`` is lock-free and safe to call while ``expand`` runs.
        * ``expand`` is copy-on-write; hold ``expand_lock`` around backward if
          you expand concurrently with training.
    """

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        k: int = 2,
        *,
        loss_coef: float = 1e-2,
        noise_eps: float = 1e-2,
        w_gate_init_std: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the gate.

        Args:
            in_features: Dimensionality of the router input.
            num_experts: Initial number of experts (>= 1). Grows via ``expand``.
            k: Number of experts to route each token to (default 2). If it ever
                exceeds the live expert count it is clamped per-forward.
            loss_coef: Multiplier on the auxiliary load-balancing loss.
            noise_eps: Floor added to the softplus noise std for numerical
                stability and to guarantee non-zero exploration noise.
            w_gate_init_std: Std of the normal init for gate columns (also used
                for the columns added by ``expand``). Kept small so freshly
                added experts start with near-uniform low logits and do not
                hijack routing before they have learned anything.
            device: Standard PyTorch factory kwarg for the parameters.
            dtype: Standard PyTorch factory kwarg for the parameters.
        """
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.in_features = in_features
        self.k = k
        self.loss_coef = loss_coef
        self.noise_eps = noise_eps
        self.w_gate_init_std = w_gate_init_std

        factory: dict[str, Any] = {"device": device, "dtype": dtype}
        # W_gate: clean routing logits.  W_noise: per-(token, expert) noise scale.
        w_gate = torch.empty(in_features, num_experts, **factory)
        nn.init.normal_(w_gate, std=w_gate_init_std)
        w_noise = torch.zeros(in_features, num_experts, **factory)
        self.w_gate = nn.Parameter(w_gate)
        self.w_noise = nn.Parameter(w_noise)
        # Coherent snapshot for lock-free readers: forward() reads this ONE
        # attribute to get both matrices, and expand() publishes replacements
        # with one attribute write, so the pair can never be observed torn
        # (a grown w_gate next to a not-yet-grown w_noise). Kept in sync by
        # expand(), _apply() (.to()/.cuda() moves), and register_parameter()
        # (attribute reassignment, load_state_dict(assign=True)). NOT kept in
        # sync by raw _parameters dict writes (torch.func.functional_call) --
        # see register_parameter's docstring.
        self._router_params: tuple[nn.Parameter, nn.Parameter] = (
            self.w_gate,
            self.w_noise,
        )

        # Writer lock. Readers never take this. Exposed publicly so callers can
        # serialise their training step against a concurrent expansion.
        self._expand_lock = threading.Lock()

        # Standard-normal loc/scale used by the differentiable load estimator.
        # Registered as non-persistent buffers so they track .to()/.cuda() moves
        # and are not written into state_dict.
        self.register_buffer("_normal_loc", torch.zeros((), **factory), persistent=False)
        self.register_buffer("_normal_scale", torch.ones((), **factory), persistent=False)

    # -- introspection --------------------------------------------------------
    @property
    def num_experts(self) -> int:
        """Live expert count. Single source of truth: ``W_gate.shape[1]``."""
        return self.w_gate.shape[1]

    @property
    def expand_lock(self) -> threading.Lock:
        """The writer lock. Acquire it to serialise training against expansion.

        Example::

            with gate.expand_lock:
                (task_loss + out.aux_loss).backward()
                optimizer.step()
        """
        return self._expand_lock

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, num_experts={self.num_experts}, "
            f"k={self.k}, loss_coef={self.loss_coef}"
        )

    def _apply(self, fn, *args: Any, **kwargs: Any):
        # .to()/.cuda()/.float() may replace the Parameter objects outright
        # (depending on torch version / overwrite-on-conversion settings) via a
        # DIRECT self._parameters[key] = ... write that bypasses
        # register_parameter, which would leave _router_params pointing at the
        # pre-move tensors. Re-publish so lock-free readers see the moved pair.
        ret = super()._apply(fn, *args, **kwargs)
        self._router_params = (self.w_gate, self.w_noise)
        return ret

    def register_parameter(self, name: str, param: Optional[nn.Parameter]) -> None:
        """Re-publish the reader snapshot whenever a router matrix is rebound.

        Every attribute-level rebind path in ``nn.Module`` funnels through
        here: plain assignment (``__setattr__`` routes Parameter values to
        ``register_parameter``) and ``load_state_dict(..., assign=True)``
        (whose assign branch calls ``setattr``). Without this hook, either of
        those would strand ``_router_params`` on the pre-rebind tensors and
        ``forward`` would silently keep routing with stale weights.

        The pair is only re-published when both matrices exist AND agree in
        shape. That guard skips mid-swap states -- :meth:`expand` registers the
        two grown parameters back-to-back after having already published the
        coherent pair itself, and an ``assign=True`` load of a different-sized
        checkpoint rebinds ``w_gate`` before ``w_noise`` -- so a torn pair can
        never be published from here. Consequence: if you manually reassign
        only ONE matrix with a new expert count, the snapshot stays on the old
        coherent pair until you reassign the other.

        NOT covered (bypasses ``__setattr__`` and this method entirely): raw
        ``module._parameters[...] = ...`` dict writes, as done by
        ``torch.func.functional_call`` / ``NamedMemberAccessor.set_tensor``.
        Under ``functional_call`` this gate's forward keeps reading the
        module's real parameters, not the substituted ones.
        """
        super().register_parameter(name, param)
        if name in ("w_gate", "w_noise"):
            wg = self._parameters.get("w_gate")
            wn = self._parameters.get("w_noise")
            if wg is not None and wn is not None and wg.shape == wn.shape:
                self._router_params = (wg, wn)

    # -- forward --------------------------------------------------------------
    def forward(self, x: Tensor) -> GateOutput:
        """Route ``x`` and compute the load-balancing auxiliary loss.

        Args:
            x: ``[batch, in_features]`` input activations. May be fp16/bf16
                under autocast; the router upcasts internally.

        Returns:
            A :class:`GateOutput`. ``gates`` is dense ``[batch, num_experts]``
            in ``x.dtype``; ``aux_loss`` is a scalar to add to your task loss.
        """
        if x.dim() != 2:
            raise ValueError(
                f"expected 2-D input [batch, in_features], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"input feature dim {x.shape[1]} != in_features {self.in_features}"
            )

        # --- CONCURRENCY: capture BOTH router matrices in one atomic read. --
        # _router_params is rebound (never mutated) by expand(), so this single
        # attribute load yields a coherent (w_gate, w_noise) pair even if an
        # expansion lands mid-forward. Reading the two parameters separately
        # would tear: expand() could swap between the reads and pair a grown
        # gate with an ungrown noise matrix. See module docstring.
        w_gate, w_noise = self._router_params
        num_experts = w_gate.shape[1]

        out_dtype = x.dtype
        device_type = x.device.type

        # --- AUTOCAST SAFETY: force the router into fp32. -------------------
        # top-k ties and the softmax over small logit gaps are precision-
        # sensitive; a fp16 router routes erratically. This mirrors Switch-
        # Transformer practice. We disable autocast for the region and upcast.
        with torch.autocast(device_type=device_type, enabled=False):
            xf = x.float()
            wg = w_gate.float()
            wn = w_noise.float()

            clean_logits = xf @ wg  # [batch, num_experts]

            if self.training:
                # Learned, input-dependent noise std (Shazeer). Floor with
                # noise_eps so std > 0 and exploration never fully vanishes.
                raw_noise = xf @ wn
                noise_std: Optional[Tensor] = F.softplus(raw_noise) + self.noise_eps
                logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            else:
                # No stochasticity in eval: routing is deterministic.
                noise_std = None
                logits = clean_logits

            k_eff = min(self.k, num_experts)
            # Grab one extra logit (k_eff + 1) when available; the (k+1)-th value
            # is the threshold used by the differentiable load estimator.
            top_n = min(k_eff + 1, num_experts)
            top_logits, top_indices = logits.topk(top_n, dim=1)  # [batch, top_n]

            # Softmax over ONLY the k_eff kept logits -> normalised gate weights.
            top_k_logits = top_logits[:, :k_eff]
            top_k_indices = top_indices[:, :k_eff]
            top_k_gates = F.softmax(top_k_logits, dim=1)  # [batch, k_eff]

            # Scatter the sparse weights back into a dense [batch, num_experts].
            gates = torch.zeros(
                xf.shape[0], num_experts, dtype=torch.float32, device=xf.device
            )
            gates.scatter_(1, top_k_indices, top_k_gates)

            aux_loss = self._aux_loss(
                gates=gates,
                clean_logits=clean_logits,
                noise_std=noise_std,
                top_logits=top_logits,
                top_indices=top_indices,
                k_eff=k_eff,
            )

        return GateOutput(
            gates=gates.to(out_dtype),
            top_k_indices=top_k_indices,
            top_k_gates=top_k_gates.to(out_dtype),
            aux_loss=aux_loss,  # keep fp32 scalar; losses are summed in fp32
        )

    # -- auxiliary loss -------------------------------------------------------
    def _aux_loss(
        self,
        *,
        gates: Tensor,
        clean_logits: Tensor,
        noise_std: Optional[Tensor],
        top_logits: Tensor,
        top_indices: Tensor,
        k_eff: int,
    ) -> Tensor:
        """Load-balancing loss = coef * (CV(importance)^2 + CV(load)^2).

        * importance: column sum of the gate weights -- how much probability
          mass each expert receives. Smooth and always differentiable.
        * load: an estimate of how many examples each expert is *assigned*.
          In training we use the differentiable estimator P(expert kept) under
          the injected Gaussian noise, so the gate learns to spread load. In
          eval there is no noise, so we fall back to the hard assignment count
          (non-differentiable, which is fine -- we do not train in eval).

        Both terms are penalised via squared coefficient of variation: minimised
        when every expert carries equal importance/load, which is what prevents
        the winner-take-all collapse MoE gates are prone to.
        """
        importance = gates.sum(dim=0)  # [num_experts]
        num_experts = gates.shape[1]

        if self.training and noise_std is not None and num_experts > k_eff:
            # Differentiable load: probability each expert would land in the
            # top-k under a fresh draw of the injected noise (Shazeer's
            # ``_prob_in_top_k``). Summed over the batch -> expected load.
            load = self._prob_in_top_k(
                clean_logits=clean_logits,
                noise_std=noise_std,
                top_logits=top_logits,
                top_indices=top_indices,
                k_eff=k_eff,
            ).sum(dim=0)
        else:
            # Hard load count. Boolean-derived, so it carries no gradient;
            # acceptable because we only optimise in training.
            load = (gates > 0).float().sum(dim=0)

        loss = _squared_cv(importance) + _squared_cv(load)
        return self.loss_coef * loss

    def _prob_in_top_k(
        self,
        *,
        clean_logits: Tensor,
        noise_std: Tensor,
        top_logits: Tensor,
        top_indices: Tensor,
        k_eff: int,
    ) -> Tensor:
        """Smooth P(expert would be in the top-k) under the injected noise.

        Returns a ``[batch, num_experts]`` tensor of probabilities. Following
        Shazeer et al.: with Gaussian noise of std ``noise_std`` added to each
        clean logit, the probability that expert i clears the selection
        threshold is ``Phi((clean_i - threshold_i) / std_i)``. An expert that is
        *currently* selected is compared against the (k+1)-th noisy logit (the
        best competitor it displaced); an expert that is *not* selected is
        compared against the k-th noisy logit (the weakest it must beat).
        """
        batch, num_experts = clean_logits.shape
        normal = torch.distributions.Normal(self._normal_loc, self._normal_scale)

        # kth largest noisy logit -> threshold an OUT expert must beat to get in.
        threshold_if_out = top_logits[:, k_eff - 1 : k_eff]  # [batch, 1]
        # (k+1)-th largest -> threshold an IN expert must beat to stay in.
        if top_logits.shape[1] > k_eff:
            threshold_if_in = top_logits[:, k_eff : k_eff + 1]  # [batch, 1]
        else:
            threshold_if_in = threshold_if_out

        # Which experts are currently selected.
        is_in = torch.zeros(
            batch, num_experts, dtype=torch.bool, device=clean_logits.device
        )
        is_in.scatter_(1, top_indices[:, :k_eff], True)

        # Pick the relevant threshold per (example, expert) and standardise.
        thresh = torch.where(is_in, threshold_if_in, threshold_if_out)
        return normal.cdf((clean_logits - thresh) / noise_std)

    # -- dynamic expansion ----------------------------------------------------
    def expand(
        self,
        n_new_experts: int,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Grow the expert dimension by ``n_new_experts`` in a copy-on-write swap.

        The existing experts' learned weights are preserved *exactly*. New
        ``w_gate`` columns are initialised small (``w_gate_init_std``) and new
        ``w_noise`` columns are zero-initialised (``softplus(0) ~= 0.69`` gives
        the fresh experts a healthy exploration-noise std). Device and dtype
        follow the current parameters, so an AMP/CUDA module stays consistent.

        If ``optimizer`` is given, its per-parameter state is migrated with
        :func:`remap_optimizer_for_expansion` so Adam-style momentum buffers for
        the existing experts survive the swap. **You almost always want to pass
        the optimizer** -- omitting it and continuing to train orphans the old
        momentum buffers (see module docstring).

        Concurrency: this method holds ``expand_lock`` and never mutates the old
        tensors, so ``forward`` may run lock-free throughout. It does NOT protect
        a concurrent backward -- serialise training yourself via ``expand_lock``.

        Transactionality: the optimizer remap validates all preconditions before
        mutating anything (see :func:`remap_optimizer_for_expansion`), and the
        new parameters are only published after it succeeds. If this method
        raises, the gate's parameters, the published router pair, and the
        optimizer are all left exactly as they were.

        Args:
            n_new_experts: How many experts to add (>= 1).
            optimizer: Optional optimizer whose state should be migrated.
        """
        if n_new_experts < 1:
            raise ValueError(f"n_new_experts must be >= 1, got {n_new_experts}")

        with self._expand_lock:
            old_gate = self.w_gate
            old_noise = self.w_noise
            old_e = old_gate.shape[1]
            new_e = old_e + n_new_experts

            # Build the new tensors off to the side; never touch the old ones.
            with torch.no_grad():
                new_gate_data = torch.empty(
                    self.in_features,
                    new_e,
                    device=old_gate.device,
                    dtype=old_gate.dtype,
                )
                nn.init.normal_(new_gate_data, std=self.w_gate_init_std)
                new_gate_data[:, :old_e].copy_(old_gate.data)

                new_noise_data = torch.zeros(
                    self.in_features,
                    new_e,
                    device=old_noise.device,
                    dtype=old_noise.dtype,
                )
                new_noise_data[:, :old_e].copy_(old_noise.data)

            new_gate = nn.Parameter(new_gate_data, requires_grad=old_gate.requires_grad)
            new_noise = nn.Parameter(new_noise_data, requires_grad=old_noise.requires_grad)

            # Migrate optimizer state BEFORE rebinding, while we still hold refs
            # to the old parameter objects (the remap needs old AND new).
            if optimizer is not None:
                remap_optimizer_for_expansion(
                    optimizer,
                    old_params=[old_gate, old_noise],
                    new_params=[new_gate, new_noise],
                )

            # Publish the new pair for lock-free readers FIRST: one atomic
            # attribute write moves w_gate and w_noise together, so a reader
            # can never observe a torn (new gate, old noise) combination.
            # register_parameter afterwards is module bookkeeping (state_dict /
            # .parameters()) that forward() never consults.
            self._router_params = (new_gate, new_noise)
            self.register_parameter("w_gate", new_gate)
            self.register_parameter("w_noise", new_noise)


# =============================================================================
# Optimizer state migration
# =============================================================================
def remap_optimizer_for_expansion(
    optimizer: torch.optim.Optimizer,
    old_params: Sequence[nn.Parameter],
    new_params: Sequence[nn.Parameter],
) -> None:
    """Migrate optimizer per-parameter state from old tensors to expanded ones.

    See the module docstring's "Optimizer-remapping strategy" section for the
    full rationale. In brief: stateful optimizers key ``optimizer.state`` on
    parameter *identity*. When a parameter is replaced by a larger one, its
    momentum buffers must be resized and re-keyed, or they are lost and the
    existing experts lose their training history.

    The migration is two-phase so it is atomic with respect to failure:

    Phase 1 (validation + build, touches NOTHING in the optimizer):
      * locate every ``old`` param inside its owning ``param_group["params"]``
        (raising if any is missing -- checked for ALL params before any state
        is popped, never interleaved with mutation);
      * for every tensor buffer that matches ``old``'s shape, allocate a zero
        tensor of ``new``'s shape and copy the old values into the leading
        slice (assumes the expansion appended entries on the trailing dims,
        which is how :meth:`DynamicNoisyTopKGate.expand` grows);
      * carry non-tensor / scalar entries (e.g. ``step``) over verbatim.

    Phase 2 (commit, plain dict/list writes that cannot fail):
      * pop ``optimizer.state[old]``, install the migrated state under
        ``optimizer.state[new]``, and replace ``old`` with ``new`` at its
        recorded position in the param group.

    If this function raises, the optimizer has not been modified at all.

    Args:
        optimizer: The optimizer to migrate in place.
        old_params: Parameters that were just replaced.
        new_params: Their replacements, same order and rank, only trailing
            dimensions larger (or equal).

    Raises:
        ValueError: On length mismatch, or if an ``old`` param is not found in
            any param group. Raised before any optimizer mutation.
    """
    if len(old_params) != len(new_params):
        raise ValueError(
            f"old/new length mismatch: {len(old_params)} vs {len(new_params)}"
        )

    # --- phase 1a: locate every old param BEFORE mutating anything ---------
    slots: list[tuple[list, int]] = []  # (param_group["params"], index) per old
    for old in old_params:
        slot: Optional[tuple[list, int]] = None
        for group in optimizer.param_groups:
            params = group["params"]
            for i, p in enumerate(params):
                if p is old:
                    slot = (params, i)
                    break
            if slot is not None:
                break
        if slot is None:
            raise ValueError(
                "old parameter not found in any optimizer param_group; cannot "
                "remap. Was this parameter actually registered with the optimizer?"
            )
        slots.append(slot)

    # --- phase 1b: build the migrated state dicts (allocations may fail) ---
    migrated_states: list[Optional[dict[str, Any]]] = []
    for old, new in zip(old_params, new_params):
        state = optimizer.state.get(old)
        if state is None:
            # No state yet (e.g. expand() before the first optimizer.step()):
            # nothing to migrate, the group swap below is still required.
            migrated_states.append(None)
            continue
        migrated: dict[str, Any] = {}
        for key, val in state.items():
            if torch.is_tensor(val) and tuple(val.shape) == tuple(old.shape):
                # Resize a moment buffer (exp_avg, exp_avg_sq, momentum, ...).
                resized = torch.zeros(new.shape, dtype=val.dtype, device=val.device)
                # Copy the old values into the matching leading slice.
                slices = tuple(slice(0, s) for s in old.shape)
                resized[slices] = val
                migrated[key] = resized
            else:
                # Scalars (step counter), or unexpectedly-shaped buffers:
                # carry over verbatim. Preserving `step` keeps Adam's
                # timeline continuous for the surviving columns.
                migrated[key] = val
        migrated_states.append(migrated)

    # --- phase 2: commit. Dict/list writes only -- nothing here can raise. -
    for (old, new), (params, i), migrated in zip(
        zip(old_params, new_params), slots, migrated_states
    ):
        optimizer.state.pop(old, None)
        if migrated is not None:
            optimizer.state[new] = migrated
        params[i] = new


# =============================================================================
# Dynamic expert code-generation framework
# =============================================================================
class ExpertValidationError(RuntimeError):
    """Raised when a generated expert fails static, sandbox, or smoke checks."""


@dataclass(frozen=True)
class ExpertRegistration:
    """Audit record returned by a successful expert registration.

    Attributes:
        index: Position of the new expert in the foundry's ``ModuleList`` (and
            the gate column it corresponds to).
        class_name: Name of the instantiated ``nn.Module`` subclass.
        source_sha256: Hex digest of the exact source string that was run.
        build_seconds: Wall-clock time spent instantiating + smoke-testing.
    """

    index: int
    class_name: str
    source_sha256: str
    build_seconds: float


# Substrings that indicate attempts to break out of the restricted namespace via
# attribute traversal. A coarse guardrail (see ExpertFoundry's security note),
# NOT a substitute for real isolation.
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__reduce__",
)


def _build_safe_builtins() -> dict[str, Any]:
    """Construct the whitelist of builtins visible to generated expert code.

    This blocks the obvious footguns -- ``open``, ``exec``, ``eval``,
    ``compile``, ``input``, ``__import__``, ``setattr``, ``delattr``, ``vars``,
    ``globals``, ``locals`` are all absent -- but it is accident-prevention,
    NOT adversarial sandboxing. Equivalent capabilities remain reachable
    through the injected ``torch`` module (``torch.save`` / ``torch.load`` are
    file I/O, ``torch.hub`` reaches the network), and the source-level
    forbidden-substring scan is trivially evaded by string concatenation fed
    to the whitelisted ``getattr``. Do not read "no open" as a security
    boundary; see the ExpertFoundry security warning for the real containment
    story. ``getattr`` is included because ``nn.Module.__init__`` machinery
    uses it.
    """
    allowed = (
        "len", "range", "min", "max", "abs", "sum", "int", "float", "bool",
        "list", "tuple", "dict", "set", "frozenset", "zip", "enumerate",
        "isinstance", "issubclass", "super", "type", "print", "reversed",
        "sorted", "map", "filter", "any", "all", "round", "slice", "property",
        "staticmethod", "classmethod", "object", "repr", "format", "hasattr",
        "getattr", "iter", "next",
        "Exception", "ValueError", "RuntimeError", "TypeError", "IndexError",
        "KeyError", "AttributeError", "ZeroDivisionError", "NotImplementedError",
        "StopIteration", "ArithmeticError",
    )
    out: dict[str, Any] = {}
    for name in allowed:
        if hasattr(_builtins, name):
            out[name] = getattr(_builtins, name)
    # __build_class__ is the implicit builtin that the `class` statement compiles
    # into -- without it no expert class can be defined. It does not enable any
    # escape beyond defining classes, so it is safe to expose.
    out["__build_class__"] = _builtins.__build_class__
    # __name__ is read by some metaclass / dataclass machinery during class body
    # execution; provide a neutral value.
    out["__name__"] = "generated_expert_module"
    return out


_SAFE_BUILTINS: dict[str, Any] = _build_safe_builtins()

# Filename stamped onto compiled expert code; used to identify classes that were
# genuinely defined by the generated source (vs injected references).
_GENERATED_FILENAME = "<generated-expert>"


class ExpertFoundry:
    """Turns raw PyTorch source strings into validated, live expert modules.

    Pipeline for :meth:`register_expert_from_source`:
        1. Static gate: compile the source (catch syntax errors), reject obvious
           sandbox-escape patterns and any imports via an AST walk.
        2. Sandboxed exec in a namespace with restricted builtins and only
           ``torch``, ``nn``, ``F``, ``math`` (+ ``Tensor``) pre-injected.
        3. Discover the ``nn.Module`` subclass (explicit ``class_name`` or the
           unique one defined).
        4. Instantiate + smoke-test in a worker thread with a wall-clock timeout.
        5. Validate the interface contract: ``forward(x) -> Tensor`` with the
           configured output dim, finite, floating dtype.
        6. Register transactionally: optimizer changes first (they touch
           nothing else), then append the expert, then grow the gate last --
           with full rollback to the pre-registration state (expert list, gate
           size, AND optimizer) if any step fails.

    .. warning::
        **The in-process sandbox is a guardrail against accidents, NOT a
        security boundary against adversarial code.** CPython's ``exec`` cannot
        be made truly safe: a determined attacker can escape restricted builtins
        via bytecode or C-extension tricks, and a busy-loop cannot be force-
        killed (Python threads are not preemptible). It does not even take
        exotic tricks: the injected ``torch`` module alone provides file I/O
        (``torch.save`` / ``torch.load``) and network access (``torch.hub``),
        and the forbidden-substring scan is defeated by string concatenation
        fed to the whitelisted ``getattr``. For genuinely untrusted
        source, run the foundry inside a locked-down subprocess or container
        (seccomp / no network / read-only FS / cgroup CPU+memory limits) and
        treat this class's checks as defense-in-depth. The timeout here abandons
        a hung worker thread (it keeps running as a daemon) and fails the
        registration cleanly, but it does not reclaim the CPU.

    The expert list is append-only, which is what makes the gate's copy-on-write
    expansion safe to read concurrently (see the gate's concurrency docs).
    """

    def __init__(
        self,
        gate: DynamicNoisyTopKGate,
        experts: nn.ModuleList,
        expert_input_dim: int,
        expert_output_dim: int,
        *,
        timeout_s: float = 10.0,
        smoke_batch: int = 4,
    ) -> None:
        """Wire the foundry to a gate and its expert list.

        Args:
            gate: The gate to grow on each successful registration. Its current
                ``num_experts`` must equal ``len(experts)``.
            experts: The live (append-only) expert module list.
            expert_input_dim: Feature dim fed to every expert's ``forward``.
            expert_output_dim: Required output feature dim of every expert.
            timeout_s: Wall-clock budget for instantiation + smoke test.
            smoke_batch: Batch size used for the dummy forward pass.
        """
        if gate.num_experts != len(experts):
            raise ValueError(
                f"gate.num_experts ({gate.num_experts}) must equal "
                f"len(experts) ({len(experts)}) at foundry construction"
            )
        self.gate = gate
        self.experts = experts
        self.expert_input_dim = expert_input_dim
        self.expert_output_dim = expert_output_dim
        self.timeout_s = timeout_s
        self.smoke_batch = smoke_batch
        # Serialise registrations so two concurrent generations can't both grow
        # the gate/experts and desynchronise their counts.
        self._register_lock = threading.Lock()

    # -- public API -----------------------------------------------------------
    def register_expert_from_source(
        self,
        source: str,
        *,
        class_name: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> ExpertRegistration:
        """Validate ``source`` and register the resulting expert live.

        Args:
            source: Python source defining exactly one (or a named) ``nn.Module``
                subclass with a zero-argument constructor.
            class_name: If given, the specific class to instantiate; otherwise
                the unique module subclass in the source is used.
            optimizer: If given, its state is migrated when the gate grows, and
                the new expert's parameters are added as a fresh param group so
                they start training immediately.

        Returns:
            An :class:`ExpertRegistration` audit record.

        Raises:
            ExpertValidationError: If any stage fails. On failure the foundry's
                state (expert list, gate size, and optimizer groups/state) is
                left exactly as it was.
        """
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()

        # ---- 1. static analysis (no code executed) ------------------------
        compiled = self._static_check(source)

        # ---- 2. sandboxed exec --------------------------------------------
        namespace = self._sandboxed_exec(compiled)

        # ---- 3. class discovery -------------------------------------------
        cls = self._discover_class(namespace, class_name)

        # ---- 4/5. timed instantiation + smoke test ------------------------
        module, build_seconds = self._instantiate_and_smoke_test(cls)

        # ---- 6. transactional registration --------------------------------
        with self._register_lock:
            # Move the new expert onto the gate's device so it is immediately
            # usable in the live (possibly CUDA) network.
            module = module.to(self.gate.w_gate.device)

            # Optimizer group FIRST: it touches nothing on the module, so a
            # failure here needs no rollback at all. A fresh param group is the
            # clean way to inject brand-new parameters into an optimizer.
            new_params = list(module.parameters()) if optimizer is not None else []
            added_group = False
            if new_params:
                try:
                    optimizer.add_param_group({"params": new_params})
                except Exception as exc:
                    raise ExpertValidationError(
                        f"failed to add expert parameters to optimizer: {exc!r}"
                    ) from exc
                added_group = True

            # Append the expert BEFORE growing the gate. Ordering matters: at
            # no point does the gate route to a column with no backing expert.
            # (Extra experts the gate doesn't yet know about are simply never
            # selected, so a reader mid-registration stays correct.)
            index = len(self.experts)
            self.experts.append(module)
            try:
                self.gate.expand(1, optimizer=optimizer)
            except Exception as exc:
                # expand() is transactional (it validates the optimizer remap
                # before mutating anything), so undoing our two reversible
                # steps restores the exact pre-registration state: pop the
                # expert, drop the param group we appended (last position,
                # guaranteed by _register_lock), and clear any stray state.
                del self.experts[index]
                if added_group:
                    optimizer.param_groups.pop()
                    for p in new_params:
                        optimizer.state.pop(p, None)
                raise ExpertValidationError(
                    f"failed to register expert into gate/optimizer: {exc!r}"
                ) from exc

        return ExpertRegistration(
            index=index,
            class_name=cls.__name__,
            source_sha256=source_sha,
            build_seconds=build_seconds,
        )

    # -- pipeline stages ------------------------------------------------------
    def _static_check(self, source: str) -> Any:
        """Compile and statically screen the source. Returns the code object."""
        for bad in _FORBIDDEN_SUBSTRINGS:
            if bad in source:
                raise ExpertValidationError(f"source contains forbidden pattern {bad!r}")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ExpertValidationError(f"syntax error in source: {exc}") from exc

        # Reject imports entirely: the sandbox pre-injects everything an expert
        # legitimately needs (torch, nn, F, math). Any import is unnecessary or
        # an escape attempt. (Dangerous dunder *attribute* access -- __globals__,
        # __subclasses__, etc. -- is handled by the _FORBIDDEN_SUBSTRINGS scan
        # above; we do NOT blanket-block dunders here because legitimate code
        # needs super().__init__().)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ExpertValidationError(
                    "imports are not allowed in generated expert source; "
                    "torch/nn/F/math are pre-injected"
                )

        try:
            return compile(tree, filename=_GENERATED_FILENAME, mode="exec")
        except (SyntaxError, ValueError) as exc:
            raise ExpertValidationError(f"failed to compile source: {exc}") from exc

    def _sandboxed_exec(self, compiled: Any) -> dict[str, Any]:
        """Exec the compiled source in a restricted namespace, return it."""
        namespace: dict[str, Any] = {
            "__builtins__": dict(_SAFE_BUILTINS),  # copy: exec must not mutate ours
            # Classes defined here take their __module__ from this __name__, which
            # is how _discover_class identifies genuinely-generated classes.
            "__name__": _GENERATED_FILENAME,
            "torch": torch,
            "nn": nn,
            "F": F,
            "math": math,
            "Tensor": Tensor,
        }
        try:
            exec(compiled, namespace)  # noqa: S102 - intentional, sandboxed
        except Exception as exc:
            raise ExpertValidationError(
                f"error while executing expert source: {exc!r}"
            ) from exc
        return namespace

    def _discover_class(
        self, namespace: dict[str, Any], class_name: Optional[str]
    ) -> type[nn.Module]:
        """Find the nn.Module subclass to instantiate."""
        if class_name is not None:
            obj = namespace.get(class_name)
            if obj is None:
                raise ExpertValidationError(
                    f"class_name {class_name!r} not defined by source"
                )
            if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
                raise ExpertValidationError(f"{class_name!r} is not an nn.Module subclass")
            return obj

        # Only consider classes DEFINED here (identified by the filename we
        # stamped at compile time), so we don't pick up injected references such
        # as nn.Module itself.
        candidates = [
            obj
            for obj in namespace.values()
            if isinstance(obj, type)
            and issubclass(obj, nn.Module)
            and obj is not nn.Module
            and getattr(obj, "__module__", None) == _GENERATED_FILENAME
        ]
        if not candidates:
            # Fallback: any module subclass bound to a name that isn't one of our
            # injected references.
            candidates = [
                obj
                for name, obj in namespace.items()
                if isinstance(obj, type)
                and issubclass(obj, nn.Module)
                and obj is not nn.Module
                and name not in ("nn", "torch", "F", "Tensor")
            ]
        if len(candidates) == 0:
            raise ExpertValidationError("source defines no nn.Module subclass")
        if len(candidates) > 1:
            names = ", ".join(sorted({c.__name__ for c in candidates}))
            raise ExpertValidationError(
                f"source defines multiple nn.Module subclasses ({names}); "
                f"pass class_name= to disambiguate"
            )
        return candidates[0]

    def _instantiate_and_smoke_test(
        self, cls: type[nn.Module]
    ) -> tuple[nn.Module, float]:
        """Instantiate and smoke-test ``cls`` under a wall-clock timeout.

        Runs in a daemon worker thread so a hung constructor/forward cannot block
        the caller forever. On timeout the thread is abandoned (Python cannot
        kill it) and we raise -- see the class-level security warning.
        """
        result: dict[str, Any] = {}

        def _work() -> None:
            try:
                t0 = time.perf_counter()
                module = cls()  # zero-arg constructor is part of the contract
                module.eval()  # smoke-test deterministically (no dropout/noise)
                self._validate_interface(module)
                result["module"] = module
                result["seconds"] = time.perf_counter() - t0
            except BaseException as exc:  # capture everything for the caller
                result["error"] = exc

        worker = threading.Thread(target=_work, daemon=True, name="expert-smoke-test")
        worker.start()
        worker.join(timeout=self.timeout_s)
        if worker.is_alive():
            raise ExpertValidationError(
                f"expert instantiation/smoke-test exceeded {self.timeout_s}s "
                f"timeout (worker thread abandoned)"
            )
        if "error" in result:
            exc = result["error"]
            raise ExpertValidationError(
                f"expert failed instantiation/smoke-test: {exc!r}"
            ) from exc
        return result["module"], result["seconds"]

    def _validate_interface(self, module: nn.Module) -> None:
        """Enforce the fixed interface contract on an instantiated expert."""
        if not callable(getattr(module, "forward", None)):
            raise ExpertValidationError("expert has no callable forward()")

        x = torch.randn(self.smoke_batch, self.expert_input_dim)
        with torch.no_grad():
            y = module(x)

        if not torch.is_tensor(y):
            raise ExpertValidationError(
                f"forward() must return a Tensor, got {type(y).__name__}"
            )
        expected = (self.smoke_batch, self.expert_output_dim)
        if tuple(y.shape) != expected:
            raise ExpertValidationError(
                f"forward() output shape {tuple(y.shape)} != expected {expected}"
            )
        if not torch.is_floating_point(y):
            raise ExpertValidationError(
                f"forward() output must be floating point, got dtype {y.dtype}"
            )
        if not torch.isfinite(y).all():
            raise ExpertValidationError("forward() output contains NaN or Inf")

        # Mixed-precision smoke test: the swarm trains under autocast, so a new
        # expert must at least not crash there. Only meaningful with CUDA.
        if torch.cuda.is_available():
            module_c = module.cuda()
            xc = x.cuda()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    yc = module_c(xc)
                if not torch.isfinite(yc.float()).all():
                    raise ExpertValidationError(
                        "forward() produced non-finite output under autocast"
                    )
            finally:
                module.cpu()  # registration moves it back to the gate's device
