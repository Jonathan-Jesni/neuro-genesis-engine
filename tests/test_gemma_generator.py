"""Tests for the Gemma-backed expert source generator.

Kept SEPARATE from the main suite: the live test needs a GPU, an HF token,
and a ~5 GB model download. The prompt-building and code-extraction tests run
anywhere with no model.

Run offline tests:   python tests/test_gemma_generator.py
Run the live test:   NGEN_RUN_GEMMA_LIVE=1 python tests/test_gemma_generator.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.moe.dynamic_gating import DynamicNoisyTopKGate, ExpertFoundry  # noqa: E402
from core.orchestrator import (  # noqa: E402
    FailureContext,
    GenerationAttempt,
    build_gemma_prompt,
    extract_python_code,
)


class _SkipTest(Exception):
    """Raised to skip a test; the bare runner reports SKIP, pytest skips."""


def _skip(reason: str) -> None:
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise _SkipTest(reason) from None


def _make_ctx(step=40, in_dim=16, out_dim=16) -> FailureContext:
    return FailureContext(
        step=step, loss=250.0, rolling_mean=3.4, rolling_std=0.25, z_score=980.0,
        num_experts=4, input_dim=in_dim, output_dim=out_dim,
        batch=torch.randn(8, in_dim),
    )


# ---------------------------------------------------------------------------
# 1. Prompt building (no model needed)
# ---------------------------------------------------------------------------
def test_prompt_first_attempt_well_formed():
    ctx = _make_ctx(in_dim=24, out_dim=12)
    prompt = build_gemma_prompt(ctx)
    # The full foundry contract must be stated: dims, single class, zero-arg
    # init, the no-imports rule, and the fenced-reply instruction.
    assert "[batch, 24]" in prompt and "[batch, 12]" in prompt
    assert "EXACTLY ONE" in prompt
    assert "__init__(self)" in prompt
    assert "NO import statements" in prompt
    assert "```python" in prompt
    # Spike stats give the model design context.
    assert "250.0" in prompt
    # A first-attempt prompt must not carry retry language.
    assert "REJECTED" not in prompt


def test_prompt_retry_embeds_source_and_reason():
    ctx = _make_ctx()
    failed_source = "class Bad(nn.Module):\n    def forward(self, x):\n        return x[:, :7]"
    reason = "forward() output shape (4, 7) != expected (4, 16)"
    prompt = build_gemma_prompt(
        ctx, GenerationAttempt(attempt_number=1, source=failed_source,
                               rejection_reason=reason)
    )
    # The prior source appears VERBATIM and the reason EXACTLY -- this is the
    # feedback channel the whole retry loop exists to carry.
    assert failed_source in prompt
    assert reason in prompt
    assert "Fix that specific issue" in prompt
    # And the full contract is restated so the fix has the spec in-context.
    assert "NO import statements" in prompt
    assert "[batch, 16]" in prompt


# ---------------------------------------------------------------------------
# 2. Code extraction from realistic Gemma-style responses (no model needed)
# ---------------------------------------------------------------------------
def test_extract_code_from_realistic_responses():
    code = (
        "class Expert(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(16, 16)\n"
        "    def forward(self, x):\n"
        "        return self.lin(x)"
    )

    # (a) The common case: prose, then a ```python fence, then more prose.
    fenced = (
        "Sure! Here's a robust expert module for your use case:\n\n"
        f"```python\n{code}\n```\n\n"
        "This module uses a linear layer. Let me know if you need changes!"
    )
    assert extract_python_code(fenced) == code

    # (b) Generic fence without the language tag.
    generic = f"Here you go:\n```\n{code}\n```"
    assert extract_python_code(generic) == code

    # (c) No fences at all: leading prose, then bare code to the end.
    unfenced = (
        "Here is the class you asked for.\n"
        "It satisfies all five requirements.\n"
        f"{code}"
    )
    assert extract_python_code(unfenced) == code

    # (d) Prose-only garbage: extraction must not crash; it returns the text
    # and the foundry rejects it downstream (that is the designed path).
    garbage = "I cannot write code that violates my guidelines."
    assert extract_python_code(garbage) == garbage

    # (e) Only the FIRST fenced block is taken (models sometimes add usage
    # examples in a second block, which would break the one-class contract).
    two_blocks = (
        f"```python\n{code}\n```\n\nUsage:\n```python\nexpert = Expert()\n```"
    )
    assert extract_python_code(two_blocks) == code


# ---------------------------------------------------------------------------
# 3. Live end-to-end (OPT-IN: real model download + GPU generation)
# ---------------------------------------------------------------------------
def test_gemma_live_end_to_end():
    if os.environ.get("NGEN_RUN_GEMMA_LIVE") != "1":
        _skip("live Gemma test is opt-in: set NGEN_RUN_GEMMA_LIVE=1")
    # NGEN_GEMMA_DEVICE overrides the device (e.g. "cpu" for the Docker
    # build-time gate: `docker build` never has GPU access on any host, so
    # the in-image proof of the live generator MUST run on CPU). When unset,
    # the original behavior holds: GPU required, else skip.
    device = os.environ.get("NGEN_GEMMA_DEVICE")
    if device is None:
        if not torch.cuda.is_available():
            _skip("live Gemma test needs a GPU (or set NGEN_GEMMA_DEVICE=cpu)")
        device = "cuda"
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    offline = (os.environ.get("HF_HUB_OFFLINE") == "1"
               or os.environ.get("TRANSFORMERS_OFFLINE") == "1")
    if not os.environ.get("HF_TOKEN") and not offline:
        # Offline mode loads from the local/baked cache -- no token needed.
        _skip("live Gemma test needs HF_TOKEN (env or .env) unless offline")
    try:
        import transformers  # noqa: F401
    except ImportError:
        _skip("live Gemma test needs transformers (pip install transformers accelerate)")

    from core.orchestrator import GemmaExpertGenerator
    from core.moe.dynamic_gating import ExpertValidationError

    if device == "cpu":
        # CPU generation of a 2B model is minutes-scale: shrink the token
        # budget (a valid expert class fits comfortably in 250 tokens) and
        # widen the wall-clock budget. Still a full end-to-end proof:
        # load-from-cache -> generate -> foundry validation -> registration.
        gen = GemmaExpertGenerator(
            device_map="cpu", max_new_tokens=250, generate_timeout_s=900.0
        )
    else:
        gen = GemmaExpertGenerator(device_map=device)
    gate = DynamicNoisyTopKGate(16, 4, k=2)
    experts = torch.nn.ModuleList(torch.nn.Linear(16, 16) for _ in range(4))
    foundry = ExpertFoundry(gate, experts, expert_input_dim=16, expert_output_dim=16)
    ctx = _make_ctx()

    # Allow the documented retry path: up to 3 attempts with real feedback.
    prior = None
    reg = None
    for attempt in range(1, 4):
        source = gen(ctx, prior)
        print(f"--- attempt {attempt} source ---\n{source}\n---")
        try:
            reg = foundry.register_expert_from_source(source)
            break
        except ExpertValidationError as exc:
            print(f"attempt {attempt} rejected: {exc}")
            prior = GenerationAttempt(attempt_number=attempt, source=source,
                                      rejection_reason=str(exc))
    assert reg is not None, "Gemma failed to produce a valid expert in 3 attempts"
    assert gate.num_experts == len(experts) == 5
    y = experts[4](torch.randn(3, 16))
    assert y.shape == (3, 16) and torch.isfinite(y).all()
    print(f"live registration OK: {reg.class_name} (attempt with index {reg.index})")


# ---------------------------------------------------------------------------
# Minimal runner (SKIP-aware) for environments without pytest.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except _SkipTest as exc:
            print(f"SKIP  {fn.__name__}: {exc}")
        except BaseException as exc:  # noqa: BLE001
            # pytest.skip raises its own exception type when pytest is present.
            if type(exc).__name__ == "Skipped":
                print(f"SKIP  {fn.__name__}: {exc}")
            else:
                failed += 1
                print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{'FAILED' if failed else 'OK'} ({failed} failures)")
    sys.exit(1 if failed else 0)
