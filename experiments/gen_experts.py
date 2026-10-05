"""Pre-generate LLM-written expert designs for the generated-vs-random comparison.

Gemma-2-2b-it writes expert modules OFFLINE (the cloud box cannot download the
model), each validated through the real foundry pipeline — static AST screen,
sandboxed exec, smoke test, interface check — with the exact rejection reason
fed back on failure (up to --retries attempts). Accepted sources are saved to
JSON; training then registers them at growth events instead of the
random-initialised template MLP (``expert_source: generated`` in train.py).

    python -m experiments.gen_experts --dim 256 --n 8 --out experiments/generated_experts/d256.json

Needs transformers + the cached google/gemma-2-2b-it weights and ~6 GB GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from core.moe.dynamic_gating import DynamicNoisyTopKGate, ExpertFoundry, ExpertValidationError
from core.orchestrator import FailureContext, GemmaExpertGenerator, GenerationAttempt


def validate(source: str, dim: int) -> tuple[str, int]:
    """Run the foundry on a throwaway gate. Returns (class_name, n_params)."""
    gate = DynamicNoisyTopKGate(dim, 1, k=1)
    experts = nn.ModuleList([nn.Linear(dim, dim)])
    foundry = ExpertFoundry(gate, experts, dim, dim)
    reg = foundry.register_expert_from_source(source)
    n_params = sum(p.numel() for p in experts[reg.index].parameters())
    return reg.class_name, n_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", type=int, default=256, help="expert input = output dim (d_model)")
    ap.add_argument("--n", type=int, default=8, help="number of accepted designs to collect")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tries", type=int, default=40, help="give up after this many attempts")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="higher than the generator default (0.2) to get distinct designs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("experiments/generated_experts/d256.json"))
    args = ap.parse_args()

    t0 = time.time()
    gen = GemmaExpertGenerator(temperature=args.temperature, device_map=args.device)
    print(f"loaded {gen.model_name} in {time.time() - t0:.0f}s", flush=True)

    designs: list[dict] = []
    tries = 0
    while len(designs) < args.n and tries < args.max_tries:
        # A realistic context: the stories->code boundary from the real runs
        # (loss jumps 2.49 -> 9.6, z ~ 100). num_experts varies so prompts differ.
        ctx = FailureContext(step=1220, loss=9.62, rolling_mean=2.49, rolling_std=0.07,
                             z_score=104.6, num_experts=4 + len(designs) % 2,
                             input_dim=args.dim, output_dim=args.dim,
                             batch=torch.zeros(1, args.dim))
        prior = None
        for attempt in range(1, args.retries + 1):
            tries += 1
            torch.manual_seed(1000 + tries)
            src = gen(ctx, prior)
            try:
                cls, n_params = validate(src, args.dim)
            except ExpertValidationError as exc:
                reason = str(exc)
                print(f"  try {tries}: rejected ({reason[:90]})", flush=True)
                prior = GenerationAttempt(attempt_number=attempt, source=src,
                                          rejection_reason=reason)
                continue
            if any(d["source"].strip() == src.strip() for d in designs):
                print(f"  try {tries}: duplicate of an earlier design, skipped", flush=True)
                break
            designs.append({"source": src, "class_name": cls, "params": n_params,
                            "attempts": attempt})
            print(f"[{len(designs)}/{args.n}] accepted {cls}: {n_params:,} params "
                  f"(attempt {attempt})", flush=True)
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": gen.model_name, "dim": args.dim, "temperature": args.temperature,
        "total_tries": tries, "designs": designs,
    }, indent=2))
    print(f"saved {len(designs)} designs ({tries} tries, {time.time() - t0:.0f}s) -> {args.out}")


if __name__ == "__main__":
    main()
