"""Placeholder demo entrypoint for the Neuro-Genesis Engine.

This is a stand-in until the real Day-5 demo lands. It doubles as a runtime
ROCm/GPU smoke test: it reports whether torch sees an AMD GPU via HIP, then
runs one forward pass of the dynamic MoE gate on whatever device is available.

Swapping in the real demo is a one-line change in the Dockerfile CMD (or just
overwrite this file) -- nothing else in the container needs to move.
"""

from __future__ import annotations

import torch

from core.moe.dynamic_gating import DynamicNoisyTopKGate


def main() -> None:
    # torch.version.hip is populated on ROCm builds (None on CUDA/CPU builds),
    # so it is the honest signal that we are on the ROCm stack we intended.
    hip = getattr(torch.version, "hip", None)
    # Under ROCm, torch reuses the CUDA API surface: torch.cuda.is_available()
    # returns True when a HIP-capable AMD GPU is visible. This is expected and
    # not a sign the image is secretly CUDA.
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    print("=== Neuro-Genesis Engine :: environment ===")
    print(f"torch version   : {torch.__version__}")
    print(f"HIP (ROCm) build: {hip!r}")
    print(f"GPU visible     : {gpu}")
    if gpu:
        print(f"device name     : {torch.cuda.get_device_name(0)}")
    print(f"running on      : {device}")

    # Minimal end-to-end sanity: build a gate, route a batch, print the shape.
    gate = DynamicNoisyTopKGate(in_features=16, num_experts=4, k=2).to(device).eval()
    x = torch.randn(8, 16, device=device)
    out = gate(x)
    print("=== forward pass ===")
    print(f"gates shape     : {tuple(out.gates.shape)}")
    print(f"aux_loss        : {out.aux_loss.item():.6f}")
    print("OK")


if __name__ == "__main__":
    main()
