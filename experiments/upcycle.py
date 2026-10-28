"""Start the MoE model from pretrained GPT-2 small ("sparse upcycling").

Each transformer block's dense MLP is copied into EVERY expert of that block's
MoE layer; attention, layer norms and embeddings are loaded as-is; the routers
keep their small random init. With identical experts the router's choice cannot
change the output (top-k weights sum to 1), so before any training the upcycled
model computes exactly GPT-2's function — `--check` verifies this against the
Hugging Face implementation.

Two steps, because the cloud box cannot download from the Hub:

    # on a machine with internet: convert HF weights to a compact dense file
    python -m experiments.upcycle convert --out pretrained/gpt2_dense_fp16.pt
    python -m experiments.upcycle check   --dense pretrained/gpt2_dense_fp16.pt

    # anywhere: train from it (train.py config key)
    --set init_from=pretrained/gpt2_dense_fp16.pt  (+ the GPT-2 model shape)

The dense file uses this repo's parameter names with a single, un-replicated
MLP per block (``blocks.<i>.mlp.fc1/fc2``), stored as float16 (~250 MB).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.model import ModelConfig, MoEGPT

GPT2_SHAPE = {"d_model": 768, "n_layer": 12, "n_head": 12, "d_ff": 3072,
              "vocab_size": 50257, "gelu_approx": "tanh"}


def convert_hf_gpt2(safetensors_path: Path, out_path: Path,
                    dtype: torch.dtype = torch.float16) -> dict:
    """HF GPT-2 checkpoint -> dense state dict in this repo's naming.

    HF stores linear layers as Conv1D with weight [in, out]; nn.Linear wants
    [out, in], so every matrix is transposed. The causal-mask buffers
    (``h.*.attn.bias``) are not weights and are dropped.
    """
    from safetensors.torch import load_file

    hf = load_file(str(safetensors_path))
    n_layer = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("h."))
    out: dict[str, torch.Tensor] = {
        "tok.weight": hf["wte.weight"],
        "pos.weight": hf["wpe.weight"],
        "ln_f.weight": hf["ln_f.weight"],
        "ln_f.bias": hf["ln_f.bias"],
    }
    for i in range(n_layer):
        h, b = f"h.{i}.", f"blocks.{i}."
        out[b + "ln1.weight"] = hf[h + "ln_1.weight"]
        out[b + "ln1.bias"] = hf[h + "ln_1.bias"]
        out[b + "qkv.weight"] = hf[h + "attn.c_attn.weight"].t()
        out[b + "qkv.bias"] = hf[h + "attn.c_attn.bias"]
        out[b + "proj.weight"] = hf[h + "attn.c_proj.weight"].t()
        out[b + "proj.bias"] = hf[h + "attn.c_proj.bias"]
        out[b + "ln2.weight"] = hf[h + "ln_2.weight"]
        out[b + "ln2.bias"] = hf[h + "ln_2.bias"]
        out[b + "mlp.fc1.weight"] = hf[h + "mlp.c_fc.weight"].t()
        out[b + "mlp.fc1.bias"] = hf[h + "mlp.c_fc.bias"]
        out[b + "mlp.fc2.weight"] = hf[h + "mlp.c_proj.weight"].t()
        out[b + "mlp.fc2.bias"] = hf[h + "mlp.c_proj.bias"]
    out = {k: v.contiguous().to(dtype) for k, v in out.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": out, "shape": dict(GPT2_SHAPE, n_layer=n_layer)}, out_path)
    return out


def load_dense_into_moe(model: MoEGPT, dense_path: Path) -> None:
    """Load a dense checkpoint into ``model``, replicating each block's MLP into
    every expert. Fails loudly on any shape mismatch or uncovered parameter, so
    a wrong ``model`` shape can never silently train from a partial load."""
    blob = torch.load(dense_path, map_location="cpu", weights_only=True)
    dense: dict[str, torch.Tensor] = blob["state"]
    params = dict(model.named_parameters())
    covered: set[str] = set()

    def put(name: str, value: torch.Tensor) -> None:
        p = params[name]
        if value.shape != p.shape:
            raise ValueError(f"{name}: checkpoint {tuple(value.shape)} vs model {tuple(p.shape)}")
        with torch.no_grad():
            p.copy_(value.to(p.dtype))
        covered.add(name)

    for key, value in dense.items():
        if key == "pos.weight":   # GPT-2 has 1024 positions; the model may use fewer
            n = params["pos.weight"].shape[0]
            if n > value.shape[0]:
                raise ValueError(f"model seq_len {n} > checkpoint positions {value.shape[0]}")
            put(key, value[:n])
        elif ".mlp." in key:      # blocks.i.mlp.fc1.weight -> every expert of block i
            block, rest = key.split(".mlp.")
            layer = model.blocks[int(block.split(".")[1])].moe
            for e in range(layer.num_experts):
                put(f"{block}.moe.experts.{e}.{rest}", value)
        else:
            put(key, value)
    missing = [n for n in params if n not in covered and ".moe.gate." not in n]
    if missing:
        raise ValueError(f"parameters not covered by {dense_path}: {missing[:5]}")


def _gpt2_moe(n_experts: int = 4, seq_len: int = 256) -> MoEGPT:
    return MoEGPT(ModelConfig(seq_len=seq_len, n_experts=n_experts, top_k=2, **GPT2_SHAPE))


def check(dense_path: Path, data_dir: Path = Path("data"), n_batches: int = 2) -> None:
    """Upcycled MoE (eval mode) vs Hugging Face GPT-2 on held-out text."""
    from transformers import GPT2LMHeadModel

    from experiments.data import HeldOut

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(dev).eval()
    moe = _gpt2_moe()
    load_dense_into_moe(moe, dense_path)
    moe = moe.to(dev).eval()
    held = HeldOut(data_dir, ["stories", "code", "math"], n_batches, 4, 256)
    with torch.no_grad():
        for dom, batches in held.batches.items():
            for x, y in batches:
                x, y = x.to(dev), y.to(dev)
                logits = hf(x).logits.float()
                l_hf = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                _, l_moe, _ = moe(x, y)
                print(f"{dom:<8} HF GPT-2 loss {l_hf.item():.4f}   upcycled MoE {l_moe.item():.4f}   "
                      f"diff {abs(l_hf.item() - l_moe.item()):.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="HF GPT-2 -> dense fp16 file")
    c.add_argument("--out", type=Path, default=Path("pretrained/gpt2_dense_fp16.pt"))
    k = sub.add_parser("check", help="compare upcycled MoE with HF GPT-2")
    k.add_argument("--dense", type=Path, default=Path("pretrained/gpt2_dense_fp16.pt"))
    args = ap.parse_args()
    if args.cmd == "convert":
        from huggingface_hub import hf_hub_download
        src = Path(hf_hub_download("openai-community/gpt2", "model.safetensors"))
        state = convert_hf_gpt2(src, args.out)
        n = sum(v.numel() for v in state.values())
        print(f"wrote {args.out}: {len(state)} tensors, {n/1e6:.1f}M params, "
              f"{args.out.stat().st_size/1e6:.0f} MB")
    else:
        check(args.dense)


if __name__ == "__main__":
    main()
