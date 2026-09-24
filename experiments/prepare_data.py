"""Download, tokenize, and cache the three-domain continual-pretraining stream.

Each domain is streamed from the HF Hub (never fully downloaded), shuffled with
a fixed seed, tokenized with GPT-2 BPE (tiktoken), and written as flat uint16
token arrays with an EOT token between documents:

    data/<domain>_train.npy   (--train-tokens tokens)
    data/<domain>_val.npy     (--val-tokens tokens, disjoint documents)
    data/meta.json            (sources, counts, settings — for the report)

Train/val are split by DOCUMENT (every --val-every-th document goes to val until
it is full), so held-out loss never sees a document the model trained on.

Usage:
    python -m experiments.prepare_data                     # all domains, default caps
    python -m experiments.prepare_data --domains stories --train-tokens 1000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tiktoken

# name -> (HF repo, text field). Order here is the order the stream presents them.
DOMAINS: dict[str, tuple[str, str]] = {
    "stories": ("roneneldan/TinyStories", "text"),
    "code": ("codeparrot/codeparrot-clean", "content"),
    "math": ("open-web-math/open-web-math", "text"),
}

EOT = 50256  # GPT-2 <|endoftext|> — fits uint16
BATCH_DOCS = 512
_ENC = None


def _encoder():
    """Load the tokenizer lazily: it downloads its vocab on first use, and a
    machine with cached data (e.g. a proxied cloud box) must not need that."""
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
        assert _ENC.eot_token == EOT
    return _ENC


def _load_hf_token() -> None:
    """Pick up HF_TOKEN from .env for higher Hub rate limits, without printing it."""
    if os.environ.get("HF_TOKEN"):
        return
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip('"').strip("'")
            return


def prepare_domain(
    name: str,
    out_dir: Path,
    train_tokens: int,
    val_tokens: int,
    val_every: int,
    seed: int,
    shuffle_buffer: int,
) -> dict:
    from datasets import load_dataset  # heavy import; keep module importable without it

    repo, field = DOMAINS[name]
    ds = load_dataset(repo, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    train = np.empty(train_tokens, dtype=np.uint16)
    val = np.empty(val_tokens, dtype=np.uint16)
    n_train = n_val = 0
    docs_train = docs_val = 0
    doc_idx = 0
    t0 = time.time()
    next_report = 0

    def flush(texts: list[str], start_idx: int) -> bool:
        nonlocal n_train, n_val, docs_train, docs_val
        for i, ids in enumerate(_encoder().encode_ordinary_batch(texts)):
            ids.append(EOT)
            to_val = (start_idx + i) % val_every == 0 and n_val < val_tokens
            if to_val:
                k = min(len(ids), val_tokens - n_val)
                val[n_val:n_val + k] = ids[:k]
                n_val += k
                docs_val += 1
            elif n_train < train_tokens:
                k = min(len(ids), train_tokens - n_train)
                train[n_train:n_train + k] = ids[:k]
                n_train += k
                docs_train += 1
            if n_train >= train_tokens and n_val >= val_tokens:
                return True
        return False

    batch: list[str] = []
    batch_start = 0
    done = False
    for ex in ds:
        text = ex[field]
        if not text:
            continue
        if not batch:
            batch_start = doc_idx
        batch.append(text)
        doc_idx += 1
        if len(batch) >= BATCH_DOCS:
            done = flush(batch, batch_start)
            batch = []
            if n_train >= next_report:
                rate = n_train / max(time.time() - t0, 1e-9)
                print(f"[{name}] train {n_train/1e6:6.1f}M / {train_tokens/1e6:.0f}M"
                      f"  val {n_val/1e6:.2f}M  ({rate/1e3:.0f}k tok/s)", flush=True)
                next_report += 2_000_000
            if done:
                break
    if batch and not done:
        flush(batch, batch_start)

    if n_train < train_tokens or n_val < val_tokens:
        print(f"[{name}] WARNING: stream exhausted early "
              f"(train {n_train}, val {n_val})", flush=True)

    np.save(out_dir / f"{name}_train.npy", train[:n_train])
    np.save(out_dir / f"{name}_val.npy", val[:n_val])
    elapsed = time.time() - t0
    print(f"[{name}] done: {n_train:,} train / {n_val:,} val tokens, "
          f"{docs_train:,} / {docs_val:,} docs, {elapsed/60:.1f} min", flush=True)
    return {
        "source": repo,
        "field": field,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "train_docs": docs_train,
        "val_docs": docs_val,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--train-tokens", type=int, default=30_000_000)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    p.add_argument("--val-every", type=int, default=50,
                   help="every Nth document goes to val until val is full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--force", action="store_true", help="re-create existing domains")
    args = p.parse_args()

    _load_hf_token()
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["tokenizer"] = "gpt2 (tiktoken)"
    meta["eot_token"] = EOT
    meta.setdefault("domains", {})

    for name in args.domains:
        have = all((args.out / f"{name}_{s}.npy").exists() for s in ("train", "val"))
        if have and not args.force:
            print(f"[{name}] already cached, skipping (use --force to rebuild)")
            continue
        meta["domains"][name] = prepare_domain(
            name, args.out, args.train_tokens, args.val_tokens,
            args.val_every, args.seed, args.shuffle_buffer,
        )
        meta["domains"][name].update(seed=args.seed, val_every=args.val_every)
        meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
