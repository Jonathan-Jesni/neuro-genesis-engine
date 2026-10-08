"""Tests for the continual-pretraining experiment code (experiments/).

Everything runs offline on CPU against tiny synthetic token data written to a
temp dir — no dataset download, no tokenizer, no GPU — so the suite is fast and
safe for CI. Each domain uses a disjoint token range, which makes domain
shifts unmistakable to the model and mixture proportions countable.

Run with:   python -m pytest tests/test_experiments.py -v
Or plainly: python tests/test_experiments.py
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data import DomainStream, base_domains, parse_phase  # noqa: E402
from experiments.model import ModelConfig, MoEGPT, count_params, expert_source  # noqa: E402
from experiments.train import (  # noqa: E402
    DEFAULTS,
    Freezer,
    ReplayBuffer,
    replay_rows,
    run,
)

VOCAB = 600
D_MODEL = 32
SEQ = 16
BATCH = 8
# Disjoint token ranges per synthetic domain.
RANGES = {"a": (0, 200), "b": (200, 400), "c": (400, 600)}


class _TmpDir(tempfile.TemporaryDirectory):
    """Temp dir that tolerates Windows file locks on cleanup.

    The data loaders memory-map the .npy files, and Windows refuses to delete a
    mapped file; collect garbage first so the maps are released, and ignore any
    straggler (CLAUDE.md: tests must not fail on Windows temp cleanup)."""

    def __init__(self) -> None:
        super().__init__(ignore_cleanup_errors=True)

    def __exit__(self, *exc) -> None:
        gc.collect()
        super().__exit__(*exc)


def _write_data(root: Path, n_tokens: int = 40_000) -> None:
    rng = np.random.default_rng(0)
    for name, (lo, hi) in RANGES.items():
        for split in ("train", "val"):
            np.save(root / f"{name}_{split}.npy",
                    rng.integers(lo, hi, size=n_tokens).astype(np.uint16))


def _tiny_model_cfg() -> dict:
    return {"vocab_size": VOCAB, "seq_len": SEQ, "d_model": D_MODEL, "n_layer": 2,
            "n_head": 2, "d_ff": 64, "n_experts": 4, "top_k": 2}


def _cfg(root: Path, **over) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update({
        "name": "t", "data_dir": str(root), "results_dir": str(root / "res"),
        "domains": ["a", "b", "c"], "tokens_per_phase": 40 * BATCH * SEQ,  # 40 steps/phase
        "batch_size": BATCH, "model": _tiny_model_cfg(), "lr": 3e-3, "warmup_steps": 5,
        "log_every": 5, "eval_every": 10_000, "eval_batches": 2, "device": "cpu", "amp": False,
        "detector": {"window": 10, "z_threshold": 4.0, "min_history": 5},
        "cooldown_steps": 20,
    })
    for k, v in over.items():
        cfg[k] = v
    return cfg


def _events(root: Path, name: str, seed: int = 0) -> list[dict]:
    p = root / "res" / f"{name}_s{seed}" / "events.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _tiny_model() -> MoEGPT:
    torch.manual_seed(0)
    return MoEGPT(ModelConfig(**_tiny_model_cfg()))


def _train_steps(model: MoEGPT, opt, freezer: Freezer, n: int, lo: int, hi: int,
                 counts: torch.Tensor | None = None) -> None:
    for _ in range(n):
        x = torch.randint(lo, hi, (4, SEQ))
        if counts is not None:
            counts += torch.bincount(x.reshape(-1), minlength=VOCAB)
        _, loss, aux = model(x, x)
        opt.zero_grad()
        (loss + aux).backward()
        opt.step()
        freezer.after_step(model)


# ---------------------------------------------------------------------------
# Data stream
# ---------------------------------------------------------------------------
def test_parse_phase_and_base_domains():
    assert parse_phase("stories") == [("stories", 1.0)]
    assert parse_phase("a:3+b:1") == [("a", 0.75), ("b", 0.25)]
    assert base_domains(["a", "a:0.9+b:0.1", "c"]) == ["a", "b", "c"]


def test_pure_stream_is_deterministic_per_seed():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        mk = lambda seed: DomainStream(root, ["a", "b"], 10, BATCH, SEQ, seed)  # noqa: E731
        same = all(torch.equal(x.x, y.x) for x, y in zip(mk(1), mk(1)))
        diff = any(not torch.equal(x.x, y.x) for x, y in zip(mk(1), mk(2)))
        assert same and diff
        doms = [b.domain_id for b in mk(1)]
        assert doms == [0] * 10 + [1] * 10


def test_mixture_phase_proportions():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        stream = DomainStream(root, ["a", "a:0.7+b:0.3"], 200, BATCH, SEQ, 0)
        rows = [row for b in itertools.islice(iter(stream), 200, 400) for row in b.x]
        share_b = np.mean([bool((row >= 200).all()) for row in rows])
        assert abs(share_b - 0.3) < 0.05, share_b
        # Every row comes wholly from ONE domain (no token-level blending).
        assert all(bool((row < 200).all()) or bool((row >= 200).all()) for row in rows)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_replay_buffer_is_a_uniform_reservoir():
    rb = ReplayBuffer(capacity=300, seq_len=4, seed=0)
    for dom in (1, 2, 3):
        for _ in range(100):
            x = torch.full((30, 4), dom)
            rb.add(x, x)
    assert rb.size == 300 and rb.seen == 9000
    shares = [(rb.buf[:, 0] == d).float().mean().item() for d in (1, 2, 3)]
    assert all(abs(s - 1 / 3) < 0.08 for s in shares), shares
    x, y = rb.sample(16)
    assert x.shape == (16, 4) and y.shape == (16, 4) and x.dtype == torch.int64


def test_fractional_replay_matches_budget_and_integer_budget_uses_no_rng():
    rng = np.random.default_rng(0)
    base, rem = divmod(0.01 * 32, 1.0)          # 1% of a 32-row batch
    draws = [replay_rows(int(base), rem, rng) for _ in range(20_000)]
    assert abs(np.mean(draws) - 0.32) < 0.02
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    base, rem = divmod(0.25 * 32, 1.0)          # 25% of 32 = exactly 8
    assert all(replay_rows(int(base), rem, rng_a) == 8 for _ in range(100))
    assert rng_a.random() == rng_b.random()     # no randomness was consumed


# ---------------------------------------------------------------------------
# Model growth
# ---------------------------------------------------------------------------
def test_grow_adds_one_expert_per_layer_with_exact_param_count():
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before, groups = count_params(m), len(opt.param_groups)
    m.grow(opt, "T")
    # Per layer: one template expert (d->d_ff->d) + one router column in w_gate and w_noise.
    per_layer = (D_MODEL * 64 + 64) + (64 * D_MODEL + D_MODEL) + 2 * D_MODEL
    assert m.num_experts == 5
    assert count_params(m) - before == 2 * per_layer
    assert len(opt.param_groups) == groups + 2      # one fresh group per layer
    src = expert_source(D_MODEL, 16, "Small")
    m.grow(opt, "T", source=src)
    assert m.num_experts == 6


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------
def _freeze_check(mode: str) -> dict[str, bool]:
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    fr = Freezer(mode)
    counts = torch.zeros(VOCAB, dtype=torch.long)
    _train_steps(m, opt, fr, 4, 0, 200, counts)
    m.grow(opt, "T")
    fr.after_growth(m, 1, counts, min_count=1)
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    _train_steps(m, opt, fr, 4, 0, VOCAB)
    moved = {n: not torch.equal(p, snap[n]) for n, p in m.named_parameters()}
    g0 = m.blocks[0].moe.gate.w_gate
    seen = counts > 0
    return {
        "old_experts": any(moved[n] for n in moved if ".moe.experts." in n
                           and not n.split(".moe.experts.")[1].startswith("4")),
        "new_expert": moved["blocks.0.moe.experts.4.fc1.weight"],
        "old_router": not torch.equal(g0[:, :4], snap["blocks.0.moe.gate.w_gate"][:, :4]),
        "new_router": not torch.equal(g0[:, 4], snap["blocks.0.moe.gate.w_gate"][:, 4]),
        "attention": moved["blocks.0.qkv.weight"],
        "emb_seen_rows": not torch.equal(m.tok.weight[seen], snap["tok.weight"][seen]),
        "emb_unseen_rows": not torch.equal(m.tok.weight[~seen], snap["tok.weight"][~seen]),
    }


def test_freeze_none_trains_everything():
    moved = _freeze_check("none")
    assert all(moved.values()), moved


def test_freeze_experts_protects_old_experts_and_router_columns_only():
    moved = _freeze_check("experts")
    assert not moved["old_experts"] and not moved["old_router"]
    assert moved["new_expert"] and moved["new_router"] and moved["attention"]


def test_freeze_all_leaves_only_new_expert_and_its_router_column():
    moved = _freeze_check("all")
    assert moved["new_expert"] and moved["new_router"]
    assert not any(moved[k] for k in ("old_experts", "old_router", "attention",
                                      "emb_seen_rows", "emb_unseen_rows")), moved


def test_freeze_all_but_emb_keeps_embeddings_trainable():
    moved = _freeze_check("all_but_emb")
    assert moved["emb_seen_rows"] and moved["emb_unseen_rows"] and moved["new_expert"]
    assert not moved["old_experts"] and not moved["attention"] and not moved["old_router"]


def test_freeze_all_seen_emb_freezes_exactly_the_seen_rows():
    moved = _freeze_check("all_seen_emb")
    assert not moved["emb_seen_rows"] and moved["emb_unseen_rows"]
    assert not moved["old_experts"] and not moved["attention"] and moved["new_expert"]


# ---------------------------------------------------------------------------
# End-to-end runs
# ---------------------------------------------------------------------------
def test_static_run_end_to_end_and_bit_reproducible():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        s1 = run(_cfg(root, name="r1"), seed=0)
        s2 = run(_cfg(root, name="r2"), seed=0)
        assert set(s1["final"]) == {"a", "b", "c"} and set(s1["forgetting"]) == {"a", "b"}
        assert s1["final"] == s2["final"] and s1["forgetting"] == s2["forgetting"]
        assert s1["experts_final"] == 4
        run_dir = root / "res" / "r1_s0"
        assert all((run_dir / f).exists() for f in
                   ("config.json", "steps.jsonl", "evals.jsonl", "events.jsonl", "summary.json"))


def test_oracle_grows_exactly_at_true_boundaries():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        s = run(_cfg(root, name="o", arm="oracle", oracle_grow=2), seed=0)
        assert [e["step"] for e in _events(root, "o")] == [40, 80]
        assert s["experts_final"] == 4 + 2 * 2


def test_signal_detector_fires_after_shift_with_confirmation():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        s = run(_cfg(root, name="sig", arm="signal", confirm_steps=3), seed=0)
        steps = [e["step"] for e in _events(root, "sig")]
        # Each boundary (40, 80) is caught; with 3-step confirmation the earliest
        # possible firing is boundary + 2.
        assert len(steps) == 2, steps
        assert 42 <= steps[0] < 50 and 82 <= steps[1] < 90, steps
        assert s["experts_final"] == 6


def test_signal_detector_stays_silent_without_a_shift():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        run(_cfg(root, name="quiet", arm="signal", confirm_steps=3, domains=["a", "a", "a"]),
            seed=0)
        assert _events(root, "quiet") == []


def test_replay_run_reports_its_budget():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        s = run(_cfg(root, name="rep", replay_frac=0.25), seed=0)
        assert s["replay_frac"] == 0.25 and s["experts_final"] == 4


def test_mixture_run_reports_forgetting_for_pure_phases_only():
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        s = run(_cfg(root, name="mix", domains=["a", "a:0.5+b:0.5"]), seed=0)
        assert set(s["final"]) == {"a", "b"} and list(s["forgetting"]) == ["a"]


def test_generated_and_size_matched_template_experts():
    design = (
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        f"        self.net = nn.Sequential(nn.LayerNorm({D_MODEL}), nn.Linear({D_MODEL}, 8),\n"
        f"                                 nn.GELU(), nn.Linear(8, {D_MODEL}))\n"
        "    def forward(self, x):\n"
        "        return self.net(x)\n"
    )
    design_params = 2 * D_MODEL + (D_MODEL * 8 + 8) + (8 * D_MODEL + D_MODEL)
    router = 2 * D_MODEL
    with _TmpDir() as td:
        root = Path(td)
        _write_data(root)
        gen_file = root / "designs.json"
        gen_file.write_text(json.dumps({"designs": [{"source": design}]}))
        base = count_params(_tiny_model())
        g = run(_cfg(root, name="gen", arm="oracle", expert_source="generated",
                     generated_file=str(gen_file)), seed=0)
        # 2 growth events x 2 layers, each adding the design + a router column.
        assert g["params_final"] - base == 2 * 2 * (design_params + router)
        assert g["expert_source"] == "generated"
        t = run(_cfg(root, name="w8", arm="oracle", grow_d_ff=8), seed=0)
        template_params = (D_MODEL * 8 + 8) + (8 * D_MODEL + D_MODEL)
        assert t["params_final"] - base == 2 * 2 * (template_params + router)


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
