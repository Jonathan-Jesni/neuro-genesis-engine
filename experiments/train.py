"""Continual-pretraining run: one arm, one seed, sequential domain stream.

Arms (``arm:`` in the config):
    static   — fixed expert count (arms A / B; B = static with more experts)
    oracle   — grow ``oracle_grow`` experts per layer at each TRUE domain
               boundary (arm C; the only arm allowed to read the domain id)
    signal   — grow when the loss-spike detector fires (arm D v1), with a
               cooldown and a detector reset after every growth event

Outputs under ``results/<run_name>/``:
    config.json   resolved config
    steps.jsonl   per-``log_every`` training stats (loss, lr, experts, routing)
    evals.jsonl   held-out loss per domain every ``eval_every`` steps + phase ends
    events.jsonl  growth events (step, reason, experts after)
    summary.json  phase-end losses, final losses, forgetting, expert count, timing

Usage:
    python -m experiments.train configs/toy_A.yaml --seed 0
    python -m experiments.train configs/toy_A.yaml --seed 0 --set model.n_experts=8 --name toy_B
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from core.orchestrator import RollingSpikeDetector
from experiments.data import DomainStream, HeldOut
from experiments.model import ModelConfig, MoEGPT, count_params

DEFAULTS: dict[str, Any] = {
    "name": None,                     # defaults to config file stem
    "arm": "static",
    "data_dir": "data",
    "results_dir": "results",
    "domains": ["stories", "code", "math"],
    "tokens_per_phase": 10_000_000,
    "batch_size": 32,
    "model": {},                      # overrides for ModelConfig
    "lr": 1e-3,
    "min_lr_frac": 0.1,
    "warmup_steps": 200,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "log_every": 20,
    "eval_every": 250,
    "eval_batches": 8,
    "device": "auto",
    "amp": True,
    # oracle arm
    "oracle_grow": 1,
    # signal arm
    "detector": {"window": 50, "z_threshold": 4.0, "min_history": 20},
    "cooldown_steps": 300,
    "signal_grow": 1,
    "max_experts": 32,
    # protection applied at every growth event (oracle or signal arms):
    #   none     — everything keeps training (plain growth)
    #   experts  — freeze pre-existing experts + their router columns
    #   all      — also freeze every shared weight (embeddings, attention, norms);
    #              only the new experts and their router columns train
    "freeze": "none",
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _apply_override(cfg: dict, expr: str) -> None:
    key, _, raw = expr.partition("=")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    _deep_update(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for o in overrides:
        _apply_override(cfg, o)
    if cfg["name"] is None:
        cfg["name"] = Path(path).stem
    return cfg


def lr_at(step: int, total: int, cfg: dict) -> float:
    base, warm = cfg["lr"], cfg["warmup_steps"]
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    floor = base * cfg["min_lr_frac"]
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model: MoEGPT, heldout: HeldOut, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    out = {}
    for dom, batches in heldout.batches.items():
        tot = 0.0
        for x, y in batches:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                _, loss, _ = model(x.to(device), y.to(device))
            tot += loss.item()
        out[dom] = tot / len(batches)
    model.train()
    return out


class Freezer:
    """Protects already-learned weights after a growth event.

    Whole tensors (old experts, shared weights) get ``requires_grad=False``, so
    AdamW never touches them (no grad => no update, no weight decay). Router
    columns cannot be frozen per column that way, so the old columns of each
    gate are snapshotted and copied back after every optimizer step. The gate
    swaps its Parameter objects on expand(), so we always re-read
    ``gate.w_gate`` / ``gate.w_noise`` rather than caching them.
    """

    def __init__(self, mode: str) -> None:
        if mode not in ("none", "experts", "all"):
            raise ValueError(f"unknown freeze mode {mode!r}")
        self.mode = mode
        self.router_snap: list[tuple[int, Tensor, Tensor]] = []   # per layer

    def after_growth(self, model: MoEGPT, n_new: int) -> None:
        if self.mode == "none":
            return
        self.router_snap = []
        for layer in model.moe_layers:
            n_old = layer.num_experts - n_new
            for e in range(n_old):
                for p in layer.experts[e].parameters():
                    p.requires_grad_(False)
            with torch.no_grad():
                self.router_snap.append((
                    n_old,
                    layer.gate.w_gate[:, :n_old].detach().clone(),
                    layer.gate.w_noise[:, :n_old].detach().clone(),
                ))
        if self.mode == "all":
            for name, p in model.named_parameters():
                if ".moe." not in name:
                    p.requires_grad_(False)

    @torch.no_grad()
    def after_step(self, model: MoEGPT) -> None:
        for layer, (n_old, wg, wn) in zip(model.moe_layers, self.router_snap):
            layer.gate.w_gate[:, :n_old].copy_(wg)
            layer.gate.w_noise[:, :n_old].copy_(wn)


class RunLog:
    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.files = {n: open(run_dir / f"{n}.jsonl", "w", encoding="utf-8")
                      for n in ("steps", "evals", "events")}

    def write(self, stream: str, rec: dict) -> None:
        f = self.files[stream]
        f.write(json.dumps(rec) + "\n")
        f.flush()

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def run(cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    amp = bool(cfg["amp"]) and device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    mcfg = ModelConfig(**cfg["model"])
    model = MoEGPT(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))

    tokens_per_step = cfg["batch_size"] * mcfg.seq_len
    steps_per_phase = max(1, cfg["tokens_per_phase"] // tokens_per_step)
    stream = DomainStream(Path(cfg["data_dir"]), cfg["domains"], steps_per_phase,
                          cfg["batch_size"], mcfg.seq_len, seed)
    heldout = HeldOut(Path(cfg["data_dir"]), cfg["domains"], cfg["eval_batches"],
                      cfg["batch_size"], mcfg.seq_len)
    total = stream.total_steps
    boundaries = set(stream.boundaries())

    run_name = f"{cfg['name']}_s{seed}"
    run_dir = Path(cfg["results_dir"]) / run_name
    log = RunLog(run_dir)
    (run_dir / "config.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2))

    arm = cfg["arm"]
    if arm not in ("static", "oracle", "signal"):
        raise ValueError(f"unknown arm {arm!r}")
    detector = RollingSpikeDetector(**cfg["detector"]) if arm == "signal" else None
    freezer = Freezer(cfg["freeze"])
    last_growth = -10**9

    print(f"[{run_name}] arm={arm} freeze={cfg['freeze']} device={device} params={count_params(model)/1e6:.1f}M "
          f"experts/layer={model.num_experts} steps={total} ({steps_per_phase}/phase)", flush=True)

    phase_end: dict[str, dict[str, float]] = {}
    t0 = time.time()
    loss_acc = aux_acc = 0.0
    n_acc = 0

    for b in stream:
        step = b.step

        # ---- growth decisions: BETWEEN steps, no gate lock held ----------
        grow_reason = None
        if arm == "oracle" and step in boundaries:
            grow_reason, n_grow = "oracle_boundary", cfg["oracle_grow"]
        if grow_reason:
            for _ in range(n_grow):
                model.grow(opt, tag=f"S{step}")
            freezer.after_growth(model, n_grow)
            log.write("events", {"step": step, "reason": grow_reason,
                                 "experts": model.num_experts, "domain": b.domain_id})

        lr = lr_at(step, total, cfg)
        for g in opt.param_groups:          # includes groups added by growth
            g["lr"] = lr

        x, y = b.x.to(device, non_blocking=True), b.y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, loss, aux = model(x, y)
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        freezer.after_step(model)

        lval = loss.item()
        loss_acc += lval
        aux_acc += aux.item()
        n_acc += 1

        # ---- signal arm: detector sees the loss AFTER the step ------------
        if detector is not None:
            spike = detector.update(lval)
            cooled = step - last_growth >= cfg["cooldown_steps"]
            if spike and cooled and model.num_experts < cfg["max_experts"]:
                for _ in range(cfg["signal_grow"]):
                    model.grow(opt, tag=f"S{step}")
                freezer.after_growth(model, cfg["signal_grow"])
                last_growth = step
                # Reset: a sustained shift would otherwise keep every later step
                # a "spike" (spikes never enter the baseline history).
                detector = RollingSpikeDetector(**cfg["detector"])
                log.write("events", {"step": step, "reason": "loss_spike",
                                     "z": spike.z_score, "loss": lval,
                                     "experts": model.num_experts, "domain": b.domain_id})

        if (step + 1) % cfg["log_every"] == 0:
            stats = model.routing_stats()
            log.write("steps", {
                "step": step, "tokens": (step + 1) * tokens_per_step,
                "domain": b.domain_id, "loss": loss_acc / n_acc, "aux": aux_acc / n_acc,
                "lr": lr, "experts": model.num_experts,
                "entropy_norm": sum(s["entropy_norm"] for s in stats) / len(stats),
                "load": [s["load"] for s in stats],
                "elapsed": time.time() - t0,
            })
            loss_acc = aux_acc = 0.0
            n_acc = 0

        is_phase_end = b.phase_step == steps_per_phase - 1
        if (step + 1) % cfg["eval_every"] == 0 or is_phase_end:
            ev = evaluate(model, heldout, device, amp)
            log.write("evals", {"step": step, "domain": b.domain_id,
                                "phase_end": is_phase_end, "experts": model.num_experts,
                                "loss": ev})
            if is_phase_end:
                phase_end[cfg["domains"][b.domain_id]] = ev
            rate = (step + 1) * tokens_per_step / (time.time() - t0)
            print(f"[{run_name}] step {step+1}/{total} dom={cfg['domains'][b.domain_id]} "
                  f"experts={model.num_experts} eval="
                  + " ".join(f"{k}:{v:.3f}" for k, v in ev.items())
                  + f"  {rate/1e3:.0f}k tok/s", flush=True)

    final = phase_end[cfg["domains"][-1]]
    forgetting = {d: final[d] - phase_end[d][d] for d in cfg["domains"][:-1]}
    summary = {
        "run": run_name, "arm": arm, "seed": seed, "freeze": cfg["freeze"],
        "phase_end": phase_end, "final": final,
        "final_avg": sum(final.values()) / len(final),
        "forgetting": forgetting,
        "forgetting_avg": sum(forgetting.values()) / max(1, len(forgetting)),
        "experts_final": model.num_experts,
        "params_final": count_params(model),
        "steps": total, "tokens": total * tokens_per_step,
        "seconds": round(time.time() - t0, 1),
        "device": str(torch.cuda.get_device_name(0)) if device.type == "cuda" else "cpu",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.close()
    print(f"[{run_name}] done: final_avg={summary['final_avg']:.3f} "
          f"forgetting_avg={summary['forgetting_avg']:+.3f} experts={model.num_experts} "
          f"({summary['seconds']/60:.1f} min)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run one continual-pretraining arm.")
    p.add_argument("config", type=Path)
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="config override, dotted keys (repeatable)")
    p.add_argument("--name", default=None, help="override run name prefix")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.name:
        cfg["name"] = args.name
    for s in args.seed:
        run(cfg, s)


if __name__ == "__main__":
    main()
