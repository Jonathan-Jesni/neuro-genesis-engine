"""Figures from results/<run>/ logs. Works on finished AND partial (crashed) runs.

    python -m experiments.analyze                      # all figures, results/ -> results/figures/
    python -m experiments.analyze --results path --fig experts

Figures:
    experts      expert count vs step, one line per run, true domain bands, growth markers
    heldout      held-out loss per domain vs step, one panel per domain, mean over seeds per arm
    detection    table of growth events vs true boundaries (latency, false triggers) -> stdout + csv
    tradeoff     stability-plasticity scatter: forgetting vs new-domain loss, one point per arm
    leak         router leak: share of each OLD domain's tokens routed to experts born later
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed arm colours so every figure in the report agrees.
ARM_COLORS = {"toy_A": "#6b7280", "toy_B": "#2563eb", "toy_C": "#d97706", "toy_D": "#059669"}
ARM_LABELS = {"toy_A": "A static", "toy_B": "B static (large)",
              "toy_C": "C oracle growth", "toy_D": "D signal growth"}
DOMAIN_SHADES = ["#f3f4f6", "#e0f2fe", "#fef3c7", "#ede9fe"]
_FALLBACK = ["#7c3aed", "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4b5563", "#0d9488", "#9333ea"]


def arm_color(arm: str) -> str:
    """Fixed colour for the four base arms; stable hash-free fallback for variants."""
    if arm in ARM_COLORS:
        return ARM_COLORS[arm]
    return _FALLBACK[sum(map(ord, arm)) % len(_FALLBACK)]


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class Run:
    def __init__(self, d: Path) -> None:
        self.dir = d
        self.name = d.name
        self.arm, _, seed = d.name.rpartition("_s")
        self.seed = int(seed) if seed.isdigit() else -1
        self.cfg = json.loads((d / "config.json").read_text())
        self.steps = _read_jsonl(d / "steps.jsonl")
        self.evals = _read_jsonl(d / "evals.jsonl")
        self.events = _read_jsonl(d / "events.jsonl")
        s = d / "summary.json"
        self.summary = json.loads(s.read_text()) if s.exists() else None
        seq = self.cfg.get("model", {}).get("seq_len", 256)
        self.steps_per_phase = max(1, self.cfg["tokens_per_phase"] // (self.cfg["batch_size"] * seq))
        self.domains: list[str] = self.cfg["domains"]

    @property
    def boundaries(self) -> list[int]:
        return [self.steps_per_phase * i for i in range(1, len(self.domains))]

    @property
    def complete(self) -> bool:
        return self.summary is not None


def load_runs(root: Path) -> list[Run]:
    return [Run(d) for d in sorted(root.iterdir()) if (d / "config.json").exists()]


def _domain_bands(ax, run: Run, label: bool = True) -> None:
    for i, dom in enumerate(run.domains):
        lo, hi = i * run.steps_per_phase, (i + 1) * run.steps_per_phase
        ax.axvspan(lo, hi, color=DOMAIN_SHADES[i % len(DOMAIN_SHADES)], zorder=0, lw=0)
        if label:
            ax.text((lo + hi) / 2, 1.0, dom, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#374151")


def fig_experts(runs: list[Run], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _domain_bands(ax, runs[0])
    seen = set()
    for r in runs:
        if not r.steps:
            continue
        xs = [s["step"] for s in r.steps]
        ys = [s["experts"] for s in r.steps]
        c = arm_color(r.arm)
        lab = ARM_LABELS.get(r.arm, r.arm) if r.arm not in seen else None
        seen.add(r.arm)
        ax.step(xs, ys, where="post", color=c, lw=1.8, alpha=0.85, label=lab)
        for e in r.events:
            ax.plot(e["step"], e["experts"], marker="v" if e["reason"] == "loss_spike" else "o",
                    color=c, ms=6, zorder=5)
    ax.set_xlabel("training step")
    ax.set_ylabel("experts per layer")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "experts.png", dpi=160)
    plt.close(fig)


def fig_heldout(runs: list[Run], out: Path) -> None:
    domains = runs[0].domains
    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 3.4), sharex=True)
    by_arm: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_arm[r.arm].append(r)
    for ax, dom in zip(axes, domains):
        _domain_bands(ax, runs[0], label=False)
        for arm, rs in sorted(by_arm.items()):
            # Align seeds on shared eval steps; partial runs contribute where they exist.
            curves = defaultdict(list)
            for r in rs:
                for e in r.evals:
                    curves[e["step"]].append(e["loss"][dom])
            if not curves:
                continue
            xs = sorted(curves)
            mean = np.array([np.mean(curves[x]) for x in xs])
            sd = np.array([np.std(curves[x]) for x in xs])
            c = arm_color(arm)
            ax.plot(xs, mean, color=c, lw=1.8, label=f"{ARM_LABELS.get(arm, arm)} (n={len(rs)})")
            if len(rs) > 1:
                ax.fill_between(xs, mean - sd, mean + sd, color=c, alpha=0.15, lw=0)
        ax.set_title(f"held-out loss: {dom}", fontsize=10)
        ax.set_xlabel("training step")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("cross-entropy (nats)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "heldout.png", dpi=160)
    plt.close(fig)


def detection_table(runs: list[Run], out: Path, window: int = 100) -> None:
    """Growth events vs true boundaries. A trigger within `window` steps after a
    boundary counts as detecting it; any other trigger is a false trigger."""
    rows = []
    for r in runs:
        if r.arm != "toy_D" and not any(e["reason"] == "loss_spike" for e in r.events):
            continue
        trig = [e["step"] for e in r.events if e["reason"] == "loss_spike"]
        last = r.steps[-1]["step"] if r.steps else 0
        reached = [b for b in r.boundaries if b <= last]
        used, lat = set(), []
        for b in reached:
            hit = next((t for t in trig if b <= t < b + window and t not in used), None)
            if hit is not None:
                used.add(hit)
                lat.append(hit - b)
        rows.append({
            "run": r.name, "complete": r.complete, "boundaries_reached": len(reached),
            "detected": len(lat), "latencies": lat,
            "false_triggers": len([t for t in trig if t not in used]),
            "triggers": trig,
        })
    if not rows:
        return
    with open(out / "detection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("detection (window=%d steps):" % window)
    for row in rows:
        print(f"  {row['run']:<12} detected {row['detected']}/{row['boundaries_reached']} "
              f"latency {row['latencies']}  false {row['false_triggers']}  "
              f"{'' if row['complete'] else '(partial run)'}")


def _by_arm(runs: list[Run], complete_only: bool = True) -> dict[str, list[Run]]:
    g: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        if r.complete or not complete_only:
            g[r.arm].append(r)
    return dict(sorted(g.items()))


def fig_tradeoff(runs: list[Run], out: Path) -> None:
    """x: how well NEW domains were learned (mean loss on each domain at the end
    of its own phase, excluding the first). y: mean forgetting. Down-left is
    better on both; the base arms sit bottom-right of the ideal, freeze arms
    top-left — the question is whether any arm reaches the corner."""
    groups = _by_arm(runs)
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm, rs in groups.items():
        doms = rs[0].domains[1:]
        xs = [np.mean([r.summary["phase_end"][d][d] for d in doms]) for r in rs]
        ys = [r.summary["forgetting_avg"] for r in rs]
        c = arm_color(arm)
        ax.errorbar(np.mean(xs), np.mean(ys),
                    xerr=np.std(xs) if len(rs) > 1 else None,
                    yerr=np.std(ys) if len(rs) > 1 else None,
                    fmt="o", color=c, ms=7, capsize=3)
        ax.annotate(f"{arm.removeprefix('toy_')} (n={len(rs)})", (np.mean(xs), np.mean(ys)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=c)
    ax.set_xlabel("new-domain loss at end of its phase (lower = learned better)")
    ax.set_ylabel("mean forgetting (lower = remembered better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=160)
    plt.close(fig)


def leak_table(runs: list[Run], out: Path) -> None:
    """At the final eval, what fraction of each old domain's routing goes to
    experts that did not exist when that domain's phase ended. Under freeze=all
    this is the only path by which an old domain can still degrade."""
    rows = []
    for r in runs:
        final = next((e for e in reversed(r.evals) if "route" in e), None)
        if final is None or not r.complete:
            continue
        born: dict[str, int] = {}
        for e in r.evals:
            if e.get("phase_end"):
                born[r.domains[e["domain"]]] = e["experts"]
        row = {"run": r.name, "arm": r.arm}
        for d in r.domains[:-1]:
            n_old = born[d]
            layers = final["route"][d]
            row[f"leak_{d}"] = float(np.mean([sum(layer[n_old:]) for layer in layers]))
        rows.append(row)
    if not rows:
        return
    with open(out / "leak.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = [k for k in rows[0] if k.startswith("leak_")]
    print("router leak at end of run (share of old-domain routing to later-born experts):")
    g: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        g[row["arm"]].append(row)
    for arm, rs in sorted(g.items()):
        cells = "  ".join(f"{k[5:]}={np.mean([x[k] for x in rs]):.2f}" for k in keys)
        print(f"  {arm:<16} n={len(rs)}  {cells}")


def main() -> None:
    p = argparse.ArgumentParser(description="Make figures from experiment logs.")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=None, help="default: <results>/figures")
    p.add_argument("--fig", nargs="+", default=["all"],
                   choices=["all", "experts", "heldout", "detection", "tradeoff", "leak"])
    args = p.parse_args()
    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no runs under {args.results}")
    want = set(args.fig)
    every = "all" in want
    if every or "experts" in want:
        fig_experts(runs, out)
    if every or "heldout" in want:
        fig_heldout(runs, out)
    if every or "detection" in want:
        detection_table(runs, out)
    if every or "tradeoff" in want:
        fig_tradeoff(runs, out)
    if every or "leak" in want:
        leak_table(runs, out)
    partial = [r.name for r in runs if not r.complete]
    print(f"wrote figures to {out}  ({len(runs)} runs"
          + (f", partial: {', '.join(partial)}" if partial else "") + ")")


if __name__ == "__main__":
    main()
