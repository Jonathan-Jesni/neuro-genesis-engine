"""Figures from results/<run>/ logs. Works on finished AND partial (crashed) runs.

    python -m experiments.analyze                      # all figures, results/ -> results/figures/
    python -m experiments.analyze --results path --fig experts

Figures:
    experts      expert count vs step, one line per run, true domain bands, growth markers
    heldout      held-out loss per domain vs step, one panel per domain, mean over seeds per arm
    detection    table of growth events vs true boundaries (latency, false triggers) -> stdout + csv
    tradeoff     stability-plasticity scatter: forgetting vs new-domain loss, one point per arm
    leak         router leak: share of each OLD domain's tokens routed to experts born later
    leakplot     scatter of router leak vs forgetting on the first domain, one dot per run
    detectplot   training loss + expert count with true boundaries and detector firings

Use --arms to restrict every figure to a readable subset, e.g.
    python -m experiments.analyze --arms toy_A toy_D toy_D_seen toy_A_replay --out figs/main
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
ARM_COLORS = {
    "toy_A": "#6b7280", "toy_B": "#2563eb", "toy_C": "#d97706", "toy_D": "#059669",
    "toy_D_fexp": "#a16207", "toy_D_fall": "#7c3aed", "toy_D_femb": "#db2777",
    "toy_D_seen": "#0891b2", "toy_C_seen": "#0e7490",
    "toy_A_replay": "#dc2626", "toy_D_replay": "#ea580c",
    "toy_D_seen_replay": "#65a30d", "toy_D_fall_replay": "#9333ea",
}
ARM_LABELS = {
    "toy_A": "Normal MoE", "toy_B": "Normal MoE, 6 experts",
    "toy_C": "Grow at true boundary (oracle)", "toy_D": "Grow on detection",
    "toy_D_fexp": "Grow + freeze old experts", "toy_D_fall": "Grow + freeze all old (1 expert)",
    "toy_D_femb": "Grow + freeze all but embeddings",
    "toy_D_seen": "Grow + freeze seen tokens", "toy_C_seen": "Oracle grow + freeze seen tokens",
    "toy_A_replay": "Normal MoE + 25% replay", "toy_D_replay": "Grow + 25% replay",
    "toy_D_seen_replay": "Grow + seen freeze + 25% replay",
    "toy_D_fall_replay": "Grow + freeze all + 25% replay",
    "toy_C_fall": "Oracle grow + freeze all (1 expert)",
    "toy_C_fall_k4": "Oracle grow + freeze all (4 experts)",
    "toy_D_fall_k2": "Grow + freeze all (2 experts)",
    "toy_D_fall_k4": "Grow + freeze all (4 experts)",
    "toy_D_fall_k8": "Grow + freeze all (8 experts)",
}
ARM_COLORS.update({
    "toy_C_fall": "#c4b5fd", "toy_C_fall_k4": "#a78bfa",
    "toy_D_fall_k2": "#6d28d9", "toy_D_fall_k4": "#4c1d95", "toy_D_fall_k8": "#1e1b4b",
})
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
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for arm, rs in groups.items():
        doms = rs[0].domains[1:]
        xs = [np.mean([r.summary["phase_end"][d][d] for d in doms]) for r in rs]
        ys = [r.summary["forgetting_avg"] for r in rs]
        c = arm_color(arm)
        ax.errorbar(np.mean(xs), np.mean(ys),
                    xerr=np.std(xs) if len(rs) > 1 else None,
                    yerr=np.std(ys) if len(rs) > 1 else None,
                    fmt="o", color=c, ms=7, capsize=3,
                    label=f"{ARM_LABELS.get(arm, arm)} (n={len(rs)})")
    ax.set_xlabel("loss on each new domain when its phase ends\n(lower = learned it better)")
    ax.set_ylabel("average forgetting of earlier domains\n(lower = remembered better)")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
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


def fig_detection(runs: list[Run], out: Path) -> None:
    """Top: training loss (mean over seeds) with the jump at each domain change.
    Bottom: experts per layer; triangles mark growth fired by the detector.
    Dashed lines are the TRUE boundaries, which the model is never told."""
    rs = [r for r in runs if r.steps]
    if not rs:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax1, ax2):
        _domain_bands(ax, rs[0], label=ax is ax1)
        for b in rs[0].boundaries:
            ax.axvline(b, color="#374151", ls="--", lw=0.9, zorder=1)
    curves = defaultdict(list)
    for r in rs:
        for st in r.steps:
            curves[st["step"]].append(st["loss"])
    xs = sorted(curves)
    ax1.plot(xs, [np.mean(curves[x]) for x in xs], color="#111827", lw=1.3)
    ax1.set_ylabel("training loss")
    for r in rs:
        xs_e = [st["step"] for st in r.steps]
        ax2.step(xs_e, [st["experts"] for st in r.steps], where="post",
                 color=arm_color(r.arm), lw=1.6, alpha=0.8)
        for e in r.events:
            ax2.plot(e["step"], e["experts"], "v", color=arm_color(r.arm), ms=7, zorder=5)
    trig = sorted({e["step"] for r in rs for e in r.events})
    ax2.set_ylabel("experts / layer")
    ax2.set_xlabel("training step   (dashed = true domain change; triangles = detector fired: "
                   + ", ".join(map(str, trig)) + ")", fontsize=8)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "detection.png", dpi=160)
    plt.close(fig)


def fig_leak_scatter(runs: list[Run], out: Path) -> None:
    """One dot per completed run that logged routing: share of FIRST-domain
    tokens routed to experts born after that domain, vs forgetting of it."""
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in runs:
        final = next((e for e in reversed(r.evals) if "route" in e), None)
        if final is None or not r.complete:
            continue
        d0 = r.domains[0]
        n_old = next(e["experts"] for e in r.evals
                     if e.get("phase_end") and r.domains[e["domain"]] == d0)
        leak = float(np.mean([sum(layer[n_old:]) for layer in final["route"][d0]]))
        pts[r.arm].append((leak, r.summary["forgetting"][d0]))
    if not pts:
        return
    allx = [x for v in pts.values() for x, _ in v]
    ally = [y for v in pts.values() for _, y in v]
    corr = np.corrcoef(allx, ally)[0, 1] if len(allx) > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm, v in sorted(pts.items()):
        ax.scatter([x for x, _ in v], [y for _, y in v], color=arm_color(arm), s=36,
                   label=ARM_LABELS.get(arm, arm), zorder=3)
    ax.set_xlabel(f"share of {runs[0].domains[0]} tokens routed to experts added later")
    ax.set_ylabel(f"forgetting on {runs[0].domains[0]} (nats)")
    ax.set_title(f"router leak vs forgetting  (r = {corr:.2f}, {len(allx)} runs)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "leak_scatter.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Make figures from experiment logs.")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=None, help="default: <results>/figures")
    p.add_argument("--fig", nargs="+", default=["all"],
                   choices=["all", "experts", "heldout", "detection", "tradeoff", "leak",
                            "leakplot", "detectplot"])
    p.add_argument("--arms", nargs="+", default=None,
                   help="only include these arms (run-name prefix before _s<seed>)")
    args = p.parse_args()
    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    if args.arms:
        runs = [r for r in runs if r.arm in set(args.arms)]
    if not runs:
        raise SystemExit(f"no runs under {args.results}"
                         + (f" matching arms {args.arms}" if args.arms else ""))
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
    if every or "leakplot" in want:
        fig_leak_scatter(runs, out)
    if every or "detectplot" in want:
        fig_detection(runs, out)
    partial = [r.name for r in runs if not r.complete]
    print(f"wrote figures to {out}  ({len(runs)} runs"
          + (f", partial: {', '.join(partial)}" if partial else "") + ")")


if __name__ == "__main__":
    main()
