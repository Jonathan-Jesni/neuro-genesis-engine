"""Print a results table (mean ± std over seeds) from results/*/summary.json.

Usage:
    python -m experiments.summarize [results_dir]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _ms(xs: list[float]) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(root.glob("*/summary.json")):
        s = json.loads(f.read_text())
        groups[s["run"].rsplit("_s", 1)[0]].append(s)
    if not groups:
        print(f"no summaries under {root}")
        return

    domains = list(next(iter(groups.values()))[0]["final"])
    head = ["run", "n", "final_avg", "forgetting_avg"] + [f"fgt_{d}" for d in domains[:-1]] + ["experts"]
    rows = []
    for name, runs in sorted(groups.items()):
        rows.append([
            name, str(len(runs)),
            _ms([r["final_avg"] for r in runs]),
            _ms([r["forgetting_avg"] for r in runs]),
            *[_ms([r["forgetting"][d] for r in runs]) for d in domains[:-1]],
            _ms([float(r["experts_final"]) for r in runs]),
        ])
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    for r in [head, ["-" * w for w in widths], *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(f"\n(lower is better for loss and forgetting; device: {runs[0].get('device')})")


if __name__ == "__main__":
    main()
