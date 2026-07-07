"""Resume an interrupted orchestrator demo run from its checkpoint.

Usage (after `python core/orchestrator.py` was interrupted, e.g. by Ctrl+C or
a compute-quota kill):

    python resume_demo.py

Everything task-defining is IMPORTED from core.orchestrator -- the base-expert
factory, the data function (same W_true via its dedicated seed), the step
budget, the checkpoint/log paths -- so the resumed run cannot drift from the
original run's architecture or data distribution.
"""

from __future__ import annotations

import sys

from core.orchestrator import (
    DEMO_CHECKPOINT,
    DEMO_CHECKPOINT_EVERY,
    DEMO_LOG,
    DEMO_STEPS,
    TrainingOrchestrator,
    demo_base_expert_factory,
    make_demo_data_fn,
    print_demo_summary,
)


def main() -> int:
    try:
        orch = TrainingOrchestrator.from_checkpoint(
            DEMO_CHECKPOINT,
            base_expert_factory=demo_base_expert_factory,
            log_path=DEMO_LOG,  # append to the same log: one reconstructable record
            checkpoint_every_n_steps=DEMO_CHECKPOINT_EVERY,
        )
    except FileNotFoundError:
        print(f"no checkpoint at {DEMO_CHECKPOINT!r} -- run the demo first "
              f"(python core/orchestrator.py)")
        return 1

    remaining = DEMO_STEPS - orch.step_count
    if remaining <= 0:
        print(f"nothing to resume: checkpoint is at step {orch.step_count} of "
              f"{DEMO_STEPS}; the run already completed.")
        orch.logger.close()
        return 0

    start_experts = orch.gate.num_experts
    start_counts = (orch.spikes_seen, orch.registrations, orch.rejections)
    print(f"resumed from {DEMO_CHECKPOINT} at step {orch.step_count} "
          f"({orch.gate.num_experts} experts, {orch.registrations} prior "
          f"registrations); running {remaining} remaining steps...")

    results = orch.run(remaining, make_demo_data_fn())
    print_demo_summary(orch, results,
                       start_experts=start_experts, start_counts=start_counts)
    orch.logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
