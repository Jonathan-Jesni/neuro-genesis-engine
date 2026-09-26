#!/usr/bin/env bash
# Run a queue of training runs sequentially; resumable and crash-tolerant.
#
#     nohup bash experiments/run_queue.sh experiments/queues/day3.txt > logs/queue.log 2>&1 &
#
# Queue file: one run per line —  <name> <config> <seed> [--set key=value ...]
# Blank lines and '#' comments are ignored. A run whose
# $RESULTS_DIR/<name>_s<seed>/summary.json already exists is SKIPPED, so after a
# session timeout just launch the same command again and it picks up where it
# stopped. A crashed run logs "CRASH" and the queue moves on.
#
# One process at a time on purpose: parallel runs crashed the W7900D (HIP
# illegal memory access), a single process never has.
#
# Env: RESULTS_DIR (default results), EXTRA (extra args appended to every run,
# e.g. EXTRA="--set tokens_per_phase=200000" for a smoke test).
set -u
QUEUE="${1:?usage: run_queue.sh <queue-file>}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EXTRA="${EXTRA:-}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

total=$(grep -cvE '^\s*(#|$)' "$QUEUE")
i=0
while read -r name cfg seed rest; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    i=$((i + 1))
    if [[ -f "$RESULTS_DIR/${name}_s${seed}/summary.json" ]]; then
        echo "[$i/$total] SKIP ${name}_s${seed} (already done)"
        continue
    fi
    echo "[$i/$total] RUN  ${name}_s${seed}  $cfg $rest"
    # shellcheck disable=SC2086  # word-splitting of $rest/$EXTRA is intended
    python -m experiments.train "$cfg" --seed "$seed" --name "$name" \
        --set results_dir="$RESULTS_DIR" $rest $EXTRA < /dev/null \
        || echo "[$i/$total] CRASH ${name}_s${seed}"
done < "$QUEUE"
echo "queue finished: $QUEUE"
