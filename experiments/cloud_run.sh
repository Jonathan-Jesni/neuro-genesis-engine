#!/usr/bin/env bash
# One-shot cloud GPU session (AMD ROCm Jupyter notebook): setup, data, all arms.
#
# Run from the repo root in the notebook's terminal:
#     bash experiments/cloud_run.sh            # full: 10M tokens/domain, seeds 0 1 2
#     TOKENS=5000000 bash experiments/cloud_run.sh   # faster if throughput is low
#
# The four arms run IN PARALLEL (the tiny model cannot saturate a big GPU alone);
# each arm runs its seeds sequentially, so seed 0 of every arm finishes first.
# Progress: tail -f logs/*.log      Table: python -m experiments.summarize
# DOWNLOAD results.tar.gz BEFORE the session timer ends — the box is not persistent.
set -euo pipefail

TOKENS="${TOKENS:-10000000}"
SEEDS="${SEEDS:-0 1 2}"
export PYTHONPATH="$PWD"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "== GPU check =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no GPU visible to torch"
print("torch", torch.__version__, "| hip", torch.version.hip, "|", torch.cuda.get_device_name(0))
if torch.version.hip is None:
    print("WARNING: not a ROCm build of torch")
EOF

echo "== deps (never torch: keep the preinstalled ROCm build) =="
pip install -q datasets tiktoken pyyaml

echo "== data (~3 min) =="
python -m experiments.prepare_data 2>&1 | grep -v -E "Warning|warn|Retrying|disconnected" || true

echo "== throughput probe (arm A, ~180 steps) =="
python -m experiments.train configs/toy_A.yaml --seed 99 --name probe \
    --set tokens_per_phase=500000 --set eval_every=100000 --set eval_batches=1 \
    --set results_dir=/tmp/probe | tail -2

mkdir -p logs results
echo "== launching arms A B C D in parallel, tokens/phase=$TOKENS seeds=[$SEEDS] =="
for arm in A B C D; do
    nohup python -m experiments.train "configs/toy_${arm}.yaml" --seed $SEEDS \
        --set tokens_per_phase="$TOKENS" > "logs/toy_${arm}.log" 2>&1 &
    echo "  arm $arm -> pid $! (logs/toy_${arm}.log)"
done

cat <<EOF

Running in background. Useful commands:
    tail -f logs/*.log
    python -m experiments.summarize
    tar czf results.tar.gz results logs     # then download it from the file browser
EOF
