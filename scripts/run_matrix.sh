#!/usr/bin/env bash
# Runs the full Green Gap experiment matrix from README's "Experiment Matrix"
# table: 3 core conditions x 3 seeds, plus 4 rank-ablation runs (r=8/r=32 for
# LoRA and QLoRA) via --rank overrides on the same two config files.
#
# Requires a rented GPU matching each condition's spec (see README's
# "Environment Setup / Phase 2"), and a zero-shot baseline accuracy measured
# beforehand.
#
# Usage:
#   GG_BASELINE_ACCURACY=0.12 GG_GPU_MODEL=A100-40GB GG_REGION=us-east-1 \
#     ./scripts/run_matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GG_BASELINE_ACCURACY:?Set GG_BASELINE_ACCURACY to the zero-shot base-model accuracy first}"
: "${GG_GPU_MODEL:?Set GG_GPU_MODEL (e.g. A100-40GB, RTX4090, L4)}"
: "${GG_REGION:?Set GG_REGION (e.g. us-east-1) for CO2e reporting}"
SEEDS=(1 2 3)

run() {
  local config=$1
  local seed=$2
  local rank=${3:-}
  local extra=()
  if [[ -n "$rank" ]]; then
    extra+=(--rank "$rank")
  fi
  echo "=== Running $config (seed=$seed${rank:+, rank=$rank}) ==="
  python scripts/run_experiment.py \
    --config "$config" \
    --baseline_accuracy "$GG_BASELINE_ACCURACY" \
    --gpu_model "$GG_GPU_MODEL" \
    --region "$GG_REGION" \
    --seed "$seed" \
    "${extra[@]}"
}

echo "--- Core runs: Full FT, LoRA r=16, QLoRA r=16 (3 seeds each) ---"
for seed in "${SEEDS[@]}"; do
  run configs/full_ft.yaml "$seed"
  run configs/lora.yaml "$seed"
  run configs/qlora.yaml "$seed"
done

echo "--- Rank ablations: LoRA/QLoRA at r=8 and r=32 (1 seed each) ---"
run configs/lora.yaml "${SEEDS[0]}" 8
run configs/lora.yaml "${SEEDS[0]}" 32
run configs/qlora.yaml "${SEEDS[0]}" 8
run configs/qlora.yaml "${SEEDS[0]}" 32

echo "All 13 runs complete. See results/metrics.csv for the aggregated table."
