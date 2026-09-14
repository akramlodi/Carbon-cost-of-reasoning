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
#
# Run only the methods appropriate for the current GPU (comma-separated):
#   GG_METHODS=lora,qlora ./scripts/run_matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GG_BASELINE_ACCURACY:?Set GG_BASELINE_ACCURACY to the zero-shot base-model accuracy first}"
: "${GG_GPU_MODEL:?Set GG_GPU_MODEL (e.g. A100-40GB, RTX4090, L4)}"
: "${GG_REGION:?Set GG_REGION (e.g. us-east-1) for CO2e reporting}"
SEEDS=(1 2 3)
GG_METHODS=${GG_METHODS:-full_ft,lora,qlora}

method_enabled() {
  local method=$1
  local requested
  IFS=',' read -r -a requested_methods <<< "$GG_METHODS"
  for requested in "${requested_methods[@]}"; do
    requested=${requested//[[:space:]]/}
    if [[ "$requested" == "$method" ]]; then
      return 0
    fi
  done
  return 1
}

IFS=',' read -r -a requested_methods <<< "$GG_METHODS"
for requested_method in "${requested_methods[@]}"; do
  requested_method=${requested_method//[[:space:]]/}
  case "$requested_method" in
    full_ft|lora|qlora) ;;
    *) echo "ERROR: GG_METHODS contains unknown method '$requested_method' (expected full_ft, lora, or qlora)" >&2; exit 1 ;;
  esac
done

run() {
  local config=$1
  local seed=$2
  local rank=${3:-}
  local extra=()
  if [[ -n "$rank" ]]; then
    extra+=(--rank "$rank")
  fi
  local method
  method=$(grep '^method:' "$config" | awk '{print $2}')

  local run_id="$method"
  if grep -q '^  enabled: true' "$config" && grep -q '^  r:' "$config"; then
    local base_rank
    base_rank=$(grep '^  r:' "$config" | awk '{print $2}')
    if [[ -n "$rank" ]]; then
      base_rank="$rank"
    fi
    run_id="${method}_r${base_rank}"
  fi
  run_id="${run_id}_seed${seed}"

  local result_path
  case "$method" in
    full_ft)
      result_path="results/runs/full_ft/${run_id}/result.json"
      ;;
    lora)
      result_path="results/runs/lora/${run_id}/result.json"
      ;;
    qlora)
      result_path="results/runs/qlora/${run_id}/result.json"
      ;;
    *)
      echo "ERROR: Unknown method '$method' in $config"
      exit 1
      ;;
  esac

  if [[ -f "$result_path" ]]; then
    echo "=== SKIPPING completed $run_id ==="
    return
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
  method_enabled full_ft && run configs/full_ft.yaml "$seed"
  method_enabled lora && run configs/lora.yaml "$seed"
  method_enabled qlora && run configs/qlora.yaml "$seed"
done

echo "--- Rank ablations: LoRA/QLoRA at r=8 and r=32 (1 seed each) ---"
method_enabled lora && run configs/lora.yaml "${SEEDS[0]}" 8
method_enabled lora && run configs/lora.yaml "${SEEDS[0]}" 32
method_enabled qlora && run configs/qlora.yaml "${SEEDS[0]}" 8
method_enabled qlora && run configs/qlora.yaml "${SEEDS[0]}" 32

echo "Selected matrix runs complete. See results/metrics.csv for the aggregated table."
