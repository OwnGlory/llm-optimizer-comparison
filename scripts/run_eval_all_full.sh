#!/usr/bin/env bash
set -euo pipefail

TASKS="piqa,arc_easy,arc_challenge,winogrande,hellaswag"
BATCH_SIZE="auto"

run_eval () {
  local RUN_NAME="$1"
  local MODEL_PATH="outputs/${RUN_NAME}/checkpoints/final"
  local OUTPUT_PATH="outputs/${RUN_NAME}/eval/all_tasks_full"

  echo "============================================================"
  echo "Running full evaluation for: ${RUN_NAME}"
  echo "Model path: ${MODEL_PATH}"
  echo "Output path: ${OUTPUT_PATH}"
  echo "============================================================"

  mkdir -p "${OUTPUT_PATH}"

  if command -v lm_eval >/dev/null 2>&1; then
    lm_eval \
      --model hf \
      --model_args pretrained="${MODEL_PATH}",dtype=bfloat16 \
      --tasks "${TASKS}" \
      --device cuda:0 \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${OUTPUT_PATH}"

  elif command -v lm-eval >/dev/null 2>&1; then
    lm-eval run \
      --model hf \
      --model_args pretrained="${MODEL_PATH}",dtype=bfloat16 \
      --tasks "${TASKS}" \
      --device cuda:0 \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${OUTPUT_PATH}"

  else
    echo "Could not find lm_eval or lm-eval command."
    exit 1
  fi
}

run_eval "adamw_baseline_200"
run_eval "muon_baseline_200"
run_eval "hybrid_baseline_200"
