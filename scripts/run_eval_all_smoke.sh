#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="outputs/adamw_baseline_200/checkpoints/final"
OUTPUT_PATH="outputs/adamw_baseline_200/eval/all_tasks_smoke"

mkdir -p "${OUTPUT_PATH}"

TASKS="piqa,arc_easy,arc_challenge,winogrande,hellaswag"

if command -v lm_eval >/dev/null 2>&1; then
  lm_eval \
    --model hf \
    --model_args pretrained="${MODEL_PATH}",dtype=bfloat16 \
    --tasks "${TASKS}" \
    --device cuda:0 \
    --batch_size 1 \
    --limit 100 \
    --output_path "${OUTPUT_PATH}"

elif command -v lm-eval >/dev/null 2>&1; then
  lm-eval run \
    --model hf \
    --model_args pretrained="${MODEL_PATH}",dtype=bfloat16 \
    --tasks "${TASKS}" \
    --device cuda:0 \
    --batch_size 1 \
    --limit 100 \
    --output_path "${OUTPUT_PATH}"

else
  echo "Could not find lm_eval or lm-eval command."
  exit 1
fi
