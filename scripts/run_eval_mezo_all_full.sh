#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="outputs/mezo_baseline_200/checkpoints/final"
OUTPUT_PATH="outputs/mezo_baseline_200/eval/all_tasks_full"
TASKS="piqa,arc_easy,arc_challenge,winogrande,hellaswag"

mkdir -p "${OUTPUT_PATH}"

lm_eval \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype=bfloat16 \
  --tasks "${TASKS}" \
  --device cuda:0 \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}"
