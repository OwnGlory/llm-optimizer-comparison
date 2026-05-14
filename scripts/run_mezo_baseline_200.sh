#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="outputs/mezo_baseline_200"
rm -rf "${OUTPUT_DIR}"

python -m src.train_mezo \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name Elriggs/openwebtext-100k \
  --dataset_split train \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_length 512 \
  --train_subset_size 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 200 \
  --learning_rate 1e-6 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --zo_eps 1e-3 \
  --zo_seed 1234 \
  --dtype bf16 \
  --save_final \
  --seed 42 \
  --log_every 1
