#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="outputs/mezo_smoke"
rm -rf "${OUTPUT_DIR}"

python -m src.train_mezo \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name Elriggs/openwebtext-100k \
  --dataset_split train \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_length 256 \
  --train_subset_size 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 10 \
  --learning_rate 1e-6 \
  --weight_decay 0.0 \
  --warmup_steps 2 \
  --zo_eps 1e-3 \
  --zo_seed 1234 \
  --dtype bf16 \
  --seed 42 \
  --log_every 1
