#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name Elriggs/openwebtext-100k \
  --dataset_split train \
  --optimizer adamw \
  --output_dir outputs/adamw_smoke \
  --max_seq_length 256 \
  --train_subset_size 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 10 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --warmup_steps 2 \
  --dtype auto \
  --gradient_checkpointing \
  --seed 42 \
  --log_every 1
