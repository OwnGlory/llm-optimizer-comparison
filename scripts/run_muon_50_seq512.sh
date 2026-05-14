#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name Elriggs/openwebtext-100k \
  --dataset_split train \
  --optimizer muon \
  --output_dir outputs/muon_50_seq512 \
  --max_seq_length 512 \
  --train_subset_size 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 50 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --muon_momentum 0.95 \
  --muon_ns_steps 5 \
  --muon_nesterov \
  --warmup_steps 5 \
  --dtype bf16 \
  --gradient_checkpointing \
  --seed 42 \
  --log_every 1
