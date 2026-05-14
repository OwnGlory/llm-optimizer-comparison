from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data import CausalLMCollator, prepare_tokenized_dataset
from src.logging_utils import JsonlLogger, save_json, save_yaml
from src.optim.muon import MuonWithAuxAdam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tuning training script.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="Elriggs/openwebtext-100k")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "muon", "hybrid"]
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--train_subset_size", type=int, default=128)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument(
        "--muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--hybrid_split_layer", type=int, default=12)
    parser.add_argument("--hybrid_muon_side", type=str, default="lower", choices=["lower", "upper"])

    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument(
        "--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"]
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_final",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(dtype_name: str, device: torch.device) -> tuple[torch.dtype, str]:
    if device.type != "cuda":
        return torch.float32, "fp32"

    if dtype_name == "bf16":
        return torch.bfloat16, "bf16"

    if dtype_name == "fp16":
        return torch.float16, "fp16"

    if dtype_name == "fp32":
        return torch.float32, "fp32"

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"

    return torch.float16, "fp16"


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def get_cuda_memory_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
            "max_memory_reserved_mb": 0.0,
        }

    return {
        "memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def get_layer_index(name: str) -> int | None:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None

    rest = name[len(prefix) :]
    layer_idx_str = rest.split(".", 1)[0]

    if not layer_idx_str.isdigit():
        return None

    return int(layer_idx_str)


def is_hidden_matrix_parameter(name: str, param: nn.Parameter) -> bool:
    if not param.requires_grad:
        return False

    if param.ndim != 2:
        return False

    if not name.startswith("model.layers."):
        return False

    excluded_keywords = [
        "embed_tokens",
        "lm_head",
        "norm",
    ]

    return not any(keyword in name for keyword in excluded_keywords)


def is_muon_parameter(name: str, param: nn.Parameter) -> bool:
    return is_hidden_matrix_parameter(name, param)


def is_hybrid_muon_parameter(
    name: str,
    param: nn.Parameter,
    split_layer: int,
    muon_side: str,
) -> bool:
    if not is_hidden_matrix_parameter(name, param):
        return False

    layer_idx = get_layer_index(name)
    if layer_idx is None:
        return False

    if muon_side == "lower":
        return layer_idx < split_layer

    if muon_side == "upper":
        return layer_idx >= split_layer

    raise ValueError(f"Unsupported hybrid_muon_side: {muon_side}")


def build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, dict[str, int | float | str]]:
    if args.optimizer == "adamw":
        trainable_params = [param for param in model.parameters() if param.requires_grad]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            foreach=False,
        )

        stats: dict[str, int | float | str] = {
            "adamw_num_tensors": len(trainable_params),
            "adamw_num_params": sum(param.numel() for param in trainable_params),
            "muon_num_tensors": 0,
            "muon_num_params": 0,
            "hybrid_split_layer": -1,
            "hybrid_muon_side": "none",
        }

        return optimizer, stats

    if args.optimizer in {"muon", "hybrid"}:
        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []

        muon_names: list[str] = []
        adamw_names: list[str] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if args.optimizer == "muon":
                use_muon = is_muon_parameter(name, param)
            else:
                use_muon = is_hybrid_muon_parameter(
                    name=name,
                    param=param,
                    split_layer=args.hybrid_split_layer,
                    muon_side=args.hybrid_muon_side,
                )

            if use_muon:
                muon_params.append(param)
                muon_names.append(name)
            else:
                adamw_params.append(param)
                adamw_names.append(name)

        optimizer = MuonWithAuxAdam(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.muon_momentum,
            ns_steps=args.muon_ns_steps,
            nesterov=args.muon_nesterov,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            adamw_eps=args.adam_eps,
        )

        stats = {
            "muon_num_tensors": len(muon_params),
            "muon_num_params": sum(param.numel() for param in muon_params),
            "adamw_num_tensors": len(adamw_params),
            "adamw_num_params": sum(param.numel() for param in adamw_params),
            "hybrid_split_layer": args.hybrid_split_layer if args.optimizer == "hybrid" else -1,
            "hybrid_muon_side": args.hybrid_muon_side if args.optimizer == "hybrid" else "none",
        }

        print("\n=== Optimizer parameter split ===")
        print(f"Optimizer:     {args.optimizer}")
        print(f"Muon tensors:  {stats['muon_num_tensors']}")
        print(f"Muon params:   {stats['muon_num_params']:,}")
        print(f"AdamW tensors: {stats['adamw_num_tensors']}")
        print(f"AdamW params:  {stats['adamw_num_params']:,}")

        if args.optimizer == "hybrid":
            print(f"Hybrid split layer: {args.hybrid_split_layer}")
            print(f"Hybrid Muon side:   {args.hybrid_muon_side}")

        print("\nFirst 10 Muon params:")
        for name in muon_names[:10]:
            print(f"  {name}")

        print("\nFirst 10 AdamW params:")
        for name in adamw_names[:10]:
            print(f"  {name}")

        return optimizer, stats

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir = output_dir / "checkpoints"

    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dtype, dtype_name = choose_dtype(args.dtype, device)

    config = vars(args).copy()
    config["device"] = str(device)
    config["resolved_dtype"] = dtype_name
    config["cuda_device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

    print("=== Training config ===")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nPreparing dataset...")
    train_dataset = prepare_tokenized_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        max_seq_length=args.max_seq_length,
        subset_size=args.train_subset_size,
        seed=args.seed,
    )

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=train_dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

    model.to(device)  # type: ignore[arg-type]
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer, optimizer_stats = build_optimizer(args, model)
    config.update(optimizer_stats)
    save_yaml(logs_dir / "config.yaml", config)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    use_amp = device.type == "cuda" and train_dtype in {torch.float16, torch.bfloat16}
    use_grad_scaler = device.type == "cuda" and train_dtype == torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    logger = JsonlLogger(logs_dir / "train.jsonl")
    data_iter = infinite_loader(train_loader)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\nStarting training...")
    total_train_start = time.perf_counter()
    last_loss = None
    step_times = []

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(range(1, args.max_steps + 1), desc="Training")

    for step in progress:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_start = time.perf_counter()
        accumulated_loss = 0.0
        non_padding_tokens = 0

        for _micro_step in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = move_batch_to_device(batch, device)

            non_padding_tokens += int(batch["attention_mask"].sum().item())

            with autocast(
                device_type="cuda",
                dtype=train_dtype,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

            loss_for_backward = loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            accumulated_loss += float(loss.detach().cpu().item())

        mean_loss = accumulated_loss / args.gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
            )
            grad_norm_value = float(grad_norm.detach().cpu().item())
        else:
            grad_norm_value = math.nan

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time_sec = time.perf_counter() - step_start
        step_times.append(step_time_sec)

        tokens_per_sec = non_padding_tokens / step_time_sec if step_time_sec > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]
        memory_stats = get_cuda_memory_stats()

        last_loss = mean_loss

        record = {
            "step": step,
            "optimizer": args.optimizer,
            "loss": mean_loss,
            "lr": current_lr,
            "grad_norm": grad_norm_value,
            "step_time_sec": step_time_sec,
            "non_padding_tokens": non_padding_tokens,
            "tokens_per_sec": tokens_per_sec,
            **memory_stats,
        }

        if step % args.log_every == 0:
            logger.log(record)

        progress.set_postfix(
            {
                "loss": f"{mean_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "max_mem_MB": f"{memory_stats['max_memory_allocated_mb']:.0f}",
            }
        )

    total_train_time_sec = time.perf_counter() - total_train_start
    logger.close()

    summary = {
        "optimizer": args.optimizer,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "train_subset_size": args.train_subset_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "resolved_dtype": dtype_name,
        "last_loss": last_loss,
        "total_train_time_sec": total_train_time_sec,
        "mean_step_time_sec": float(np.mean(step_times)) if step_times else None,
        "min_step_time_sec": float(np.min(step_times)) if step_times else None,
        "max_step_time_sec": float(np.max(step_times)) if step_times else None,
        **optimizer_stats,
        **get_cuda_memory_stats(),
    }

    save_json(metrics_dir / "train_summary.json", summary)

    if args.save_final:
        final_checkpoint_dir = checkpoints_dir / "final"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving final checkpoint to {final_checkpoint_dir}")
        model.save_pretrained(final_checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_checkpoint_dir)

    print("\n=== Training summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
