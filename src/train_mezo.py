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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import CausalLMCollator, prepare_tokenized_dataset
from src.logging_utils import JsonlLogger, save_json, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MeZO full fine-tuning script.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="Elriggs/openwebtext-100k")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--train_subset_size", type=int, default=128)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=2)

    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--zo_seed", type=int, default=1234)

    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"]
    )
    parser.add_argument("--save_final", action=argparse.BooleanOptionalAction, default=False)

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


def cosine_lr(
    base_lr: float,
    step: int,
    max_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    if max_steps <= warmup_steps:
        return base_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def trainable_floating_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        param
        for param in model.parameters()
        if param.requires_grad and torch.is_floating_point(param)
    ]


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def perturb_parameters(
    params: list[nn.Parameter],
    seed: int,
    scale: float,
    device: torch.device,
) -> None:
    generator = make_generator(seed, device)

    for param in params:
        noise = torch.randn(
            param.shape,
            device=param.device,
            dtype=param.dtype,
            generator=generator,
        )
        param.add_(noise, alpha=scale)


@torch.no_grad()
def zo_update(
    params: list[nn.Parameter],
    seed: int,
    projected_gradient: float,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    generator = make_generator(seed, device)

    for param in params:
        if weight_decay != 0.0:
            param.mul_(1.0 - learning_rate * weight_decay)

        noise = torch.randn(
            param.shape,
            device=param.device,
            dtype=param.dtype,
            generator=generator,
        )
        param.add_(noise, alpha=-learning_rate * projected_gradient)


@torch.no_grad()
def compute_mean_loss(
    model: nn.Module,
    micro_batches: list[dict[str, torch.Tensor]],
    train_dtype: torch.dtype,
    use_amp: bool,
) -> tuple[float, int]:
    total_loss = 0.0
    non_padding_tokens = 0

    for batch in micro_batches:
        non_padding_tokens += int(batch["attention_mask"].sum().item())

        with autocast(
            device_type="cuda",
            dtype=train_dtype,
            enabled=use_amp,
        ):
            outputs = model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        total_loss += float(loss.detach().cpu().item())

    mean_loss = total_loss / len(micro_batches)

    return mean_loss, non_padding_tokens


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
    config["optimizer"] = "mezo"
    config["device"] = str(device)
    config["resolved_dtype"] = dtype_name
    config["cuda_device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

    print("=== MeZO training config ===")
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
    model.eval()

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    params = trainable_floating_parameters(model)

    config["total_params"] = total_params
    config["trainable_params"] = trainable_params
    config["mezo_num_tensors"] = len(params)
    config["mezo_num_params"] = sum(param.numel() for param in params)

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"MeZO tensors:         {config['mezo_num_tensors']}")
    print(f"MeZO parameters:      {config['mezo_num_params']:,}")

    save_yaml(logs_dir / "config.yaml", config)

    logger = JsonlLogger(logs_dir / "train.jsonl")
    data_iter = infinite_loader(train_loader)

    use_amp = device.type == "cuda" and train_dtype in {torch.float16, torch.bfloat16}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\nStarting MeZO training...")
    total_train_start = time.perf_counter()
    step_times: list[float] = []
    last_loss: float | None = None
    last_projected_gradient: float | None = None

    progress = tqdm(range(1, args.max_steps + 1), desc="MeZO training")

    for step in progress:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_start = time.perf_counter()
        current_lr = cosine_lr(
            base_lr=args.learning_rate,
            step=step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
        )

        micro_batches = [
            move_batch_to_device(next(data_iter), device)
            for _ in range(args.gradient_accumulation_steps)
        ]

        zo_seed = args.zo_seed + step

        perturb_parameters(
            params=params,
            seed=zo_seed,
            scale=args.zo_eps,
            device=device,
        )
        loss_plus, non_padding_tokens = compute_mean_loss(
            model=model,
            micro_batches=micro_batches,
            train_dtype=train_dtype,
            use_amp=use_amp,
        )

        perturb_parameters(
            params=params,
            seed=zo_seed,
            scale=-2.0 * args.zo_eps,
            device=device,
        )
        loss_minus, _ = compute_mean_loss(
            model=model,
            micro_batches=micro_batches,
            train_dtype=train_dtype,
            use_amp=use_amp,
        )

        perturb_parameters(
            params=params,
            seed=zo_seed,
            scale=args.zo_eps,
            device=device,
        )

        projected_gradient = (loss_plus - loss_minus) / (2.0 * args.zo_eps)

        zo_update(
            params=params,
            seed=zo_seed,
            projected_gradient=projected_gradient,
            learning_rate=current_lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time_sec = time.perf_counter() - step_start
        step_times.append(step_time_sec)

        loss_estimate = 0.5 * (loss_plus + loss_minus)
        tokens_per_sec = non_padding_tokens / step_time_sec if step_time_sec > 0 else 0.0
        memory_stats = get_cuda_memory_stats()

        last_loss = loss_estimate
        last_projected_gradient = projected_gradient

        record = {
            "step": step,
            "optimizer": "mezo",
            "loss": loss_estimate,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "zo_projected_gradient": projected_gradient,
            "zo_eps": args.zo_eps,
            "zo_seed": zo_seed,
            "lr": current_lr,
            "grad_norm": 0.0,
            "step_time_sec": step_time_sec,
            "non_padding_tokens": non_padding_tokens,
            "tokens_per_sec": tokens_per_sec,
            **memory_stats,
        }

        if step % args.log_every == 0:
            logger.log(record)

        progress.set_postfix(
            {
                "loss": f"{loss_estimate:.4f}",
                "lr": f"{current_lr:.2e}",
                "zo_g": f"{projected_gradient:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "max_mem_MB": f"{memory_stats['max_memory_allocated_mb']:.0f}",
            }
        )

    total_train_time_sec = time.perf_counter() - total_train_start
    logger.close()

    summary = {
        "optimizer": "mezo",
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "train_subset_size": args.train_subset_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "resolved_dtype": dtype_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "zo_eps": args.zo_eps,
        "last_loss": last_loss,
        "last_projected_gradient": last_projected_gradient,
        "total_train_time_sec": total_train_time_sec,
        "mean_step_time_sec": float(np.mean(step_times)) if step_times else None,
        "min_step_time_sec": float(np.min(step_times)) if step_times else None,
        "max_step_time_sec": float(np.max(step_times)) if step_times else None,
        "mezo_num_tensors": len(params),
        "mezo_num_params": sum(param.numel() for param in params),
        "muon_num_params": 0,
        "adamw_num_params": 0,
        **get_cuda_memory_stats(),
    }

    save_json(metrics_dir / "train_summary.json", summary)

    if args.save_final:
        final_checkpoint_dir = checkpoints_dir / "final"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving final checkpoint to {final_checkpoint_dir}")
        model.save_pretrained(final_checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_checkpoint_dir)

    print("\n=== MeZO training summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
