from __future__ import annotations

import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def format_num(value: int) -> str:
    return f"{value:,}".replace(",", "_")


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("\n=== Tokenizer ===")
    print("Tokenizer class:", tokenizer.__class__.__name__)
    print("Vocab size:", len(tokenizer))

    print("\n=== Model ===")
    print("Model class:", model.__class__.__name__)
    print("Hidden size:", getattr(model.config, "hidden_size", None))
    print("Intermediate size:", getattr(model.config, "intermediate_size", None))
    print("Number of hidden layers:", getattr(model.config, "num_hidden_layers", None))
    print("Number of attention heads:", getattr(model.config, "num_attention_heads", None))
    print("Number of key-value heads:", getattr(model.config, "num_key_value_heads", None))
    print("Max position embeddings:", getattr(model.config, "max_position_embeddings", None))

    total_params = 0
    trainable_params = 0
    layer_param_counts: dict[int, int] = defaultdict(int)
    non_layer_param_count = 0

    print("\n=== Named parameters ===")
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if param.requires_grad:
            trainable_params += numel

        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            layer_param_counts[layer_idx] += numel
        else:
            non_layer_param_count += numel

        print(
            f"{name:80s} "
            f"shape={str(tuple(param.shape)):24s} "
            f"numel={format_num(numel):>15s} "
            f"requires_grad={param.requires_grad}"
        )

    print("\n=== Parameter summary ===")
    print(f"Total parameters:     {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")
    print(f"Non-layer parameters: {format_num(non_layer_param_count)}")

    print("\n=== Parameters per transformer layer ===")
    for layer_idx in sorted(layer_param_counts):
        print(f"Layer {layer_idx:02d}: {format_num(layer_param_counts[layer_idx])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
