from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def find_text_column(dataset: Dataset) -> str:
    """Find the most likely text column in a Hugging Face dataset."""
    preferred_names = ["text", "content", "raw_content"]

    for column_name in preferred_names:
        if column_name in dataset.column_names:
            return column_name

    sample = dataset[0]
    for column_name, value in sample.items():
        if isinstance(value, str):
            return column_name

    raise ValueError(f"Could not find a text column. Available columns: {dataset.column_names}")


def prepare_tokenized_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    split: str,
    max_seq_length: int,
    subset_size: int | None,
    seed: int,
) -> Dataset:
    """Load and tokenize a text dataset for causal language modeling."""
    dataset = load_dataset(dataset_name, split=split)

    if subset_size is not None and subset_size > 0 and subset_size < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(subset_size))

    text_column = find_text_column(dataset)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(examples: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [text if text is not None else "" for text in examples[text_column]]

        return tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized


@dataclass
class CausalLMCollator:
    """Create input_ids, attention_mask and labels for causal LM training."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [feature["input_ids"] for feature in features],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [feature["attention_mask"] for feature in features],
            dtype=torch.long,
        )

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
