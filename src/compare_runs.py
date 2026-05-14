from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple training runs.")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/comparison.csv")
    parser.add_argument("--output_markdown", type=str, default="outputs/comparison.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def maybe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def main() -> None:
    args = parse_args()

    rows: list[dict[str, Any]] = []

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        summary_path = run_dir / "metrics" / "train_summary.json"
        validation_path = run_dir / "metrics" / "validation_report.json"

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")

        summary = load_json(summary_path)
        validation = maybe_load_json(validation_path)

        row = {
            "run": run_dir.name,
            "optimizer": summary.get("optimizer"),
            "seq_len": summary.get("max_seq_length"),
            "steps": summary.get("max_steps"),
            "dtype": summary.get("resolved_dtype"),
            "grad_accum": summary.get("gradient_accumulation_steps"),
            "last_loss": summary.get("last_loss"),
            "initial_loss_mean": validation.get("initial_loss_mean"),
            "final_loss_mean": validation.get("final_loss_mean"),
            "delta_loss": validation.get("delta_loss"),
            "rel_delta_loss_%": validation.get("relative_delta_loss_percent"),
            "initial_ppl": validation.get("initial_perplexity"),
            "final_ppl": validation.get("final_perplexity"),
            "delta_ppl": validation.get("delta_perplexity"),
            "mean_step_time_sec": summary.get("mean_step_time_sec"),
            "mean_tok_per_sec": validation.get("mean_tokens_per_sec_excluding_warmup"),
            "max_memory_allocated_mb": summary.get("max_memory_allocated_mb"),
            "max_memory_reserved_mb": summary.get("max_memory_reserved_mb"),
            "memory_ratio": validation.get("max_memory_allocated_ratio"),
            "step_time_cv": validation.get("cv_step_time_excluding_warmup"),
            "overall_pass": validation.get("overall_pass"),
            "loss_did_not_explode": validation.get("loss_did_not_explode"),
            "muon_num_params": summary.get("muon_num_params"),
            "adamw_num_params": summary.get("adamw_num_params"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    output_csv = Path(args.output_csv)
    output_markdown = Path(args.output_markdown)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    markdown = df.to_markdown(index=False)
    output_markdown.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"\nSaved CSV to: {output_csv}")
    print(f"Saved Markdown to: {output_markdown}")


if __name__ == "__main__":
    main()
