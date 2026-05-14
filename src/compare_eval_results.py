from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation result CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--run_names", nargs="+", required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_markdown", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.inputs) != len(args.run_names):
        raise ValueError("--inputs and --run_names must have the same length.")

    frames = []

    for input_path, run_name in zip(args.inputs, args.run_names, strict=True):
        df = pd.read_csv(input_path)
        df["run"] = run_name
        frames.append(df)

    long_df = pd.concat(frames, ignore_index=True)

    # Keep only real quality metrics.
    long_df = long_df[long_df["metric"].isin(["acc", "acc_norm"])].copy()

    # Wide table: one row per task/metric, one column per run.
    value_table = long_df.pivot_table(
        index=["task", "metric"],
        columns="run",
        values="value",
        aggfunc="first",
    ).reset_index()

    stderr_table = long_df.pivot_table(
        index=["task", "metric"],
        columns="run",
        values="stderr",
        aggfunc="first",
    ).reset_index()

    output_csv = Path(args.output_csv)
    output_markdown = Path(args.output_markdown)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    value_table.to_csv(output_csv, index=False)

    markdown = value_table.to_markdown(index=False)
    output_markdown.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"\nSaved value table CSV: {output_csv}")
    print(f"Saved value table Markdown: {output_markdown}")

    stderr_output_csv = output_csv.with_name(output_csv.stem + "_stderr.csv")
    stderr_output_markdown = output_markdown.with_name(output_markdown.stem + "_stderr.md")

    stderr_table.to_csv(stderr_output_csv, index=False)
    stderr_output_markdown.write_text(
        stderr_table.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    print(f"Saved stderr table CSV: {stderr_output_csv}")
    print(f"Saved stderr table Markdown: {stderr_output_markdown}")


if __name__ == "__main__":
    main()
