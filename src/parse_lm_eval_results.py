from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse lm-evaluation-harness JSON results.")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_markdown", type=str, required=True)
    return parser.parse_args()


def find_latest_result_json(eval_dir: Path) -> Path:
    json_files = sorted(
        eval_dir.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not json_files:
        raise FileNotFoundError(f"No JSON result files found under {eval_dir}")

    return json_files[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_metric_key(key: str) -> tuple[str, str | None]:
    if "," in key:
        metric_name, filter_name = key.split(",", 1)
        return metric_name, filter_name

    return key, None


def is_metric_value(key: str, value: Any) -> bool:
    if key in {"alias", "samples", "sample_len"}:
        return False

    metric_name, _filter_name = split_metric_key(key)

    if metric_name.endswith("_stderr"):
        return False

    return isinstance(value, int | float)


def find_stderr(metrics: dict[str, Any], metric_key: str) -> float | None:
    metric_name, filter_name = split_metric_key(metric_key)

    candidates = []

    if filter_name is not None:
        candidates.append(f"{metric_name}_stderr,{filter_name}")

    candidates.append(f"{metric_name}_stderr")

    for candidate in candidates:
        value = metrics.get(candidate)
        if isinstance(value, int | float):
            return float(value)

    return None


def main() -> None:
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    output_csv = Path(args.output_csv)
    output_markdown = Path(args.output_markdown)

    result_path = find_latest_result_json(eval_dir)
    data = load_json(result_path)

    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"Could not find 'results' dict in {result_path}")

    rows: list[dict[str, Any]] = []

    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue

        for metric_key, value in task_metrics.items():
            if not is_metric_value(metric_key, value):
                continue

            metric_name, filter_name = split_metric_key(metric_key)

            rows.append(
                {
                    "task": task_name,
                    "metric": metric_name,
                    "filter": filter_name,
                    "value": float(value),
                    "stderr": find_stderr(task_metrics, metric_key),
                    "source_json": str(result_path),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["task", "metric", "filter"], na_position="last")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    markdown = df[["task", "metric", "filter", "value", "stderr"]].to_markdown(index=False)
    output_markdown.write_text(markdown + "\n", encoding="utf-8")

    print(f"Parsed: {result_path}")
    print()
    print(markdown)
    print()
    print(f"Saved CSV: {output_csv}")
    print(f"Saved Markdown: {output_markdown}")


if __name__ == "__main__":
    main()
