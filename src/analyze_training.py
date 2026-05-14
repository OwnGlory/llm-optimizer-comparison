from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze training logs.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--expected_steps", type=int, default=None)
    parser.add_argument("--warmup_ignore_steps", type=int, default=5)
    parser.add_argument("--memory_warn_ratio", type=float, default=0.90)
    parser.add_argument("--loss_window", type=int, default=20)
    parser.add_argument("--rolling_window", type=int, default=20)
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return pd.DataFrame.from_records(records)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def get_total_gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None

    return torch.cuda.get_device_properties(0).total_memory / 1024**2


def safe_exp(value: float) -> float:
    if value > 80:
        return math.inf
    return float(math.exp(value))


def make_line_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df[x_column], df[y_column], marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_loss_plot(df: pd.DataFrame, output_path: Path, rolling_window: int) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["loss"], marker="o", linewidth=1, label="loss")

    rolling = df["loss"].rolling(window=rolling_window, min_periods=1).mean()
    plt.plot(
        df["step"],
        rolling,
        linewidth=2,
        label=f"rolling mean, window={rolling_window}",
    )

    plt.title("Training loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_rolling_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    rolling_window: int,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df[x_column], df[y_column], marker="o", linewidth=1, alpha=0.55, label=y_column)

    rolling = df[y_column].rolling(window=rolling_window, min_periods=1).mean()
    plt.plot(
        df[x_column],
        rolling,
        linewidth=2,
        label=f"rolling mean, window={rolling_window}",
    )

    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_perplexity_plot(df: pd.DataFrame, output_path: Path, rolling_window: int) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["perplexity"], marker="o", linewidth=1, alpha=0.55, label="perplexity")

    rolling = df["perplexity"].rolling(window=rolling_window, min_periods=1).mean()
    plt.plot(
        df["step"],
        rolling,
        linewidth=2,
        label=f"rolling mean, window={rolling_window}",
    )

    plt.title("Training perplexity")
    plt.xlabel("step")
    plt.ylabel("perplexity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"

    train_log_path = logs_dir / "train.jsonl"
    summary_path = metrics_dir / "train_summary.json"

    checks: dict[str, Any] = {}

    checks["train_log_exists"] = train_log_path.exists()
    checks["summary_exists"] = summary_path.exists()

    if not checks["train_log_exists"]:
        raise FileNotFoundError(f"Missing train log: {train_log_path}")

    if not checks["summary_exists"]:
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    df = read_jsonl(train_log_path)
    summary = load_json(summary_path)

    if "loss" not in df.columns:
        raise ValueError("train.jsonl does not contain 'loss' column.")

    df["perplexity"] = df["loss"].apply(lambda value: safe_exp(float(value)))

    expected_steps = args.expected_steps
    if expected_steps is None:
        expected_steps = int(summary.get("max_steps", len(df)))

    checks["num_log_rows"] = int(len(df))
    checks["expected_steps"] = int(expected_steps)
    checks["has_expected_number_of_rows"] = bool(len(df) == expected_steps)

    checks["loss_is_finite"] = bool(np.isfinite(df["loss"]).all())
    checks["lr_is_finite"] = bool(np.isfinite(df["lr"]).all())
    checks["step_time_is_positive"] = bool((df["step_time_sec"] > 0).all())
    checks["tokens_per_sec_is_positive"] = bool((df["tokens_per_sec"] > 0).all())

    steady_df = df[df["step"] > args.warmup_ignore_steps].copy()
    if len(steady_df) == 0:
        steady_df = df.copy()

    mean_step_time = float(steady_df["step_time_sec"].mean())
    median_step_time = float(steady_df["step_time_sec"].median())
    std_step_time = float(steady_df["step_time_sec"].std(ddof=0))
    cv_step_time = float(std_step_time / mean_step_time) if mean_step_time > 0 else math.inf
    p95_step_time = float(steady_df["step_time_sec"].quantile(0.95))

    checks["mean_step_time_sec_excluding_warmup"] = mean_step_time
    checks["median_step_time_sec_excluding_warmup"] = median_step_time
    checks["std_step_time_sec_excluding_warmup"] = std_step_time
    checks["cv_step_time_excluding_warmup"] = cv_step_time
    checks["p95_step_time_sec_excluding_warmup"] = p95_step_time
    checks["step_time_stable_cv_lt_0_20"] = bool(cv_step_time < 0.20)

    checks["mean_tokens_per_sec_excluding_warmup"] = float(steady_df["tokens_per_sec"].mean())
    checks["median_tokens_per_sec_excluding_warmup"] = float(steady_df["tokens_per_sec"].median())

    max_memory_allocated_mb = float(df["max_memory_allocated_mb"].max())
    max_memory_reserved_mb = float(df["max_memory_reserved_mb"].max())
    total_gpu_memory_mb = get_total_gpu_memory_mb()

    checks["max_memory_allocated_mb"] = max_memory_allocated_mb
    checks["max_memory_reserved_mb"] = max_memory_reserved_mb
    checks["total_gpu_memory_mb"] = total_gpu_memory_mb

    if total_gpu_memory_mb is not None:
        memory_ratio = max_memory_allocated_mb / total_gpu_memory_mb
        checks["max_memory_allocated_ratio"] = float(memory_ratio)
        checks["memory_below_warn_ratio"] = bool(memory_ratio < args.memory_warn_ratio)
    else:
        checks["max_memory_allocated_ratio"] = None
        checks["memory_below_warn_ratio"] = None

    loss_window = min(args.loss_window, len(df))
    initial_loss_mean = float(df["loss"].head(loss_window).mean())
    final_loss_mean = float(df["loss"].tail(loss_window).mean())
    delta_loss = final_loss_mean - initial_loss_mean
    relative_delta_loss_percent = (
        100.0 * delta_loss / initial_loss_mean if initial_loss_mean != 0 else math.nan
    )

    initial_perplexity = safe_exp(initial_loss_mean)
    final_perplexity = safe_exp(final_loss_mean)
    delta_perplexity = final_perplexity - initial_perplexity
    relative_delta_perplexity_percent = (
        100.0 * delta_perplexity / initial_perplexity
        if initial_perplexity not in {0.0, math.inf}
        else math.nan
    )

    checks["loss_window"] = int(loss_window)
    checks["initial_loss_mean"] = initial_loss_mean
    checks["final_loss_mean"] = final_loss_mean
    checks["delta_loss"] = delta_loss
    checks["relative_delta_loss_percent"] = relative_delta_loss_percent
    checks["initial_perplexity"] = initial_perplexity
    checks["final_perplexity"] = final_perplexity
    checks["delta_perplexity"] = delta_perplexity
    checks["relative_delta_perplexity_percent"] = relative_delta_perplexity_percent

    checks["min_loss"] = float(df["loss"].min())
    checks["max_loss"] = float(df["loss"].max())
    checks["min_perplexity"] = float(df["perplexity"].min())
    checks["max_perplexity"] = float(df["perplexity"].max())

    loss_exploded = (
        not checks["loss_is_finite"]
        or final_loss_mean > 2.0 * initial_loss_mean
        or checks["max_loss"] > 3.0 * initial_loss_mean
    )
    checks["loss_did_not_explode"] = bool(not loss_exploded)

    checks["overall_pass"] = bool(
        checks["train_log_exists"]
        and checks["summary_exists"]
        and checks["has_expected_number_of_rows"]
        and checks["loss_is_finite"]
        and checks["lr_is_finite"]
        and checks["step_time_is_positive"]
        and checks["tokens_per_sec_is_positive"]
        and checks["step_time_stable_cv_lt_0_20"]
        and checks["loss_did_not_explode"]
        and (checks["memory_below_warn_ratio"] is not False)
    )

    figures_dir.mkdir(parents=True, exist_ok=True)

    make_loss_plot(df, figures_dir / "loss_vs_step.png", rolling_window=args.rolling_window)
    make_perplexity_plot(
        df,
        figures_dir / "perplexity_vs_step.png",
        rolling_window=args.rolling_window,
    )
    make_line_plot(
        df,
        x_column="step",
        y_column="lr",
        output_path=figures_dir / "lr_vs_step.png",
        title="Learning rate schedule",
        ylabel="learning rate",
    )
    make_rolling_plot(
        df,
        x_column="step",
        y_column="step_time_sec",
        output_path=figures_dir / "step_time_vs_step.png",
        title="Step time",
        ylabel="seconds",
        rolling_window=args.rolling_window,
    )
    make_rolling_plot(
        df,
        x_column="step",
        y_column="tokens_per_sec",
        output_path=figures_dir / "tokens_per_sec_vs_step.png",
        title="Training throughput",
        ylabel="tokens/sec",
        rolling_window=args.rolling_window,
    )
    make_line_plot(
        df,
        x_column="step",
        y_column="max_memory_allocated_mb",
        output_path=figures_dir / "max_memory_vs_step.png",
        title="Max CUDA memory allocated",
        ylabel="MB",
    )

    report_path = metrics_dir / "validation_report.json"
    save_json(report_path, checks)

    print("\n=== Validation report ===")
    for key, value in checks.items():
        print(f"{key}: {value}")

    print("\nSaved figures to:")
    for path in sorted(figures_dir.glob("*.png")):
        print(f"  {path}")

    print(f"\nSaved validation report to: {report_path}")


if __name__ == "__main__":
    main()
