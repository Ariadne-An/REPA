#!/usr/bin/env python3
"""
Utility to visualize metrics stored in a JSONL training log produced by REPA.

Example:
    python scripts/visualize_training_log.py \
        --input logs/trackA_h200_bs128_bf16.jsonl \
        --output logs/trackA_h200_bs128_bf16.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict]:
    """Load the JSONL log into a list of dicts."""
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_no}: {exc}") from exc
            metrics = obj.get("metrics", {})
            record = {"step": obj.get("step"), "phase": obj.get("phase")}
            record.update(metrics)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def build_series(records: List[Dict], phase_filter: str | None = "train") -> Dict[str, List[float]]:
    """Return per-metric series filtered by phase and ordered by step."""
    filtered = [
        r for r in records
        if phase_filter is None or r.get("phase") == phase_filter
    ]
    if not filtered:
        raise ValueError(f"No records found for phase={phase_filter!r}")
    filtered.sort(key=lambda r: r.get("step", 0))

    metric_keys: List[str] = sorted(
        {k for r in filtered for k in r.keys() if k not in {"phase"}}
    )
    series: Dict[str, List[float]] = {key: [] for key in metric_keys}

    for record in filtered:
        for key in metric_keys:
            if key == "phase":
                continue
            val = record.get(key)
            if val is None:
                val = math.nan
            series[key].append(val)
    return series


def pick_present(series: Dict[str, List[float]], candidates: Iterable[str]) -> List[str]:
    """Return the subset of candidate keys that exist in the series."""
    return [key for key in candidates if key in series]


def plot_series(series: Dict[str, List[float]], output: Path) -> None:
    """Create subplots for the most relevant metrics (single phase view)."""
    steps = series.get("step")
    if not steps:
        raise ValueError("Series does not contain 'step' values to use as the x-axis.")

    plots = []
    loss_keys = pick_present(
        series,
        ["loss/total_avg", "loss/diffusion_avg", "loss/token_avg", "loss/manifold_avg"],
    )
    if loss_keys:
        plots.append(("Loss terms", loss_keys))

    val_loss_keys = pick_present(
        series,
        ["val/total", "val/diffusion", "val/token", "val/manifold"],
    )
    if val_loss_keys:
        plots.append(("Validation loss terms", val_loss_keys))

    lr_keys = pick_present(series, ["lr"])
    if lr_keys:
        plots.append(("Learning rate", lr_keys))

    time_keys = pick_present(series, ["train/step_time_s_avg"])
    if time_keys:
        plots.append(("Step time (s)", time_keys))

    val_repeat_keys = pick_present(series, ["val/repeats"])
    if val_repeat_keys:
        plots.append(("Validation repeats", val_repeat_keys))

    if not plots:
        raise ValueError("No recognized metrics found to plot.")

    cols = 2 if len(plots) > 1 else 1
    rows = math.ceil(len(plots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False, sharex=True)
    axes_flat = axes.flatten()

    for ax, (title, keys) in zip(axes_flat, plots):
        for key in keys:
            ax.plot(steps, series[key], label=key)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide any unused axes.
    for ax in axes_flat[len(plots):]:
        ax.axis("off")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def plot_train_val_loss_pairs(
    train_series: Dict[str, List[float]],
    val_series: Dict[str, List[float]],
    output: Path,
) -> None:
    """Create four subplots comparing train/val losses for each loss type."""
    train_steps = train_series.get("step")
    val_steps = val_series.get("step")
    if not train_steps or not val_steps:
        raise ValueError("Both train and validation series must include 'step' keys.")

    loss_pairs: List[Tuple[str, str, str]] = [
        ("Total Loss", "loss/total_avg", "val/total"),
        ("Diffusion Loss", "loss/diffusion_avg", "val/diffusion"),
        ("Token Loss", "loss/token_avg", "val/token"),
        ("Manifold Loss", "loss/manifold_avg", "val/manifold"),
    ]

    missing = [
        title
        for title, train_key, val_key in loss_pairs
        if train_key not in train_series or val_key not in val_series
    ]
    if missing:
        raise ValueError(f"Missing metrics for: {', '.join(missing)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, squeeze=False)
    axes_flat = axes.flatten()
    for ax, (title, train_key, val_key) in zip(axes_flat, loss_pairs):
        ax.plot(train_steps, train_series[train_key], label="train", color="#1f77b4")
        ax.plot(val_steps, val_series[val_key], label="val", color="#d62728")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Train vs Validation Losses", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a REPA JSONL training log.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the JSONL log file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination for the figure (defaults to the input file name with .png).",
    )
    parser.add_argument(
        "--mode",
        choices=["phase", "loss_compare"],
        default="loss_compare",
        help="Visualization mode: per-phase metrics or combined loss comparison.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        help="Phase to filter on for --mode phase (set to 'None' to include everything).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".png")
    records = read_jsonl(args.input)
    if args.mode == "loss_compare":
        train_series = build_series(records, phase_filter="train")
        val_series = build_series(records, phase_filter="val")
        plot_train_val_loss_pairs(train_series, val_series, output=output)
    else:
        phase_filter = None if args.phase.lower() == "none" else args.phase
        series = build_series(records, phase_filter=phase_filter)
        plot_series(series, output=output)


if __name__ == "__main__":
    main()
