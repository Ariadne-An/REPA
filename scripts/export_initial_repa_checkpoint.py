#!/usr/bin/env python3
"""
Export an "untrained" SD15 + REPA checkpoint (LoRA + AlignHead initial state).

This is useful for evaluation baselines: the exported file contains the
freshly initialised SD15UNetAligned state_dict before any training updates.

Usage:
    python scripts/export_initial_repa_checkpoint.py \
        --config configs/sd15_repa_档A.yaml \
        --output checkpoints/sd15_repa_init.pt \
        --base-model-dir checkpoints/sd15_base
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from models.sd15_unet_aligned import SD15UNetAligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export initial SD15+REPA checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config with align/LoRA settings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output .pt file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional path to save metadata JSON (defaults to <output>.json).",
    )
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default=None,
        help="Optional local snapshot path for SD1.5 weights (uses HF cache if omitted).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for initialisation (only matters for xFormers probing).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    pretrained_path = args.base_model_dir or config.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")

    model = SD15UNetAligned(
        pretrained_model_name=pretrained_path,
        align_layers=config.get("align_layers", ["mid"]),
        dino_dim=config.get("dino_D", 1024),
        use_lora=config.get("use_lora", True),
        lora_rank=config.get("lora_rank", 8),
        lora_targets=config.get("lora_targets", "attn"),
        lora_alpha=config.get("lora_alpha", config.get("lora_rank", 8)),
        device=args.device,
    )
    model = model.to("cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".json")
    meta = {
        "config": args.config,
        "pretrained_model": pretrained_path,
        "align_layers": config.get("align_layers", ["mid"]),
        "lora_rank": config.get("lora_rank", 8),
        "lora_targets": config.get("lora_targets", "attn"),
    }
    metadata_path.write_text(json.dumps(meta, indent=2))

    print(f"✅ Exported initial checkpoint to {output_path}")
    print(f"ℹ️  Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

