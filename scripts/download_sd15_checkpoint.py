#!/usr/bin/env python3
"""
Download the vanilla Stable Diffusion 1.5 pipeline weights for offline use.

Usage:
    python scripts/download_sd15_checkpoint.py \
        --repo runwayml/stable-diffusion-v1-5 \
        --output-dir checkpoints/sd15_base
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SD1.5 base checkpoint.")
    parser.add_argument(
        "--repo",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face repo id for Stable Diffusion.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional git revision/commit/tag to pin.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to store the downloaded snapshot.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns to limit files (e.g. 'unet/*' 'vae/*').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
    )

    print(f"✅ Stable Diffusion checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()

