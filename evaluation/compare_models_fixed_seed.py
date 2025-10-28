#!/usr/bin/env python3
"""
Compare models with FIXED seeds for fair comparison.
Generate images from SD1.5 base, REPA init, and REPA trained using identical seeds.
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
import pandas as pd


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_imagenet_prompts(csv_path, num_samples=300):
    """Load ImageNet prompts from CSV."""
    df = pd.read_csv(csv_path)
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    # Use class_name as prompt (format: "tench, Tinca tinca")
    return df['class_name'].tolist()


def generate_images(
    model_path: str,
    prompts: list,
    output_dir: Path,
    base_seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = "cuda",
):
    """Generate images with fixed seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Loading model: {model_path}")
    print(f"{'='*80}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    print(f"Generating {len(prompts)} images with fixed seeds...")

    for idx, prompt in enumerate(tqdm(prompts)):
        # Use same seed for each image index across all models
        seed = base_seed + idx
        set_seed(seed)

        # Generate with fixed seed
        generator = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Save image
        output_path = output_dir / f"{idx:06d}.png"
        image.save(output_path)

    print(f"✓ Saved {len(prompts)} images to {output_dir}")

    # Clean up
    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="eval_outputs/comparison_fixed_seed")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--sd15-base", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--repa-init", type=str, default="converted/sd15_repa_init")
    parser.add_argument("--repa-trained", type=str, default="converted/sd15_repa_step24k")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    # Load prompts
    print(f"Loading {args.num_samples} prompts from {args.csv_path}...")
    prompts = load_imagenet_prompts(args.csv_path, args.num_samples)

    # Save prompts for reference
    prompts_file = output_root / "prompts.json"
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    with prompts_file.open("w") as f:
        json.dump({"prompts": prompts, "base_seed": args.base_seed}, f, indent=2)
    print(f"✓ Saved prompts to {prompts_file}")

    # Generate images for each model
    models = [
        ("sd15_base", args.sd15_base),
        ("sd15_repa_init", args.repa_init),
        ("sd15_repa_trained", args.repa_trained),
    ]

    for model_name, model_path in models:
        output_dir = output_root / "samples" / model_name
        generate_images(
            model_path=model_path,
            prompts=prompts,
            output_dir=output_dir,
            base_seed=args.base_seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )

    print(f"\n{'='*80}")
    print("✅ All models completed!")
    print(f"{'='*80}")
    print(f"Images saved to: {output_root}/samples/")
    print(f"Compare images with same index (e.g., 000042.png) across different models")


if __name__ == "__main__":
    main()
