"""
Generate images directly from REPA checkpoint without converting to full SD1.5 pipeline.
This saves disk space by not creating 8GB converted models.
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
from diffusers import StableDiffusionPipeline
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from collections import defaultdict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_merge_lora(checkpoint_path, use_ema=False):
    """Load checkpoint and merge LoRA weights into a state_dict."""
    checkpoint_path = Path(checkpoint_path)

    # Load EMA or regular weights
    if use_ema:
        ema_path = checkpoint_path.parent / "ema.pt"
        print(f"Loading EMA weights from {ema_path}")
        ema_dict = torch.load(ema_path, map_location="cpu")
        state_dict = ema_dict["shadow"]
    else:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = safe_load_file(checkpoint_path)

    # Separate LoRA and base weights
    lora_dict = {}
    base_dict = {}

    for key, value in state_dict.items():
        if "lora_" in key:
            lora_dict[key] = value
        elif not key.startswith("align_head."):
            base_dict[key] = value

    print(f"Found {len(lora_dict)} LoRA keys, {len(base_dict)} base keys")

    # Group LoRA weights by layer
    lora_groups = defaultdict(dict)
    for key, value in lora_dict.items():
        parts = key.split(".")
        # Find lora_A or lora_B
        for i, part in enumerate(parts):
            if part.startswith("lora_"):
                base_name = ".".join(parts[:i])
                param_type = part  # lora_A or lora_B
                lora_groups[base_name][param_type] = value
                break

    # Merge LoRA into base weights
    merged_dict = base_dict.copy()

    for base_name, lora_params in lora_groups.items():
        if "lora_A" in lora_params and "lora_B" in lora_params:
            lora_A = lora_params["lora_A"]
            lora_B = lora_params["lora_B"]

            # LoRA scaling: typically alpha/rank, but here alpha=rank=8, so scale=1.0
            scale = 1.0

            # Merge: W' = W + scale * (B @ A)
            delta = (lora_B @ lora_A) * scale

            if base_name in merged_dict:
                merged_dict[base_name] = merged_dict[base_name] + delta.to(merged_dict[base_name].dtype)

    print(f"Merged LoRA weights, final dict has {len(merged_dict)} keys")
    return merged_dict


def generate_images(checkpoint_path, output_dir, num_samples=300, use_ema=True,
                   base_seed=42, num_inference_steps=50, guidance_scale=7.5):
    """Generate images from checkpoint."""

    # Load prompts - use random sampling to ensure class diversity
    df = pd.read_csv('/workspace/data/val_50k.csv')
    # Sample randomly to get diverse classes
    df_sampled = df.sample(n=num_samples, random_state=base_seed)
    prompts = df_sampled['class_name'].tolist()
    print(f"Loaded {len(prompts)} prompts from {df['class_id'].nunique()} classes")
    print(f"Sampled prompts span {df_sampled['class_id'].nunique()} unique classes")

    # Load merged weights
    merged_dict = load_and_merge_lora(checkpoint_path, use_ema=use_ema)

    # Load base SD1.5 pipeline
    print("Loading base SD1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Replace UNet weights
    print("Replacing UNet weights...")
    unet_state = pipe.unet.state_dict()
    for key in merged_dict.keys():
        if key in unet_state:
            unet_state[key] = merged_dict[key].to(torch.float16)

    pipe.unet.load_state_dict(unet_state)
    pipe = pipe.to('cuda')

    # Generate images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} images...")
    for idx, prompt in enumerate(tqdm(prompts)):
        seed = base_seed + idx
        set_seed(seed)
        generator = torch.Generator(device='cuda').manual_seed(seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        output_path = output_dir / f'{idx:06d}.png'
        image.save(output_path)

    print(f"Saved {num_samples} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    generate_images(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_ema=args.use_ema,
        base_seed=args.base_seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
