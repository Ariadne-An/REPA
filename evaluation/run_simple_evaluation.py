#!/usr/bin/env python3
"""
Simplified evaluation script for SD1.5 + REPA checkpoints.
Uses torch-fidelity for all metrics computation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
from torch_fidelity import calculate_metrics

import sys
sys.path.insert(0, '/workspace/REPA')
from models.sd15_unet_aligned import SD15UNetAligned


def load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_class_names(path: Path) -> Dict[int, str]:
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    return {idx: name for idx, name in enumerate(data)}


def prepare_prompts(
    csv_path: Path,
    class_names: Dict[int, str],
    num_samples: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path)
    if num_samples is not None and num_samples < len(df):
        df = df.sample(num_samples, random_state=seed).reset_index(drop=True)
    prompts, seeds = [], []
    rng = random.Random(seed)
    for _, row in df.iterrows():
        class_id = int(row["class_id"])
        prompt = f"a photo of a {class_names[class_id]}"
        prompts.append(prompt)
        seeds.append(rng.randint(0, 2**31 - 1))
    return prompts, seeds


def load_pipeline(
    config: Dict,
    base_model_dir: str,
    checkpoint_path: Optional[Path],
    torch_dtype: torch.dtype,
    device: str,
) -> StableDiffusionPipeline:
    """Load SD pipeline with optional REPA checkpoint."""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_dir,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if checkpoint_path is not None:
        if checkpoint_path.is_dir():
            # Diffusers format - trained checkpoint with merged LoRA
            # Load entire pipeline and replace unet directly
            print(f"📥 Loading trained model from: {checkpoint_path}")
            trained_pipe = StableDiffusionPipeline.from_pretrained(str(checkpoint_path))
            pipe.unet = trained_pipe.unet
            print("✅ Loaded trained U-Net")
            del trained_pipe
        else:
            # PyTorch checkpoint - aligned U-Net with LoRA weights
            aligned_unet = SD15UNetAligned(
                pretrained_model_name=base_model_dir,
                align_layers=config.get("align_layers", ["mid"]),
                dino_dim=config.get("dino_D", 1024),
                use_lora=config.get("use_lora", True),
                lora_rank=config.get("lora_rank", 8),
                lora_targets=config.get("lora_targets", "attn"),
                lora_alpha=config.get("lora_alpha", config.get("lora_rank", 8)),
                device="cpu",
            )

            state_dict = torch.load(checkpoint_path, map_location="cpu")
            result = aligned_unet.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded checkpoint ({len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys)")
            pipe.unet = aligned_unet.unet

    pipe = pipe.to(device)
    # Ensure all components have correct dtype
    pipe.unet = pipe.unet.to(torch_dtype)
    pipe.vae = pipe.vae.to(torch_dtype)
    pipe.text_encoder = pipe.text_encoder.to(torch_dtype)
    if hasattr(pipe.unet, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
        except:
            pass

    return pipe


def generate_images(
    pipe: StableDiffusionPipeline,
    name: str,
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    batch_size: int,
) -> Path:
    """Generate images and save to directory."""
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(prompts)} images to {variant_dir}")

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Gen {name}"):
        batch_prompts = prompts[i:i+batch_size]
        batch_seeds = seeds[i:i+batch_size]

        # Generate
        with torch.no_grad():
            images = pipe(
                batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=[torch.Generator(device=pipe.device).manual_seed(s) for s in batch_seeds],
            ).images

        # Save
        for j, img in enumerate(images):
            img.save(variant_dir / f"img_{i+j:05d}.png")

    return variant_dir


def compute_metrics_with_torch_fidelity(
    fake_dir: Path,
    real_dir: Path,
    batch_size: int = 32,
) -> Dict:
    """Compute all metrics using torch-fidelity."""
    print(f"Computing metrics: FID, IS, Precision, Recall")

    # Check if real dir has images
    real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    if len(real_images) == 0:
        print("⚠️  No real images found, skipping FID/Precision/Recall")
        metrics = calculate_metrics(
            input1=str(fake_dir),
            isc=True,
            batch_size=batch_size,
        )
        return {
            "inception_score": float(metrics.get("inception_score_mean", 0)),
            "inception_score_std": float(metrics.get("inception_score_std", 0)),
        }

    # Compute all metrics
    metrics = calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        fid=True,
        isc=True,
        prc=True,
        batch_size=batch_size,
    )

    return {
        "fid": float(metrics.get("frechet_inception_distance", -1)),
        "inception_score": float(metrics.get("inception_score_mean", 0)),
        "inception_score_std": float(metrics.get("inception_score_std", 0)),
        "precision": float(metrics.get("precision", 0)),
        "recall": float(metrics.get("recall", 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SD1.5 + REPA checkpoints.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--imagenet-classes", type=str, required=True)
    parser.add_argument("--real-images-dir", type=str, default=None)
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument("--initial-checkpoint", type=str, required=True)
    parser.add_argument("--trained-checkpoint", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    class_names = load_class_names(Path(args.imagenet_classes))
    prompts, seeds = prepare_prompts(Path(args.csv_path), class_names, args.num_samples, args.seed)

    device = torch.device(args.device)
    torch_dtype = str_to_dtype(args.dtype)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("sd15_base", None),
        ("sd15_repa_init", Path(args.initial_checkpoint)),
        ("sd15_repa_trained", Path(args.trained_checkpoint)),
    ]

    # Generate images for all variants
    generated_dirs = []
    for name, ckpt in variants:
        print(f"\n{'='*60}")
        print(f"=== Generating with {name} ===")
        print(f"{'='*60}")
        pipe = load_pipeline(config, args.base_model_dir, ckpt, torch_dtype, device)
        variant_dir = generate_images(
            pipe=pipe,
            name=name,
            prompts=prompts,
            seeds=seeds,
            output_dir=output_root / "samples",
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
        )
        generated_dirs.append((name, variant_dir))
        pipe.to("cpu")
        del pipe
        torch.cuda.empty_cache()

    # Compute metrics for all variants
    metrics = {}
    real_dir = Path(args.real_images_dir) if args.real_images_dir else output_root / "real_images_placeholder"

    print(f"\n{'='*60}")
    print("=== Computing Metrics ===")
    print(f"{'='*60}")

    for name, fake_dir in generated_dirs:
        print(f"\n--- Metrics for {name} ---")
        scores = compute_metrics_with_torch_fidelity(fake_dir, real_dir, batch_size=32)
        metrics[name] = scores
        print(json.dumps(scores, indent=2))

    # Save results
    results_path = output_root / "metrics.json"
    with results_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"📊 Results saved to {results_path}")
    print(f"🖼️  Generated images in {output_root / 'samples'}")
    print(f"{'='*60}\n")

    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Model':<25} {'FID':<10} {'IS':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    for name, scores in metrics.items():
        fid = scores.get('fid', -1)
        isc = scores.get('inception_score', 0)
        prec = scores.get('precision', 0)
        recall = scores.get('recall', 0)
        fid_str = f"{fid:.2f}" if fid >= 0 else "N/A"
        print(f"{name:<25} {fid_str:<10} {isc:<12.2f} {prec:<12.3f} {recall:<12.3f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
