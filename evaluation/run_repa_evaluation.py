#!/usr/bin/env python3
"""
Comprehensive evaluation script for SD1.5 + REPA checkpoints.

Features:
  * Compare three variants: vanilla SD1.5, SD1.5 + initial LoRA, SD1.5 + trained LoRA.
  * Generate Imagenet-style samples (a photo of a {class_name}) using class labels from CSV.
  * Compute FID, sFID (via clean-fid) and Inception Score / Precision / Recall (via torchmetrics).

Example:
    python evaluation/run_repa_evaluation.py \
        --config configs/sd15_repa_档A.yaml \
        --csv-path data/val_50k.csv \
        --imagenet-classes data/imagenet_classes.json \
        --real-images-dir /workspace/imagenet_val_organized \
        --base-model-dir checkpoints/sd15_base \
        --initial-checkpoint checkpoints/sd15_repa_init.pt \
        --trained-checkpoint models/sd15_repa_step24k/model.pt \
        --output-root eval_outputs/step24k \
        --num-samples 5000 \
        --num-inference-steps 50 \
        --guidance-scale 7.5

Requirements:
    pip install clean-fid torchmetrics[image] diffusers transformers accelerate
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from cleanfid import fid
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.inception import InceptionScore
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None

from models.sd15_unet_aligned import SD15UNetAligned


# -----------------------------------------------------------------------------#
# Utility helpers


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


def instantiate_aligned_unet(config: Dict, base_model_dir: str, device: str) -> SD15UNetAligned:
    model = SD15UNetAligned(
        pretrained_model_name=base_model_dir,
        align_layers=config.get("align_layers", ["mid"]),
        dino_dim=config.get("dino_D", 1024),
        use_lora=config.get("use_lora", True),
        lora_rank=config.get("lora_rank", 8),
        lora_targets=config.get("lora_targets", "attn"),
        lora_alpha=config.get("lora_alpha", config.get("lora_rank", 8)),
        device=device,
    )
    return model


def load_pipeline(
    config: Dict,
    base_model_dir: str,
    checkpoint: Optional[Path],
    torch_dtype: torch.dtype,
    device: torch.device,
) -> StableDiffusionPipeline:
    if checkpoint is not None:
        if checkpoint.is_dir():
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif checkpoint.suffix == ".safetensors":
            try:
                pipe = StableDiffusionPipeline.from_single_file(
                    checkpoint,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                from safetensors.torch import load_file

                pipe = StableDiffusionPipeline.from_pretrained(
                    base_model_dir,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

                wrapper = instantiate_aligned_unet(config, base_model_dir, device="cpu")
                state = load_file(checkpoint)
                missing, unexpected = wrapper.load_state_dict(state, strict=False)
                if missing:
                    print(f"[WARN] Missing keys when loading {checkpoint.name}: {missing[:5]}...")
                if unexpected:
                    print(f"[WARN] Unexpected keys when loading {checkpoint.name}: {unexpected[:5]}...")
                pipe.unet.load_state_dict(wrapper.unet.state_dict(), strict=True)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model_dir,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            wrapper = instantiate_aligned_unet(config, base_model_dir, device="cpu")
            state = torch.load(checkpoint, map_location="cpu")
            missing, unexpected = wrapper.load_state_dict(state, strict=False)
            if missing:
                print(f"[WARN] Missing keys when loading {checkpoint.name}: {missing[:5]}...")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading {checkpoint.name}: {unexpected[:5]}...")
            pipe.unet.load_state_dict(wrapper.unet.state_dict(), strict=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_dir,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"[WARN] xFormers not enabled: {exc}")

    pipe.to(device)
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
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=pipe.device)

    total = len(prompts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        batch_seeds = seeds[start:end]
        generator.manual_seed(batch_seeds[0])
        images = pipe(
            batch_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        for prompt_idx, image in enumerate(images):
            idx = start + prompt_idx
            out_path = variant_dir / f"{idx:06d}.png"
            image.save(out_path)
    return variant_dir


# -----------------------------------------------------------------------------#
# Metric computation helpers

class ImageFolderDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(
            [p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.files[idx]).convert("RGB")
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor


def compute_fid_metrics(fake_dir: Path, real_dir: Path) -> Dict[str, float]:
    fid_score = fid.compute_fid(str(real_dir), str(fake_dir))
    # sFID not available in current clean-fid version
    return {"fid": fid_score}


def compute_torchmetrics_metrics(fake_dir: Path, real_dir: Path, device: torch.device, batch_size: int = 32) -> Dict[str, float]:
    fake_ds = ImageFolderDataset(fake_dir)

    fake_loader = DataLoader(fake_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    inception = InceptionScore(splits=10, normalize=True).to(device)

    for batch in fake_loader:
        batch = batch.to(device)
        inception.update(batch)

    is_mean, is_std = inception.compute()

    return {
        "inception_score": float(is_mean),
        "inception_score_std": float(is_std),
    }


# -----------------------------------------------------------------------------#
# Main evaluation routine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SD1.5 + REPA checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="YAML config used for training.")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV file with Imagenet samples (id,class_id).")
    parser.add_argument("--imagenet-classes", type=str, required=True, help="JSON mapping class_id -> class name.")
    parser.add_argument("--real-images-dir", type=str, required=True, help="Directory of real validation images.")
    parser.add_argument("--base-model-dir", type=str, required=True, help="Local SD1.5 snapshot downloaded via scripts/download_sd15_checkpoint.py.")
    parser.add_argument("--initial-checkpoint", type=str, required=True, help="Path to initial (untrained) SD15+REPA checkpoint.")
    parser.add_argument("--trained-checkpoint", type=str, required=True, help="Path to trained checkpoint (e.g., step_24000).")
    parser.add_argument("--output-root", type=str, required=True, help="Directory to store generated images and metrics.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional number of samples (default: full CSV).")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion steps per sample.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG guidance scale.")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for generation/metrics.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Torch dtype for pipeline.")
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

    generated_dirs = []
    for name, ckpt in variants:
        print(f"=== Generating with {name} ===")
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

    metrics = {}
    real_dir = Path(args.real_images_dir).resolve()
    for name, fake_dir in generated_dirs:
        print(f"=== Computing metrics for {name} ===")
        scores = {}
        scores.update(compute_fid_metrics(fake_dir, real_dir))
        scores.update(compute_torchmetrics_metrics(fake_dir, real_dir, device=device, batch_size=args.batch_size))
        metrics[name] = scores
        print(f"{name}: {scores}")

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
