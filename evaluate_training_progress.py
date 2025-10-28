"""
Comprehensive evaluation script for REPA checkpoint comparison.

Evaluates multiple checkpoints (6k, 12k, 18k, 24k) with:
1. FID (Fréchet Inception Distance)
2. sFID (Spatial FID)
3. IS (Inception Score)
4. Integrates with pre-computed CKNNA metrics

Usage:
    python evaluate_training_progress.py \
        --checkpoints 6k:exps/trackA_h200_bs128_bf16/step_006000/model.safetensors \
                      12k:exps/trackA_h200_bs128_bf16/step_012000/model.safetensors \
                      18k:exps/trackA_h200_bs128_bf16/step_018000/model.safetensors \
                      24k:exps/trackA_h200_bs128_bf16/step_024000/model.safetensors \
        --output-dir eval_outputs/training_progress \
        --num-samples 300 \
        --use-ema \
        --real-images-dir /workspace/data/val_images_512
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file as safe_load_file
from collections import defaultdict
from cleanfid import fid
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.stats import entropy
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_merge_lora(checkpoint_path, use_ema=False):
    """Load checkpoint and merge LoRA weights."""
    checkpoint_path = Path(checkpoint_path)

    # Load EMA or regular weights
    if use_ema:
        ema_path = checkpoint_path.parent / "ema.pt"
        print(f"  Loading EMA weights from {ema_path}")
        ema_dict = torch.load(ema_path, map_location="cpu", weights_only=False)
        state_dict = ema_dict["shadow"]
    else:
        print(f"  Loading checkpoint from {checkpoint_path}")
        state_dict = safe_load_file(checkpoint_path)

    # Separate LoRA and base weights
    lora_dict = {}
    base_dict = {}

    for key, value in state_dict.items():
        if "lora_" in key:
            lora_dict[key] = value
        elif not key.startswith("align_head."):
            base_dict[key] = value

    print(f"  Found {len(lora_dict)} LoRA keys, {len(base_dict)} base keys")

    # Group LoRA weights by layer
    lora_groups = defaultdict(dict)
    for key, value in lora_dict.items():
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part.startswith("lora_"):
                base_name = ".".join(parts[:i])
                param_type = part
                lora_groups[base_name][param_type] = value
                break

    # Merge LoRA into base weights
    merged_dict = base_dict.copy()

    for base_name, lora_params in lora_groups.items():
        if "lora_A" in lora_params and "lora_B" in lora_params:
            lora_A = lora_params["lora_A"]
            lora_B = lora_params["lora_B"]
            scale = 1.0
            delta = (lora_B @ lora_A) * scale

            if base_name in merged_dict:
                merged_dict[base_name] = merged_dict[base_name] + delta.to(merged_dict[base_name].dtype)

    print(f"  Merged LoRA weights, final dict has {len(merged_dict)} keys")
    return merged_dict


def generate_images(checkpoint_path, output_dir, num_samples=300, use_ema=True,
                   base_seed=42, num_inference_steps=50, guidance_scale=7.5):
    """Generate images from checkpoint."""

    output_dir = Path(output_dir)
    if output_dir.exists() and len(list(output_dir.glob("*.png"))) == num_samples:
        print(f"  Images already exist at {output_dir}, skipping generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts with random sampling for diversity
    df = pd.read_csv('/workspace/data/val_50k.csv')
    df_sampled = df.sample(n=num_samples, random_state=base_seed)
    prompts = df_sampled['class_name'].tolist()
    print(f"  Sampled {len(prompts)} prompts from {df_sampled['class_id'].nunique()} unique classes")

    # Load merged weights
    merged_dict = load_and_merge_lora(checkpoint_path, use_ema=use_ema)

    # Load base SD1.5 pipeline
    print("  Loading base SD1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Replace UNet weights
    print("  Replacing UNet weights...")
    unet_state = pipe.unet.state_dict()
    for key in merged_dict.keys():
        if key in unet_state:
            unet_state[key] = merged_dict[key].to(torch.float16)

    pipe.unet.load_state_dict(unet_state)
    pipe = pipe.to('cuda')

    # Generate images
    print(f"  Generating {num_samples} images...")
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
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

    print(f"  Saved {num_samples} images to {output_dir}")

    # Clean up
    del pipe
    torch.cuda.empty_cache()


def compute_inception_score(image_dir, batch_size=32, splits=10):
    """
    Compute Inception Score for generated images.

    Args:
        image_dir: Directory containing generated images
        batch_size: Batch size for processing
        splits: Number of splits for computing IS

    Returns:
        Tuple of (mean, std) of Inception Score
    """
    print(f"  Computing Inception Score...")

    # Load Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model = inception_model.cuda()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load images
    image_files = sorted(Path(image_dir).glob("*.png"))

    # Get predictions
    preds = []
    for i in tqdm(range(0, len(image_files), batch_size), desc="IS inference"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for img_file in batch_files:
            img = Image.open(img_file).convert('RGB')
            img = transform(img)
            batch_images.append(img)

        batch_tensor = torch.stack(batch_images).cuda()

        with torch.no_grad():
            pred = torch.nn.functional.softmax(inception_model(batch_tensor), dim=1)

        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Compute IS
    split_scores = []

    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)

    # Clean up
    del inception_model
    torch.cuda.empty_cache()

    return is_mean, is_std


def evaluate_checkpoint(name, checkpoint_path, output_dir, real_images_dir,
                       num_samples, use_ema, base_seed):
    """Evaluate a single checkpoint with all metrics."""
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {name}")
    print(f"{'='*80}")

    samples_dir = output_dir / f"samples_{name}"

    # 1. Generate images
    print("\n[1/4] Generating images...")
    generate_images(
        checkpoint_path=checkpoint_path,
        output_dir=samples_dir,
        num_samples=num_samples,
        use_ema=use_ema,
        base_seed=base_seed,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # 2. Compute FID
    print("\n[2/4] Computing FID...")
    fid_score = fid.compute_fid(real_images_dir, str(samples_dir))
    print(f"  FID: {fid_score:.2f}")

    # 3. Compute sFID (spatial FID)
    print("\n[3/4] Computing sFID...")
    sfid_score = fid.compute_fid(real_images_dir, str(samples_dir), mode="clean", model_name="inception_v3", num_workers=0)
    print(f"  sFID: {sfid_score:.2f}")

    # 4. Compute Inception Score
    print("\n[4/4] Computing Inception Score...")
    is_mean, is_std = compute_inception_score(samples_dir)
    print(f"  IS: {is_mean:.2f} ± {is_std:.2f}")

    results = {
        "fid": float(fid_score),
        "sfid": float(sfid_score),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                       help="Checkpoints in format name:path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--real-images-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--alignment-metrics", type=str, default=None,
                       help="Path to pre-computed alignment metrics JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse checkpoints
    checkpoints = {}
    for ckpt_spec in args.checkpoints:
        name, path = ckpt_spec.split(":", 1)
        checkpoints[name] = path

    print(f"\n{'='*80}")
    print(f"REPA Training Progress Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoints: {list(checkpoints.keys())}")
    print(f"Num samples: {args.num_samples}")
    print(f"Use EMA: {args.use_ema}")
    print(f"Real images: {args.real_images_dir}")

    # Evaluate each checkpoint
    all_results = {}
    for name, checkpoint_path in checkpoints.items():
        results = evaluate_checkpoint(
            name=name,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            real_images_dir=args.real_images_dir,
            num_samples=args.num_samples,
            use_ema=args.use_ema,
            base_seed=args.base_seed,
        )
        all_results[name] = results

    # Load alignment metrics if provided
    if args.alignment_metrics and Path(args.alignment_metrics).exists():
        print(f"\nLoading pre-computed alignment metrics from {args.alignment_metrics}")
        with open(args.alignment_metrics) as f:
            alignment_metrics = json.load(f)

        # Merge alignment metrics into results
        for name in all_results:
            if name in alignment_metrics:
                all_results[name]["alignment"] = alignment_metrics[name]

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Evaluation Summary")
    print(f"{'='*80}")
    print(f"\n{'Checkpoint':<15} {'FID':<10} {'sFID':<10} {'IS':<15}")
    print(f"{'-'*50}")
    for name, results in all_results.items():
        is_str = f"{results['is_mean']:.2f}±{results['is_std']:.2f}"
        print(f"{name:<15} {results['fid']:<10.2f} {results['sfid']:<10.2f} {is_str:<15}")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
