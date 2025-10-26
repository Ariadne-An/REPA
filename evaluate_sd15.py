import argparse
from pathlib import Path
from typing import Optional

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm

from models.sd15_unet_aligned import SD15UNetAligned
from utils_sd15 import (
    apply_ema_if_available,
    chunk_list,
    compute_fid,
    ensure_dir,
    load_checkpoint_state,
    load_imagenet_classes,
    load_yaml_config,
    sample_prompts,
    save_images,
    set_global_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SD15-REPA checkpoints via sampling/FID.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Directory containing saved checkpoint (step_xxxxxx).")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Where to save generated images/logs.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during generation.")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Override inference steps (defaults to config eval settings).")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Override CFG scale (defaults to config eval settings).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights if available.")
    parser.add_argument("--imagenet_classes", type=str, default="imagenet_classes.json", help="Path to ImageNet class mapping for prompt sampling.")
    parser.add_argument("--fid_stats", type=str, default=None, help="Optional clean-fid statistics (.npz) to compute FID.")
    return parser.parse_args()


def load_aligned_unet(config, checkpoint_dir: Path, device: torch.device, use_ema: bool) -> SD15UNetAligned:
    model = SD15UNetAligned(
        pretrained_model_name=config["pretrained_model_name_or_path"],
        align_layers=config["align_layers"],
        dino_dim=config["dino_D"],
        use_lora=config.get("use_lora", True),
        lora_rank=config.get("lora_rank", 32),
        lora_targets=config.get("lora_targets", "attn+conv"),
        device=str(device),
    )

    state = load_checkpoint_state(checkpoint_dir, device=device)
    incompat = model.load_state_dict(state, strict=False)
    if getattr(incompat, "missing_keys", None):
        print(f"[WARN] Missing keys when loading checkpoint: {incompat.missing_keys}")
    if getattr(incompat, "unexpected_keys", None):
        print(f"[WARN] Unexpected keys when loading checkpoint: {incompat.unexpected_keys}")

    if use_ema and apply_ema_if_available(model, checkpoint_dir, device):
        print("Loaded EMA weights.")

    # Hooks are not needed for inference, remove them to reduce overhead.
    if hasattr(model, "hook_manager"):
        model.hook_manager.remove_hooks()

    return model


def build_pipeline(config, unet, device) -> StableDiffusionPipeline:
    dtype = torch.bfloat16 if config.get("mixed_precision", "bf16") == "bf16" else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        config["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet = unet.to(dtype=dtype, device=device)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint)
    output_dir = ensure_dir(Path(args.output_dir))
    samples_dir = ensure_dir(output_dir / "samples")

    model = load_aligned_unet(config, checkpoint_dir, device, args.use_ema)
    pipeline = build_pipeline(config, model.unet, device)

    num_steps = args.num_inference_steps or config.get("eval_num_inference_steps", 250)
    guidance_scale = args.guidance_scale or config.get("eval_cfg_scale", 7.5)

    class_names = load_imagenet_classes(args.imagenet_classes)
    prompts = sample_prompts(class_names, args.num_samples, seed=args.seed)

    next_index = 0
    progress = tqdm(total=args.num_samples, desc="Generating", ncols=100)
    for chunk in chunk_list(prompts, args.batch_size):
        generator = torch.Generator(device=device).manual_seed(args.seed + next_index)
        outputs = pipeline(
            prompt=chunk,
            negative_prompt=[""] * len(chunk),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        save_images(outputs.images, samples_dir, start_index=next_index, prefix="sample")
        next_index += len(outputs.images)
        progress.update(len(outputs.images))
        if next_index >= args.num_samples:
            break

    progress.close()
    print(f"Saved {next_index} samples to {samples_dir}")

    if args.fid_stats:
        fid_value = compute_fid(samples_dir, Path(args.fid_stats))
        print(f"FID: {fid_value:.4f}")


if __name__ == "__main__":
    main()
