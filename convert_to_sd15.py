"""
Convert REPA checkpoint to standard Stable Diffusion 1.5 format.

This script:
1. Loads the REPA checkpoint (U-Net with LoRA + AlignHead)
2. Merges LoRA weights into base U-Net
3. Removes AlignHead (not needed for inference)
4. Saves as standard diffusers SD1.5 format
5. Optionally saves as single safetensors file (for WebUI)

Usage:
    # Convert to diffusers format
    python convert_to_sd15.py \
        --checkpoint exps/trackA_h200_bs128_bf16/step_024000/model.safetensors \
        --output_dir models/sd15_repa_24k

    # Also save as single file (WebUI compatible)
    python convert_to_sd15.py \
        --checkpoint exps/trackA_h200_bs128_bf16/step_024000/model.safetensors \
        --output_dir models/sd15_repa_24k \
        --save_single_file
"""

import argparse
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to REPA checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for converted model")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base SD1.5 model")
    parser.add_argument("--save_single_file", action="store_true",
                        help="Also save as single safetensors file (WebUI compatible)")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA weights instead of model weights")
    return parser.parse_args()


def load_checkpoint(checkpoint_path, use_ema=False):
    """Load REPA checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if use_ema:
        ema_path = checkpoint_path.parent / "ema.pt"
        if ema_path.exists():
            print(f"✓ Loading EMA weights from {ema_path}")
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
            if isinstance(ema_state, dict) and "shadow" in ema_state:
                ema_state = ema_state["shadow"]

            # We still need LoRA tensors (stored only in main checkpoint)
            print(f"✓ Loading base checkpoint for LoRA tensors: {checkpoint_path}")
            base_state = load_checkpoint(checkpoint_path, use_ema=False)

            merged_state = {}
            for key, value in base_state.items():
                if key in ema_state:
                    merged_state[key] = ema_state[key]
                else:
                    merged_state[key] = value
            return merged_state

    print(f"✓ Loading model weights from {checkpoint_path}")

    # Support both .pt and .safetensors formats
    if checkpoint_path.suffix == ".pt":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        state_dict = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def extract_unet_state_dict(state_dict):
    """
    Extract and convert U-Net state dict.

    Input format: unet.base_model.model.xxx (LoRA wrapped)
    Output format: xxx (standard diffusers)
    """
    unet_state_dict = {}

    for key, value in state_dict.items():
        # Skip AlignHead weights
        if "align_heads" in key or "align_head" in key:
            continue

        # Skip LoRA-specific keys (we only want merged weights)
        if "lora_A" in key or "lora_B" in key:
            continue

        # Remove LoRA wrapper prefix
        new_key = key
        if key.startswith("unet.base_model.model."):
            new_key = key.replace("unet.base_model.model.", "")
        elif key.startswith("unet."):
            new_key = key.replace("unet.", "")

        # Remove .base_layer. from LoRA-wrapped layers
        # e.g., xxx.base_layer.weight → xxx.weight
        new_key = new_key.replace(".base_layer.", ".")

        unet_state_dict[new_key] = value

    return unet_state_dict


def merge_lora_weights(state_dict):
    """
    Merge LoRA weights into base weights.

    For each layer with LoRA:
        W_merged = W_base + α/r * (lora_B @ lora_A)

    where α = lora_alpha (usually 32), r = lora_rank (usually 32)
    """
    print("\n✓ Merging LoRA weights into base U-Net...")

    # Find all LoRA layers
    lora_layers = defaultdict(dict)
    def sanitize(key: str) -> str:
        key = key.replace("unet.base_model.model.", "").replace("unet.", "").replace(".base_layer.", ".")
        key = key.replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
        return key

    for key in state_dict.keys():
        if "lora_A" in key:
            lora_layers[sanitize(key)]["A"] = state_dict[key]
        elif "lora_B" in key:
            lora_layers[sanitize(key)]["B"] = state_dict[key]

    print(f"  Found {len(lora_layers)} LoRA layers")

    # Infer lora_alpha and lora_rank
    if lora_layers:
        sample_key = list(lora_layers.keys())[0]
        lora_A = lora_layers[sample_key]["A"]
        lora_rank = lora_A.shape[0]
        # Assume alpha = rank (common practice)
        lora_alpha = lora_rank
        scaling = lora_alpha / lora_rank
        print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}, scaling: {scaling}")

    # Merge LoRA weights
    merged_state_dict = {}

    for key, value in state_dict.items():
        # Skip LoRA A/B weights (we'll merge them)
        if "lora_A" in key or "lora_B" in key:
            continue

        # Skip AlignHead
        if "align_heads" in key or "align_head" in key:
            continue

        # Check if this layer has LoRA
        clean_key = key.replace("unet.base_model.model.", "").replace("unet.", "")
        target_key = clean_key.replace(".base_layer.", ".")
        lookup_key = target_key.replace(".weight", "")

        if lookup_key in lora_layers:
            lora_info = lora_layers[lookup_key]
            if "A" in lora_info and "B" in lora_info:
                lora_A = lora_info["A"]
                lora_B = lora_info["B"]

                # Merge: W_new = W_base + (alpha/rank) * (lora_B @ lora_A)
                # With alpha=rank, this simplifies to: W_new = W_base + lora_B @ lora_A
                lora_weight = lora_B @ lora_A
                merged_weight = value + lora_weight
                merged_state_dict[target_key] = merged_weight
                continue

        # No LoRA contribution; store base weight (sanitized key)
        merged_state_dict[target_key] = value

    return merged_state_dict


def save_diffusers_format(unet_state_dict, base_model, output_dir):
    """Save as diffusers format (full pipeline)."""
    print(f"\n✓ Loading base SD1.5 pipeline from {base_model}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    print("✓ Replacing U-Net weights...")
    # Load into U-Net
    missing, unexpected = pipeline.unet.load_state_dict(unet_state_dict, strict=False)

    if missing:
        print(f"  ⚠️  Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for k in missing:
                print(f"    - {k}")

    if unexpected:
        print(f"  ⚠️  Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for k in unexpected:
                print(f"    - {k}")

    print(f"\n✓ Saving to {output_dir}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline.save_pretrained(output_dir)
    print("✅ Diffusers format saved!")


def save_single_file(unet_state_dict, base_model, output_path):
    """Save as single safetensors file (WebUI compatible)."""
    print(f"\n✓ Creating single-file checkpoint...")

    # Load full pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    # Replace U-Net
    pipeline.unet.load_state_dict(unet_state_dict, strict=False)

    # Collect all state dicts
    full_state_dict = {}

    # U-Net
    for k, v in pipeline.unet.state_dict().items():
        full_state_dict[f"model.diffusion_model.{k}"] = v

    # VAE
    for k, v in pipeline.vae.state_dict().items():
        full_state_dict[f"first_stage_model.{k}"] = v

    # Text Encoder
    for k, v in pipeline.text_encoder.state_dict().items():
        full_state_dict[f"cond_stage_model.transformer.{k}"] = v

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"✓ Saving to {output_path}...")
    save_file(full_state_dict, output_path)
    print("✅ Single-file checkpoint saved!")


def main():
    args = parse_args()

    print("="*80)
    print("REPA → Standard SD1.5 Converter")
    print("="*80)

    # Step 1: Load checkpoint
    state_dict = load_checkpoint(args.checkpoint, use_ema=args.use_ema)
    print(f"  Loaded {len(state_dict)} keys")

    # Step 2: Merge LoRA
    merged_dict = merge_lora_weights(state_dict)
    print(f"  After merging: {len(merged_dict)} keys")

    # Step 3: Extract U-Net state dict
    unet_state_dict = extract_unet_state_dict(merged_dict)
    print(f"  U-Net keys: {len(unet_state_dict)}")

    # Step 4: Save as diffusers format
    save_diffusers_format(unet_state_dict, args.base_model, args.output_dir)

    # Step 5: Optionally save as single file
    if args.save_single_file:
        single_file_path = Path(args.output_dir) / "model.safetensors"
        save_single_file(unet_state_dict, args.base_model, single_file_path)

    print("\n" + "="*80)
    print("✅ Conversion complete!")
    print("="*80)
    print(f"\nYou can now use the model:")
    print(f"  • Diffusers: load from '{args.output_dir}'")
    if args.save_single_file:
        print(f"  • WebUI/ComfyUI: use '{args.output_dir}/model.safetensors'")


if __name__ == "__main__":
    main()
