"""
Prepare CLIP text embeddings for ImageNet classes.

This script:
1. Loads ImageNet class names from JSON
2. Generates text prompts using template "a photo of a {class_name}"
3. Encodes prompts with SD-1.5 CLIP text encoder
4. Adds null prompt "" at idx=1000 for CFG
5. Saves as PyTorch tensor [1001, 77, 768]

Usage:
    python preprocessing/prepare_clip_embeddings.py \
        --imagenet_classes data/imagenet_classes.json \
        --output_path data/clip_embeddings_1001.pt \
        --template "a photo of a {}"
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


def load_imagenet_classes(json_path):
    """
    Load ImageNet class names from JSON.

    Args:
        json_path: Path to imagenet_classes.json

    Returns:
        class_names: List of 1000 class names (sorted by class_id)
    """
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    # Convert to list sorted by class_id
    class_names = [class_dict[str(i)][1] if isinstance(class_dict[str(i)], list) else class_dict[str(i)] for i in range(1000)]

    print(f"📊 Loaded {len(class_names)} classes")
    print(f"   Examples: {class_names[:5]}")

    return class_names


def generate_prompts(class_names, template="a photo of a {}"):
    """
    Generate text prompts from class names.

    Args:
        class_names: List of class names
        template: Template string with {} placeholder

    Returns:
        prompts: List of 1000 prompts
    """
    prompts = [template.format(name) for name in class_names]

    print(f"📝 Generated {len(prompts)} prompts")
    print(f"   Template: {template}")
    print(f"   Examples: {prompts[:3]}")

    return prompts


def encode_prompts(prompts, tokenizer, text_encoder, device='cuda'):
    """
    Encode text prompts with CLIP text encoder.

    Args:
        prompts: List of text prompts
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        device: Device

    Returns:
        embeddings: [N, 77, 768] tensor
    """
    embeddings_list = []

    text_encoder = text_encoder.to(device).eval()

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Encoding prompts"):
            # Tokenize
            tokens = tokenizer(
                prompt,
                padding='max_length',
                max_length=77,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)

            # Encode (explicitly pass attention_mask for robustness)
            output = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state  # [1, 77, 768]

            embeddings_list.append(output.cpu())

    # Stack
    embeddings = torch.cat(embeddings_list, dim=0)  # [N, 77, 768]

    print(f"✅ Encoded {len(prompts)} prompts")
    print(f"   Shape: {embeddings.shape}")

    return embeddings


def prepare_clip_embeddings(imagenet_classes_path, output_path, template, device='cuda'):
    """
    Prepare CLIP text embeddings for ImageNet classes.

    Args:
        imagenet_classes_path: Path to imagenet_classes.json
        output_path: Output .pt file path
        template: Text prompt template
        device: Device
    """
    # Load class names
    class_names = load_imagenet_classes(imagenet_classes_path)

    # Generate prompts
    prompts = generate_prompts(class_names, template=template)

    # Add null prompt for CFG
    prompts.append("")  # idx=1000
    print(f"📝 Added null prompt at idx=1000 for CFG")

    # Load CLIP tokenizer and text encoder from SD-1.5
    print("\n📥 Loading CLIP text encoder from SD-1.5...")
    tokenizer = CLIPTokenizer.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='tokenizer'
    )
    text_encoder = CLIPTextModel.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='text_encoder'
    )
    print("✅ Loaded CLIP text encoder")

    # Encode prompts
    embeddings = encode_prompts(prompts, tokenizer, text_encoder, device=device)

    # Verify shape
    assert embeddings.shape == (1001, 77, 768), f"Expected [1001, 77, 768], got {embeddings.shape}"

    # Convert to fp16 to save space
    embeddings = embeddings.half()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings, output_path)

    print(f"\n💾 Saved to: {output_path}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify
    print("\n🔍 Verifying...")
    loaded = torch.load(output_path)
    print(f"   Loaded shape: {loaded.shape}")
    print(f"   Loaded dtype: {loaded.dtype}")

    # Check null prompt
    null_emb = loaded[1000]
    if torch.isnan(null_emb).any():
        print("   ❌ WARNING: Null prompt contains NaN!")
    else:
        print("   ✅ Null prompt is valid")

    print("\n🎉 CLIP embeddings prepared successfully!")
    print(f"   Next: Run scripts/selfcheck.py to verify all preprocessing")


def main():
    parser = argparse.ArgumentParser(description="Prepare CLIP text embeddings")
    parser.add_argument(
        '--imagenet_classes',
        type=str,
        required=True,
        help="Path to imagenet_classes.json"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/clip_embeddings_1001.pt',
        help="Output .pt file path (default: data/clip_embeddings_1001.pt)"
    )
    parser.add_argument(
        '--template',
        type=str,
        default='a photo of a {}',
        help='Text prompt template (default: "a photo of a {}")'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Device (default: cuda)"
    )

    args = parser.parse_args()

    print("="*80)
    print("CLIP Text Embeddings Preparation")
    print("="*80)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Check input file
    if not Path(args.imagenet_classes).exists():
        print(f"❌ ERROR: ImageNet classes file not found: {args.imagenet_classes}")
        print("\n📝 Please create this file with format:")
        print('   {"0": "tench", "1": "goldfish", ..., "999": "toilet tissue"}')
        print("\n   You can download from:")
        print("   https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a")
        return 1

    # Prepare embeddings
    prepare_clip_embeddings(
        args.imagenet_classes,
        args.output_path,
        args.template,
        args.device
    )

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
