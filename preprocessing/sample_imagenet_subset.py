"""
Sample a uniform subset from ImageNet for training.

This script samples N images per class from ImageNet train split,
ensuring balanced class distribution.

Usage:
    python preprocessing/sample_imagenet_subset.py \
        --imagenet_dir /data/ILSVRC/Data/CLS-LOC/train \
        --samples_per_class 200 \
        --output_csv data/train_200k.csv \
        --seed 42
"""

import argparse
import os
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def get_imagenet_classes(imagenet_dir):
    """
    Get list of ImageNet synsets (class folders).

    Args:
        imagenet_dir: Path to ImageNet train directory

    Returns:
        synsets: Sorted list of synset IDs (e.g., ['n01440764', ...])
    """
    synsets = sorted([d.name for d in Path(imagenet_dir).iterdir() if d.is_dir()])

    if len(synsets) != 1000:
        print(f"⚠️  WARNING: Expected 1000 classes, found {len(synsets)}")

    return synsets


def sample_images_per_class(imagenet_dir, samples_per_class, seed=42):
    """
    Sample images uniformly across all classes.

    Args:
        imagenet_dir: Path to ImageNet train directory
        samples_per_class: Number of samples per class
        seed: Random seed

    Returns:
        samples: List of dicts with keys: id, img_path, class_id, synset
    """
    random.seed(seed)

    synsets = get_imagenet_classes(imagenet_dir)
    print(f"📊 Found {len(synsets)} classes")

    samples = []
    insufficient_classes = []

    for class_id, synset in enumerate(tqdm(synsets, desc="Sampling classes")):
        synset_dir = Path(imagenet_dir) / synset

        # Get all images in this class
        image_paths = list(synset_dir.glob('*.JPEG'))

        if len(image_paths) < samples_per_class:
            print(f"⚠️  WARNING: {synset} has only {len(image_paths)} images (< {samples_per_class})")
            insufficient_classes.append((synset, len(image_paths)))
            sampled_paths = image_paths  # Use all available
        else:
            # Random sample
            sampled_paths = random.sample(image_paths, samples_per_class)

        # Add to samples
        for img_path in sampled_paths:
            sample_id = img_path.stem  # Filename without extension
            samples.append({
                'id': sample_id,
                'img_path': str(img_path),
                'class_id': class_id,
                'synset': synset
            })

    # Summary
    print(f"\n✅ Sampled {len(samples)} images")
    print(f"   Target: {len(synsets) * samples_per_class}")

    if insufficient_classes:
        print(f"\n⚠️  {len(insufficient_classes)} classes had insufficient samples:")
        for synset, count in insufficient_classes[:10]:  # Show first 10
            print(f"   - {synset}: {count} images")
        if len(insufficient_classes) > 10:
            print(f"   ... and {len(insufficient_classes) - 10} more")

    return samples


def save_csv(samples, output_csv):
    """Save samples to CSV file."""
    df = pd.DataFrame(samples)

    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved to: {output_csv}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")

    # Show class distribution
    class_counts = df['class_id'].value_counts()
    print(f"\n📊 Class distribution:")
    print(f"   Min samples per class: {class_counts.min()}")
    print(f"   Max samples per class: {class_counts.max()}")
    print(f"   Mean samples per class: {class_counts.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Sample ImageNet subset")
    parser.add_argument(
        '--imagenet_dir',
        type=str,
        required=True,
        help="Path to ImageNet train directory (e.g., /data/ILSVRC/Data/CLS-LOC/train)"
    )
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=200,
        help="Number of samples per class (default: 200)"
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='data/train_200k.csv',
        help="Output CSV path (default: data/train_200k.csv)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help="Verify that all image files exist"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.imagenet_dir):
        print(f"❌ ERROR: ImageNet directory not found: {args.imagenet_dir}")
        return 1

    print("="*80)
    print("ImageNet Subset Sampling")
    print("="*80)
    print(f"Input directory: {args.imagenet_dir}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Sample images
    samples = sample_images_per_class(
        args.imagenet_dir,
        args.samples_per_class,
        seed=args.seed
    )

    # Optional: Verify all files exist
    if args.verify:
        print("\n🔍 Verifying image files...")
        missing_files = []
        for sample in tqdm(samples, desc="Verifying"):
            if not os.path.exists(sample['img_path']):
                missing_files.append(sample['img_path'])

        if missing_files:
            print(f"❌ ERROR: {len(missing_files)} files not found")
            for path in missing_files[:10]:
                print(f"   - {path}")
            return 1
        else:
            print(f"✅ All {len(samples)} files exist")

    # Save to CSV
    save_csv(samples, args.output_csv)

    print("\n🎉 Done! Next steps:")
    print("   1. Run: python preprocessing/encode_vae_latents.py --csv_path", args.output_csv)
    print("   2. Run: python preprocessing/build_dino_cache.py --csv_path", args.output_csv)
    print("   3. Run: python preprocessing/prepare_clip_embeddings.py")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
