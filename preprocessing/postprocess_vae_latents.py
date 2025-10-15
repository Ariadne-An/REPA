"""
Post-process VAE latents from official REPA preprocessing.

This script:
1. Reads VAE latents from the ZIP file produced by dataset_tools.py encode
2. Applies the 0.18215 scaling factor (required for SD-1.5)
3. Converts to LMDB format with proper metadata

Usage:
    # Step 1: Run official preprocessing
    python preprocessing/dataset_tools.py convert --source=... --dest=data/images --resolution=512x512
    python preprocessing/dataset_tools.py encode --source=data/images --dest=data/vae-sd-raw

    # Step 2: Post-process with scaling
    python preprocessing/postprocess_vae_latents.py \
        --source data/vae-sd-raw \
        --dest data/vae_latents \
        --latent_scale 0.18215
"""

import argparse
import json
import lmdb
import numpy as np
import torch
import zipfile
from pathlib import Path
from tqdm import tqdm
import io


def read_vae_zip(zip_path):
    """
    Read VAE latents from official REPA ZIP file.

    Args:
        zip_path: Path to ZIP file from dataset_tools.py encode

    Yields:
        (sample_id, latent): tuple of (str, np.ndarray [8, H, W])
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Read dataset.json for metadata
        with zf.open('dataset.json') as f:
            metadata = json.load(f)

        # Iterate through all files
        for fname in tqdm(zf.namelist(), desc="Reading latents"):
            if fname == 'dataset.json' or not fname.endswith('.npy'):
                continue

            # Read latent
            with zf.open(fname) as f:
                latent_bytes = f.read()
                # Official format: .npy files
                latent = np.load(io.BytesIO(latent_bytes))

                # Sample ID: remove extension
                sample_id = Path(fname).stem

                yield sample_id, latent, metadata


def postprocess_and_save_lmdb(source_dir, dest_dir, latent_scale):
    """
    Post-process VAE latents and save to LMDB.

    Args:
        source_dir: Directory containing vae-sd.zip from official preprocessing
        dest_dir: Output LMDB directory
        latent_scale: Scaling factor (0.18215 for SD-1.5)
    """
    # Find ZIP file
    source_path = Path(source_dir)
    zip_files = list(source_path.glob('*.zip'))

    if len(zip_files) == 0:
        raise FileNotFoundError(f"No ZIP file found in {source_dir}")
    elif len(zip_files) > 1:
        print(f"⚠️  Multiple ZIP files found, using: {zip_files[0]}")

    zip_path = zip_files[0]
    print(f"📂 Input ZIP: {zip_path}")

    # Create LMDB
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Estimate map_size (assume ~1M samples, each 4*64*64*2 bytes = 32KB)
    map_size = 50 * 1024**3  # 50GB should be enough

    env = lmdb.open(str(dest_path), map_size=map_size)

    print(f"📂 Output LMDB: {dest_path}")
    print(f"🔢 Latent scale: {latent_scale}")

    # Process latents
    num_samples = 0

    with env.begin(write=True) as txn:
        for sample_id, latent, metadata in read_vae_zip(zip_path):
            # latent shape from official: [8, H, W] (mean + std concatenated)
            # We need to sample: z = mean + std * randn()

            assert latent.shape[0] == 8, f"Expected 8 channels (mean+std), got {latent.shape[0]}"

            mean = latent[:4]  # [4, H, W]
            std = latent[4:]   # [4, H, W]

            # Sample latent
            # Note: For deterministic preprocessing, we use mean only (no sampling)
            # This is consistent with training where we sample on-the-fly
            z = mean  # [4, H, W]

            # Apply scaling
            z_scaled = z * latent_scale

            # Convert to fp16 to save space
            z_fp16 = z_scaled.astype(np.float16)

            # Save to LMDB
            key = sample_id.encode('utf-8')
            value = z_fp16.tobytes()
            txn.put(key, value)

            num_samples += 1

            if num_samples % 10000 == 0:
                print(f"  Processed {num_samples} samples...")

        # Save metadata
        meta = {
            'latent_scale': latent_scale,
            'vae_version': 'sd-v1-5',
            'shape': list(z_fp16.shape),  # [4, H, W]
            'dtype': 'float16',
            'num_samples': num_samples,
            'source_metadata': metadata
        }

        txn.put(b'__meta__', json.dumps(meta).encode('utf-8'))

        print(f"\n✅ Processed {num_samples} samples")
        print(f"   Latent shape: {z_fp16.shape}")
        print(f"   Saved to: {dest_path}")

    env.close()

    # Verify
    print("\n🔍 Verifying...")
    env = lmdb.open(str(dest_path), readonly=True)
    with env.begin() as txn:
        # Check meta
        meta_bytes = txn.get(b'__meta__')
        meta = json.loads(meta_bytes.decode('utf-8'))
        print(f"   Meta: {meta}")

        # Check a sample
        cursor = txn.cursor()
        cursor.first()
        sample_key, sample_value = cursor.item()
        if sample_key != b'__meta__':
            latent = np.frombuffer(sample_value, dtype=np.float16).reshape(meta['shape'])
            print(f"   Sample latent shape: {latent.shape}")
            print(f"   Sample latent range: [{latent.min():.4f}, {latent.max():.4f}]")

    env.close()

    print("\n🎉 Post-processing complete!")
    print(f"   Next: Run build_dino_cache.py to extract DINO tokens")


def main():
    parser = argparse.ArgumentParser(description="Post-process VAE latents with scaling")
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Source directory containing VAE ZIP file from official preprocessing"
    )
    parser.add_argument(
        '--dest',
        type=str,
        required=True,
        help="Destination LMDB directory"
    )
    parser.add_argument(
        '--latent_scale',
        type=float,
        default=0.18215,
        help="Latent scaling factor (default: 0.18215 for SD-1.5)"
    )

    args = parser.parse_args()

    print("="*80)
    print("VAE Latent Post-Processing (Apply Scaling + Convert to LMDB)")
    print("="*80)

    postprocess_and_save_lmdb(args.source, args.dest, args.latent_scale)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
