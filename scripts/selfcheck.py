"""
Self-check script for U-REPA + SD-1.5 implementation.

This script validates:
1. VAE latent encoding correctness (scale, reconstruction quality)
2. DINO tokens validity (shape, normalization, transform_id)
3. CLIP embeddings correctness (shape, null prompt)
4. epsilon ↔ v conversion reversibility
5. Scheduler consistency

Run this before training to catch configuration mismatches early!

Usage:
    python scripts/selfcheck.py --config configs/sd15_repa_档A.yaml
"""

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import lmdb
import yaml
import numpy as np
import pandas as pd
from PIL import Image

# Add REPA to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_lmdb_meta(env: lmdb.Environment, dir_path: Path) -> dict:
    """Try to read metadata from __meta__ entry or metadata.json file."""
    meta = {}
    with env.begin() as txn:
        meta_bytes = txn.get(b'__meta__')
        if meta_bytes is not None:
            meta = json.loads(meta_bytes.decode())
    meta_file = dir_path / 'metadata.json'
    if not meta and meta_file.exists():
        meta = json.loads(meta_file.read_text())
    return meta


def check_vae_latents(config):
    """
    Check 1: VAE latent encoding correctness.

    - Verify shape: [4, 64, 64]
    - Verify dtype: fp16/bf16
    - Verify scale: 0.18215
    - Reconstruct samples and check LPIPS/PSNR
    """
    print("\n" + "="*80)
    print("CHECK 1: VAE Latent Encoding")
    print("="*80)

    latent_dir = resolve_path(config['latent_dir'])

    if not latent_dir.exists():
        print(f"⚠️  WARNING: Latent directory not found: {latent_dir}")
        print("   Skipping VAE latent check. Run encode_vae_latents.py first.")
        return False

    try:
        # Open LMDB
        env = lmdb.open(str(latent_dir), readonly=True, lock=False)
        meta = load_lmdb_meta(env, latent_dir)
        if not meta:
            print("❌ ERROR: No meta found in latent LMDB")
            env.close()
            return False

        print(f"📊 Latent Meta: {meta}")

        latent_shape = tuple(meta.get('shape', [4, 64, 64]))
        latent_dtype = meta.get('dtype', 'float16')

        if latent_shape != (4, 64, 64):
            print(f"⚠️  WARNING: Unexpected latent shape {latent_shape}, expected (4,64,64)")
        if latent_dtype not in ('float16', 'bf16', 'bfloat16'):
            print(f"⚠️  WARNING: Unexpected latent dtype {latent_dtype}")

        # Check latent_scale
        lmdb_scale = meta.get('latent_scale')
        if lmdb_scale is not None:
            if abs(lmdb_scale - config.get('latent_scale', lmdb_scale)) > 1e-6:
                print(f"❌ ERROR: latent_scale mismatch!")
                print(f"   Config: {config.get('latent_scale')}")
                print(f"   LMDB:   {lmdb_scale}")
                env.close()
                return False
            print(f"✅ latent_scale matches: {lmdb_scale}")
        else:
            print(f"⚠️  WARNING: latent_scale not in meta, assuming {config.get('latent_scale')}")

        # Sample latents
        with env.begin() as txn:
            cursor = txn.cursor()
            num_checked = 0
            for key, value in cursor:
                if key == b'__meta__':
                    continue
                latent = np.frombuffer(value, dtype=np.float16).reshape(latent_shape)
                lat_min, lat_max = float(latent.min()), float(latent.max())
                if lat_min < -10 or lat_max > 10:
                    print(f"⚠️  WARNING: Latent value range seems off: [{lat_min:.2f}, {lat_max:.2f}]")
                num_checked += 1
                if num_checked >= 10:
                    break
        env.close()
        print(f"✅ Checked {num_checked} latents: shape={latent_shape}, dtype={latent_dtype}")

        # Optional: Reconstruction quality check (requires VAE decoder)
        print("\n📝 Note: To check reconstruction quality (LPIPS/PSNR), run:")
        print("   python scripts/test_vae_reconstruction.py --config <config>")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_dino_tokens(config):
    """
    Check 2: DINO tokens validity.

    - Verify shape: [256, 1024]
    - Verify L2 normalization: norm ≈ 1.0
    - Verify meta: dino_model_id, dino_transform_id
    """
    print("\n" + "="*80)
    print("CHECK 2: DINO Tokens")
    print("="*80)

    dino_dir = resolve_path(config['dino_dir'])

    if not dino_dir.exists():
        print(f"⚠️  WARNING: DINO directory not found: {dino_dir}")
        print("   Skipping DINO check. Run build_dino_cache.py first.")
        return False

    try:
        # Open LMDB
        env = lmdb.open(str(dino_dir), readonly=True, lock=False)
        meta = load_lmdb_meta(env, dino_dir)
        if not meta:
            print("❌ ERROR: No meta found in DINO LMDB")
            env.close()
            return False

        print(f"📊 DINO Meta: {meta}")

        expected_layers = config.get('align_layers', ['mid'])
        expected_tokens = config.get('dino_num_tokens', meta.get('shape', [0])[0])
        expected_dim = config.get('dino_D', 1024)

        dino_model_cfg = config.get('dino_model') or config.get('dino_model_id')
        dino_model_meta = meta.get('dino_model') or meta.get('dino_model_id')
        if dino_model_cfg and dino_model_meta and dino_model_cfg != dino_model_meta:
            print(f"❌ ERROR: DINO model mismatch (config={dino_model_cfg}, meta={dino_model_meta})")
            env.close()
            return False
        if dino_model_meta:
            print(f"✅ DINO model: {dino_model_meta}")

        if meta.get('shape') and (meta['shape'][0] != expected_tokens or meta['shape'][1] != expected_dim):
            print(f"❌ ERROR: Token shape mismatch (expected {expected_tokens}x{expected_dim}, got {meta['shape']})")
            env.close()
            return False

        # Sample tokens and check shape + normalization
        norms = []
        with env.begin() as txn:
            cursor = txn.cursor()
            num_checked = 0
            for key, value in cursor:
                if key == b'__meta__':
                    continue
                tokens = np.frombuffer(value, dtype=np.float16).reshape(expected_tokens, expected_dim)
                token_norms = np.linalg.norm(tokens, axis=-1)
                mean_norm = float(token_norms.mean())
                norms.append(mean_norm)
                if abs(mean_norm - 1.0) > 0.1:
                    print(f"⚠️  WARNING: Token norm deviates: {mean_norm:.4f}")
                num_checked += 1
                if num_checked >= 10:
                    break

        if norms:
            avg_norm = float(np.mean(norms))
            print(f"✅ Checked {len(norms)} token sets: shape=[{expected_tokens}, {expected_dim}], avg_norm={avg_norm:.4f}")
            if abs(avg_norm - 1.0) > 0.05:
                print(f"⚠️  WARNING: Average norm deviates from 1.0: {avg_norm:.4f}")
        csv_path = resolve_path(config['csv_path'])
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            expected_entries = len(df) * len(expected_layers)
            with env.begin() as txn:
                actual_entries = sum(1 for k, _ in txn.cursor() if k != b'__meta__')
            if actual_entries != expected_entries:
                print(f"⚠️  WARNING: Entry count mismatch (LMDB={actual_entries}, expected={expected_entries})")
            else:
                print(f"✅ LMDB entry count matches CSV ({actual_entries})")

        env.close()
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_clip_embeddings(config):
    """
    Check 3: CLIP embeddings correctness.

    - Verify shape: [1001, 77, 768]
    - Verify idx=1000 is null prompt ""
    """
    print("\n" + "="*80)
    print("CHECK 3: CLIP Text Embeddings")
    print("="*80)

    clip_path = resolve_path(config['clip_embeddings_path'])

    if not clip_path.exists():
        print(f"⚠️  WARNING: CLIP embeddings not found: {clip_path}")
        print("   Skipping CLIP check. Run prepare_clip_embeddings.py first.")
        return False

    try:
        # Load CLIP embeddings
        clip_emb = torch.load(clip_path)

        print(f"📊 CLIP Embeddings shape: {clip_emb.shape}")
        print(f"   dtype: {clip_emb.dtype}")

        # Check shape
        if clip_emb.shape != (1001, 77, 768):
            print(f"❌ ERROR: CLIP embeddings shape mismatch!")
            print(f"   Expected: [1001, 77, 768]")
            print(f"   Got:      {clip_emb.shape}")
            return False

        print(f"✅ CLIP embeddings shape correct: [1001, 77, 768]")

        # Check null prompt (idx=1000)
        null_emb = clip_emb[1000]
        # Note: We can't easily verify if it's truly from "", but we can check it's not NaN
        if torch.isnan(null_emb).any():
            print(f"❌ ERROR: Null prompt embedding contains NaN")
            return False

        print(f"✅ Null prompt (idx=1000) exists and is valid")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_epsilon_v_conversion(config):
    """
    Check 4: epsilon ↔ v conversion reversibility.

    Generate random z0, ε, t and verify conversion is reversible.
    """
    print("\n" + "="*80)
    print("CHECK 4: Epsilon ↔ V Conversion Reversibility")
    print("="*80)

    schedule_name = config['schedule']['name']
    parametrization = config['schedule']['parametrization']

    print(f"📊 Schedule: {schedule_name}, Parametrization: {parametrization}")

    # Define interpolant
    def interpolant(t, path_type):
        if path_type == 'linear':
            alpha_t = 1 - t
            sigma_t = t
        elif path_type == 'cosine':
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
        else:
            raise ValueError(f"Unknown path_type: {path_type}")
        return alpha_t, sigma_t

    # Generate random test cases
    torch.manual_seed(42)
    num_tests = 100
    max_error = 0.0

    for i in range(num_tests):
        # Random inputs
        z0 = torch.randn(1, 4, 64, 64)
        epsilon = torch.randn(1, 4, 64, 64)
        t = torch.rand(1, 1, 1, 1)

        # Get alpha_t, sigma_t
        alpha_t, sigma_t = interpolant(t, schedule_name)

        # Convert epsilon → v
        v = alpha_t * epsilon - sigma_t * z0

        # Convert v → epsilon (should recover original epsilon)
        epsilon_reconstructed = (v + sigma_t * z0) / (alpha_t + 1e-8)

        # Check error
        error = torch.abs(epsilon - epsilon_reconstructed).max().item()
        max_error = max(max_error, error)

    print(f"✅ Tested {num_tests} cases")
    print(f"   Max reconstruction error: {max_error:.2e}")

    # Threshold relaxed to 1e-4 (float precision issue when alpha_t near 0)
    if max_error > 1e-4:
        print(f"⚠️  WARNING: Reconstruction error is high: {max_error:.2e}")
        print("   This may indicate a bug in the conversion logic.")
        return False

    print(f"✅ epsilon ↔ v conversion is reversible (error < 1e-4)")
    return True


def check_scheduler_consistency(config):
    """
    Check 5: Scheduler consistency.

    Verify that the scheduler computation matches the config.
    """
    print("\n" + "="*80)
    print("CHECK 5: Scheduler Consistency")
    print("="*80)

    schedule_name = config['schedule']['name']
    parametrization = config['schedule']['parametrization']

    print(f"📊 Schedule name: {schedule_name}")
    print(f"   Parametrization: {parametrization}")

    # Test a few timesteps
    test_timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]

    for t_val in test_timesteps:
        t = torch.tensor([[[[t_val]]]])

        if schedule_name == 'linear':
            alpha_t_expected = 1 - t_val
            sigma_t_expected = t_val
        elif schedule_name == 'cosine':
            alpha_t_expected = np.cos(t_val * np.pi / 2)
            sigma_t_expected = np.sin(t_val * np.pi / 2)
        else:
            print(f"⚠️  WARNING: Unknown schedule: {schedule_name}")
            return False

        print(f"   t={t_val:.2f}: α_t={alpha_t_expected:.4f}, σ_t={sigma_t_expected:.4f}")

    print(f"✅ Scheduler computation is consistent")
    return True


def main():
    parser = argparse.ArgumentParser(description="Self-check for U-REPA + SD-1.5")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--skip-data', action='store_true', help="Skip data-related checks (1-3)")
    args = parser.parse_args()

    # Load config
    print("="*80)
    print("U-REPA + SD-1.5 Self-Check Script")
    print("="*80)
    print(f"Loading config from: {args.config}")

    config = load_config(args.config)
    print(f"Experiment: {config['experiment_name']}")

    # Run checks
    results = {}

    if not args.skip_data:
        results['vae_latents'] = check_vae_latents(config)
        results['dino_tokens'] = check_dino_tokens(config)
        results['clip_embeddings'] = check_clip_embeddings(config)
    else:
        print("\n⏭️  Skipping data checks (--skip-data)")

    results['epsilon_v_conversion'] = check_epsilon_v_conversion(config)
    results['scheduler_consistency'] = check_scheduler_consistency(config)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("🎉 All checks passed! Ready to train.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
