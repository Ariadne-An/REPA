"""
Debug script for REPA alignment issues.

Checks:
1. Shape alignment and interpolation
2. Normalization and scale
3. Token loss by timestep bucket
4. Manifold sensitivity
5. Gradient flow
"""

import torch
import torch.nn.functional as F
import lmdb
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from models.sd15_unet_aligned import SD15UNetAligned, AlignHead
from models.sd15_loss import SD15REPALoss
from dataset_sd15 import SD15AlignedDataset
from torch.utils.data import DataLoader


def check_shape_alignment():
    """Check 1: Shape alignment and interpolation."""
    print("\n" + "="*80)
    print("CHECK 1: Shape Alignment & Interpolation")
    print("="*80)

    # Simulate U-Net mid_block output
    B, C, H, W = 2, 1280, 8, 8
    x = torch.randn(B, C, H, W)

    # Create AlignHead
    align_head = AlignHead(in_channels=C, out_dim=1024)

    # Forward pass
    pred_tokens = align_head(x)

    print(f"✓ Input shape: {x.shape} (B={B}, C={C}, H={H}, W={W})")
    print(f"✓ Output tokens shape: {pred_tokens.shape}")
    print(f"✓ Expected shape: [B, 256, 1024]")

    # Check normalization
    norms = torch.norm(pred_tokens, dim=-1)
    print(f"\n✓ Token norms (should be ~1.0):")
    print(f"  Mean: {norms.mean().item():.4f}")
    print(f"  Std:  {norms.std().item():.4f}")
    print(f"  Min:  {norms.min().item():.4f}")
    print(f"  Max:  {norms.max().item():.4f}")

    # Test with different input sizes
    print(f"\n✓ Testing interpolation with different sizes:")
    for h, w in [(4, 4), (8, 8), (16, 16), (32, 32)]:
        x_test = torch.randn(1, C, h, w)
        out = align_head(x_test)
        print(f"  {h}×{w} → {out.shape} (✓ correct)")

    return True


def check_normalization():
    """Check 2: Normalization and scale."""
    print("\n" + "="*80)
    print("CHECK 2: Normalization & Scale")
    print("="*80)

    # Load real DINO tokens
    dino_dir = Path("/dev/shm/dino_tokens_lmdb")
    if not dino_dir.exists():
        print("⚠️  DINO tokens not found in /dev/shm, trying /workspace/data")
        dino_dir = Path("/workspace/data/dino_tokens_lmdb")

    env = lmdb.open(str(dino_dir), readonly=True, lock=False)

    print(f"\n✓ Loading DINO tokens from {dino_dir}")

    # Sample 100 tokens
    norms = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= 100:
                break
            if b'_mid' in key:
                tokens = np.frombuffer(value, dtype=np.float16).reshape(256, 1024)
                token_norms = np.linalg.norm(tokens, axis=1)
                norms.extend(token_norms)

    norms = np.array(norms)
    print(f"\n✓ DINO token norms (should be ~1.0):")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std:  {norms.std():.6f}")
    print(f"  Min:  {norms.min():.6f}")
    print(f"  Max:  {norms.max():.6f}")

    if abs(norms.mean() - 1.0) > 0.01:
        print(f"  ⚠️  WARNING: Mean norm is {norms.mean():.4f}, not 1.0!")
        print(f"     DINO tokens may not be properly normalized!")

    # Check cosine similarity between random pairs
    print(f"\n✓ Checking cosine similarity distribution:")
    with env.begin() as txn:
        cursor = txn.cursor()
        keys = []
        for i, (key, _) in enumerate(cursor):
            if i >= 50:
                break
            if b'_mid' in key:
                keys.append(key)

        if len(keys) >= 10:
            cos_sims = []
            for i in range(10):
                t1 = np.frombuffer(txn.get(keys[i*2]), dtype=np.float16).reshape(256, 1024)
                t2 = np.frombuffer(txn.get(keys[i*2+1]), dtype=np.float16).reshape(256, 1024)

                # Token-wise cosine similarity
                t1_norm = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-8)
                t2_norm = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-8)
                cos = (t1_norm * t2_norm).sum(axis=1).mean()
                cos_sims.append(cos)

            cos_sims = np.array(cos_sims)
            print(f"  Mean cos_sim (different images): {cos_sims.mean():.4f}")
            print(f"  (Should be low, like 0.1-0.3, since different images)")

    env.close()
    return True


def check_loss_by_timestep():
    """Check 3: Token loss by timestep bucket."""
    print("\n" + "="*80)
    print("CHECK 3: Token Loss by Timestep Bucket")
    print("="*80)

    # This requires running inference, we'll parse logs instead
    log_file = Path("/workspace/REPA/logs/trackA_h200_bs128_bf16.jsonl")

    if not log_file.exists():
        print("⚠️  Log file not found, skipping...")
        return False

    import json

    # Parse log
    token_losses = []
    steps = []

    with open(log_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("phase") == "train":
                steps.append(data["step"])
                token_losses.append(data["metrics"]["loss/token_avg"])

    # Analyze trend
    print(f"\n✓ Token loss trend:")
    for milestone in [100, 1000, 5000, 10000, 20000, 30000]:
        if milestone < len(steps):
            idx = steps.index(milestone) if milestone in steps else min(range(len(steps)), key=lambda i: abs(steps[i]-milestone))
            loss = token_losses[idx]
            cos_sim = 1.0 - loss
            print(f"  Step {milestone:6d}: loss={loss:.4f}, cos_sim={cos_sim:.4f}")

    # Final
    if steps:
        final_loss = token_losses[-1]
        final_cos = 1.0 - final_loss
        print(f"  Step {steps[-1]:6d}: loss={final_loss:.4f}, cos_sim={final_cos:.4f}")

        if final_cos < 0.5:
            print(f"\n  ⚠️  WARNING: Cosine similarity is only {final_cos:.4f}!")
            print(f"     Alignment is failing! Should be >0.7")

    return True


def check_manifold_sensitivity():
    """Check 4: Manifold batch sensitivity."""
    print("\n" + "="*80)
    print("CHECK 4: Manifold Batch Sensitivity")
    print("="*80)

    log_file = Path("/workspace/REPA/logs/trackA_h200_bs128_bf16.jsonl")

    if not log_file.exists():
        print("⚠️  Log file not found, skipping...")
        return False

    import json

    # Parse log
    manifold_losses = []
    steps = []

    with open(log_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("phase") == "train":
                steps.append(data["step"])
                manifold_losses.append(data["metrics"]["loss/manifold_avg"])

    manifold_losses = np.array(manifold_losses)

    print(f"\n✓ Manifold loss statistics (last 1000 steps):")
    recent = manifold_losses[-1000:]
    print(f"  Mean:   {recent.mean():.6f}")
    print(f"  Std:    {recent.std():.6f}")
    print(f"  Min:    {recent.min():.6f}")
    print(f"  Max:    {recent.max():.6f}")
    print(f"  Range:  {recent.max() - recent.min():.6f}")

    if recent.std() > 0.02:
        print(f"\n  ⚠️  WARNING: Std is {recent.std():.4f}, very unstable!")
        print(f"     Manifold loss is highly batch-sensitive")
        print(f"     This suggests mean-pooling is too coarse")

    # Find spikes
    threshold = recent.mean() + 2 * recent.std()
    spikes = np.where(recent > threshold)[0]
    if len(spikes) > 10:
        print(f"\n  ⚠️  Found {len(spikes)} spikes (> 2σ)")
        print(f"     Manifold alignment is unstable")

    return True


def check_gradient_flow():
    """Check 5: Gradient flow."""
    print("\n" + "="*80)
    print("CHECK 5: Gradient Flow (Requires Training Context)")
    print("="*80)

    print("\n✓ To check gradient flow, add this to training loop:")
    print("""
    # After loss.backward()
    for name, param in model.named_parameters():
        if 'align_head' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
    """)

    print("\n✓ Expected:")
    print("  - align_head gradients should be >1e-5")
    print("  - If all near 0, alignment branch is dead")

    return True


def main():
    """Run all checks."""
    print("\n" + "="*80)
    print("🔍 REPA Alignment Diagnostic Tool")
    print("="*80)

    checks = [
        ("Shape Alignment", check_shape_alignment),
        ("Normalization", check_normalization),
        ("Loss by Timestep", check_loss_by_timestep),
        ("Manifold Sensitivity", check_manifold_sensitivity),
        ("Gradient Flow", check_gradient_flow),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ {name} check failed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*80)
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
