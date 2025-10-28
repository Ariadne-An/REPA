#!/usr/bin/env python3
"""
Compute FID and Inception Score metrics on already-generated images.
"""

import json
from pathlib import Path
import torch
from cleanfid import fid
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image


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


def compute_fid_metrics(fake_dir: Path, real_dir: Path):
    print(f"  Computing FID for {fake_dir.name}...")
    fid_score = fid.compute_fid(str(real_dir), str(fake_dir))
    return {"fid": fid_score}


def compute_inception_score(fake_dir: Path, device: torch.device, batch_size: int = 32):
    print(f"  Computing Inception Score for {fake_dir.name}...")
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_dir = Path("/workspace/REPA/eval_outputs/repa_quality_final/samples")
    real_dir = Path("/workspace/data/val_images_512")

    variants = [
        "sd15_base",
        "sd15_repa_init",
        "sd15_repa_trained",
    ]

    metrics = {}

    for variant_name in variants:
        print(f"\n=== Computing metrics for {variant_name} ===")
        fake_dir = samples_dir / variant_name

        if not fake_dir.exists():
            print(f"  WARNING: {fake_dir} does not exist, skipping...")
            continue

        scores = {}
        scores.update(compute_fid_metrics(fake_dir, real_dir))
        scores.update(compute_inception_score(fake_dir, device, batch_size=32))

        metrics[variant_name] = scores
        print(f"  {variant_name}: FID={scores['fid']:.2f}, IS={scores['inception_score']:.2f}±{scores['inception_score_std']:.2f}")

    output_path = Path("/workspace/REPA/eval_outputs/repa_quality_final/metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
