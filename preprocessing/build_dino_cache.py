"""
Build DINO token LMDB aligned with CSV manifest.

This script:
1. Reads ImageNet samples from dataset.zip following the order in CSV.
2. Applies official DINOv2 transforms and extracts [256, 1024] tokens.
3. L2-normalizes tokens, stores them in fp16 inside LMDB with keys
   formatted as `{sample_id}_{layer}` (default layer: "mid").
4. Writes metadata.json and LMDB __meta__ entry for downstream sanity checks.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class DINOv2Encoder:
    """Minimal wrapper around dinov2_vitl14."""

    def __init__(self, ckpt_path: str | None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        print("📥 Loading DINOv2 ViT-L/14...")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        if self.ckpt_path:
            state = torch.load(self.ckpt_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
        self.model = self.model.eval().to(self.device)
        print("✅ DINOv2 ready on", self.device)

    @staticmethod
    def transforms():
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(images)
        if isinstance(output, dict):
            if "x_norm_patchtokens" in output:
                tokens = output["x_norm_patchtokens"]
            elif "x" in output:
                tokens = output["x"][:, 1:]
            else:
                raise KeyError("Unexpected DINO output keys")
        else:
            tokens = output[:, 1:]
        B, N, D = tokens.shape
        if (N, D) != (256, 1024):
            raise ValueError(f"Unexpected token shape {tokens.shape}, expected [B, 256, 1024]")
        return tokens


class CSVZipDataset(Dataset):
    """Dataset that reads images from zip according to CSV records."""

    def __init__(self, zip_path: Path, records: List[Tuple[str, str]], transform):
        self.zip_path = zip_path
        self.records = records
        self.transform = transform
        self._zip_handles: Dict[int, zipfile.ZipFile] = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample_id, rel_path = self.records[idx]
        zf = self._get_zip()
        with zf.open(rel_path, "r") as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return sample_id, img

    def _get_zip(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            if not hasattr(self, "_zip_single"):
                self._zip_single = zipfile.ZipFile(self.zip_path, "r")
            return self._zip_single
        wid = worker_info.id
        if wid not in self._zip_handles:
            self._zip_handles[wid] = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_handles[wid]

    def __del__(self):
        if hasattr(self, "_zip_single"):
            self._zip_single.close()
        for handle in self._zip_handles.values():
            handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Build DINOv2 token LMDB aligned with CSV.")
    parser.add_argument("--zip-path", type=Path, required=True, help="dataset.zip produced by preprocessing")
    parser.add_argument("--csv-path", type=Path, required=True, help="CSV manifest with id,img_path columns")
    parser.add_argument("--dest", type=Path, required=True, help="Output LMDB directory")
    parser.add_argument("--dino-ckpt", type=Path, default=None, help="Optional local dinov2_vitl14 ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--align-layers", nargs="+", default=["mid"],
                        help="Layer names for key suffixes (default: mid)")
    parser.add_argument("--map-size-gb", type=float, default=200.0,
                        help="LMDB map size in GB (default 200)")
    return parser.parse_args()


def load_records(csv_path: Path) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    required = {"id", "img_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must contain columns: {required}")
    records = list(df[["id", "img_path"]].itertuples(index=False, name=None))
    print(f"📑 Loaded {len(records)} records from {csv_path}")
    return records


def write_metadata(dest: Path, info: Dict):
    meta_path = dest / "metadata.json"
    meta_path.write_text(json.dumps(info, indent=2))


def main():
    args = parse_args()

    records = load_records(args.csv_path)
    dataset = CSVZipDataset(args.zip_path, records, DINOv2Encoder.transforms())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda batch: list(zip(*batch)),
    )

    encoder = DINOv2Encoder(
        ckpt_path=str(args.dino_ckpt) if args.dino_ckpt else None,
        device=args.device,
    )

    map_size = int(args.map_size_gb * (1024 ** 3))
    args.dest.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(args.dest), map_size=map_size, subdir=True)

    total = 0
    layers: Sequence[str] = args.align_layers

    with torch.no_grad():
        for sample_ids, batch_imgs in tqdm(dataloader, desc="Extracting DINO tokens"):
            imgs = torch.stack(batch_imgs).to(encoder.device)
            tokens = encoder.extract(imgs)
            tokens = F.normalize(tokens, dim=-1).to(torch.float16).cpu().numpy()

            with env.begin(write=True) as txn:
                for sid, token in zip(sample_ids, tokens):
                    for layer in layers:
                        key = f"{sid}_{layer}".encode("utf-8")
                        txn.put(key, token.tobytes())
            total += len(sample_ids)

    meta_info = {
        "layers": list(layers),
        "shape": [256, 1024],
        "grid": 16,
        "dtype": "float16",
        "num_samples": len(records),
        "csv_path": str(args.csv_path),
        "zip_path": str(args.zip_path),
        "dino_model": "dinov2_vitl14",
    }
    write_metadata(args.dest, meta_info)

    with env.begin(write=True) as txn:
        txn.put(b"__meta__", json.dumps(meta_info, indent=2).encode("utf-8"))

    print(f"✅ Saved tokens for {total} samples into {args.dest}")


if __name__ == "__main__":
    main()
