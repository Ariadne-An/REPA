import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from encoders import StabilityVAEEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode ImageNet pixels into SD-VAE latents and store them in LMDB."
    )
    parser.add_argument("--zip-path", type=str, required=True, help="Path to dataset.zip")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV file with 'id' and 'img_path' columns")
    parser.add_argument("--output", type=str, required=True, help="Directory for LMDB output")
    parser.add_argument("--model-url", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model identifier")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per VAE step")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--map-size-gb", type=float, default=32.0, help="LMDB map size in GB")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    return parser.parse_args()


class ZipImageDataset(Dataset):
    def __init__(self, zip_path: str, records: List[Tuple[str, str]]):
        self.zip_path = zip_path
        self.records = records
        self._zip_files: Dict[int, zipfile.ZipFile] = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample_id, rel_path = self.records[idx]
        zf = self._get_zip_file()
        with zf.open(rel_path, "r") as f:
            img = Image.open(f).convert("RGB")
            array = np.array(img)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return sample_id, tensor

    def _get_zip_file(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            if not hasattr(self, "_zip_single"):
                self._zip_single = zipfile.ZipFile(self.zip_path, "r")
            return self._zip_single
        worker_id = worker_info.id
        if worker_id not in self._zip_files:
            self._zip_files[worker_id] = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_files[worker_id]

    def __del__(self):
        if hasattr(self, "_zip_single"):
            self._zip_single.close()
        for zf in self._zip_files.values():
            zf.close()


def save_lmdb(env: lmdb.Environment, items: List[Tuple[str, np.ndarray]]):
    with env.begin(write=True) as txn:
        for sample_id, array in items:
            txn.put(sample_id.encode("utf-8"), array.tobytes())


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    if not {"id", "img_path"}.issubset(df.columns):
        raise ValueError("CSV must contain 'id' and 'img_path' columns")
    records = list(df[["id", "img_path"]].itertuples(index=False, name=None))

    dataset = ZipImageDataset(args.zip_path, records)
    def collate(batch):
        sample_ids, tensors = zip(*batch)
        return list(sample_ids), list(tensors)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae = StabilityVAEEncoder(vae_name=args.model_url, batch_size=args.batch_size)
    vae.init(device)

    map_size = int(args.map_size_gb * (1024 ** 3))
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_path), map_size=map_size)

    total_samples = len(records)
    processed = 0
    last_latent_shape = None

    with torch.no_grad():
        for sample_ids, images in tqdm(dataloader, desc="Encoding VAE latents"):
            tensor = torch.stack(images).float().to(device)
            tensor = tensor / 127.5 - 1.0
            latents = vae.encode_pixels(tensor).cpu().numpy().astype(np.float16)
            save_lmdb(env, list(zip(sample_ids, latents)))
            processed += len(sample_ids)
            last_latent_shape = latents.shape[1:]

    meta = {
        "shape": list(last_latent_shape) if last_latent_shape else [4, 64, 64],
        "dtype": "float16",
        "latent_scale": 0.18215,
        "num_samples": processed,
        "vae_model": args.model_url,
        "csv_path": args.csv_path,
        "zip_path": args.zip_path,
        "preprocessing_complete": True,
    }
    (output_path / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Saved latents for {processed}/{total_samples} samples to {output_path}")


if __name__ == "__main__":
    main()
