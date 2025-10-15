"""
Build DINO token cache from preprocessed images.

This script:
1. Reads images from the ZIP file produced by dataset_tools.py convert
2. Applies DINOv3 official transforms
3. Extracts DINO tokens [196, 1024]
4. L2 normalizes each token
5. Saves to LMDB format

Usage:
    python preprocessing/build_dino_cache.py \
        --source data/images \
        --dest data/dino_tokens \
        --dino_ckpt checkpoints/dinov3_vitl16.pth \
        --batch_size 64 \
        --num_workers 4
"""

import argparse
import json
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import zipfile
from pathlib import Path
from tqdm import tqdm
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DINOv3Encoder:
    """
    DINOv3 ViT-L/16 encoder wrapper.

    This class provides:
    - from_local_ckpt() to load local checkpoint
    - get_transforms() to get official transforms
    - Forward pass to extract tokens [B, 196, 1024]
    """

    def __init__(self, ckpt_path=None, device='cuda'):
        self.device = device
        self.model = None
        self.model_id = "dinov3_vitl16"
        self.transform_id = "dinov3_vitl16_official"

        if ckpt_path:
            self.load_from_local(ckpt_path)
        else:
            self.load_from_hub()

    def load_from_hub(self):
        """Load DINOv3 from torch.hub."""
        print("📥 Loading DINOv3 from torch.hub...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.model = self.model.eval().to(self.device)
        print("✅ Loaded DINOv3 ViT-L/14 from torch.hub")

    def load_from_local(self, ckpt_path):
        """Load DINOv3 from local checkpoint."""
        print(f"📥 Loading DINOv3 from: {ckpt_path}")

        # Load model architecture from hub (without weights)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        # Load local checkpoint
        state_dict = torch.load(ckpt_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        print(f"✅ Loaded DINOv3 from local checkpoint")

    @staticmethod
    def get_transforms():
        """
        Get official DINOv3 transforms.

        Returns:
            transforms: torchvision transforms
        """
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_tokens(self, images):
        """
        Extract DINO tokens from images.

        Args:
            images: [B, 3, 224, 224] tensor

        Returns:
            tokens: [B, 196, 1024] tensor (NOT normalized yet)
        """
        self.model.eval()

        # Forward pass
        # DINOv2/v3 returns [B, N, D] where N=196 (14x14), D=1024 (for ViT-L)
        output = self.model.forward_features(images)

        # Get patch tokens (exclude CLS token if present)
        if 'x_norm_patchtokens' in output:
            tokens = output['x_norm_patchtokens']  # [B, 196, 1024]
        elif isinstance(output, dict) and 'x' in output:
            x = output['x']
            tokens = x[:, 1:]  # Remove CLS token
        else:
            # Fallback: assume output is [B, N+1, D], remove first token (CLS)
            tokens = output[:, 1:]

        # Verify shape
        assert tokens.shape[1] == 196, f"Expected 196 tokens, got {tokens.shape[1]}"
        assert tokens.shape[2] == 1024, f"Expected D=1024, got {tokens.shape[2]}"

        return tokens


class ZipImageDataset(Dataset):
    """Dataset to read images from ZIP file."""

    def __init__(self, zip_path, transform=None, max_images=None):
        self.zip_path = zip_path
        self.transform = transform

        # Read file list
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.file_list = [f for f in zf.namelist() if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]

        if max_images:
            self.file_list = self.file_list[:max_images]

        print(f"📊 Found {len(self.file_list)} images in ZIP")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        # Read image from ZIP
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(fname) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')

        # Apply transform
        if self.transform:
            img = self.transform(img)

        # Sample ID: remove directory and extension
        sample_id = Path(fname).stem

        return img, sample_id


def build_dino_cache(source_dir, dest_dir, dino_encoder, batch_size=64, num_workers=4):
    """
    Build DINO token cache.

    Args:
        source_dir: Directory containing dataset.zip from official convert
        dest_dir: Output LMDB directory
        dino_encoder: DINOv3Encoder instance
        batch_size: Batch size for processing
        num_workers: Number of workers for dataloader
    """
    # Find ZIP file
    source_path = Path(source_dir)
    zip_files = list(source_path.glob('dataset.zip'))

    if len(zip_files) == 0:
        raise FileNotFoundError(f"No dataset.zip found in {source_dir}")

    zip_path = zip_files[0]
    print(f"📂 Input ZIP: {zip_path}")

    # Create dataset
    transform = DINOv3Encoder.get_transforms()
    dataset = ZipImageDataset(zip_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create LMDB
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Estimate map_size (assume ~1M samples, each 196*1024*2 bytes = 400KB)
    map_size = 500 * 1024**3  # 500GB

    env = lmdb.open(str(dest_path), map_size=map_size)

    print(f"📂 Output LMDB: {dest_path}")
    print(f"🔢 Batch size: {batch_size}")
    print(f"🔢 Num workers: {num_workers}")

    # Process batches
    num_samples = 0

    with env.begin(write=True) as txn:
        for images, sample_ids in tqdm(dataloader, desc="Extracting DINO tokens"):
            # Move to device
            images = images.to(dino_encoder.device)

            # Extract tokens [B, 196, 1024]
            tokens = dino_encoder.extract_tokens(images)

            # L2 normalize each token
            tokens = F.normalize(tokens, dim=-1)  # [B, 196, 1024]

            # Save to LMDB
            tokens_cpu = tokens.cpu().numpy().astype(np.float16)

            for i, sample_id in enumerate(sample_ids):
                key = sample_id.encode('utf-8')
                value = tokens_cpu[i].tobytes()  # [196, 1024] fp16
                txn.put(key, value)

                num_samples += 1

        # Save metadata
        meta = {
            'dino_model_id': dino_encoder.model_id,
            'dino_transform_id': dino_encoder.transform_id,
            'D': 1024,
            'grid': 14,
            'num_tokens': 196,
            'dtype': 'float16',
            'num_samples': num_samples,
            'normalized': True
        }

        txn.put(b'__meta__', json.dumps(meta).encode('utf-8'))

        print(f"\n✅ Processed {num_samples} samples")
        print(f"   Token shape: [196, 1024]")
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
            tokens = np.frombuffer(sample_value, dtype=np.float16).reshape(196, 1024)
            print(f"   Sample token shape: {tokens.shape}")
            print(f"   Sample token norm: {np.linalg.norm(tokens[0]):.4f} (should be ~1.0)")

    env.close()

    print("\n🎉 DINO cache built successfully!")


def main():
    parser = argparse.ArgumentParser(description="Build DINO token cache")
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Source directory containing dataset.zip from official convert"
    )
    parser.add_argument(
        '--dest',
        type=str,
        required=True,
        help="Destination LMDB directory"
    )
    parser.add_argument(
        '--dino_ckpt',
        type=str,
        default=None,
        help="Path to local DINOv3 checkpoint (if None, load from torch.hub)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help="Number of workers for dataloader (default: 4)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Device (default: cuda)"
    )

    args = parser.parse_args()

    print("="*80)
    print("DINO Token Cache Builder")
    print("="*80)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Load DINO encoder
    dino_encoder = DINOv3Encoder(ckpt_path=args.dino_ckpt, device=args.device)

    # Build cache
    build_dino_cache(
        args.source,
        args.dest,
        dino_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n🎯 Next: Run prepare_clip_embeddings.py to generate CLIP text embeddings")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
