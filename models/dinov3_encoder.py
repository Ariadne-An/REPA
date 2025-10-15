"""
DINOv3 encoder for U-REPA SD-1.5.

This module provides a wrapper around DINOv2 ViT-L/14 for extracting
visual tokens that serve as alignment targets.

Key Features:
- Load from torch.hub or local checkpoint
- Official DINOv3 transforms (Resize→CenterCrop→Normalize)
- Extract 196 tokens [14×14 grid, D=1024]
- Handles 16×16→14×14 spatial downsampling (DINOv2 vitl14 outputs 256 tokens)
- L2 normalize tokens for alignment

Usage:
    encoder = DINOv3Encoder(ckpt_path='checkpoints/dinov3_vitl16.pth')
    transform = encoder.get_transforms()
    tokens = encoder.extract_tokens(images)  # [B, 196, 1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class DINOv3Encoder(nn.Module):
    """
    DINOv3 ViT-L/14 encoder wrapper.

    This class provides:
    - from_local_ckpt() to load local checkpoint
    - get_transforms() to get official transforms
    - extract_tokens() to get patch tokens [B, 196, 1024]

    Attributes:
        model: DINOv3 ViT-L/14 model
        model_id: Model identifier for validation
        transform_id: Transform identifier for validation
        device: Device (cuda/cpu)
    """

    def __init__(self, ckpt_path=None, device='cuda'):
        """
        Initialize DINOv3 encoder.

        Args:
            ckpt_path: Path to local checkpoint (if None, load from torch.hub)
            device: Device to load model on
        """
        super().__init__()

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
        """
        Load DINOv3 from local checkpoint.

        Args:
            ckpt_path: Path to checkpoint file
        """
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

        These transforms match the official DINOv3 evaluation protocol:
        1. Resize to 256 with bicubic interpolation
        2. Center crop to 224
        3. ToTensor (converts to [0,1])
        4. Normalize with ImageNet stats

        Returns:
            transforms.Compose: Transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_tokens(self, images, normalize=True):
        """
        Extract DINO tokens from images.

        Args:
            images: [B, 3, 224, 224] tensor (already transformed)
            normalize: Whether to L2 normalize tokens (default: True)

        Returns:
            tokens: [B, 196, 1024] tensor (14×14 grid)
        """
        self.model.eval()

        # Move to correct device if needed
        if images.device != self.device:
            images = images.to(self.device)

        # Forward pass
        # DINOv2/v3 returns dict with multiple outputs
        output = self.model.forward_features(images)

        # Get patch tokens (exclude CLS token if present)
        # Different DINOv2/v3 versions may have different output formats
        if 'x_norm_patchtokens' in output:
            # Official DINOv3 format
            tokens = output['x_norm_patchtokens']  # [B, N, 1024]
        elif isinstance(output, dict) and 'x' in output:
            # Alternative format: extract from 'x' and remove CLS
            x = output['x']
            tokens = x[:, 1:]  # Remove CLS token (first token)
        else:
            # Fallback: assume output is [B, N+1, D], remove first token (CLS)
            tokens = output[:, 1:]

        # Handle different token counts (DINOv2 vitl14 outputs 16×16=256)
        B, N, D = tokens.shape
        if N == 256:  # 16×16 grid from vitl14
            # Reshape to spatial: [B, 256, 1024] → [B, 1024, 16, 16]
            tokens = tokens.transpose(1, 2).reshape(B, D, 16, 16)
            # Downsample to 14×14: [B, 1024, 16, 16] → [B, 1024, 14, 14]
            tokens = F.interpolate(tokens, size=(14, 14), mode='bilinear', align_corners=False)
            # Reshape back: [B, 1024, 14, 14] → [B, 196, 1024]
            tokens = tokens.reshape(B, D, 196).transpose(1, 2)
        elif N != 196:
            raise ValueError(f"Unexpected token count: {N}. Expected 256 (16×16) or 196 (14×14)")

        # Verify final shape
        assert tokens.shape == (B, 196, 1024), \
            f"Expected [{B}, 196, 1024], got {tokens.shape}"

        # L2 normalize each token
        if normalize:
            tokens = F.normalize(tokens, dim=-1)  # [B, 196, 1024]

        return tokens

    def forward(self, images, normalize=True):
        """
        Forward pass (alias for extract_tokens).

        Args:
            images: [B, 3, 224, 224] tensor
            normalize: Whether to L2 normalize tokens

        Returns:
            tokens: [B, 196, 1024] tensor
        """
        return self.extract_tokens(images, normalize=normalize)


def test_dinov3_encoder():
    """Test DINOv3Encoder with random input."""
    print("="*80)
    print("Testing DINOv3Encoder")
    print("="*80)

    # Create encoder
    encoder = DINOv3Encoder(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Test transforms
    transform = encoder.get_transforms()
    print(f"✅ Created transforms: {transform}")

    # Test extraction with random images
    B = 4
    images = torch.randn(B, 3, 224, 224).to(encoder.device)

    print(f"\n🔍 Testing extraction with batch_size={B}...")
    tokens = encoder.extract_tokens(images, normalize=True)

    print(f"✅ Extracted tokens: {tokens.shape}")
    print(f"   Token norm (should be ~1.0): {tokens[0, 0].norm().item():.4f}")

    # Test without normalization
    tokens_unnorm = encoder.extract_tokens(images, normalize=False)
    print(f"   Unnormalized token norm: {tokens_unnorm[0, 0].norm().item():.4f}")

    print("\n🎉 DINOv3Encoder test passed!")


if __name__ == '__main__':
    test_dinov3_encoder()
