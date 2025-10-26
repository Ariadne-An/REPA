"""
SD-1.5 U-Net with U-REPA alignment for representation learning.

This module implements the core U-REPA architecture:
1. AlignHead: Project U-Net features [B, C, H, W] → DINO tokens [B, H*W, D]
2. HookManager: Extract intermediate features using forward hooks
3. SD15UNetAligned: Wrapper around diffusers SD-1.5 U-Net with alignment

Key Design:
- Alignment at mid_block (layer 18, resolution 8×8, C=1280)
- Optional alignment at enc_last/dec_first (layer 12/24)
- Project-then-upsample: Conv1×1 projection → bilinear upsample to 14×14
- LoRA fine-tuning on alignment layers only

Usage:
    model = SD15UNetAligned(
        pretrained_model_name='runwayml/stable-diffusion-v1-5',
        align_layers=['mid'],
        use_lora=True,
        lora_rank=32
    )

    # Training forward pass
    out = model(
        noisy_latent,
        timesteps,
        text_embeddings,
        return_align_tokens=True
    )
    # out['pred']: [B, 4, 64, 64] prediction (epsilon or v)
    # out['align_tokens']['mid']: [B, 196, 1024] projected tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from typing import Dict, List, Optional, Tuple


class AlignHead(nn.Module):
    """
    Alignment head: U-Net features → DINO tokens.

    Architecture:
        1. Conv 1×1: [B, C, H, W] → [B, D, H, W]
        2. Bilinear upsample: [B, D, H, W] → [B, D, 14, 14]
        3. Reshape: [B, D, 14, 14] → [B, 196, D]

    This follows U-REPA's "project-then-upsample" strategy.
    """

    def __init__(self, in_channels: int, out_dim: int = 1024):
        """
        Initialize AlignHead.

        Args:
            in_channels: Input channels (C)
            out_dim: Output dimension (D, default: 1024 for DINOv2)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim

        # Conv 1×1 projection + normalization + activation
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] U-Net features

        Returns:
            tokens: [B, 196, D] projected tokens
        """
        # Project: [B, C, H, W] → [B, D, H, W]
        x = self.proj(x)

        # Upsample to 14×14 if needed
        if x.shape[-2:] != (14, 14):
            x = F.interpolate(
                x,
                size=(14, 14),
                mode='bilinear',
                align_corners=False
            )

        # Reshape: [B, D, 14, 14] → [B, 196, D]
        tokens = x.flatten(2).transpose(1, 2)  # [B, 196, D]
        tokens = F.normalize(tokens, dim=-1)

        return tokens


class HookManager:
    """
    Manages forward hooks to extract intermediate U-Net features.

    U-Net structure (diffusers SD-1.5):
    - down_blocks[0-3]: Encoder blocks (layers 0-11)
    - mid_block: Middle block (layer 18)
    - up_blocks[0-3]: Decoder blocks (layers 24-35)

    Alignment targets:
    - "enc_last": down_blocks[3] output (layer 12, 8×8, C=1280)
    - "mid": mid_block output (layer 18, 8×8, C=1280)
    - "dec_first": up_blocks[0] output (layer 24, 8×8, C=1280)
    """

    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize HookManager.

        Args:
            model: UNet2DConditionModel
            target_layers: List of layers to hook ['enc_last', 'mid', 'dec_first']
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {}  # Store extracted features
        self.hooks = []  # Store hook handles

    def _create_hook(self, layer_name: str):
        """Create a forward hook for the specified layer."""
        def hook_fn(module, input, output):
            # Store the output feature
            # Handle both tensor and dict outputs
            if isinstance(output, dict):
                # Some blocks return dict with 'sample' key
                self.features[layer_name] = output['sample']
            elif isinstance(output, tuple):
                # Some blocks return tuple (sample, res_samples)
                self.features[layer_name] = output[0]
            else:
                self.features[layer_name] = output

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on target layers."""
        for layer_name in self.target_layers:
            if layer_name == 'enc_last':
                # Hook on down_blocks[3]
                module = self.model.down_blocks[3]
            elif layer_name == 'mid':
                # Hook on mid_block
                module = self.model.mid_block
            elif layer_name == 'dec_first':
                # Hook on up_blocks[0]
                module = self.model.up_blocks[0]
            else:
                raise ValueError(f"Unknown layer: {layer_name}")

            # Register hook
            hook = module.register_forward_hook(self._create_hook(layer_name))
            self.hooks.append(hook)

        print(f"✅ Registered hooks on: {self.target_layers}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}

    def get_features(self) -> Dict[str, torch.Tensor]:
        """
        Get extracted features.

        Returns:
            features: Dict mapping layer_name → [B, C, H, W] features
        """
        return self.features

    def clear_features(self):
        """Clear stored features."""
        self.features = {}


class SD15UNetAligned(nn.Module):
    """
    SD-1.5 U-Net with U-REPA alignment.

    This class wraps diffusers UNet2DConditionModel and adds:
    1. AlignHead modules for specified layers
    2. HookManager to extract intermediate features
    3. Optional LoRA fine-tuning on alignment layers

    Attributes:
        unet: UNet2DConditionModel from diffusers
        align_heads: Dict of AlignHead modules
        hook_manager: HookManager instance
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank (if use_lora=True)
    """

    # Layer configurations
    LAYER_CONFIGS = {
        'enc_last': {'channels': 1280, 'resolution': 8},
        'mid': {'channels': 1280, 'resolution': 8},
        'dec_first': {'channels': 1280, 'resolution': 8},
    }

    def __init__(
        self,
        pretrained_model_name: str = 'runwayml/stable-diffusion-v1-5',
        align_layers: List[str] = ['mid'],
        dino_dim: int = 1024,
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_targets: str = 'attn+conv',  # 'attn', 'conv', 'attn+conv'
        device: str = 'cuda'
    ):
        """
        Initialize SD15UNetAligned.

        Args:
            pretrained_model_name: HuggingFace model name
            align_layers: List of layers to align ['mid', 'enc_last', 'dec_first']
            dino_dim: DINO token dimension (default: 1024)
            use_lora: Whether to use LoRA
            lora_rank: LoRA rank
            lora_targets: LoRA target modules ('attn', 'conv', 'attn+conv')
            device: Device
        """
        super().__init__()

        self.pretrained_model_name = pretrained_model_name
        self.align_layers = align_layers
        self.dino_dim = dino_dim
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_targets = lora_targets
        self.device = device

        # Validate align_layers
        for layer in align_layers:
            if layer not in self.LAYER_CONFIGS:
                raise ValueError(
                    f"Invalid align_layer: {layer}. "
                    f"Must be one of {list(self.LAYER_CONFIGS.keys())}"
                )

        # Load U-Net
        print(f"📥 Loading U-Net from: {pretrained_model_name}")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder='unet'
        )
        print("✅ Loaded U-Net")

        # Freeze U-Net parameters (will be unfrozen selectively by LoRA)
        for param in self.unet.parameters():
            param.requires_grad = False

        # Create AlignHead for each target layer
        self.align_heads = nn.ModuleDict()
        for layer_name in align_layers:
            in_channels = self.LAYER_CONFIGS[layer_name]['channels']
            self.align_heads[layer_name] = AlignHead(
                in_channels=in_channels,
                out_dim=dino_dim
            )

        print(f"✅ Created AlignHeads for: {align_layers}")

        # Create HookManager
        self.hook_manager = HookManager(self.unet, align_layers)
        self.hook_manager.register_hooks()

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora()

        # Move to device
        self.to(device)

    def _apply_lora(self):
        """
        Apply LoRA to alignment layers.

        This uses the peft library to add LoRA adapters to:
        - Attention layers (to_q, to_k, to_v, to_out)
        - Conv layers (conv1, conv2, conv_shortcut)

        Only layers involved in alignment are unfrozen.
        """
        from peft import LoraConfig, get_peft_model

        print(f"🔧 Applying LoRA (rank={self.lora_rank}, targets={self.lora_targets})...")

        # Determine target modules based on lora_targets
        target_modules = []
        if 'attn' in self.lora_targets:
            target_modules.extend([
                'to_q', 'to_k', 'to_v', 'to_out.0'
            ])
        if 'conv' in self.lora_targets:
            target_modules.extend([
                'conv1', 'conv2', 'conv_shortcut'
            ])

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,  # Typical: alpha = rank
            target_modules=target_modules,
            lora_dropout=0.0,
            bias='none',
            task_type=None  # For non-task-specific models
        )

        # Apply LoRA to U-Net
        # Note: We only want to apply LoRA to alignment layers, but peft
        # applies to all matching modules. We'll freeze non-alignment layers
        # after applying LoRA.
        self.unet = get_peft_model(self.unet, lora_config)

        # Freeze non-alignment layers
        self._freeze_non_alignment_layers()

        print(f"✅ Applied LoRA to U-Net")
        print(f"   Trainable params: {self._count_trainable_params()}")

    def _freeze_non_alignment_layers(self):
        """
        Freeze all U-Net parameters except those in alignment layers.

        Alignment layers:
        - 'enc_last': down_blocks[3]
        - 'mid': mid_block
        - 'dec_first': up_blocks[0]
        """
        # First, freeze everything
        for param in self.unet.parameters():
            param.requires_grad = False

        # Then unfreeze alignment layers
        for layer_name in self.align_layers:
            if layer_name == 'enc_last':
                module = self.unet.down_blocks[3]
            elif layer_name == 'mid':
                module = self.unet.mid_block
            elif layer_name == 'dec_first':
                module = self.unet.up_blocks[0]

            # Unfreeze LoRA parameters in this module
            for name, param in module.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True

        print(f"✅ Unfroze LoRA params in: {self.align_layers}")

    def _count_trainable_params(self) -> Tuple[int, int]:
        """
        Count trainable parameters.

        Returns:
            (trainable, total) parameter counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        return_align_tokens: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            noisy_latent: [B, 4, 64, 64] noisy latent
            timesteps: [B] timesteps
            text_embeddings: [B, 77, 768] CLIP text embeddings
            return_align_tokens: Whether to return alignment tokens

        Returns:
            outputs: Dict with keys:
                - 'pred': [B, 4, 64, 64] prediction (epsilon or v)
                - 'align_tokens': Dict[layer_name → [B, 196, 1024]] (if return_align_tokens=True)
        """
        # Clear previous features
        self.hook_manager.clear_features()

        # Forward through U-Net
        # Note: diffusers UNet returns ModelOutput object with 'sample' attribute
        unet_output = self.unet(
            noisy_latent,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=True
        )

        pred = unet_output.sample  # [B, 4, 64, 64]

        outputs = {'pred': pred}

        # Extract and project features if requested
        if return_align_tokens:
            features = self.hook_manager.get_features()
            align_tokens = {}

            for layer_name in self.align_layers:
                if layer_name not in features:
                    raise RuntimeError(
                        f"Feature not found for layer: {layer_name}. "
                        f"Available: {list(features.keys())}"
                    )

                # Get feature [B, C, H, W]
                feat = features[layer_name]

                # Project to tokens [B, 196, 1024]
                tokens = self.align_heads[layer_name](feat)

                align_tokens[layer_name] = tokens

            outputs['align_tokens'] = align_tokens

        return outputs

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Get list of trainable parameters.

        Returns:
            params: List of trainable parameters (LoRA + AlignHeads)
        """
        params = []

        # LoRA parameters in U-Net
        for param in self.unet.parameters():
            if param.requires_grad:
                params.append(param)

        # AlignHead parameters
        for param in self.align_heads.parameters():
            params.append(param)

        return params

    def print_trainable_params(self):
        """Print summary of trainable parameters."""
        trainable, total = self._count_trainable_params()

        print("="*80)
        print("Trainable Parameters")
        print("="*80)
        print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"Total: {total:,}")
        print()

        # Breakdown by module
        unet_trainable = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        align_trainable = sum(p.numel() for p in self.align_heads.parameters())

        print(f"U-Net (LoRA): {unet_trainable:,}")
        print(f"AlignHeads: {align_trainable:,}")
        print("="*80)


def test_sd15_unet_aligned():
    """Test SD15UNetAligned with random input."""
    print("="*80)
    print("Testing SD15UNetAligned")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = SD15UNetAligned(
        pretrained_model_name='runwayml/stable-diffusion-v1-5',
        align_layers=['mid'],
        use_lora=True,
        lora_rank=32,
        device=device
    )

    model.print_trainable_params()

    # Create dummy inputs
    B = 2
    noisy_latent = torch.randn(B, 4, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (B,)).to(device)
    text_embeddings = torch.randn(B, 77, 768).to(device)

    print(f"\n🔍 Testing forward pass (batch_size={B})...")

    # Forward without alignment tokens
    out1 = model(noisy_latent, timesteps, text_embeddings, return_align_tokens=False)
    print(f"✅ pred shape: {out1['pred'].shape}")

    # Forward with alignment tokens
    out2 = model(noisy_latent, timesteps, text_embeddings, return_align_tokens=True)
    print(f"✅ pred shape: {out2['pred'].shape}")
    for layer_name, tokens in out2['align_tokens'].items():
        print(f"   {layer_name}: {tokens.shape}")

    print("\n🎉 SD15UNetAligned test passed!")


if __name__ == '__main__':
    test_sd15_unet_aligned()
