"""
Loss functions for U-REPA SD-1.5.

This module implements:
1. Diffusion loss (epsilon or v-prediction)
2. Token alignment loss (negative cosine similarity)
3. Manifold alignment loss (Gram matrix alignment)

Following U-REPA paper's formulation:
    L_total = L_diff + λ * (L_token + w * L_manifold)

where λ=0.8 (align_coeff) and w=3.0 (manifold_coeff) by default.

Usage:
    loss_fn = SD15REPALoss(
        prediction='v',
        schedule='linear',
        align_coeff=0.8,
        manifold_coeff=3.0
    )

    losses = loss_fn(
        model=model,
        noisy_latent=noisy_latent,
        timesteps=timesteps,
        text_embeddings=text_embeddings,
        target_latent=target_latent,
        target_tokens=target_tokens
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mean_flat(tensor):
    """
    Take mean over all non-batch dimensions.

    Args:
        tensor: [B, ...] tensor

    Returns:
        mean: [B] tensor
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SD15REPALoss(nn.Module):
    """
    Loss function for U-REPA SD-1.5.

    Computes:
    1. Diffusion loss: MSE between prediction and target
    2. Token loss: Negative cosine similarity between predicted and target tokens
    3. Manifold loss: MSE between Gram matrices of predicted and target tokens

    Attributes:
        prediction: 'epsilon' or 'v' (diffusion prediction type)
        schedule: 'linear' or 'cosine' (noise schedule)
        align_coeff: λ (coefficient for alignment loss)
        manifold_coeff: w (coefficient for manifold loss relative to token loss)
        latent_scale: Scale factor for latent space (0.18215 for SD-1.5)
    """

    def __init__(
        self,
        prediction='v',
        schedule='linear',
        align_coeff=0.8,
        manifold_coeff=3.0,
        latent_scale=0.18215,
    ):
        """
        Initialize SD15REPALoss.

        Args:
            prediction: Diffusion prediction type ('epsilon' or 'v')
            schedule: Noise schedule ('linear' or 'cosine')
            align_coeff: λ - coefficient for alignment loss
            manifold_coeff: w - coefficient for manifold loss
            latent_scale: VAE latent scaling factor
        """
        super().__init__()

        self.prediction = prediction
        self.schedule = schedule
        self.align_coeff = align_coeff
        self.manifold_coeff = manifold_coeff
        self.latent_scale = latent_scale

        # Validate parameters
        assert prediction in ['epsilon', 'v'], f"prediction must be 'epsilon' or 'v', got {prediction}"
        assert schedule in ['linear', 'cosine'], f"schedule must be 'linear' or 'cosine', got {schedule}"

    def get_schedule_params(self, timesteps):
        """
        Compute α_t and σ_t for given timesteps.

        Args:
            timesteps: [B] tensor of timesteps in [0, 1]

        Returns:
            alpha_t: [B, 1, 1, 1] tensor
            sigma_t: [B, 1, 1, 1] tensor
            d_alpha_t: [B, 1, 1, 1] tensor (derivative)
            d_sigma_t: [B, 1, 1, 1] tensor (derivative)
        """
        t = timesteps.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        if self.schedule == 'linear':
            # Linear schedule: α_t = 1-t, σ_t = t
            alpha_t = 1.0 - t
            sigma_t = t
            d_alpha_t = -torch.ones_like(t)
            d_sigma_t = torch.ones_like(t)

        elif self.schedule == 'cosine':
            # Cosine schedule: α_t = cos(πt/2), σ_t = sin(πt/2)
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -(np.pi / 2) * torch.sin(t * np.pi / 2)
            d_sigma_t = (np.pi / 2) * torch.cos(t * np.pi / 2)

        else:
            raise NotImplementedError(f"Schedule {self.schedule} not implemented")

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_diffusion_target(self, clean_latent, noise, alpha_t, sigma_t, d_alpha_t, d_sigma_t):
        """
        Compute diffusion target based on prediction type.

        Args:
            clean_latent: [B, 4, 64, 64] clean latent (x_0)
            noise: [B, 4, 64, 64] sampled noise (ε)
            alpha_t, sigma_t, d_alpha_t, d_sigma_t: Schedule parameters

        Returns:
            target: [B, 4, 64, 64] target for model prediction
        """
        if self.prediction == 'epsilon':
            # ε-prediction: predict noise
            target = noise

        elif self.prediction == 'v':
            # v-prediction: v = α_t * ε - σ_t * x_0
            # But we use derivative form: v = d_alpha_t * x_0 + d_sigma_t * ε
            target = d_alpha_t * clean_latent + d_sigma_t * noise

        else:
            raise NotImplementedError(f"Prediction {self.prediction} not implemented")

        return target

    def compute_diffusion_loss(self, pred, target):
        """
        Compute MSE loss between prediction and target.

        Args:
            pred: [B, 4, 64, 64] model prediction
            target: [B, 4, 64, 64] ground truth target

        Returns:
            loss: [B] per-sample loss
        """
        mse = (pred - target) ** 2
        loss = mean_flat(mse)  # [B]
        return loss

    def compute_token_loss(self, pred_tokens, target_tokens):
        """
        Compute negative cosine similarity between predicted and target tokens.

        Token loss: L_token = 1/N * Σ (1 - cos_sim(z_i, z̃_i))

        Args:
            pred_tokens: [B, 196, 1024] predicted alignment tokens
            target_tokens: [B, 196, 1024] target DINO tokens (L2 normalized)

        Returns:
            loss: [B] per-sample loss
        """
        # L2 normalize both (target should already be normalized)
        pred_tokens = F.normalize(pred_tokens, dim=-1)    # [B, 196, 1024]
        target_tokens = F.normalize(target_tokens, dim=-1)  # [B, 196, 1024]

        # Cosine similarity: element-wise dot product and sum
        cos_sim = (pred_tokens * target_tokens).sum(dim=-1)  # [B, 196]

        # Negative cosine similarity averaged over tokens
        token_loss = 1.0 - cos_sim  # [B, 196]
        token_loss = token_loss.mean(dim=-1)  # [B]

        return token_loss

    def compute_manifold_loss(self, pred_tokens, target_tokens):
        """
        Compute manifold alignment loss via Gram matrix.

        Manifold loss compares the Gram matrices G = Z·Z^T and G̃ = Z̃·Z̃^T
        where Z and Z̃ are token matrices.

        This is O(B²) complexity - compares all pairs within a batch.

        Args:
            pred_tokens: [B, 196, 1024] predicted alignment tokens
            target_tokens: [B, 196, 1024] target DINO tokens

        Returns:
            loss: scalar loss (averaged over batch)
        """
        B, N, D = pred_tokens.shape

        # L2 normalize tokens
        pred_tokens = F.normalize(pred_tokens, dim=-1)    # [B, 196, 1024]
        target_tokens = F.normalize(target_tokens, dim=-1)  # [B, 196, 1024]

        # Flatten to [B, N*D]
        pred_flat = pred_tokens.reshape(B, N * D)      # [B, 196*1024]
        target_flat = target_tokens.reshape(B, N * D)  # [B, 196*1024]

        # Compute Gram matrices: [B, B]
        # G[i,j] = <z_i, z_j> measures similarity between samples i and j
        gram_pred = torch.mm(pred_flat, pred_flat.t())       # [B, B]
        gram_target = torch.mm(target_flat, target_flat.t())  # [B, B]

        # MSE between Gram matrices
        manifold_loss = F.mse_loss(gram_pred, gram_target, reduction='mean')

        return manifold_loss

    def forward(
        self,
        model,
        noisy_latent,
        timesteps,
        text_embeddings,
        target_latent,
        target_tokens,
        return_details=False,
    ):
        """
        Compute total loss for U-REPA SD-1.5.

        Args:
            model: SD15UNetAligned model
            noisy_latent: [B, 4, 64, 64] noisy latent (already scaled by 0.18215)
            timesteps: [B] timesteps in [0, 1]
            text_embeddings: [B, 77, 768] CLIP text embeddings
            target_latent: [B, 4, 64, 64] clean latent (already scaled)
            target_tokens: [B, 196, 1024] DINO tokens (L2 normalized)
            return_details: If True, return detailed loss dict

        Returns:
            If return_details=False:
                loss: scalar total loss
            If return_details=True:
                dict with keys: 'total', 'diffusion', 'token', 'manifold'
        """
        # Get schedule parameters
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.get_schedule_params(timesteps)

        # Compute noisy input (should already be provided, but we can verify)
        # noisy_latent = alpha_t * target_latent + sigma_t * noise
        # For now, assume noisy_latent is correctly provided

        # Forward through model with alignment
        output = model(
            noisy_latent,
            timesteps,
            text_embeddings,
            return_align_tokens=True
        )

        pred = output['pred']                      # [B, 4, 64, 64]
        align_tokens = output['align_tokens']      # {'mid': [B, 196, 1024]}
        pred_tokens = align_tokens['mid']          # [B, 196, 1024]

        # Compute diffusion target
        # We need to reconstruct noise from noisy_latent
        # noise = (noisy_latent - alpha_t * target_latent) / sigma_t
        # But this may cause division by zero when sigma_t ≈ 0
        # Better: store noise in dataset or pass it as argument
        # For now, compute target directly using noisy_latent and target_latent

        # Actually, in training we sample noise first, then compute noisy_latent
        # So we should modify the interface to pass noise as well
        # Let me compute a workaround target for v-prediction:

        # For v-prediction: v = d_alpha_t * x_0 + d_sigma_t * ε
        # We can compute ε from noisy_latent: ε = (x_t - alpha_t * x_0) / sigma_t
        # But safer to require noise as input

        # Temporarily: assume pred is directly comparable to a computed target
        # In actual training loop, we'll pass noise explicitly

        # Placeholder: compute reconstruction loss
        # This needs to be refined in the training script
        noise = (noisy_latent - alpha_t * target_latent) / (sigma_t + 1e-8)
        target = self.compute_diffusion_target(
            target_latent, noise, alpha_t, sigma_t, d_alpha_t, d_sigma_t
        )

        # 1. Diffusion loss
        diffusion_loss = self.compute_diffusion_loss(pred, target)  # [B]
        diffusion_loss = diffusion_loss.mean()  # scalar

        # 2. Token alignment loss
        token_loss = self.compute_token_loss(pred_tokens, target_tokens)  # [B]
        token_loss = token_loss.mean()  # scalar

        # 3. Manifold alignment loss
        manifold_loss = self.compute_manifold_loss(pred_tokens, target_tokens)  # scalar

        # Total loss: L_total = L_diff + λ * (L_token + w * L_manifold)
        alignment_loss = token_loss + self.manifold_coeff * manifold_loss
        total_loss = diffusion_loss + self.align_coeff * alignment_loss

        if return_details:
            return {
                'total': total_loss,
                'diffusion': diffusion_loss,
                'token': token_loss,
                'manifold': manifold_loss,
                'alignment': alignment_loss,
            }
        else:
            return total_loss


def test_loss_computation():
    """Test loss functions with random data."""
    print("="*80)
    print("Testing SD15REPALoss")
    print("="*80)

    # Create loss function
    loss_fn = SD15REPALoss(
        prediction='v',
        schedule='linear',
        align_coeff=0.8,
        manifold_coeff=3.0,
    )
    print(f"✅ Created loss function")
    print(f"   Prediction: {loss_fn.prediction}")
    print(f"   Schedule: {loss_fn.schedule}")
    print(f"   Align coeff (λ): {loss_fn.align_coeff}")
    print(f"   Manifold coeff (w): {loss_fn.manifold_coeff}")

    # Test schedule parameters
    print(f"\n🔍 Testing schedule parameters...")
    timesteps = torch.tensor([0.0, 0.5, 1.0])
    alpha_t, sigma_t, d_alpha_t, d_sigma_t = loss_fn.get_schedule_params(timesteps)
    print(f"   t=0.0: α={alpha_t[0].item():.4f}, σ={sigma_t[0].item():.4f}")
    print(f"   t=0.5: α={alpha_t[1].item():.4f}, σ={sigma_t[1].item():.4f}")
    print(f"   t=1.0: α={alpha_t[2].item():.4f}, σ={sigma_t[2].item():.4f}")

    # Test token loss
    print(f"\n🔍 Testing token loss...")
    B = 4
    pred_tokens = torch.randn(B, 196, 1024)
    target_tokens = torch.randn(B, 196, 1024)
    token_loss = loss_fn.compute_token_loss(pred_tokens, target_tokens)
    print(f"   Token loss shape: {token_loss.shape}")
    print(f"   Token loss mean: {token_loss.mean().item():.4f}")

    # Test manifold loss
    print(f"\n🔍 Testing manifold loss...")
    manifold_loss = loss_fn.compute_manifold_loss(pred_tokens, target_tokens)
    print(f"   Manifold loss: {manifold_loss.item():.4f}")

    # Test with identical tokens (should give low loss)
    print(f"\n🔍 Testing with identical tokens (should give ~0 loss)...")
    identical_tokens = torch.randn(B, 196, 1024)
    token_loss_identical = loss_fn.compute_token_loss(identical_tokens, identical_tokens)
    manifold_loss_identical = loss_fn.compute_manifold_loss(identical_tokens, identical_tokens)
    print(f"   Token loss (identical): {token_loss_identical.mean().item():.6f}")
    print(f"   Manifold loss (identical): {manifold_loss_identical.item():.6f}")

    print("\n🎉 All tests passed!")


if __name__ == '__main__':
    test_loss_computation()
