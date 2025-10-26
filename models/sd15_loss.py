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
    target = noise if prediction='epsilon' else scheduler.get_velocity(x0, noise, t)
    loss = SD15REPALoss()(model, noisy_latent, timesteps, text_embeds, target, dino_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    Combines diffusion loss与 token+流形对齐损失，最终形式
    L_total = L_diff + λ * (L_token + w * L_manifold)。
    """

    def __init__(
        self,
        align_coeff=0.8,
        manifold_coeff=3.0,
    ):
        """
        Initialize SD15REPALoss.

        Args:
            align_coeff: λ - coefficient for alignment loss
            manifold_coeff: w - coefficient for manifold loss
        """
        super().__init__()

        self.align_coeff = align_coeff
        self.manifold_coeff = manifold_coeff

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
        target,
        target_tokens,
        return_details=False,
    ):
        """
        Compute total loss for U-REPA SD-1.5.

        Args:
            model: SD15UNetAligned model
            noisy_latent: [B, 4, 64, 64] noisy latent
            timesteps: [B] integer timesteps in [0, num_train_steps)
            text_embeddings: [B, 77, 768] CLIP text embeddings
            target: [B, 4, 64, 64] diffusion target (noise or v)
            target_tokens: dict or tensor of DINO tokens
            return_details: If True, return detailed loss dict
        """
        model_timesteps = timesteps.long()

        # Forward through model with alignment
        output = model(
            noisy_latent,
            model_timesteps,
            text_embeddings,
            return_align_tokens=True
        )

        pred = output['pred']                      # [B, 4, 64, 64]
        align_tokens = output['align_tokens']      # {'mid': [B, 196, 1024]}

        # 1. Diffusion loss
        diffusion_loss = self.compute_diffusion_loss(pred, target)  # [B]
        diffusion_loss = diffusion_loss.mean()  # scalar

        if isinstance(target_tokens, dict):
            target_map = target_tokens
        else:
            target_map = {'mid': target_tokens}

        common_layers = sorted(set(align_tokens.keys()) & set(target_map.keys()))
        if not common_layers:
            raise ValueError("No overlapping layers between predicted tokens and target tokens.")

        token_losses = []
        manifold_losses = []
        for layer in common_layers:
            pred_tokens = align_tokens[layer]
            tgt_tokens = target_map[layer].to(pred_tokens.device).type_as(pred_tokens)
            token_losses.append(self.compute_token_loss(pred_tokens, tgt_tokens))
            manifold_losses.append(self.compute_manifold_loss(pred_tokens, tgt_tokens))

        token_loss = torch.stack(token_losses).mean()
        manifold_loss = torch.stack(manifold_losses).mean()

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
    B = 2
    pred = torch.randn(B, 4, 64, 64)
    target = torch.randn_like(pred)
    pred_tokens = {
        'mid': torch.randn(B, 196, 1024),
        'enc_last': torch.randn(B, 196, 1024),
    }
    target_tokens = {
        'mid': torch.randn(B, 196, 1024),
        'enc_last': torch.randn(B, 196, 1024),
    }

    class DummyModel(torch.nn.Module):
        def forward(self, latents, timesteps, encoder_hidden_states, return_align_tokens=False):
            return {
                'pred': pred,
                'align_tokens': pred_tokens,
            }

    model = DummyModel()
    loss_fn = SD15REPALoss()
    loss = loss_fn(
        model=model,
        noisy_latent=torch.randn_like(pred),
        timesteps=torch.randint(0, 1000, (B,)),
        text_embeddings=torch.randn(B, 77, 768),
        target=target,
        target_tokens=target_tokens,
    )
    assert loss.requires_grad


if __name__ == '__main__':
    test_loss_computation()
