#!/usr/bin/env python3
"""
Alignment diagnostics for SD1.5 + REPA checkpoints.

Metrics:
  1. Linear probing accuracy (logistic regression) on ImageNet subset.
  2. Centered KNN Alignment (CKNNA) between model tokens and DINO tokens.
  3. t-wise CKNNA at multiple diffusion timesteps (e.g., t ∈ {0, 0.25, 0.5}).

Outputs JSON with metrics per checkpoint.

Example:
    python evaluation/run_alignment_metrics.py \
        --config configs/sd15_repa_档A.yaml \
        --csv-path data/val_50k.csv \
        --latent-dir data/val_vae_latents_lmdb \
        --dino-dir data/val_dino_tokens_lmdb \
        --clip-embeddings data/clip_embeddings_1001.pt \
        --base-model-dir checkpoints/sd15_base \
        --checkpoints \
            vanilla:None \
            init:checkpoints/sd15_repa_init.pt \
            step24k:models/sd15_repa_step24k/model.pt \
        --output eval_outputs/alignment_step24k.json \
        --sample-size 5000
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, '/workspace/REPA')

from dataset_sd15 import SD15AlignedDataset
from models.sd15_unet_aligned import SD15UNetAligned
from models.sd15_loss import SD15REPALoss
from diffusers import DDPMScheduler


# -----------------------------------------------------------------------------#
# Utilities


def load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def prepare_dataset(
    csv_path: str,
    latent_dir: str,
    dino_dir: str,
    clip_embeddings: str,
    align_layers: List[str],
    sample_size: Optional[int],
    seed: int,
) -> SD15AlignedDataset:
    dataset = SD15AlignedDataset(
        csv_path=csv_path,
        latent_dir=latent_dir,
        dino_dir=dino_dir,
        clip_embeddings_path=clip_embeddings,
        align_layers=align_layers,
        cfg_dropout=0.0,
        seed=seed,
    )
    if sample_size is None or sample_size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:sample_size]
    return Subset(dataset, indices)


def instantiate_model(config: Dict, base_model_dir: str, checkpoint: Optional[Path], device: torch.device) -> SD15UNetAligned:
    model = SD15UNetAligned(
        pretrained_model_name=base_model_dir,
        align_layers=config.get("align_layers", ["mid"]),
        dino_dim=config.get("dino_D", 1024),
        use_lora=config.get("use_lora", True),
        lora_rank=config.get("lora_rank", 8),
        lora_targets=config.get("lora_targets", "attn"),
        lora_alpha=config.get("lora_alpha", config.get("lora_rank", 8)),
        device=str(device),
    )
    if checkpoint is not None:
        if str(checkpoint).endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(checkpoint, device=str(device))
        else:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys for {checkpoint.name}: {missing[:5]}")
        if unexpected:
            print(f"[WARN] Unexpected keys for {checkpoint.name}: {unexpected[:5]}")
    model.eval()
    return model


def gather_features(
    model: SD15UNetAligned,
    dataloader: DataLoader,
    scheduler: DDPMScheduler,
    prediction_type: str,
    timesteps: Iterable[int],
    device: torch.device,
    null_prompt_embedding: Optional[torch.Tensor] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Returns:
        {timestep: {"model": np.ndarray[B, D], "model_uncond": np.ndarray[B, D],
                    "dino": np.ndarray[B, D], "labels": np.ndarray[B]}}
    model: conditioned features (for CKNNA)
    model_uncond: unconditioned features using null prompt (for Linear Probing)
    """
    outputs = {}
    align_layer = model.align_layers[0]  # assume single layer for mean pooling
    for timestep in timesteps:
        outputs[timestep] = {"model": [], "model_uncond": [], "dino": [], "labels": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            latent = batch["latent"].to(device)
            # VAE stores [mean, std] as 8 channels, but we only need mean (first 4 channels)
            if latent.shape[1] == 8:
                latent = latent[:, :4]  # Take only the mean part
            labels = batch["class_id"].numpy()
            dino_tokens = batch["dino_tokens"][align_layer].numpy()
            dino_mean = dino_tokens.mean(axis=1)  # [B,1024]

            batch_size = latent.shape[0]

            for timestep in timesteps:
                if timestep == -1:
                    noisy_latent = latent
                    ts = torch.zeros(batch_size, dtype=torch.long, device=device)
                else:
                    ts = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
                    noise = torch.randn_like(latent)
                    noisy_latent = scheduler.add_noise(latent, noise, ts)

                # 1. Conditioned forward (with real prompt) - for CKNNA
                enc_hidden = batch["encoder_hidden_states"].to(device)
                output = model(
                    noisy_latent,
                    ts,
                    enc_hidden,
                    return_align_tokens=True,
                )
                align_tokens = output["align_tokens"][align_layer]  # [B,256,1024]
                align_mean = align_tokens.mean(dim=1)  # [B,1024]
                align_mean = torch.nn.functional.normalize(align_mean, dim=-1)
                outputs[timestep]["model"].append(align_mean.cpu().numpy())

                # 2. Unconditioned forward (with null prompt) - for Linear Probing
                if null_prompt_embedding is not None:
                    # Match dtype and device of the conditioned embedding
                    enc_hidden_null = null_prompt_embedding.unsqueeze(0).expand(batch_size, -1, -1)
                    enc_hidden_null = enc_hidden_null.to(device=device, dtype=enc_hidden.dtype)
                    output_uncond = model(
                        noisy_latent,
                        ts,
                        enc_hidden_null,
                        return_align_tokens=True,
                    )
                    align_tokens_uncond = output_uncond["align_tokens"][align_layer]  # [B,256,1024]
                    align_mean_uncond = align_tokens_uncond.mean(dim=1)  # [B,1024]
                    align_mean_uncond = torch.nn.functional.normalize(align_mean_uncond, dim=-1)
                    outputs[timestep]["model_uncond"].append(align_mean_uncond.cpu().numpy())

                outputs[timestep]["dino"].append(dino_mean)
                outputs[timestep]["labels"].append(labels)

    for timestep in timesteps:
        outputs[timestep]["model"] = np.concatenate(outputs[timestep]["model"], axis=0)
        if null_prompt_embedding is not None:
            outputs[timestep]["model_uncond"] = np.concatenate(outputs[timestep]["model_uncond"], axis=0)
        outputs[timestep]["dino"] = np.concatenate(outputs[timestep]["dino"], axis=0)
        outputs[timestep]["labels"] = np.concatenate(outputs[timestep]["labels"], axis=0)
    return outputs


def linear_probing_accuracy(features: np.ndarray, labels: np.ndarray, test_size: float = 0.2, seed: int = 42) -> float:
    # Check if stratification is possible (need at least 2 samples per class)
    from collections import Counter
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    stratify_labels = labels if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=stratify_labels
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=0)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def compute_cknna(emb_x: np.ndarray, emb_y: np.ndarray, k: int = 10) -> float:
    """
    Approximation: compute overlap ratio between k-NN graphs of X and Y.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs_x = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_x)
    nbrs_y = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_y)

    neigh_x = nbrs_x.kneighbors(return_distance=False)
    neigh_y = nbrs_y.kneighbors(return_distance=False)

    # remove self index (first column)
    neigh_x = neigh_x[:, 1:]
    neigh_y = neigh_y[:, 1:]

    scores = []
    for idx in range(len(emb_x)):
        set_x = set(neigh_x[idx])
        set_y = set(neigh_y[idx])
        overlap = len(set_x & set_y) / len(set_x | set_y)
        scores.append(overlap)
    return float(np.mean(scores))


# -----------------------------------------------------------------------------#
# Main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alignment metrics for SD1.5+REPA.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--dino-dir", type=str, required=True)
    parser.add_argument("--clip-embeddings", type=str, required=True)
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of name:path entries (use name:None for vanilla base). Example: vanilla:None init:ckpt_init.pt trained:ckpt.pt",
    )
    parser.add_argument("--output", type=str, required=True, help="JSON output path.")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timesteps", type=str, default="-1,250,500", help="Comma-separated timesteps; use -1 for clean latent.")
    parser.add_argument("--scheduler-num-train-steps", type=int, default=1000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--skip-linear", action="store_true", help="Skip linear probing evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    timesteps = [int(t.strip()) for t in args.timesteps.split(",")]

    config = load_yaml(Path(args.config))
    device = torch.device(args.device)

    dataset = prepare_dataset(
        csv_path=args.csv_path,
        latent_dir=args.latent_dir,
        dino_dir=args.dino_dir,
        clip_embeddings=args.clip_embeddings,
        align_layers=config.get("align_layers", ["mid"]),
        sample_size=args.sample_size,
        seed=args.seed,
    )

    # Load null prompt embedding (empty string CLIP embedding)
    # This is typically stored as the last embedding (class_id=1000) in the CLIP embeddings file
    clip_embeddings_all = torch.load(args.clip_embeddings, map_location="cpu", weights_only=True)
    null_prompt_embedding = clip_embeddings_all[-1]  # [77, 768] for SD1.5
    print(f"✓ Loaded null prompt embedding: shape={null_prompt_embedding.shape}, dtype={null_prompt_embedding.dtype}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    scheduler = DDPMScheduler.from_pretrained(
        config.get("pretrained_model_name_or_path", args.base_model_dir),
        subfolder="scheduler",
    )
    scheduler.config.num_train_timesteps = args.scheduler_num_train_steps
    prediction_type = config["schedule"]["parametrization"]

    results = {}

    for entry in args.checkpoints:
        name, path_str = entry.split(":", 1)
        checkpoint = None if path_str.lower() == "none" else Path(path_str)
        print(f"\n=== Evaluating {name} ===")

        model = instantiate_model(config, args.base_model_dir, checkpoint, device=device)

        features = gather_features(
            model=model,
            dataloader=dataloader,
            scheduler=scheduler,
            prediction_type=prediction_type,
            timesteps=timesteps,
            device=device,
            null_prompt_embedding=null_prompt_embedding,
        )

        result_entry = {}
        if not args.skip_linear:
            clean_t = timesteps[0]
            feat_clean_uncond = features[clean_t]["model_uncond"]
            labels_clean = features[clean_t]["labels"]
            acc_model = linear_probing_accuracy(feat_clean_uncond, labels_clean, seed=args.seed)
            acc_dino = linear_probing_accuracy(features[clean_t]["dino"], labels_clean, seed=args.seed)
            result_entry["linear_probing"] = {
                "model_tokens_acc": acc_model,
                "dino_tokens_acc": acc_dino,
            }

        # CKNNA for each timestep
        # Use CONDITIONED features (real prompts) for CKNNA to measure training-time alignment
        knn_scores = {}
        for t in timesteps:
            knn_scores[str(t)] = compute_cknna(features[t]["model"], features[t]["dino"], k=args.knn_k)
        result_entry["cknna"] = knn_scores

        results[name] = result_entry
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Alignment metrics saved to {output_path}")


if __name__ == "__main__":
    main()
