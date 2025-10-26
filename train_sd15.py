import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset_sd15 import SD15AlignedDataset
from models.sd15_unet_aligned import SD15UNetAligned
from models.sd15_loss import SD15REPALoss
from utils_sd15 import ModelEMA, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD1.5 + U-REPA alignment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sd15_repa_档A.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for output directory.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from.")
    return parser.parse_args()


def build_dataloader(config: Dict) -> DataLoader:
    dataset = SD15AlignedDataset(
        csv_path=config["csv_path"],
        latent_dir=config["latent_dir"],
        dino_dir=config["dino_dir"],
        clip_embeddings_path=config["clip_embeddings_path"],
        align_layers=config["align_layers"],
        cfg_dropout=config.get("cfg_dropout", 0.1),
        seed=config.get("seed", 42),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    return loader


def build_model(config: Dict, device: torch.device) -> SD15UNetAligned:
    model = SD15UNetAligned(
        pretrained_model_name=config["pretrained_model_name_or_path"],
        align_layers=config["align_layers"],
        dino_dim=config["dino_D"],
        use_lora=config.get("use_lora", True),
        lora_rank=config.get("lora_rank", 32),
        lora_targets=config.get("lora_targets", "attn+conv"),
        device=str(device),
    )
    return model


def build_optimizer(model: SD15UNetAligned, config: Dict) -> AdamW:
    params = model.get_trainable_params()
    optimizer = AdamW(
        params,
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=config.get("weight_decay", 0.0),
    )
    return optimizer


def build_scheduler(optimizer: AdamW, config: Dict) -> LambdaLR:
    max_steps = config["max_steps"]
    warmup_ratio = config.get("warmup_ratio", 0.1)
    warmup_steps = int(max_steps * warmup_ratio)
    schedule_type = config.get("lr_schedule", "cosine")

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        if schedule_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule_type == "linear":
            return 1.0 - progress
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def prepare_batch(batch: Dict, device: torch.device) -> Dict:
    latent = batch["latent"].to(device)
    encoder_hidden_states = batch["encoder_hidden_states"].to(device)
    class_id = batch["class_id"].to(device)
    dino_tokens = {k: v.to(device) for k, v in batch["dino_tokens"].items()}
    return {
        "latent": latent,
        "encoder_hidden_states": encoder_hidden_states,
        "class_id": class_id,
        "dino_tokens": dino_tokens,
    }


def save_checkpoint(
    accelerator: Accelerator,
    ema: ModelEMA,
    step: int,
    output_dir: Path,
) -> Path:
    if not accelerator.is_main_process:
        return output_dir

    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accelerator.save_state(str(ckpt_dir))

    if ema is not None:
        torch.save(ema.state_dict(), ckpt_dir / "ema.pt")

    return ckpt_dir


def maybe_load_checkpoint(
    accelerator: Accelerator,
    ema: ModelEMA,
    resume_dir: Optional[str],
) -> int:
    """
    Load a previously saved accelerator state (and EMA) if provided.
    Returns the global step to resume from.
    """
    if resume_dir is None:
        return 0

    accelerator.print(f"Resuming from checkpoint: {resume_dir}")
    accelerator.load_state(resume_dir)

    ema_path = Path(resume_dir) / "ema.pt"
    if ema is not None and ema_path.exists():
        state = torch.load(ema_path, map_location=accelerator.device)
        ema.load_state_dict(state)

    metadata_path = Path(resume_dir) / "training_state.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location="cpu")
        return metadata.get("global_step", 0)

    return 0


def store_training_state(accelerator: Accelerator, step: int, target_dir: Path):
    if not accelerator.is_main_process:
        return
    torch.save({"global_step": step}, target_dir / "training_state.pt")


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    output_dir = Path(config.get("output_dir", "exps"))
    project_dir = output_dir / "accelerate"
    project_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("grad_accum_steps", 1),
        mixed_precision=config.get("mixed_precision", "bf16"),
        log_with=None if config.get("report_to", "none") == "none" else config["report_to"],
        project_config=ProjectConfiguration(project_dir=str(project_dir)),
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.get("seed", 42), device_specific=True)

    if accelerator.log_with is not None:
        accelerator.init_trackers(
            project_name=config["experiment_name"],
            config=config,
        )

    dataloader = build_dataloader(config)
    model = build_model(config, accelerator.device)
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_scheduler(optimizer, config)

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    loss_fn = SD15REPALoss(
        align_coeff=config["align_coeff"],
        manifold_coeff=config["manifold_coeff"],
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )
    prediction_type = config["schedule"]["parametrization"]

    ema = None
    if config.get("ema_decay", None):
        ema = ModelEMA(accelerator.unwrap_model(model), decay=config["ema_decay"])

    start_step = maybe_load_checkpoint(
        accelerator,
        ema,
        args.resume or config.get("resume_from_checkpoint"),
    )
    global_step = start_step
    max_steps = config["max_steps"]
    log_interval = config.get("log_interval", 100)
    save_interval = config.get("save_interval", 10000)

    progress_bar = tqdm(
        total=max_steps,
        initial=global_step,
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                batch = prepare_batch(batch, accelerator.device)
                clean_latent = batch["latent"]
                encoder_hidden_states = batch["encoder_hidden_states"]
                dino_tokens = batch["dino_tokens"]

                batch_size = clean_latent.size(0)
                timesteps = torch.randint(
                    0,
                    1000,
                    (batch_size,),
                    device=clean_latent.device,
                    dtype=torch.long,
                )

                with torch.no_grad():
                    noise = torch.randn_like(clean_latent)
                    noisy_latent = noise_scheduler.add_noise(clean_latent, noise, timesteps)
                    if prediction_type == "v":
                        target = noise_scheduler.get_velocity(clean_latent, noise, timesteps)
                    elif prediction_type == "epsilon":
                        target = noise
                    else:
                        raise ValueError(f"Unsupported prediction type: {prediction_type}")

                loss_components = loss_fn(
                    model=model,
                    noisy_latent=noisy_latent,
                    timesteps=timesteps,
                    text_embeddings=encoder_hidden_states,
                    target=target,
                    target_tokens=dino_tokens,
                    return_details=True,
                )
                loss = loss_components["total"]

                accelerator.backward(loss)

                if accelerator.sync_gradients and config.get("gradient_clip", None):
                    accelerator.clip_grad_norm_(model.parameters(), config["gradient_clip"])

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None and accelerator.sync_gradients:
                    ema.update(accelerator.unwrap_model(model))

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if accelerator.is_main_process and global_step % log_interval == 0:
                    log_payload = {
                        "loss/total": loss_components["total"].item(),
                        "loss/diffusion": loss_components["diffusion"].item(),
                        "loss/token": loss_components["token"].item(),
                        "loss/manifold": loss_components["manifold"].item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.log(log_payload, step=global_step)

                if (
                    accelerator.is_main_process
                    and save_interval > 0
                    and global_step % save_interval == 0
                ):
                    ckpt_dir = save_checkpoint(accelerator, ema, global_step, output_dir)
                    store_training_state(accelerator, global_step, ckpt_dir)

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    ckpt_dir = save_checkpoint(accelerator, ema, global_step, output_dir)
    store_training_state(accelerator, global_step, ckpt_dir)
    accelerator.print("Training finished.")


if __name__ == "__main__":
    main()
