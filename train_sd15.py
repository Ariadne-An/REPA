import argparse
import json
import math
import time
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
    parser.add_argument("--dryrun", action="store_true", help="Run a short dry run overriding steps and intervals.")

    # Data overrides
    parser.add_argument("--train-csv", type=str, default=None, help="Override train CSV path")
    parser.add_argument("--latent-dir", type=str, default=None, help="Override train latent LMDB path")
    parser.add_argument("--dino-dir", type=str, default=None, help="Override train DINO LMDB path")
    parser.add_argument("--clip-embeddings", type=str, default=None, help="Override CLIP embeddings path")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train dataloader workers")

    # Validation overrides
    parser.add_argument("--val-csv", type=str, default=None, help="Override validation CSV path")
    parser.add_argument("--val-latent-dir", type=str, default=None, help="Override validation latent LMDB path")
    parser.add_argument("--val-dino-dir", type=str, default=None, help="Override validation DINO LMDB path")
    parser.add_argument("--val-clip-embeddings", type=str, default=None, help="Override validation CLIP embeddings path")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Override validation batch size")
    parser.add_argument("--val-num-workers", type=int, default=None, help="Override validation dataloader workers")
    parser.add_argument("--val-interval", type=int, default=None, help="Override validation interval (steps)")
    parser.add_argument("--val-num-batches", type=int, default=None, help="Override validation batches per eval")
    parser.add_argument("--val-repeat", type=int, default=None, help="Repeat validation loop N times for averaging")
    parser.add_argument("--val-cfg-dropout", type=float, default=None, help="Override validation CFG dropout rate")

    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train (overrides max_steps)")
    parser.add_argument("--local-log", type=str, default=None, help="Path to append local JSON logs")

    return parser.parse_args()


def apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    overrides = {
        "csv_path": args.train_csv,
        "latent_dir": args.latent_dir,
        "dino_dir": args.dino_dir,
        "clip_embeddings_path": args.clip_embeddings,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_csv_path": args.val_csv,
        "val_latent_dir": args.val_latent_dir,
        "val_dino_dir": args.val_dino_dir,
        "val_clip_embeddings_path": args.val_clip_embeddings,
        "local_log_path": args.local_log,
        "val_cfg_dropout": args.val_cfg_dropout,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.val_batch_size is not None:
        config["val_batch_size"] = args.val_batch_size
    if args.val_num_workers is not None:
        config["val_num_workers"] = args.val_num_workers

    if args.val_interval is not None:
        config["val_interval"] = args.val_interval
    if args.val_num_batches is not None:
        config["val_num_batches"] = args.val_num_batches
    if args.val_repeat is not None:
        config["val_repeat"] = max(1, args.val_repeat)

    if args.epochs is not None:
        config["epochs_override"] = args.epochs

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.resume is not None:
        config["resume_from_checkpoint"] = args.resume

    return config


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

    num_workers = config.get("num_workers", 8)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
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
        lora_alpha=config.get("lora_alpha"),
        device=str(device),
    )
    try:
        model.unet.enable_xformers_memory_efficient_attention()
        print("✅ Enabled xFormers memory-efficient attention")
    except Exception as exc:
        print(f"⚠️  xFormers attention not enabled: {exc}")

    return model


def build_optimizer(model: SD15UNetAligned, config: Dict) -> AdamW:
    base_lr = config.get("learning_rate", 1e-4)
    lora_lr = config.get("lora_lr", base_lr)
    align_head_lr = config.get("align_head_lr", base_lr)
    lora_weight_decay = config.get("lora_weight_decay", 0.0)
    align_head_weight_decay = config.get("align_head_weight_decay", config.get("weight_decay", 0.01))

    align_params = list(model.align_heads.parameters())
    align_param_ids = {id(p) for p in align_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in align_param_ids]

    param_groups = []
    if lora_params:
        param_groups.append(
            {"params": lora_params, "lr": lora_lr, "weight_decay": lora_weight_decay}
        )
    if align_params:
        param_groups.append(
            {"params": align_params, "lr": align_head_lr, "weight_decay": align_head_weight_decay}
        )

    optimizer = AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
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


def build_val_dataloader(config: Dict) -> Optional[DataLoader]:
    required_keys = [
        "val_csv_path",
        "val_latent_dir",
        "val_dino_dir",
        "val_clip_embeddings_path",
    ]
    if not all(k in config for k in required_keys):
        return None

    val_cfg_dropout = config.get("val_cfg_dropout", config.get("cfg_dropout", 0.1))
    val_seed = config.get("val_seed", config.get("seed", 42) + 1)

    dataset = SD15AlignedDataset(
        csv_path=config["val_csv_path"],
        latent_dir=config["val_latent_dir"],
        dino_dir=config["val_dino_dir"],
        clip_embeddings_path=config["val_clip_embeddings_path"],
        align_layers=config["align_layers"],
        cfg_dropout=val_cfg_dropout,
        seed=val_seed,
    )

    val_workers = config.get("val_num_workers", config.get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_size=config.get("val_batch_size", config["batch_size"]),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
        drop_last=False,
    )
    return loader


def append_local_log(path: Optional[Path], record: Dict):
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def sample_noisy_latent(clean_latent, timesteps, scheduler, prediction_type):
    noise = torch.randn_like(clean_latent)
    noisy_latent = scheduler.add_noise(clean_latent, noise, timesteps)
    if prediction_type == "v":
        target = scheduler.get_velocity(clean_latent, noise, timesteps)
    elif prediction_type == "epsilon":
        target = noise
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    return noisy_latent, target


def compute_align_coeff(step: int, max_coeff: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return max_coeff
    progress = min(max(step, 0) / warmup_steps, 1.0)
    return max_coeff * progress


def run_validation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loss_fn: SD15REPALoss,
    dataloader: Optional[DataLoader],
    scheduler: DDPMScheduler,
    prediction_type: str,
    val_num_batches: int,
    global_step: int,
    align_coeff_max: float,
    align_warmup_steps: int,
    val_repeat: int = 1,
    local_log_path: Optional[Path] = None,
):
    if dataloader is None or val_num_batches <= 0:
        return

    model.eval()
    metric_keys = ["total", "diffusion", "token", "manifold"]
    sums = {k: torch.zeros([], device=accelerator.device) for k in metric_keys}
    batches = 0
    repeats_done = 0

    with torch.no_grad():
        for _ in range(max(1, val_repeat)):
            processed = 0
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= val_num_batches:
                    break

                batch = prepare_batch(batch, accelerator.device)
                clean_latent = batch["latent"]
                encoder_hidden_states = batch["encoder_hidden_states"]
                dino_tokens = batch["dino_tokens"]

                batch_size = clean_latent.size(0)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=clean_latent.device,
                    dtype=torch.long,
                )

                noisy_latent, target = sample_noisy_latent(
                    clean_latent, timesteps, scheduler, prediction_type
                )

                current_align_coeff = compute_align_coeff(global_step, align_coeff_max, align_warmup_steps)
                loss_fn.align_coeff = current_align_coeff

                loss_components = loss_fn(
                    model=model,
                    noisy_latent=noisy_latent,
                    timesteps=timesteps,
                    text_embeddings=encoder_hidden_states,
                    target=target,
                    target_tokens=dino_tokens,
                    return_details=True,
                )

                for key in metric_keys:
                    sums[key] += loss_components[key].detach()

                batches += 1
                processed += 1

            if processed == 0:
                break
            repeats_done += 1

    if batches == 0:
        model.train()
        return

    log_payload = {}
    for key in metric_keys:
        avg = sums[key] / batches
        avg = accelerator.gather_for_metrics(avg).mean()
        log_payload[f"val/{key}"] = avg.item()

    log_payload["val/repeats"] = repeats_done
    accelerator.log(log_payload, step=global_step)
    if accelerator.is_main_process and local_log_path is not None:
        append_local_log(local_log_path, {"step": global_step, "phase": "val", "metrics": log_payload})
    model.train()


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
    config = apply_overrides(config, args)
    if "val_interval" not in config:
        config["val_interval"] = 3000

    use_epoch_override = True
    if config.get("debug", False) or args.dryrun:
        dry_steps = config.get("debug_steps", 20)
        config["max_steps"] = min(config.get("max_steps", dry_steps), dry_steps)
        config["save_interval"] = max(1, min(config.get("save_interval", dry_steps), dry_steps))
        config["eval_interval"] = min(config.get("eval_interval", dry_steps), dry_steps)
        config["val_interval"] = max(1, min(config.get("val_interval", dry_steps), dry_steps))
        config["val_num_batches"] = min(config.get("val_num_batches", 1), 1)
        config["val_repeat"] = 1
        config["batch_size"] = min(config.get("batch_size", 4), 4)
        config["val_batch_size"] = min(config.get("val_batch_size", config["batch_size"]), config["batch_size"])
        use_epoch_override = False

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

    local_log_path: Optional[Path] = None
    requested_local_log = config.get("local_log_path")
    if requested_local_log:
        local_log_path = Path(requested_local_log)
        if accelerator.is_main_process:
            local_log_path.parent.mkdir(parents=True, exist_ok=True)

    if accelerator.log_with is not None:
        accelerator.init_trackers(
            project_name=config["experiment_name"],
            config=config,
        )

    dataloader = build_dataloader(config)
    steps_per_epoch = max(1, len(dataloader))

    target_epochs = None
    if use_epoch_override:
        target_epochs = config.get("epochs_override")
        if target_epochs is None:
            target_epochs = config.get("epoch_count")
        if target_epochs is None:
            target_epochs = config.get("num_epochs")
        if target_epochs:
            config["max_steps"] = steps_per_epoch * target_epochs
            config["resolved_epochs"] = target_epochs

    val_dataloader = build_val_dataloader(config)
    model = build_model(config, accelerator.device)
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_scheduler(optimizer, config)

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader, lr_scheduler
        )
    else:
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler
        )

    align_coeff_max = config.get("align_coeff", 0.5)
    align_warmup_steps = config.get("align_warmup_steps", 0)
    initial_align_coeff = 0.0 if align_warmup_steps > 0 else align_coeff_max
    loss_fn = SD15REPALoss(
        align_coeff=initial_align_coeff,
        manifold_coeff=config["manifold_coeff"],
        manifold_mask_diag=config.get("manifold_mask_diag", False),
        manifold_upper_only=config.get("manifold_upper_only", False),
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )
    prediction_type = config["schedule"]["parametrization"]
    val_num_batches = config.get("val_num_batches", 0)
    val_interval = config.get("val_interval", config.get("eval_interval", 0))
    val_repeat = max(1, config.get("val_repeat", 1))

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
    log_sums = {"total": 0.0, "diffusion": 0.0, "token": 0.0, "manifold": 0.0}
    log_counter = 0
    step_time_sum = 0.0

    progress_bar = tqdm(
        total=max_steps,
        initial=global_step,
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    start_time = time.perf_counter()
    step_start = start_time
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
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=clean_latent.device,
                    dtype=torch.long,
                )

                noisy_latent, target = sample_noisy_latent(
                    clean_latent, timesteps, noise_scheduler, prediction_type
                )

                current_align_coeff = compute_align_coeff(global_step, align_coeff_max, align_warmup_steps)
                loss_fn.align_coeff = current_align_coeff

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
                    step_time = time.perf_counter() - step_start
                    step_start = time.perf_counter()
                    for metric in log_sums:
                        if metric in loss_components:
                            log_sums[metric] += float(loss_components[metric].item())
                    log_counter += 1
                    step_time_sum += step_time

                    if accelerator.is_main_process and global_step % log_interval == 0:
                        log_payload = {
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if log_counter > 0:
                            for metric, total in log_sums.items():
                                log_payload[f"loss/{metric}_avg"] = total / log_counter
                            log_payload["train/step_time_s_avg"] = step_time_sum / log_counter
                            log_sums = {k: 0.0 for k in log_sums}
                            log_counter = 0
                            step_time_sum = 0.0
                        accelerator.log(log_payload, step=global_step)
                        if accelerator.is_main_process:
                            append_local_log(
                                local_log_path,
                                {"step": global_step, "phase": "train", "metrics": log_payload},
                            )

                if (
                    accelerator.is_main_process
                    and save_interval > 0
                    and global_step % save_interval == 0
                ):
                    ckpt_dir = save_checkpoint(accelerator, ema, global_step, output_dir)
                    store_training_state(accelerator, global_step, ckpt_dir)

                if (
                    val_dataloader is not None
                    and val_interval > 0
                    and global_step % val_interval == 0
                ):
                    run_validation(
                        accelerator,
                        model,
                        loss_fn,
                        val_dataloader,
                        noise_scheduler,
                        prediction_type,
                        val_num_batches,
                        global_step,
                        align_coeff_max,
                        align_warmup_steps,
                        val_repeat=val_repeat,
                        local_log_path=local_log_path,
                    )

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process and log_counter > 0:
        flush_payload = {
            "lr": lr_scheduler.get_last_lr()[0],
            "train/step_time_s_avg": step_time_sum / max(1, log_counter),
        }
        for metric, total in log_sums.items():
            flush_payload[f"loss/{metric}_avg"] = total / max(1, log_counter)
        accelerator.log(flush_payload, step=global_step)
        append_local_log(
            local_log_path,
            {"step": global_step, "phase": "train", "metrics": flush_payload},
        )

    ckpt_dir = save_checkpoint(accelerator, ema, global_step, output_dir)
    store_training_state(accelerator, global_step, ckpt_dir)
    accelerator.print("Training finished.")


if __name__ == "__main__":
    main()
