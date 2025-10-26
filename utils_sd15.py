import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import yaml


def load_yaml_config(path: str) -> Dict:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path_obj.open("r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    """
    Simple EMA tracker for torch.nn.Module.

    Stores a copy of the model parameters and updates them with exponential decay.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        current_state = model.state_dict()
        for key, param in current_state.items():
            shadow_param = self.shadow[key]
            if torch.is_floating_point(param):
                if shadow_param.device != param.device:
                    shadow_param = shadow_param.to(param.device)
                shadow_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
                self.shadow[key] = shadow_param
            else:
                self.shadow[key] = param.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.decay = state_dict["decay"]
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=False)


def load_checkpoint_state(checkpoint_dir: Path, device: torch.device = torch.device("cpu")) -> Dict:
    ckpt_file = checkpoint_dir / "pytorch_model.bin"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {ckpt_file}")
    return torch.load(ckpt_file, map_location=device)


def apply_ema_if_available(model: torch.nn.Module, checkpoint_dir: Path, device: torch.device) -> bool:
    ema_path = checkpoint_dir / "ema.pt"
    if not ema_path.exists():
        return False
    ema_state = torch.load(ema_path, map_location=device)
    ema_helper = ModelEMA(model, decay=ema_state.get("decay", 0.9995))
    ema_helper.load_state_dict(ema_state)
    ema_helper.copy_to(model)
    return True


def chunk_list(items: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def load_imagenet_classes(json_path: str) -> List[str]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"imagenet_classes json not found: {json_path}")
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data[str(i)] for i in sorted(int(k) for k in data.keys())]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("imagenet_classes must be list or {idx: name} mapping.")


def sample_prompts(class_names: List[str], num_samples: int, seed: int = 0, template: str = "a photo of a {}") -> List[str]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(num_samples):
        cls = rng.choice(class_names)
        prompts.append(template.format(cls))
    return prompts


def save_images(images: List, output_dir: Path, start_index: int = 0, prefix: str = "sample"):
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        idx = start_index + offset
        image.save(output_dir / f"{prefix}_{idx:06d}.png")


def compute_fid(samples_dir: Path, stats_path: Path) -> float:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError("clean-fid is required for FID computation. Install via `pip install clean-fid`.") from exc

    return fid.compute_fid(samples_dir=str(samples_dir), dataset=str(stats_path))
