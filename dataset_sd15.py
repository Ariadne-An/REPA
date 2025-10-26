import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _as_float32_tensor(buffer: memoryview, shape: List[int]) -> torch.Tensor:
    """Decode a LMDB value (fp16) into a float32 torch.Tensor."""
    np_array = np.frombuffer(buffer, dtype=np.float16).reshape(shape)
    # Copy to make sure the tensor owns its memory (LMDB buffer is invalid outside txn)
    return torch.from_numpy(np_array.astype(np.float32, copy=True))


class SD15AlignedDataset(Dataset):
    """
    Dataset wrapper that loads SD-VAE latents, DINO tokens and CLIP text embeddings
    prepared during preprocessing.

    Each sample returns:
        - latent: [4, 64, 64] clean latent (float32)
        - dino_tokens: dict[layer_name -> [256, 1024]] (float32)
        - encoder_hidden_states: [77, 768] CLIP text embedding (float32)
        - class_id: int (after CFG dropout)
    """

    def __init__(
        self,
        csv_path: str,
        latent_dir: str,
        dino_dir: str,
        clip_embeddings_path: str,
        align_layers: Optional[List[str]] = None,
        cfg_dropout: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.align_layers = align_layers or ['mid']
        self.cfg_dropout = float(cfg_dropout)
        self.rng = np.random.RandomState(seed)

        self.df = pd.read_csv(csv_path)
        if 'id' not in self.df.columns or 'class_id' not in self.df.columns:
            raise ValueError(
                f"CSV file {csv_path} must contain 'id' and 'class_id' columns."
            )

        self.latent_dir = latent_dir
        self.dino_dir = dino_dir

        self._latent_env = None
        self._dino_env = None
        self.latent_shape = self._load_lmdb_shape(latent_dir, default=(4, 64, 64))
        self.token_shape = self._load_lmdb_shape(dino_dir, default=(256, 1024))

        clip_embeddings = torch.load(clip_embeddings_path, map_location='cpu')
        if clip_embeddings.ndim != 3 or clip_embeddings.shape[1:] != (77, 768):
            raise ValueError(
                f"CLIP embeddings should have shape [N, 77, 768], got {clip_embeddings.shape}"
            )
        self.clip_embeddings = clip_embeddings.float()
        self.null_class_id = self.clip_embeddings.shape[0] - 1

    @staticmethod
    def _open_lmdb(path: str) -> lmdb.Environment:
        if path is None:
            raise ValueError("LMDB path is None.")
        return lmdb.open(
            path,
            subdir=os.path.isdir(path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=32,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row['id'])
        class_id = int(row['class_id'])

        latent = self._read_latent(sample_id)
        dino_tokens = self._read_dino_tokens(sample_id)

        if self.rng.rand() < self.cfg_dropout:
            class_id = self.null_class_id

        encoder_hidden_states = self.clip_embeddings[class_id]

        return {
            'latent': latent,
            'dino_tokens': dino_tokens,
            'encoder_hidden_states': encoder_hidden_states.clone(),
            'class_id': torch.tensor(class_id, dtype=torch.long),
        }

    def _read_latent(self, sample_id: str) -> torch.Tensor:
        env = self._get_latent_env()
        with env.begin(write=False) as txn:
            value = txn.get(sample_id.encode('utf-8'))
            if value is None:
                raise KeyError(f"Latent not found for id={sample_id}")
            return _as_float32_tensor(memoryview(value), list(self.latent_shape))

    def _read_dino_tokens(self, sample_id: str) -> Dict[str, torch.Tensor]:
        tokens: Dict[str, torch.Tensor] = {}
        env = self._get_dino_env()
        with env.begin(write=False) as txn:
            for layer_name in self.align_layers:
                key = f"{sample_id}_{layer_name}".encode('utf-8')
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"DINO tokens not found for key={key!r}")
                tokens[layer_name] = _as_float32_tensor(memoryview(value), list(self.token_shape))
        return tokens

    def __del__(self):
        for env in [getattr(self, '_latent_env', None), getattr(self, '_dino_env', None)]:
            if isinstance(env, lmdb.Environment):
                env.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_latent_env'] = None
        state['_dino_env'] = None
        return state

    def _get_latent_env(self) -> lmdb.Environment:
        if self._latent_env is None:
            self._latent_env = self._open_lmdb(self.latent_dir)
        return self._latent_env

    def _get_dino_env(self) -> lmdb.Environment:
        if self._dino_env is None:
            self._dino_env = self._open_lmdb(self.dino_dir)
        return self._dino_env

    def _load_lmdb_shape(self, dir_path: str, default: tuple) -> tuple:
        env = self._open_lmdb(dir_path)
        meta = {}
        meta_path = Path(dir_path) / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            with env.begin() as txn:
                meta_bytes = txn.get(b"__meta__")
                if meta_bytes:
                    meta = json.loads(meta_bytes.decode())
        env.close()
        return tuple(meta.get("shape", default))
