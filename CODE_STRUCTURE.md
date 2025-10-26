# Code Structure & Module Interfaces

本文档详细描述每个模块的接口、输入输出、依赖关系和实现细节。

---

## 1. 目录结构

```
REPA/
├── models/
│   ├── sit.py                      # 原版 REPA SiT 模型 (保留)
│   ├── sd15_unet_aligned.py        # ★ 新增: SD U-Net + 对齐头
│   └── dinov2_encoder.py           # ★ 新增: DINOv2 适配器
├── loss.py                          # 原版 REPA 损失 (保留)
├── loss_sd15.py                     # ★ 新增: SD-1.5 专用损失
├── dataset.py                       # 原版 REPA 数据集 (保留)
├── dataset_sd15.py                  # ★ 新增: SD-1.5 数据集
├── train.py                         # 原版 REPA 训练 (保留)
├── train_sd15.py                    # ★ 新增: SD-1.5 训练脚本
├── evaluate_sd15.py                 # ★ 新增: 评测脚本
├── preprocessing/
│   ├── prepare_clip_embeddings.py  # ★ 新增
│   ├── sample_imagenet_subset.py   # ★ 新增
│   ├── encode_vae_latents.py       # ★ 新增
│   └── build_dino_cache.py         # ★ 新增
├── configs/
│   ├── sd15_repa_档A.yaml          # ★ 新增
│   └── sd15_repa_档B.yaml          # ★ 新增
└── utils_sd15.py                    # ★ 新增: 辅助函数
```

---

## 2. 核心模块详细设计

### 2.1 `models/sd15_unet_aligned.py`

#### 2.1.1 `AlignHead` 类

**功能**: 将 U-Net 特征投影并上采样到 DINO token 空间。

**接口**:
```python
class AlignHead(nn.Module):
    def __init__(self, in_ch: int = 1280, D: int = 1024):
        """
        Args:
            in_ch: 输入通道数 (U-Net 层的输出通道)
            D: 输出维度 (DINO token 维度)
        """

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W] - U-Net 特征 (如 [B, 1280, 8, 8])

        Returns:
            tokens: [B, N, D] - 对齐后的 tokens (如 [B, 256, 1024])
                    已 L2 归一化
        """
```

**实现细节**:
```python
self.proj = nn.Sequential(
    nn.Conv2d(in_ch, D, kernel_size=1, bias=False),  # 1×1 conv
    nn.GroupNorm(32, D),
    nn.GELU()
)

def forward(self, feat):
    x = self.proj(feat)  # [B, D, H, W]

    # 上采样到 16×16 (DINOv2 token grid)
    if x.shape[2] != 14 or x.shape[3] != 14:
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)

    # Flatten 到 token 序列
    x = x.flatten(2).transpose(1, 2)  # [B, 256, D]

    # L2 normalize
    x = F.normalize(x, dim=-1)

    return x
```

**校验**:
- 输入 shape: `[B, 1280, 8, 8]`
- 输出 shape: `[B, 256, 1024]`
- 输出 norm: `torch.norm(output, dim=-1).mean() ≈ 1.0`

---

#### 2.1.2 `HookManager` 类

**功能**: 管理 U-Net 的 forward hook，提取中间特征。

**接口**:
```python
class HookManager:
    def __init__(self):
        self.features = {}
        self.handles = []

    def register_hook(self, module: nn.Module, name: str):
        """
        注册 hook 到指定模块。

        Args:
            module: 要 hook 的模块 (如 unet.mid_block.resnets[-1])
            name: hook 名称 (如 'mid')
        """

    def get_feature(self, name: str) -> torch.Tensor:
        """
        获取 hook 提取的特征 (不清空)。

        Args:
            name: hook 名称

        Returns:
            feat: [B, C, H, W]
        """

    def pop_feature(self, name: str) -> torch.Tensor:
        """
        获取并清空 hook 特征。
        """

    def clear(self):
        """
        清空所有 hook 特征缓存。
        """

    def remove_all_hooks(self):
        """
        移除所有 hook (清理时调用)。
        """
```

**实现细节**:
```python
def register_hook(self, module, name):
    def hook_fn(module, input, output):
        self.features[name] = output.detach()  # detach 避免梯度干扰

    handle = module.register_forward_hook(hook_fn)
    self.handles.append(handle)
```

---

#### 2.1.3 `SD15UNetAligned` 类

**功能**: 封装 SD-1.5 U-Net + 对齐头 + Hook 管理。

**接口**:
```python
class SD15UNetAligned(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        align_layers: List[str] = ['mid'],
        D: int = 1024,
        lora_config: Optional[LoraConfig] = None
    ):
        """
        Args:
            unet: 预训练的 SD-1.5 U-Net
            align_layers: 要对齐的层名称列表 (如 ['mid'], ['enc_last', 'mid', 'dec_first'])
            D: DINO token 维度
            lora_config: LoRA 配置 (可选)
        """

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        标准 SD U-Net forward，同时提取对齐特征。

        Args:
            latents: [B, 4, 64, 64] - noisy latents
            timesteps: [B] - timestep indices (0-999)
            encoder_hidden_states: [B, 77, 768] - CLIP text embeddings

        Returns:
            output: [B, 4, 64, 64] - predicted noise or v
            (同时在内部缓存对齐特征，通过 get_aligned_tokens 获取)
        """

    def get_aligned_tokens(self) -> Dict[str, torch.Tensor]:
        """
        获取对齐后的 tokens (从 hook 特征经过对齐头)。

        Returns:
            tokens: dict of {layer_name: [B, 256, D]}
        """

    def clear_features(self):
        """
        清空 hook 缓存 (每个训练步结束后调用)。
        """
```

**实现细节**:
```python
def __init__(self, unet, align_layers, D, lora_config):
    super().__init__()
    self.unet = unet
    self.align_layers = align_layers
    self.D = D

    # 应用 LoRA (可选)
    if lora_config is not None:
        self.unet = apply_lora(self.unet, lora_config)

    # 创建对齐头
    self.align_heads = nn.ModuleDict()
    for layer_name in align_layers:
        # 假设所有层通道数都是 1280 (已验证)
        self.align_heads[layer_name] = AlignHead(in_ch=1280, D=D)

    # Hook 管理器
    self.hook_manager = HookManager()

    # 注册 hooks
    self._register_hooks()

def _register_hooks(self):
    """根据 align_layers 注册 hooks"""
    for layer_name in self.align_layers:
        if layer_name == 'mid':
            module = self.unet.mid_block.resnets[-1]
        elif layer_name == 'enc_last':
            module = self.unet.down_blocks[3].resnets[-1]
        elif layer_name == 'dec_first':
            module = self.unet.up_blocks[0].resnets[-1]
        else:
            raise ValueError(f"Unknown align layer: {layer_name}")

        self.hook_manager.register_hook(module, layer_name)

def forward(self, latents, timesteps, encoder_hidden_states, return_dict=False):
    # 标准 forward
    output = self.unet(
        latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=return_dict
    )

    # 如果 return_dict=False，output 是 tuple (sample,)
    if not return_dict:
        return output[0] if isinstance(output, tuple) else output
    else:
        return output

def get_aligned_tokens(self):
    tokens = {}
    for layer_name in self.align_layers:
        feat = self.hook_manager.get_feature(layer_name)  # [B, 1280, 8, 8]
        tokens[layer_name] = self.align_heads[layer_name](feat)  # [B, 256, 1024]
    return tokens
```

**LoRA 应用**:
```python
def apply_lora(unet, lora_config):
    """
    应用 LoRA 到 U-Net (仅白名单层)。

    Args:
        unet: SD U-Net
        lora_config: LoraConfig (from peft)

    Returns:
        unet_with_lora: LoRA-injected U-Net
    """
    from peft import get_peft_model, LoraConfig

    # 自动发现 target_modules (白名单 + 关键词匹配)
    target_modules = get_lora_target_modules(
        unet,
        lora_targets=lora_config.lora_targets,  # 'attn+conv' or 'attention-only'
        allowed_prefixes=['mid_block', 'down_blocks.3', 'up_blocks.0']
    )

    lora_config_dict = {
        'r': lora_config.rank,
        'lora_alpha': lora_config.rank,  # 通常 alpha=rank
        'target_modules': target_modules,
        'lora_dropout': 0.0,
        'bias': 'none'
    }

    return get_peft_model(unet, LoraConfig(**lora_config_dict))

def get_lora_target_modules(unet, lora_targets='attn+conv', allowed_prefixes=[]):
    """
    自动发现 LoRA target_modules (白名单 + 关键词匹配)。

    Args:
        unet: SD U-Net
        lora_targets: 'attn+conv' or 'attention-only'
        allowed_prefixes: 白名单前缀列表

    Returns:
        target_modules: list of module names (如 ['to_q', 'to_k', 'to_v', 'conv1'])
    """
    attn_keywords = ['to_q', 'to_k', 'to_v', 'to_out.0']
    conv_keywords = ['conv1', 'conv2', 'conv_shortcut']

    keywords = attn_keywords
    if lora_targets == 'attn+conv':
        keywords += conv_keywords

    target_modules = set()

    for name, module in unet.named_modules():
        # 检查白名单前缀
        if not any(name.startswith(p) for p in allowed_prefixes):
            continue

        # 检查关键词
        for kw in keywords:
            if kw in name:
                # 提取模块类型 (如 'to_q' from 'mid_block.attentions.0.to_q')
                module_type = name.split('.')[-1]
                target_modules.add(module_type)
                break

    return list(target_modules)
```

---

### 2.2 `loss_sd15.py`

#### 2.2.1 `manifold_gram_loss` 函数

**功能**: 计算样本-样本 Gram 矩阵对齐损失。

**接口**:
```python
def manifold_gram_loss(
    x_tokens: torch.Tensor,
    y_tokens: torch.Tensor,
    upper_only: bool = False,
    mask_diag: bool = False
) -> torch.Tensor:
    """
    Args:
        x_tokens: [B, N, D] - 模型预测的 tokens
        y_tokens: [B, N, D] - 目标 DINO tokens
        upper_only: 是否只用上三角 (省计算)
        mask_diag: 是否去掉对角线

    Returns:
        loss: scalar tensor
    """
```

**实现**: (已在 IMPLEMENTATION_PLAN.md 中给出)

---

#### 2.2.2 `SD15Loss` 类

**功能**: 计算 SD-1.5 的总损失 (去噪 + 对齐)。

**接口**:
```python
class SD15Loss:
    def __init__(
        self,
        prediction: str = 'v',
        path_type: str = 'linear',
        weighting: str = 'uniform',
        align_coeff: float = 0.8,
        manifold_coeff: float = 3.0,
        manifold_upper_only: bool = False,
        manifold_mask_diag: bool = False
    ):
        """
        Args:
            prediction: 'v' or 'epsilon'
            path_type: 'linear' or 'cosine'
            weighting: 'uniform' or 'lognormal'
            align_coeff: λ (对齐损失权重)
            manifold_coeff: w (流形损失相对 token 损失的权重)
        """

    def __call__(
        self,
        unet: SD15UNetAligned,
        clean_latents: torch.Tensor,
        dino_tokens: Dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            unet: SD15UNetAligned 模型
            clean_latents: [B, 4, 64, 64] - 干净的 VAE latents
            dino_tokens: dict of {layer_name: [B, 256, 1024]} - 目标 DINO tokens
            encoder_hidden_states: [B, 77, 768] - CLIP text embeddings
            timesteps: [B] - 可选，指定 timesteps

        Returns:
            loss_dict: {
                'loss': total loss,
                'loss_diff': denoising loss,
                'loss_token': token alignment loss,
                'loss_manifold': manifold loss,
                'loss_align': token + manifold
            }
        """
```

**关键逻辑**:
```python
def __call__(self, unet, clean_latents, dino_tokens, encoder_hidden_states, timesteps=None):
    # 1. 采样 timesteps (如果未提供)
    # 2. 计算 alpha_t, sigma_t
    # 3. 加噪声: noisy_latents = alpha_t * clean + sigma_t * noise
    # 4. 计算 target (v or epsilon)
    # 5. U-Net forward: output = unet(noisy_latents, timesteps, encoder_hidden_states)
    # 6. 计算 loss_diff = MSE(output, target)
    # 7. 获取对齐 tokens: pred_tokens = unet.get_aligned_tokens()
    # 8. 对每个层计算 loss_token 和 loss_manifold
    # 9. 组合: loss = loss_diff + λ * (loss_token + w * loss_manifold)
    # 10. 返回 loss_dict
```

---

### 2.3 `dataset_sd15.py`

#### 2.3.1 `SD15AlignedDataset` 类

**功能**: 加载预缓存的 latent, DINO tokens, CLIP embeddings。

**接口**:
```python
class SD15AlignedDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        latent_dir: str,
        dino_dir: str,
        clip_embeddings_path: str,
        align_layers: List[str] = ['mid'],
        cfg_dropout: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            csv_path: 采样 CSV 文件路径 (如 train_200k.csv)
            latent_dir: VAE latent 缓存目录 (LMDB)
            dino_dir: DINO tokens 缓存目录 (LMDB)
            clip_embeddings_path: CLIP embeddings 文件路径 (.pt)
            align_layers: 对齐层名称列表 (决定返回哪些 DINO tokens)
            cfg_dropout: CFG label dropout 概率
            seed: 随机种子
        """

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            {
                'latent': [4, 64, 64] - VAE latent (fp16/bf16)
                'dino_tokens': dict of {layer_name: [256, 1024]} - DINO tokens
                'encoder_hidden_states': [77, 768] - CLIP text embedding
                'class_id': int - ImageNet class id
            }
        """
```

**实现细节**:
```python
def __init__(self, csv_path, latent_dir, dino_dir, clip_embeddings_path, align_layers, cfg_dropout, seed):
    # 加载 CSV
    self.df = pd.read_csv(csv_path)

    # 打开 LMDB
    self.latent_env = lmdb.open(latent_dir, readonly=True, lock=False)
    self.dino_env = lmdb.open(dino_dir, readonly=True, lock=False)

    # 加载 CLIP embeddings
    self.clip_embeddings = torch.load(clip_embeddings_path)  # [1001, 77, 768]
    assert self.clip_embeddings.shape == (1001, 77, 768)

    self.align_layers = align_layers
    self.cfg_dropout = cfg_dropout
    self.rng = np.random.RandomState(seed)

def __getitem__(self, idx):
    row = self.df.iloc[idx]
    sample_id = row['id']
    class_id = row['class_id']

    # 读取 VAE latent
    with self.latent_env.begin() as txn:
        latent_bytes = txn.get(sample_id.encode())
        latent = torch.frombuffer(latent_bytes, dtype=torch.float16).reshape(4, 64, 64)

    # 读取 DINO tokens
    dino_tokens = {}
    with self.dino_env.begin() as txn:
        for layer_name in self.align_layers:
            key = f"{sample_id}_{layer_name}".encode()
            tokens_bytes = txn.get(key)
            tokens = torch.frombuffer(tokens_bytes, dtype=torch.float16).reshape(256, 1024)
            dino_tokens[layer_name] = tokens

    # CFG: 随机 dropout label
    if self.rng.rand() < self.cfg_dropout:
        class_id = 1000  # null class (最后一个)

    # 获取 CLIP text embedding
    encoder_hidden_states = self.clip_embeddings[class_id]  # [77, 768]

    return {
        'latent': latent,
        'dino_tokens': dino_tokens,
        'encoder_hidden_states': encoder_hidden_states,
        'class_id': class_id
    }
```

**校验**:
- 确保 latent shape: `[4, 64, 64]`
- 确保 dino_tokens shape: `{layer: [256, 1024]}`
- 确保 encoder_hidden_states shape: `[77, 768]`
- 确保 CFG dropout 生效 (10% 的样本 class_id=1000)

---

### 2.4 `train_sd15.py`

**功能**: 主训练脚本。

**命令行参数**:
```bash
python train_sd15.py \
  --config configs/sd15_repa_档A.yaml \
  --output_dir exps/sd15_repa_档A \
  --report_to wandb \
  --seed 42
```

**主要流程**:
```python
def main():
    # 1. 加载配置
    args = parse_args()
    config = load_config(args.config)

    # 2. 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
        log_with='wandb' if args.report_to == 'wandb' else None
    )

    # 3. 加载 SD U-Net
    unet = UNet2DConditionModel.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='unet',
        torch_dtype=torch.bfloat16
    )

    # 4. 封装为 SD15UNetAligned
    lora_config = LoraConfig(rank=32, lora_targets='attn+conv') if config.use_lora else None
    unet_aligned = SD15UNetAligned(
        unet=unet,
        align_layers=config.align_layers,
        D=config.dino_D,
        lora_config=lora_config
    )

    # 5. 初始化损失函数
    criterion = SD15Loss(
        prediction=config.prediction,
        align_coeff=config.align_coeff,
        manifold_coeff=config.manifold_coeff
    )

    # 6. 加载数据集
    dataset = SD15AlignedDataset(
        csv_path=config.csv_path,
        latent_dir=config.latent_dir,
        dino_dir=config.dino_dir,
        clip_embeddings_path=config.clip_embeddings_path,
        align_layers=config.align_layers,
        cfg_dropout=config.cfg_dropout
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # 7. 优化器
    optimizer = torch.optim.AdamW(
        unet_aligned.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # 8. LR scheduler
    num_training_steps = config.max_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # 9. EMA
    ema_unet = EMAModel(unet_aligned.parameters(), decay=0.9995)

    # 10. Accelerator prepare
    unet_aligned, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet_aligned, optimizer, dataloader, lr_scheduler
    )

    # 11. 训练循环
    global_step = 0
    for epoch in range(config.epochs):
        for batch in dataloader:
            # Forward
            loss_dict = criterion(
                unet=unet_aligned,
                clean_latents=batch['latent'],
                dino_tokens=batch['dino_tokens'],
                encoder_hidden_states=batch['encoder_hidden_states']
            )

            # Backward
            accelerator.backward(loss_dict['loss'])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet_aligned.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # EMA update
            if accelerator.sync_gradients:
                ema_unet.step(unet_aligned.parameters())

            # Clear hook cache
            unet_aligned.clear_features()

            # Logging
            if global_step % config.log_interval == 0:
                accelerator.log({
                    'loss': loss_dict['loss'].item(),
                    'loss_diff': loss_dict['loss_diff'].item(),
                    'loss_token': loss_dict['loss_token'].item(),
                    'loss_manifold': loss_dict['loss_manifold'].item(),
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

            # Evaluation
            if global_step % config.eval_interval == 0 and global_step > 0:
                evaluate(unet_aligned, ema_unet, config, accelerator, global_step)

            # Save checkpoint
            if global_step % config.save_interval == 0 and global_step > 0:
                save_checkpoint(unet_aligned, ema_unet, optimizer, global_step, config)

            global_step += 1
            if global_step >= config.max_steps:
                break
```

---

### 2.5 预处理脚本

#### 2.5.1 `preprocessing/sample_imagenet_subset.py`

**功能**: 均匀采样 ImageNet 子集。

**用法**:
```bash
python preprocessing/sample_imagenet_subset.py \
  --imagenet_dir /data/ILSVRC/Data/CLS-LOC/train \
  --samples_per_class 200 \
  --output_csv train_200k.csv \
  --seed 42
```

**输出**: `train_200k.csv` (字段: id, img_path, class_id, synset)

---

#### 2.5.2 `preprocessing/encode_vae_latents.py`

**功能**: 批量编码 VAE latents。

**用法**:
```bash
python preprocessing/encode_vae_latents.py \
  --csv_path train_200k.csv \
  --output_dir /data/vae_latents \
  --batch_size 32 \
  --num_workers 8
```

**输出**: LMDB 数据库 (key=sample_id, value=[4,64,64] latent)

---

#### 2.5.3 `preprocessing/build_dino_cache.py`

**功能**: 批量提取 DINO tokens。

**用法**:
```bash
python preprocessing/build_dino_cache.py \
  --csv_path train_200k.csv \
  --dino_ckpt /path/to/dinov2_vitl14.pth \
  --output_dir /data/dino_tokens \
  --batch_size 64 \
  --num_workers 8
```

**输出**: LMDB 数据库 (key=sample_id_mid, value=[256,1024] tokens)

---

#### 2.5.4 `preprocessing/prepare_clip_embeddings.py`

**功能**: 生成 CLIP text embeddings。

**用法**:
```bash
python preprocessing/prepare_clip_embeddings.py \
  --imagenet_classes imagenet_classes.json \
  --output_path clip_embeddings_1001.pt
```

**输出**: `clip_embeddings_1001.pt` (shape=[1001, 77, 768])

---

## 3. 配置文件

### 3.1 `configs/sd15_repa_档A.yaml`

```yaml
# 档 A: 快速验证
experiment_name: "sd15_repa_档A"

# Data
csv_path: "train_200k.csv"
latent_dir: "/data/vae_latents"
dino_dir: "/data/dino_tokens"
clip_embeddings_path: "clip_embeddings_1001.pt"

# Model
align_layers: ["mid"]  # 默认只对齐 mid
dino_D: 1024  # DINOv2 ViT-L/14
use_lora: true
lora_rank: 32
lora_targets: "attn+conv"

# Loss
prediction: "v"  # or "epsilon"
path_type: "linear"
weighting: "uniform"
align_coeff: 0.8  # λ
manifold_coeff: 3.0  # w
manifold_upper_only: false
manifold_mask_diag: false

# Training
batch_size: 48
max_steps: 60000
learning_rate: 2.0e-4
weight_decay: 0.01
ema_decay: 0.9995
cfg_dropout: 0.1

# Logging & Saving
log_interval: 100
eval_interval: 5000
save_interval: 10000

# Misc
seed: 42
mixed_precision: "bf16"
gradient_clip: 1.0
```

---

## 4. 依赖关系图

```
train_sd15.py
    ├─> SD15UNetAligned (models/sd15_unet_aligned.py)
    │       ├─> AlignHead
    │       ├─> HookManager
    │       └─> apply_lora (peft)
    ├─> SD15Loss (loss_sd15.py)
    │       └─> manifold_gram_loss
    ├─> SD15AlignedDataset (dataset_sd15.py)
    │       ├─> VAE latents (LMDB)
    │       ├─> DINO tokens (LMDB)
    │       └─> CLIP embeddings (.pt)
    └─> Accelerator (accelerate)
```

---

## 5. 数据流图

```
[ImageNet 原图]
    ↓
[sample_imagenet_subset.py] → train_200k.csv
    ↓
[encode_vae_latents.py] → vae_latents/ (LMDB)
[build_dino_cache.py] → dino_tokens/ (LMDB)
[prepare_clip_embeddings.py] → clip_embeddings_1001.pt
    ↓
[SD15AlignedDataset] → batch: {latent, dino_tokens, encoder_hidden_states}
    ↓
[train_sd15.py]
    ├─> SD15UNetAligned.forward(latent, timestep, encoder_hidden_states)
    ├─> SD15UNetAligned.get_aligned_tokens() → pred_tokens
    ├─> SD15Loss(unet, latent, dino_tokens, encoder_hidden_states) → loss_dict
    └─> backward + update
```

---

## 6. 关键接口约定

### 6.1 Tensor Shapes

| 变量名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| `latent` | [B, 4, 64, 64] | bf16/fp16 | VAE latent (512×512 → 64×64) |
| `dino_tokens` | {layer: [B, 256, 1024]} | bf16/fp16 | DINO tokens (16×16=256) |
| `encoder_hidden_states` | [B, 77, 768] | bf16/fp16 | CLIP text embeddings |
| `timesteps` | [B] | long | Timestep indices (0-999) |
| `pred_tokens` | {layer: [B, 256, 1024]} | bf16/fp16 | 对齐后的 tokens (L2-normalized) |
| `Gram` | [B, B] | bf16/fp16 | 样本-样本相似度矩阵 |

### 6.2 Loss Dict

```python
loss_dict = {
    'loss': torch.Tensor,         # 总损失
    'loss_diff': torch.Tensor,    # 去噪损失
    'loss_token': torch.Tensor,   # Token-wise 对齐损失
    'loss_manifold': torch.Tensor,# 流形损失
    'loss_align': torch.Tensor    # token + manifold
}
```

### 6.3 Config Dict

```python
config = {
    'csv_path': str,
    'latent_dir': str,
    'dino_dir': str,
    'clip_embeddings_path': str,
    'align_layers': List[str],
    'dino_D': int,
    'use_lora': bool,
    'lora_rank': int,
    'prediction': str,  # 'v' or 'epsilon'
    'align_coeff': float,  # λ
    'manifold_coeff': float,  # w
    'batch_size': int,
    'max_steps': int,
    'learning_rate': float,
    ...
}
```

---

## 7. 单元测试

### 7.1 `AlignHead` 测试

```python
def test_align_head():
    head = AlignHead(in_ch=1280, D=1024)
    feat = torch.randn(2, 1280, 8, 8)
    tokens = head(feat)

    assert tokens.shape == (2, 256, 1024)
    assert torch.allclose(torch.norm(tokens, dim=-1).mean(), torch.tensor(1.0), atol=0.01)
```

### 7.2 `manifold_gram_loss` 测试

```python
def test_manifold_loss():
    x = torch.randn(4, 256, 1024)
    y = torch.randn(4, 256, 1024)
    loss = manifold_gram_loss(x, y)

    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0
```

### 7.3 `SD15UNetAligned` 测试

```python
def test_sd15_unet_aligned():
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet')
    unet_aligned = SD15UNetAligned(unet, align_layers=['mid'], D=1024)

    latents = torch.randn(2, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (2,))
    encoder_hidden_states = torch.randn(2, 77, 768)

    output = unet_aligned(latents, timesteps, encoder_hidden_states)
    assert output.shape == (2, 4, 64, 64)

    tokens = unet_aligned.get_aligned_tokens()
    assert 'mid' in tokens
    assert tokens['mid'].shape == (2, 256, 1024)
```

---

## 8. 调试工具

### 8.1 `utils_sd15.py`

```python
def visualize_aligned_tokens(pred_tokens, dino_tokens, save_path):
    """
    用 t-SNE 可视化对齐效果。

    Args:
        pred_tokens: [B, 256, D]
        dino_tokens: [B, 256, D]
        save_path: str
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Flatten
    X = torch.cat([pred_tokens, dino_tokens], dim=0).flatten(1).cpu().numpy()
    labels = ['pred'] * pred_tokens.shape[0] + ['dino'] * dino_tokens.shape[0]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    for label in ['pred', 'dino']:
        mask = [l == label for l in labels]
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def compute_cca_score(pred_tokens, dino_tokens):
    """
    计算 CCA 相关性分数。

    Args:
        pred_tokens: [B, N, D]
        dino_tokens: [B, N, D]

    Returns:
        cca_score: float (0-1, 越高越好)
    """
    from sklearn.cross_decomposition import CCA

    X = pred_tokens.flatten(1).cpu().numpy()
    Y = dino_tokens.flatten(1).cpu().numpy()

    cca = CCA(n_components=min(10, X.shape[1]))
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)

    return np.corrcoef(X_c.T, Y_c.T).diagonal(offset=X_c.shape[1]).mean()
```

---

这就是完整的代码结构设计！每个模块的接口、输入输出、依赖关系都已明确。接下来请你检查一下，有什么需要补充或修改的吗？
