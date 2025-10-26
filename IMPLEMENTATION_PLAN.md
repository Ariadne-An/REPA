# U-REPA + SD-1.5 Implementation Plan

## 目标概述

在 REPA 官方仓库的训练与评测主干上，新增一个 SD-1.5 U-Net 模型分支；训练输入用 SD-VAE latent，对齐目标用 DINO tokens；按 U-REPA 的三条关键策略实现对齐（中段层、先投影后上采样、token+流形损失），最终用 FID/sFID 和表征可视化验证有效性。

**参考论文**: U-REPA: Aligning Diffusion U-Nets to ViTs (arXiv 2503.18414)
**代码未开源**: 本实现基于论文方法从零实现

---

## 1. 核心设计决策

### 1.1 架构选择

| 项目 | 决策 | 理由 |
|------|------|------|
| **基础模型** | SD-1.5 U-Net (runwayml/stable-diffusion-v1-5) | 成熟稳定，生态完善 |
| **分辨率** | 512×512 (latent 64×64×4) | 更接近实际应用，方便后续编辑任务 |
| **对齐目标** | DINOv2 ViT-L/14 (D=1024, 14×14 grid) | 最强语义编码器 |
| **对齐层** | mid_block (默认) + 可选 enc_last/dec_first | U-REPA 论文核心发现：中段最优 |
| **微调方式** | LoRA (rank=32) + 可选 partial full-FT | 集中火力在对齐层，减少显存 |

### 1.2 损失函数设计

```
L_total = L_diff + λ * (L_token + w * L_manifold)
```

- **L_diff**: 标准扩散去噪损失 (v-prediction 默认，epsilon 可选)
- **L_token**: Token-wise 负余弦相似度 (1 - cosine_similarity)
- **L_manifold**: 样本-样本 Gram 矩阵对齐 (O(B²) 而非 O(B×N²))
- **λ = 0.8**: 对齐损失权重
- **w = 3.0**: 流形损失相对 token 损失的权重

### 1.3 关键技术点

#### (A) v-prediction 开关 + 保留 ε-prediction

**动机**: U-REPA 论文采用 v 目标，SD-1.5 生态广泛使用 ε 目标。

**实现**:
- 默认 `--prediction=v` (follow 论文)
- 支持 `--prediction=epsilon` (作为 fallback)
- 转换公式:
  ```python
  # v = alpha_t * epsilon - sigma_t * x_0
  # epsilon = (v + sigma_t * x_0) / alpha_t
  ```

#### (B) 流形损失: 样本-样本 Gram 对齐

**动机**: 对齐样本间几何关系，而非 token-token 的 O(N²) 计算。

**实现**:
```python
x_agg = F.normalize(x_tokens.mean(1), dim=-1)  # [B, D]
y_agg = F.normalize(y_tokens.mean(1), dim=-1)
Mx = x_agg @ x_agg.T  # [B, B]
My = y_agg @ y_agg.T
L_manifold = F.mse_loss(Mx, My)
```

**优点**:
- 计算量从 O(B×196²) → O(B²)
- 保留全局几何关系
- 与 token-wise loss 互补

#### (C) 对齐层选择

**决策**: 默认只对齐 `mid_block` (8×8 分辨率)

**层结构** (SD-1.5 U-Net, latent 64×64):
```
down_blocks[0,1,2,3]  # encoder, 分辨率: 64→32→16→8
mid_block             # middle, 分辨率: 8
up_blocks[0,1,2,3]    # decoder, 分辨率: 8→16→32→64
```

**对齐层定义**:
- `enc_last` = `down_blocks[3].resnets[-1]` 输出 (8×8, C=1280)
- `mid` = `mid_block.resnets[-1]` 输出 (8×8, C=1280)
- `dec_first` = `up_blocks[0].resnets[-1]` 输出 (8×8, C=1280)

**扩展策略**:
- 默认: 只用 `mid`
- 可选: 同时对齐 `enc_last + mid + dec_first`，取平均

#### (D) 对齐头: 先投影后上采样

**设计**:
```python
class AlignHead(nn.Module):
    def __init__(self, in_ch=1280, D=1024):
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, D, 1, bias=False),  # 1×1 conv
            nn.GroupNorm(32, D),
            nn.GELU()
        )

    def forward(self, feat):
        x = self.proj(feat)  # [B, D, H, W] (e.g., [B, 1024, 8, 8])
        if x.shape[2] != 14:
            x = F.interpolate(x, (14, 14), mode='bilinear', align_corners=False)
        x = x.flatten(2).transpose(1, 2)  # [B, 196, D]
        return F.normalize(x, dim=-1)  # L2 normalize
```

**关键点**:
- 先投影 C→D (1280→1024)
- 后上采样到 14×14 (与 DINOv2 token grid 对齐)
- 输出 L2 归一化的 tokens [B, 196, 1024]

#### (E) 条件输入: 复用 SD-1.5 原生 cross-attention

**决策**: 不修改 U-Net 结构，保持与 SD-1.5 生态兼容。

**ImageNet class → text prompt**:
- 模板: `"a photo of a {class_name}"`
- Null prompt (CFG): `""`
- 预缓存 1001 个 CLIP text embeddings (1000 类 + 1 个 null)

**优点**:
- 训练/推理接口统一
- 支持 CFG (Classifier-Free Guidance)
- 与 SD 编辑方法兼容

---

## 2. 训练配置

### 2.1 硬件与环境

- **GPU**: 1×A100 80GB
- **精度**: bfloat16 (bf16)
- **Batch size**: 48–64
- **梯度累积**: 1 (单卡不需要)
- **速度预估**: ~8–10 it/s (512 分辨率, mid-only)

### 2.2 训练方案

#### 档 A: 快速验证 (2 小时)

| 项目 | 配置 |
|------|------|
| **数据量** | 200k 张 (均匀采样, 1000 类 × 200 张/类) |
| **步数** | 60k steps |
| **Batch** | 48–64 |
| **时间** | ~1.7–2.1 小时 |
| **目标** | 验证方法有效性，观察 FID 下降趋势 |

#### 档 B: 主结果 (4 小时)

| 项目 | 配置 |
|------|------|
| **数据量** | 500k 张 (均匀采样, 1000 类 × 500 张/类) |
| **步数** | 120k steps |
| **Batch** | 48–64 |
| **时间** | ~3.3–4.2 小时 |
| **目标** | 论文级主结果，FID 曲线明显改善 |

#### 可选: 档 C (如果档 B 仍在下降)

- 延长到 200k steps (~7 小时)
- 或启用 3 层对齐 (enc_last + mid + dec_first)

### 2.3 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| **learning_rate** | 2e-4 (LoRA) / 1e-4 (full-FT) | LoRA 可以用更大的 LR |
| **optimizer** | AdamW (β1=0.9, β2=0.999, wd=0.01) | 标准配置 |
| **lr_schedule** | Warmup 10% + Cosine decay | 稳定收敛 |
| **EMA** | 0.9995 | 评测使用 EMA 权重 |
| **λ (align_coeff)** | 0.8 | 对齐损失权重 |
| **w (manifold_coeff)** | 3.0 | 流形损失权重 |
| **CFG dropout** | 10% | Label dropout 概率 |
| **prediction** | v (默认) / epsilon (可选) | 预测目标类型 |

### 2.4 LoRA 配置

| 项目 | 配置 |
|------|------|
| **rank** | 32 |
| **target_modules** | attention + conv (默认) / attention-only (可选) |
| **白名单前缀** | `mid_block.*`, `down_blocks.3.*`, `up_blocks.0.*` |
| **自动发现** | 遍历 named_modules，匹配关键词 `.to_q`, `.to_k`, `.to_v`, `.to_out.0`, `.conv1`, `.conv2` |

---

## 3. 数据预处理

### 3.1 ImageNet 采样

**输入**: ILSVRC 标准目录结构
```
ILSVRC/
  Data/CLS-LOC/train/{synset}/xxx.JPEG
  Data/CLS-LOC/val/{synset}/xxx.JPEG
```

**输出**: 采样 CSV 文件
- `train_200k.csv` (档 A)
- `train_500k.csv` (档 B)

**字段**: `id, img_path, class_id, synset`

**采样策略**:
- 均匀采样: 每类固定数量
- 如某类不足，记录日志并跳过

### 3.2 SD-VAE Latent 缓存

**编码器**: `runwayml/stable-diffusion-v1-5` 的 VAE

**流程**:
1. 图像 resize/crop 到 512×512
2. 归一化到 [-1, 1]
3. 通过 VAE encoder 得到 latent [4, 64, 64]
4. 保存为 fp16/bf16

**存储**:
- 格式: LMDB 分片 (每 10k 样本一个 shard)
- Key: 样本 id
- Value: [4, 64, 64] 的 latent (fp16/bf16)
- 总大小: ~42 GB (200k 张) / ~105 GB (500k 张)

### 3.3 DINOv2 Tokens 缓存

**编码器**: DINOv2 ViT-L/14 (D=1024)

**流程**:
1. 对干净图像 (像素域) 应用官方 transforms:
   ```python
   transforms.Compose([
       transforms.Resize(256, interpolation=3),  # bicubic
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
   ])
   ```
2. 前向得到 tokens [196, 1024]
3. L2 归一化每个 token
4. 保存为 fp16/bf16

**存储**:
- 格式: LMDB 分片
- Key: 样本 id (与 VAE latent 对齐)
- Value: [196, 1024] 的 tokens (fp16/bf16)
- 总大小: ~160 GB (200k 张) / ~400 GB (500k 张)

**元信息**: `{"encoder": "dinov2-vit-l-16", "D": 1024, "grid": 14, "patch": 16}`

**启动校验**: 强制检查 D=1024, N=196，不符报错

### 3.4 CLIP Text Embeddings 缓存

**编码器**: SD-1.5 的 CLIP text encoder

**流程**:
1. 加载 ImageNet class name 映射 (`imagenet_classes.json`)
2. 对每个类生成 prompt: `"a photo of a {class_name}"`
3. 生成 null prompt: `""`
4. 通过 CLIP text encoder 得到 embeddings [77, 768]
5. 保存 1001 个 embeddings (1000 类 + 1 null)

**存储**:
- 格式: `.pt` 文件 (单文件，很小)
- 形状: [1001, 77, 768] (fp16)
- 总大小: ~117 MB

---

## 4. 代码结构设计

### 4.1 新增文件

```
REPA/
├── models/
│   ├── sd15_unet_aligned.py      # SD U-Net wrapper + 对齐头 + Hook 管理
│   └── dinov2_encoder.py         # DINOv2 适配器 (支持本地权重加载)
├── loss_sd15.py                   # SD-1.5 专用损失函数
├── dataset_sd15.py                # SD15AlignedDataset (加载 latent + DINO + CLIP)
├── train_sd15.py                  # 训练脚本 (主入口)
├── evaluate_sd15.py               # 评测脚本 (FID + 可视化)
├── preprocessing/
│   ├── prepare_clip_embeddings.py # 生成 CLIP text embeddings
│   ├── sample_imagenet_subset.py  # 均匀采样 ImageNet
│   ├── encode_vae_latents.py      # 批量编码 VAE latents
│   └── build_dino_cache.py        # 批量提取 DINO tokens
└── configs/
    ├── sd15_repa_档A.yaml         # 档 A 配置
    └── sd15_repa_档B.yaml         # 档 B 配置
```

### 4.2 核心模块接口

#### (1) `models/sd15_unet_aligned.py`

```python
class AlignHead(nn.Module):
    """对齐头: 1280 → 1024, ↑14×14"""
    def __init__(self, in_ch=1280, D=1024):
        ...

    def forward(self, feat):
        # [B, C, H, W] → [B, 196, D] (L2-normalized)
        ...

class SD15UNetAligned(nn.Module):
    """SD-1.5 U-Net + 对齐头 + Hook 管理"""
    def __init__(self, unet, align_layers=['mid'], D=1024, lora_config=None):
        ...

    def forward(self, latents, timesteps, encoder_hidden_states):
        # 标准 SD U-Net forward
        # 同时从 hook 提取特征并应用对齐头
        ...

    def get_aligned_tokens(self):
        # 返回 dict: {layer_name: [B, 196, D]}
        ...

    def clear_hooks(self):
        # 清空 hook 缓存
        ...
```

#### (2) `loss_sd15.py`

```python
def manifold_gram_loss(x_tokens, y_tokens, upper_only=False):
    """样本-样本 Gram 矩阵对齐"""
    ...

class SD15Loss:
    """SD-1.5 专用损失函数"""
    def __init__(self, prediction='v', align_coeff=0.8, manifold_coeff=3.0):
        ...

    def __call__(self, unet, clean_latents, dino_tokens, encoder_hidden_states):
        # 返回 dict: {'loss', 'loss_diff', 'loss_token', 'loss_manifold', 'loss_align'}
        ...
```

#### (3) `dataset_sd15.py`

```python
class SD15AlignedDataset(Dataset):
    """加载预缓存的 latent + DINO tokens + CLIP embeddings"""
    def __init__(self, csv_path, latent_dir, dino_dir, clip_embeddings_path, cfg_dropout=0.1):
        ...

    def __getitem__(self, idx):
        # 返回 dict:
        # {
        #     'latent': [4, 64, 64],
        #     'dino_tokens': {'mid': [196, 1024]},
        #     'encoder_hidden_states': [77, 768],
        #     'class_id': int
        # }
        ...
```

#### (4) `train_sd15.py`

```python
def main():
    # 1. 加载配置
    # 2. 初始化 SD U-Net + 对齐头
    # 3. 可选: 应用 LoRA
    # 4. 加载数据集
    # 5. 训练循环:
    #    - forward: unet + get_aligned_tokens
    #    - compute loss: SD15Loss
    #    - backward + update
    #    - EMA update
    #    - 每 5k 步评测 (FID + 可视化)
    # 6. 保存 checkpoint (主权重 + EMA)
    ...
```

---

## 5. 评测与验证

### 5.1 定量评测

**FID/sFID**:
- 工具: `clean-fid` (推荐) 或 REPA 原版脚本
- 样本: 固定从 ImageNet val 采样 10k/50k
- 采样参数: NFE=250, CFG=7.5, guidance_high=0.7 (复用 REPA 默认)
- 频率: 每 5k 步评测一次

**曲线**:
- FID vs. iteration (与 baseline 对比)
- 达到目标 FID 所需步数 (收敛加速证据)

### 5.2 对齐可视化

**(A) t-SNE/UMAP**:
- 取 mid_block 的 x_tokens 和 y_tokens (DINO)
- 降维到 2D，可视化分布是否靠拢

**(B) CCA (Canonical Correlation Analysis)**:
- 计算 x_tokens 和 y_tokens 的相关性
- 对比 baseline (无对齐) vs. REPA (有对齐)

**(C) 生成对比**:
- 固定 seed，生成 SD vs. SD-REPA 的图像
- 主观评价质量/多样性

### 5.3 消融实验 (可选)

| 实验 | 配置 | 目标 |
|------|------|------|
| **Baseline** | λ=0 (无对齐) | 证明对齐有效 |
| **Token-only** | w=0 (无流形) | 验证流形损失作用 |
| **Multi-layer** | enc_last + mid + dec_first | 探索多层对齐收益 |
| **Lambda sweep** | λ ∈ {0.5, 0.8, 1.0} | 找最优对齐权重 |

---

## 6. 关键校验点

### 6.1 预处理阶段

- [ ] ImageNet 采样: 确保 1000 类均匀分布
- [ ] VAE latent: shape=[4, 64, 64], dtype=fp16/bf16, 范围合理
- [ ] DINO tokens: shape=[196, 1024], L2-normalized (norm≈1)
- [ ] CLIP embeddings: shape=[1001, 77, 768], 包含 null prompt

### 6.2 训练启动

- [ ] U-Net 加载成功 (runwayml/stable-diffusion-v1-5)
- [ ] 对齐头初始化: in_ch=1280, D=1024, GroupNorm(32, 1024)
- [ ] Hook 注册成功: mid_block.resnets[-1]
- [ ] LoRA 插入: 只在白名单层 (mid_block.*, down_blocks.3.*, up_blocks.0.*)
- [ ] 数据加载: batch shape 正确，无 NaN/Inf

### 6.3 训练过程

- [ ] 损失下降: L_diff, L_token, L_manifold 都在下降
- [ ] 梯度正常: 无爆炸/消失 (grad_norm < 10)
- [ ] 显存稳定: ~60-70GB (batch=48-64, bf16)
- [ ] 速度符合预期: 8-10 it/s

### 6.4 评测阶段

- [ ] FID 计算成功: 使用相同参数 (NFE, CFG)
- [ ] 可视化生成: t-SNE 显示分布靠拢
- [ ] 生成质量: 视觉检查无明显崩坏

---

## 7. 风险与应对

### 7.1 已知风险

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| **DINO 权重加载失败** | 中 | 高 | 提供 from_local_ckpt 兼容多种格式 |
| **显存不足** | 低 | 中 | 降低 batch (48→32) 或启用 grad_ckpt |
| **训练不稳定** | 中 | 高 | 降低 λ (0.8→0.5) 或关闭 manifold (w=0) |
| **FID 不降** | 中 | 高 | 消融实验找问题 (λ, w, 对齐层) |
| **采样速度慢** | 低 | 低 | 优化评测频率 (5k→10k) |

### 7.2 调试策略

**如果 FID 不降**:
1. 检查 baseline (λ=0) 的 FID，确保数据/采样没问题
2. 可视化对齐 tokens (t-SNE)，确保确实在靠拢
3. 降低 λ 到 0.5，减少对齐损失干扰
4. 关闭 manifold (w=0)，只用 token loss

**如果训练崩溃**:
1. 检查 NaN/Inf (梯度/损失)
2. 降低学习率 (2e-4 → 1e-4)
3. 增加 warmup (10% → 20%)
4. 检查 DINO tokens 是否正确归一化

---

## 8. 时间线与里程碑

### Phase 1: 预处理 (1-2 天)

- [ ] 编写并测试采样脚本 (半天)
- [ ] 生成档 A 的 CSV (200k 张) (1 小时)
- [ ] 编码 VAE latents (200k 张) (2-3 小时)
- [ ] 提取 DINO tokens (200k 张) (3-4 小时)
- [ ] 生成 CLIP embeddings (1 分钟)

### Phase 2: 代码实现 (1-2 天)

- [ ] 实现对齐头 + Hook 管理 (半天)
- [ ] 实现损失函数 (半天)
- [ ] 实现数据集类 (半天)
- [ ] 实现训练脚本 (1 天)
- [ ] 单元测试 (每个模块独立验证) (半天)

### Phase 3: 小规模测试 (半天)

- [ ] 用 1k 样本跑 100 步 (30 分钟)
- [ ] 验证所有模块无 bug
- [ ] 验证损失正常下降

### Phase 4: 档 A 训练 (2 小时)

- [ ] 200k 样本，60k 步
- [ ] 观察 FID 趋势
- [ ] 生成对齐可视化

### Phase 5: 档 B 训练 (4 小时)

- [ ] 500k 样本，120k 步
- [ ] 完整 FID 曲线
- [ ] 主结果输出

---

## 9. 成功标准

### 最低标准 (Must-have)

- [ ] 代码能成功运行，无崩溃
- [ ] FID 相比 baseline (λ=0) 有明显改善 (如 baseline=5.0，REPA=3.5)
- [ ] 对齐可视化 (t-SNE) 显示分布靠拢

### 理想标准 (Should-have)

- [ ] FID 接近或超越论文报告 (FID<2.0)
- [ ] 收敛速度明显加快 (达到目标 FID 所需步数减半)
- [ ] 生成质量视觉上明显更好

### 额外收获 (Nice-to-have)

- [ ] 消融实验完整 (token/manifold/multi-layer)
- [ ] 超参数搜索找到最优配置
- [ ] 代码开源，复现性强

---

## 10. 附录

### A. ImageNet Class Name 映射

需要准备 `imagenet_classes.json`:
```json
{
  "0": "tench",
  "1": "goldfish",
  "2": "great white shark",
  ...
  "999": "toilet tissue"
}
```

来源:
- [ImageNet class index](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
- 或从 torchvision.datasets.ImageNet 提取

### B. DINOv2 权重获取

**选项 1: torch.hub** (推荐)
```python
import torch
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# 保存权重
torch.save(dino.state_dict(), 'dinov2_vitl14.pth')
```

**选项 2: Hugging Face**
```python
from transformers import AutoModel
dino = AutoModel.from_pretrained('facebook/dinov2-large')
```

**选项 3: 手动下载**
- 从 [facebookresearch/dinov2 releases](https://github.com/facebookresearch/dinov2/releases) 下载

### C. 参考资料

- **U-REPA 论文**: https://arxiv.org/abs/2503.18414
- **REPA 原版代码**: https://github.com/sihyun-yu/REPA
- **SD-1.5**: https://huggingface.co/runwayml/stable-diffusion-v1-5
- **DINOv2**: https://github.com/facebookresearch/dinov2
- **Diffusers 文档**: https://huggingface.co/docs/diffusers

---

## 更新日志

- **2025-10-15**: 初始方案定稿
- (后续更新记录在此)
