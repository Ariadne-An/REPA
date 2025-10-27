# 🚀 SD1.5-REPA 快速上手指南

**5 分钟学会使用你训练的模型！**

---

## 📍 你的模型位置

```
/workspace/REPA/models/sd15_repa_step24k/
├── model.safetensors        # 4.0 GB - 单文件格式
├── unet/                    # Diffusers 格式
├── vae/
├── text_encoder/
└── ...
```

---

## 方法 1: Python 脚本 (最简单) ⭐

### 基础使用

创建一个 Python 文件 `generate.py`:

```python
from diffusers import StableDiffusionPipeline
import torch

# 1. 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "/workspace/REPA/models/sd15_repa_step24k",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")

# 2. 生成图片
prompt = "a beautiful sunset over mountains, professional photography, 4k"

image = pipe(
    prompt=prompt,
    num_inference_steps=50,      # 推理步数 (25-100，越多越好但越慢)
    guidance_scale=7.5,          # CFG scale (7-8.5 推荐)
    height=512,                  # 高度
    width=512,                   # 宽度
).images[0]

# 3. 保存
image.save("output.png")
print("✅ 生成完成！查看 output.png")
```

**运行**：
```bash
python generate.py
```

---

### 进阶用法：批量生成

```python
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "/workspace/REPA/models/sd15_repa_step24k",
    torch_dtype=torch.float16,
).to("cuda")

# 批量生成 4 张不同的图
prompts = [
    "a cute cat sitting on a window",
    "a professional photo of a dog",
    "a beautiful landscape with lake",
    "a modern architecture building",
]

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(42 + i),  # 固定种子可复现
    ).images[0]

    image.save(f"output_{i+1}.png")
    print(f"✅ 生成 {i+1}/4")
```

---

### 优化：加速生成

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "/workspace/REPA/models/sd15_repa_step24k",
    torch_dtype=torch.float16,
).to("cuda")

# 🚀 加速优化
# 1. xFormers (节省显存，加速)
pipe.enable_xformers_memory_efficient_attention()

# 2. VAE slicing (节省显存)
pipe.enable_vae_slicing()

# 3. Attention slicing (进一步节省显存)
pipe.enable_attention_slicing()

# 现在生成会更快，显存占用更少
image = pipe(
    "a beautiful photo",
    num_inference_steps=25,  # 可以用更少步数
).images[0]

image.save("fast_output.png")
```

---

### 高级：控制随机性

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "/workspace/REPA/models/sd15_repa_step24k",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a professional portrait photo"

# 生成 4 张不同种子的图片，找最好的
seeds = [42, 123, 456, 789]

for seed in seeds:
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    image.save(f"output_seed_{seed}.png")

print("✅ 生成 4 个种子，选择你最喜欢的！")
```

---

## 方法 2: AUTOMATIC1111 WebUI 🖼️

如果你有 WebUI 安装：

### 步骤 1: 复制模型

```bash
# 找到你的 WebUI 安装路径
WEBUI_PATH="/path/to/stable-diffusion-webui"

# 复制模型文件
cp /workspace/REPA/models/sd15_repa_step24k/model.safetensors \
   $WEBUI_PATH/models/Stable-diffusion/sd15_repa_step24k.safetensors
```

### 步骤 2: 使用

1. 启动 WebUI
2. 左上角 "Stable Diffusion checkpoint" → 选择 `sd15_repa_step24k`
3. 输入 prompt，点击 Generate

**推荐参数**：
- Sampling Steps: 30-50
- CFG Scale: 7.0-8.0
- Sampling Method: DPM++ 2M Karras (推荐) 或 Euler a

---

## 方法 3: ComfyUI 🎨

### 步骤 1: 复制模型

```bash
COMFYUI_PATH="/path/to/ComfyUI"

cp /workspace/REPA/models/sd15_repa_step24k/model.safetensors \
   $COMFYUI_PATH/models/checkpoints/sd15_repa_step24k.safetensors
```

### 步骤 2: 使用

1. 在 ComfyUI 界面中
2. 找到 "Load Checkpoint" 节点
3. 选择 `sd15_repa_step24k.safetensors`
4. 连接到你的工作流

---

## 📊 参数调节指南

### Inference Steps (推理步数)

| Steps | 质量 | 速度 | 适用场景 |
|-------|------|------|---------|
| 20-25 | ⭐⭐⭐ | 🚀🚀🚀 | 快速预览 |
| 30-40 | ⭐⭐⭐⭐ | 🚀🚀 | **推荐日常使用** |
| 50-75 | ⭐⭐⭐⭐⭐ | 🚀 | 高质量最终输出 |
| 100+ | ⭐⭐⭐⭐⭐ | 🐌 | 精细调整 |

**建议**：从 30 开始，不满意再增加到 50。

---

### CFG Scale (Guidance Scale)

| CFG | 效果 | 适用场景 |
|-----|------|---------|
| 3-5 | 创意、多样性高、可能偏离 prompt | 艺术创作 |
| **7-8** | **平衡，推荐** | **日常使用** |
| 9-12 | 严格遵循 prompt，可能过饱和 | 精确控制 |
| 15+ | 过度饱和，不自然 | 不推荐 |

**建议**：默认用 7.5，如果觉得不够准确就调到 8.5。

---

### Negative Prompt (负面提示词)

加入负面 prompt 可以提升质量：

```python
image = pipe(
    prompt="a beautiful portrait photo",
    negative_prompt="ugly, blurry, low quality, distorted, bad anatomy",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
```

**推荐负面词**：
```
ugly, blurry, low quality, low res, distorted, bad anatomy,
deformed, disfigured, watermark, text, signature
```

---

## 🎯 实战例子

### 例子 1: 人物肖像

```python
prompt = """
a professional portrait photo of a young woman,
natural lighting, soft focus, high quality,
detailed face, photorealistic
"""

negative_prompt = """
ugly, blurry, low quality, bad anatomy,
deformed face, distorted features
"""

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=8.0,
    height=512,
    width=512,
).images[0]
```

---

### 例子 2: 风景照片

```python
prompt = """
a beautiful mountain landscape with lake,
golden hour lighting, professional photography,
8k, highly detailed, stunning colors
"""

negative_prompt = "low quality, blurry, distorted, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5,
    height=512,
    width=768,  # 横向构图
).images[0]
```

---

### 例子 3: 艺术风格

```python
prompt = """
a cat in the style of Van Gogh, oil painting,
impressionist, vibrant colors, masterpiece
"""

negative_prompt = "photorealistic, photograph, low quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=9.0,  # 更高的 CFG 确保风格
).images[0]
```

---

## 🔧 常见问题

### Q1: 显存不足 (OOM)

**解决方案**：

```python
# 启用显存优化
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# 或者降低分辨率
image = pipe(prompt, height=448, width=448).images[0]
```

---

### Q2: 生成速度慢

**解决方案**：

```python
# 1. 使用 fp16
pipe = pipe.to("cuda", dtype=torch.float16)

# 2. 启用 xFormers
pipe.enable_xformers_memory_efficient_attention()

# 3. 减少步数
image = pipe(prompt, num_inference_steps=25).images[0]
```

---

### Q3: 生成质量不满意

**调整策略**：

1. **Prompt 不够好**：
   - 添加更多细节描述
   - 使用专业术语（如 "professional photography", "highly detailed"）
   - 参考成功的 prompt 范例

2. **参数不合适**：
   - 增加步数：30 → 50
   - 调整 CFG：7.5 → 8.5
   - 添加 negative prompt

3. **种子问题**：
   - 尝试多个不同种子（42, 123, 456...）
   - 选择最好的结果

---

### Q4: 能否生成更大分辨率？

SD1.5 训练在 512×512，但可以生成更大：

```python
# 方法 1: 直接生成 (可能有重复)
image = pipe(prompt, height=768, width=768).images[0]

# 方法 2: 生成后放大 (推荐)
small = pipe(prompt, height=512, width=512).images[0]

# 使用 upscaler (需要额外安装)
from PIL import Image
large = small.resize((1024, 1024), Image.LANCZOS)
large.save("large_output.png")
```

---

## 💡 Pro Tips

### Tip 1: 保存配置

把你喜欢的配置保存下来：

```python
# config.py
DEFAULT_CONFIG = {
    "num_inference_steps": 40,
    "guidance_scale": 7.5,
    "height": 512,
    "width": 512,
    "negative_prompt": "ugly, blurry, low quality",
}

# 使用
image = pipe(prompt="your prompt", **DEFAULT_CONFIG).images[0]
```

---

### Tip 2: 批量处理

```python
import os
from tqdm import tqdm

prompts = [
    "prompt 1",
    "prompt 2",
    "prompt 3",
    # ... 更多
]

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

for i, prompt in enumerate(tqdm(prompts)):
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"{output_dir}/image_{i:03d}.png")
```

---

### Tip 3: Prompt 模板

```python
TEMPLATES = {
    "portrait": "a professional portrait photo of {subject}, natural lighting, high quality",
    "landscape": "a beautiful {subject} landscape, golden hour, professional photography",
    "art": "{subject} in the style of {artist}, masterpiece, highly detailed",
}

# 使用
prompt = TEMPLATES["portrait"].format(subject="a young woman")
image = pipe(prompt).images[0]
```

---

## 🎨 创意示例

### 生成网格对比

```python
from PIL import Image

prompts = ["cat", "dog", "bird", "fish"]
images = []

for prompt in prompts:
    img = pipe(f"a photo of a {prompt}", num_inference_steps=30).images[0]
    images.append(img)

# 创建 2x2 网格
grid = Image.new('RGB', (1024, 1024))
for i, img in enumerate(images):
    x = (i % 2) * 512
    y = (i // 2) * 512
    grid.paste(img, (x, y))

grid.save("grid.png")
```

---

## 📚 下一步

学会基础使用后，你可以：

1. **探索 LoRA**：在你的模型基础上再加 LoRA
2. **尝试 ControlNet**：精确控制构图
3. **Img2Img**：图生图功能
4. **Inpainting**：局部修复

---

## 🆘 获取帮助

如果遇到问题：

1. 检查 GPU 是否可用：`torch.cuda.is_available()`
2. 查看显存占用：`nvidia-smi`
3. 尝试示例脚本：`python test_converted_model.py`

---

**🎉 开始创作吧！**

记住：
- 从简单的 prompt 开始
- 多尝试不同参数
- 保存好的配置
- 最重要的是：玩得开心！
