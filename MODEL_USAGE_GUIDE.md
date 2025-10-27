# 🎨 SD1.5-REPA 模型使用指南

恭喜！你已经成功训练并转换了 REPA 微调的 Stable Diffusion 1.5 模型。

## 📦 转换后的文件

转换后生成了两种格式：

```
models/sd15_repa_step24k/
├── model.safetensors          # 单文件格式 (4.0 GB) - WebUI/ComfyUI 用
├── unet/                      # Diffusers 格式
├── vae/
├── text_encoder/
├── tokenizer/
├── scheduler/
└── model_index.json
```

---

## 🚀 使用方式

### **方式 1：Diffusers (Python 代码)**

最简单的方式，适合写脚本：

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "models/sd15_repa_step24k",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 生成图片
image = pipe(
    prompt="a beautiful landscape with mountains and lake",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]

image.save("output.png")
```

**运行测试**：
```bash
python test_converted_model.py \
  --model_path models/sd15_repa_step24k \
  --prompt "a photo of a cat" \
  --num_images 4 \
  --output test_output.png
```

---

### **方式 2：AUTOMATIC1111 WebUI**

把你的模型放到 WebUI 的 models 文件夹：

```bash
# 复制单文件 checkpoint
cp models/sd15_repa_step24k/model.safetensors \
   /path/to/stable-diffusion-webui/models/Stable-diffusion/sd15_repa_step24k.safetensors

# 重启 WebUI，在模型选择器中选择 sd15_repa_step24k
```

---

### **方式 3：ComfyUI**

```bash
# 复制到 ComfyUI models 目录
cp models/sd15_repa_step24k/model.safetensors \
   /path/to/ComfyUI/models/checkpoints/sd15_repa_step24k.safetensors

# 在 ComfyUI 界面中的 Load Checkpoint 节点选择这个模型
```

---

### **方式 4：用于其他项目**

如果你的项目使用 diffusers，直接指向路径：

```python
from diffusers import StableDiffusionPipeline

# 方法 1：本地路径
pipe = StableDiffusionPipeline.from_pretrained(
    "/workspace/REPA/models/sd15_repa_step24k"
)

# 方法 2：或者先上传到 HuggingFace Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="models/sd15_repa_step24k",
    repo_id="your-username/sd15-repa",
    repo_type="model",
)

# 然后别人可以直接用
pipe = StableDiffusionPipeline.from_pretrained("your-username/sd15-repa")
```

---

## 🔍 模型特点

你训练的这个模型：

### ✅ **优势**
- 基于 U-REPA 对齐训练，理论上：
  - 更好的语义理解
  - 更准确的 prompt 对齐
  - 可能更快的收敛（训练到 24k steps）

### ⚠️ **注意事项**
1. **不包含 Safety Checker**
   - 转换时移除了安全检查器
   - 如果需要，可以手动添加

2. **基于 SD1.5**
   - 分辨率：最佳 512×512
   - 不支持 SDXL 的功能

3. **微调过的模型**
   - 可能在某些风格上有偏好
   - 如果效果不理想，可以尝试其他 checkpoint（如 step_30000）

---

## 📊 对比其他 Checkpoints

你有多个可用的 checkpoint：

```bash
ls exps/trackA_h200_bs128_bf16/
# step_006000
# step_012000
# step_018000
# step_024000  ← 你选的这个
# step_030000
# step_036000
```

**建议**：
- `step_024000`: 早期，可能更接近原始 SD1.5 风格
- `step_030000`: 中期，平衡
- `step_036000`: 最终，对齐训练最充分

如果想试试其他的：
```bash
python convert_to_sd15.py \
  --checkpoint exps/trackA_h200_bs128_bf16/step_030000/model.safetensors \
  --output_dir models/sd15_repa_step30k \
  --save_single_file
```

---

## 🔧 高级用法

### **使用 EMA Weights**

EMA (Exponential Moving Average) 权重通常更稳定：

```bash
python convert_to_sd15.py \
  --checkpoint exps/trackA_h200_bs128_bf16/step_024000/model.safetensors \
  --output_dir models/sd15_repa_step24k_ema \
  --use_ema \
  --save_single_file
```

### **只转换 U-Net（不包含 VAE、Text Encoder）**

如果你只想要微调的 U-Net：

```python
# 手动加载
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "models/sd15_repa_step24k/unet",
    torch_dtype=torch.float16
)

# 配合原始 SD1.5 的其他组件使用
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,  # 替换 U-Net
    torch_dtype=torch.float16
)
```

---

## 🐛 常见问题

### Q: 模型生成的图片质量不好？

**A**: 尝试：
1. 调整 `guidance_scale` (建议 7.0-8.0)
2. 增加 `num_inference_steps` (50-100)
3. 使用更详细的 prompt
4. 尝试其他 checkpoint (step_030000, step_036000)

### Q: 可以用 LoRA 或 ControlNet 吗？

**A**: 可以！这是一个标准的 SD1.5 模型，支持所有 SD1.5 的插件：
- LoRA
- ControlNet
- T2I-Adapter
- IP-Adapter
等等

### Q: 能和原始 SD1.5 混合吗？

**A**: 可以！使用 checkpoint merger：

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载两个模型
model1 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model2 = StableDiffusionPipeline.from_pretrained("models/sd15_repa_step24k")

# 混合 U-Net 权重（0.5 = 50/50 混合）
for key in model1.unet.state_dict():
    model1.unet.state_dict()[key] = (
        0.5 * model1.unet.state_dict()[key] +
        0.5 * model2.unet.state_dict()[key]
    )

model1.save_pretrained("models/sd15_mixed")
```

---

## 📈 性能建议

### **速度优化**

```python
pipe = StableDiffusionPipeline.from_pretrained(...)

# 1. 使用 xFormers
pipe.enable_xformers_memory_efficient_attention()

# 2. 使用 torch.compile (PyTorch 2.0+)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# 3. 使用 FP16
pipe = pipe.to("cuda", dtype=torch.float16)
```

### **内存优化**

```python
# CPU offload (节省显存)
pipe.enable_model_cpu_offload()

# 或者顺序 CPU offload
pipe.enable_sequential_cpu_offload()
```

---

## 📝 总结

✅ **你现在拥有**：
1. 标准 Diffusers 格式的完整模型
2. WebUI/ComfyUI 兼容的单文件 checkpoint
3. 可以在任何支持 SD1.5 的工具中使用

✅ **可以做**：
- Python 脚本生成图片
- WebUI/ComfyUI 图形界面使用
- 配合 LoRA/ControlNet 等插件
- 上传到 HuggingFace Hub 分享

🎉 **开始创作吧！**
