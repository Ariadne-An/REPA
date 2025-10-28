# 生成扭曲图片时使用的权重路径总结

## 1. 评估命令参数

```bash
PYTHONPATH=/workspace/REPA python evaluation/run_repa_evaluation.py \
    --trained-checkpoint converted/sd15_repa_step24k \  # ← 使用的路径
    --output-root eval_outputs/repa_quality_final \
    --num-samples 1000
```

## 2. converted/sd15_repa_step24k/ 目录内容

```
converted/sd15_repa_step24k/
├── model.safetensors (4.0 GB)  # diffusers格式的UNet权重
├── model_index.json
├── quick_test.png (410K)        # 转换时生成的测试图片（已经是扭曲的）
├── unet/                         # UNet配置文件
├── vae/                          # VAE配置文件
├── text_encoder/                 # CLIP text encoder
├── tokenizer/                    # CLIP tokenizer
├── scheduler/                    # 调度器配置
└── feature_extractor/            # 特征提取器
```

## 3. 这个模型是从哪个checkpoint转换来的？

**源checkpoint路径**: `exps/trackA_h200_bs128_bf16/step_024000/model.safetensors`

**转换时间**: Oct 28 04:52（model.safetensors的时间戳）

**转换脚本**: 很可能是`convert_to_sd15.py`

## 4. step_024000目录里的权重文件

```
exps/trackA_h200_bs128_bf16/step_024000/
├── model.safetensors (3.22 GB)    # ← 转换时用的这个（普通训练权重）
├── ema.pt (3.22 GB)                # ← 应该用这个（EMA权重，质量更好）
├── optimizer.bin (0.02 GB)
├── scheduler.bin (0.00 GB)
├── training_state.pt (0.00 GB)
└── random_states_0.pkl (0.00 GB)
```

## 5. 评估脚本加载模型的逻辑

在 `evaluation/run_repa_evaluation.py` 第102-116行：

```python
def load_pipeline(checkpoint: Path, ...):
    if checkpoint.is_dir():
        # converted/sd15_repa_step24k 是目录，走这个分支
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint,  # 直接加载diffusers格式的模型
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

## 6. 问题总结

1. **使用的权重**: `converted/sd15_repa_step24k/model.safetensors`
2. **权重来源**: 从 `exps/trackA_h200_bs128_bf16/step_024000/model.safetensors` 转换
3. **潜在问题**:
   - step_024000目录有两个权重文件：`model.safetensors`（普通权重）和 `ema.pt`（EMA权重）
   - 转换时可能用了普通权重而不是EMA权重
   - EMA权重通常质量更好，因为它是训练过程中权重的指数移动平均
4. **证据**: 转换时生成的`quick_test.png`就已经是扭曲的

## 7. 后续需要做的

需要检查：
1. 转换脚本是否正确读取了EMA权重
2. EMA权重的格式（ema.pt里的实际内容）
3. 如果EMA权重格式正确，重新转换并评估
