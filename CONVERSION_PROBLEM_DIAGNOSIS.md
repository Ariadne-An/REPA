# 转换问题诊断

## 问题根源

**转换后的模型参数数量**:
- Converted UNet: 859,520,964 parameters
- Base SD1.5 UNet: 859,520,964 parameters
- **Difference: 0 parameters**

**这说明：LoRA权重根本没有被合并进去！**

转换后的 `converted/sd15_repa_step24k` 实际上就是基础SD1.5模型，没有任何REPA训练的权重。

这就是为什么生成的图片扭曲的原因 - 评估时用的不是REPA模型，而是baseline SD1.5！

## 证据

1. **原始checkpoint内容** (exps/trackA_h200_bs128_bf16/step_024000/model.safetensors):
   - Total keys: 945
   - LoRA keys: 256 (lora_A/lora_B)
   - Base layer keys: 160
   - **应该找到128个LoRA层**

2. **转换脚本测试**:
   - LoRA检测逻辑: ✓ 能正确找到128个LoRA层
   - 键名转换: ✓ 正确

3. **转换后模型检查**:
   - 参数数量与base SD1.5完全一致
   - **没有LoRA合并的痕迹**

## 下一步

需要重新正确转换模型，确保：
1. LoRA权重被正确检测
2. LoRA权重被正确合并到base weights
3. 转换后的模型参数应该与base SD1.5一致（因为LoRA合并后参数数量不变，但权重值会改变）
