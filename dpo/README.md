原始模型↓ SFT LoRA 训练
  ↓ merge_and_unload → SFT merged
  ↓ 用 SFT merged 生成 rejected（新方案）
  ↓ DPO LoRA 训练（在 SFT merged 基础上）
  ↓ merge_and_unload → DPO merged
  ↓ GRPO LoRA 训练（在 DPO merged 基础上）
  ↓ 最终模型