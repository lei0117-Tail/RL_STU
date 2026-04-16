"""
DPO 训练脚本（串联方案：SFT merge → DPO，自生成数据）
========================================================
与 train_dpo.py 的区别：
  ❌ 并联：原始模型 → 新 LoRA（DPO 不知道 SFT 学了什么）
  ✅ 串联：原始模型 → 加载SFT LoRA → merge合并 → 新 LoRA（DPO 站在 SFT 肩膀上）

merge_and_unload() 的作用：
  把 LoRA 插件的权重"烧录"进基础模型，得到一个融合了 SFT 知识的完整模型
  之后 DPOTrainer 在这个更聪明的模型上套新 LoRA 继续训练

训练结果保存到：dpo/finance-qwen-3b-dpo-merged-final/

运行前提：
  sft/finance-qwen-3b-lora-final/ 目录必须存在（先跑完 sft/train_finance_mac.py）

运行方式：
  # 第一步：先生成 DPO 数据（只需一次）
  .venv/bin/python dpo/generate_dpo_data.py

  # 第二步：串联 DPO 训练
  .venv/bin/python dpo/train_dpo_merged.py
"""

import json
import os
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

load_dotenv()

# ==========================================
# 路径配置
# ==========================================
_root    = os.path.dirname(os.path.dirname(__file__))
_dpo_dir = os.path.dirname(__file__)

BASE_MODEL_PATH = os.path.join(_root, "models/Qwen2.5-3B")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "Qwen/Qwen2.5-3B"

SFT_LORA_PATH    = os.path.join(_root, "sft/finance-qwen-3b-lora-final")
DATA_FILE        = os.path.join(_dpo_dir, "dpo_finance_data.jsonl")
_new_models      = os.path.join(_root, "new_models")
SFT_MERGED_PATH  = os.path.join(_new_models, "sft-merged")                # SFT merge 后的完整模型（可选）
OUTPUT_DIR       = os.path.join(_new_models, "checkpoints/dpo-merged-lora")
FINAL_OUTPUT     = os.path.join(_new_models, "dpo-merged-final")

# ==========================================
# 前置检查
# ==========================================
if not os.path.isdir(SFT_LORA_PATH):
    raise FileNotFoundError(
        f"找不到 SFT LoRA：{SFT_LORA_PATH}\n"
        "请先运行：.venv/bin/python sft/train_finance_mac.py"
    )

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"找不到 DPO 数据文件：{DATA_FILE}\n"
        "请先运行：.venv/bin/python dpo/generate_dpo_data.py"
    )

# ==========================================
# 1. 加载数据集
# ==========================================
print(f"加载 DPO 数据：{DATA_FILE}")
records = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line.strip()))

dataset = Dataset.from_list(records)
print(f"共 {len(dataset)} 条 DPO 训练数据")

# ==========================================
# 2. 加载 SFT 合并模型（串联起点）
# ==========================================
# 优先复用已有的 SFT merge 模型（由 tools/merge_sft_lora.py 生成）
# 若不存在则现场执行 merge，并询问是否保存以备后用
if os.path.isdir(SFT_MERGED_PATH):
    # ── 快速路径：直接加载已合并的完整模型，跳过 merge 耗时步骤 ──
    print(f"\n[串联方案] 发现已有 SFT 合并模型，直接加载：{SFT_MERGED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED_PATH,
        dtype=torch.bfloat16,
        device_map="mps"
    )
    print(f"  ✅ 加载完成！模型 = 原始 Qwen + SFT 金融知识（已合并）\n")
else:
    # ── 慢速路径：现场执行 原始模型 → SFT LoRA → merge ──
    print(f"\n[串联方案] 未发现 SFT 合并模型，现场执行 merge ...")
    print(f"  提示：可先运行 tools/merge_sft_lora.py 生成合并模型，下次直接加载更快")

    tokenizer = AutoTokenizer.from_pretrained(SFT_LORA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Step 1/3: 加载原始模型 {BASE_MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="mps"
    )

    print(f"  Step 2/3: 挂载 SFT LoRA 权重 {SFT_LORA_PATH} ...")
    model = PeftModel.from_pretrained(model, SFT_LORA_PATH)

    print(f"  Step 3/3: merge_and_unload，将 SFT 权重烧录进模型 ...")
    model = model.merge_and_unload()
    print(f"  ✅ 合并完成！现在模型 = 原始 Qwen + SFT 金融知识（完整权重）\n")

# ==========================================
# 3. 在合并后的模型上配置新 LoRA
# ==========================================
# 这个新 LoRA 是在"SFT之后的模型"基础上训练的
# 等于：DPO 的起点 = 原始模型 + SFT知识，而不是原始模型
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================
# 4. DPO 训练参数
# ==========================================
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    max_steps=200,
    logging_steps=10,
    save_steps=100,
    optim="adamw_torch",
    report_to="none",
    max_length=512,
    beta=0.1,
)

# ==========================================
# 5. 启动训练
# ==========================================
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# ==========================================
# 5b. 断点续训检测
# ==========================================
resume_from = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        resume_from = max(checkpoints, key=lambda x: int(x.rsplit("-", 1)[-1]))
        print(f"⏩ 发现断点，从 {resume_from} 继续训练")
    else:
        print("🆕 未发现断点，从头开始训练")
else:
    print("🆕 未发现断点，从头开始训练")

print("🚀 串联 DPO 训练开始！")
print("   起点 = 原始模型 + SFT金融知识（已 merge），效果优于并联方案")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 6. 保存
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"\n🎉 串联 DPO 训练完成！")
print(f"   串联 DPO LoRA 保存在：{FINAL_OUTPUT}")
print(f"   （此 LoRA 建立在 SFT 知识之上，推理时仍需搭配原始基础模型使用）")

# ==========================================
# 7. 清理中间 checkpoint
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n🧹 已清理训练中间文件：{OUTPUT_DIR}")

