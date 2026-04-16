"""
DPO 训练脚本（串联方案：SFT merge → DPO hh-rlhf）
====================================================
与 train_dpo_hh.py 的区别：
  ❌ 并联：原始模型 → 新 LoRA
  ✅ 串联：原始模型 → SFT LoRA → merge → 新 LoRA

训练结果保存到：dpo/finance-qwen-3b-dpo-hh-merged-final/

运行方式：
  .venv/bin/python dpo/train_dpo_hh_merged.py
"""

import os
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import load_dataset
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

SFT_LORA_PATH   = os.path.join(_root, "sft/Qwen2.5-3B-sft-lora-final")
_new_models     = os.path.join(_root, "new_models")
SFT_MERGED_PATH = os.path.join(_new_models, "sft-merged")                 # SFT merge 后的完整模型（可选）
OUTPUT_DIR      = os.path.join(_new_models, "checkpoints/dpo-hh-merged-lora")
FINAL_OUTPUT    = os.path.join(_new_models, "dpo-hh-merged-final")

# ==========================================
# 前置检查
# ==========================================
if not os.path.isdir(SFT_LORA_PATH):
    raise FileNotFoundError(
        f"找不到 SFT LoRA：{SFT_LORA_PATH}\n"
        "请先运行：.venv/bin/python sft/train_finance_mac.py"
    )

# ==========================================
# 1. 加载并处理 hh-rlhf 数据集
# ==========================================
print("下载 Anthropic/hh-rlhf 数据集...")
raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")

def parse_hh(example):
    def split_prompt_response(text: str):
        idx = text.rfind("\n\nAssistant:")
        if idx == -1:
            return text, ""
        prompt   = text[:idx + len("\n\nAssistant:")]
        response = text[idx + len("\n\nAssistant:"):].strip()
        return prompt, response

    prompt_c, chosen_resp   = split_prompt_response(example["chosen"])
    _, rejected_resp        = split_prompt_response(example["rejected"])
    return {"prompt": prompt_c, "chosen": chosen_resp, "rejected": rejected_resp}

dataset = raw_dataset.map(parse_hh, remove_columns=raw_dataset.column_names)
dataset = dataset.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
print(f"有效数据：{len(dataset)} 条")

# ==========================================
# 2. 加载 SFT 合并模型（串联起点）
# ==========================================
if os.path.isdir(SFT_MERGED_PATH):
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
    print(f"\n[串联方案] 未发现 SFT 合并模型，现场执行 merge ...")
    print(f"  提示：可先运行 tools/merge_sft_lora.py 生成合并模型，下次直接加载更快")

    tokenizer = AutoTokenizer.from_pretrained(SFT_LORA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Step 1/3: 加载原始模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="mps"
    )

    print(f"  Step 2/3: 挂载 SFT LoRA ...")
    model = PeftModel.from_pretrained(model, SFT_LORA_PATH)

    print(f"  Step 3/3: merge_and_unload ...")
    model = model.merge_and_unload()
    print(f"  ✅ 合并完成！模型 = 原始 Qwen + SFT 金融知识\n")

# ==========================================
# 3. 新 LoRA 配置
# ==========================================
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

print("🚀 串联 DPO (hh-rlhf) 训练开始！")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 6. 保存
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"\n🎉 串联 DPO (hh-rlhf) 训练完成！")
print(f"   串联 DPO-HH LoRA 保存在：{FINAL_OUTPUT}")

# ==========================================
# 7. 清理中间 checkpoint
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n🧹 已清理训练中间文件：{OUTPUT_DIR}")

