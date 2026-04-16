"""
DPO 训练脚本（方案一：Anthropic/hh-rlhf 数据集）
===================================================
使用 HuggingFace 上经典的 hh-rlhf 数据集直接训练 DPO。

数据集格式（hh-rlhf 原始格式）：
  chosen  : 完整对话文本（Human + Assistant，最后是好回答）
  rejected: 完整对话文本（Human + Assistant，最后是坏回答）

训练结果保存到：dpo/finance-qwen-3b-dpo-hh-final/

运行方式：
  .venv/bin/python dpo/train_dpo_hh.py
"""

import os
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import DPOConfig, DPOTrainer

load_dotenv()

# ==========================================
# 路径配置
# ==========================================
_root    = os.path.dirname(os.path.dirname(__file__))
_dpo_dir = os.path.dirname(__file__)

# 模型选择（通过 .env 中的 SELECT_MODEL 控制）
SELECT_MODEL    = os.getenv("SELECT_MODEL", "Qwen2.5-3B")
_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"Qwen/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{BASE_MODEL_PATH}")

_new_models  = os.path.join(_root, "new_models")
OUTPUT_DIR   = os.path.join(_new_models, f"checkpoints/{SELECT_MODEL}-dpo-hh-lora")
FINAL_OUTPUT = os.path.join(_new_models, f"{SELECT_MODEL}-dpo-hh-lora-final")

# ==========================================
# 1. 加载 hh-rlhf 数据集并转换格式
# ==========================================
print("下载 Anthropic/hh-rlhf 数据集（~200MB，第一次需要一点时间）...")
raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
print(f"原始数据：{len(raw_dataset)} 条")

def parse_hh(example):
    """
    hh-rlhf 原始格式是多轮对话拼接的字符串，例如：
      "\n\nHuman: xxx\n\nAssistant: yyy\n\nHuman: zzz\n\nAssistant: good answer"
    需要拆出 prompt（最后一个 Assistant: 之前的部分）和回答
    """
    def split_prompt_response(text: str):
        # 找最后一个 "\n\nAssistant:" 的位置
        idx = text.rfind("\n\nAssistant:")
        if idx == -1:
            return text, ""
        prompt   = text[:idx + len("\n\nAssistant:")]
        response = text[idx + len("\n\nAssistant:"):].strip()
        return prompt, response

    prompt_c, chosen_resp   = split_prompt_response(example["chosen"])
    prompt_r, rejected_resp = split_prompt_response(example["rejected"])

    # prompt 应该相同（同一个问题），取 chosen 的 prompt
    return {
        "prompt":   prompt_c,
        "chosen":   chosen_resp,
        "rejected": rejected_resp,
    }

print("转换数据格式...")
dataset = raw_dataset.map(parse_hh, remove_columns=raw_dataset.column_names)

# 过滤掉回答为空或过短的
dataset = dataset.filter(
    lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10
)
print(f"有效数据：{len(dataset)} 条")

# ==========================================
# 2. 加载基础模型
# ==========================================
print(f"加载基础模型：{BASE_MODEL_PATH}")

# 检测模型架构
try:
    _cfg  = AutoConfig.from_pretrained(BASE_MODEL_PATH)
    _arch = getattr(_cfg, 'model_type', '')
except Exception:
    _arch = ''
_is_multimodal = _arch in ("gemma4", "paligemma", "llava", "idefics", "mllama")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)

# ==========================================
# 3. LoRA 配置（全新 LoRA，独立于 SFT）
# ==========================================
if _is_multimodal and _arch == "gemma4":
    _target_modules = r"model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
else:
    _target_modules = ["q_proj", "v_proj"]

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=_target_modules,
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
    beta=0.1,               # DPO 温度系数
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

print("🚀 hh-rlhf DPO 训练开始！")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 6. 保存（独立目录）
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"\n🎉 DPO (hh-rlhf) 训练完成！")
print(f"   LoRA 插件保存在：{FINAL_OUTPUT}")

# ==========================================
# 7. 清理中间 checkpoint
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n🧹 已清理训练中间文件：{OUTPUT_DIR}")

