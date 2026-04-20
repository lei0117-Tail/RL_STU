"""
DPO 训练脚本（方案二：自生成数据）
=====================================
使用 generate_dpo_data.py 生成的 dpo_finance_data.jsonl 进行训练。

数据格式：
  chosen  = finance-alpaca 标准答案（高质量）
  rejected = 原始 Qwen 生成的回答（较差）

训练结果保存到：dpo/Qwen2.5-3B-dpo-final/（全新目录，不覆盖 SFT）

运行方式：
  # 第一步：先生成数据（只需执行一次）
  .venv/bin/python dpo/generate_dpo_data.py

  # 第二步：DPO 训练
  .venv/bin/python dpo/train_dpo.py
"""

import json
import os
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import Dataset
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
HF_MODEL_ORG    = os.getenv("HF_MODEL_ORG", "Qwen")               # HF 组织名，gemma 系列填 google
_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"{HF_MODEL_ORG}/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{BASE_MODEL_PATH}")

SFT_LORA_PATH   = os.path.join(_root, f"sft/{SELECT_MODEL}-lora-final")   # 对应模型的 SFT LoRA
DATA_FILE       = os.path.join(_dpo_dir, "dpo_finance_data.jsonl")
_new_models     = os.path.join(_root, "new_models")
OUTPUT_DIR      = os.path.join(_new_models, f"checkpoints/{SELECT_MODEL}-dpo-lora")
FINAL_OUTPUT    = os.path.join(_new_models, f"{SELECT_MODEL}-dpo-lora-final")

# ==========================================
# 检查数据文件
# ==========================================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"找不到 DPO 数据文件：{DATA_FILE}\n"
        "请先运行：.venv/bin/python dpo/generate_dpo_data.py"
    )

# ==========================================
# 1. 加载 DPO 数据集
# ==========================================
print(f"加载 DPO 数据：{DATA_FILE}")
records = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line.strip()))

dataset = Dataset.from_list(records)
print(f"共 {len(dataset)} 条 DPO 训练数据")


# ==========================================
# 2. 加载基础模型（从 SFT LoRA 的基础模型出发）
# ==========================================
print(f"加载基础模型：{BASE_MODEL_PATH}")

# 检测模型架构，多模态模型（如 gemma4）需要特殊处理
try:
    _cfg  = AutoConfig.from_pretrained(BASE_MODEL_PATH)
    _arch = getattr(_cfg, 'model_type', '')
except Exception:
    _arch = ''
_is_multimodal = _arch in ("gemma4", "paligemma", "llava", "idefics", "mllama")

choose_path = SFT_LORA_PATH if os.path.isdir(SFT_LORA_PATH) else BASE_MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(choose_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)


# ==========================================
# 3. 配置新的 LoRA 插件（全新一套，不依赖 SFT 的权重）
# ==========================================
# DPO 会生成独立的 LoRA 权重文件，保存在 FINAL_OUTPUT 目录
# gemma4 多模态模型需用正则只注入文本解码器，跳过视觉/音频塔的 Gemma4ClippableLinear
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
    learning_rate=5e-5,             # DPO 学习率要比 SFT 小，避免过拟合
    max_steps=200,
    logging_steps=10,
    save_steps=100,
    optim="adamw_torch",
    report_to="none",
    max_length=512,
    beta=0.1,                       # DPO 温度系数，控制偏好强度，越大越保守
)

# ==========================================
# 5. 启动 DPO 训练
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

print("🚀 DPO 训练开始！新的 LoRA 权重将保存到独立目录，不影响 SFT 结果")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 6. 保存 DPO LoRA 插件（全新文件，与 SFT 完全独立）
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"\n🎉 DPO 训练完成！")
print(f"   DPO LoRA 插件保存在：{FINAL_OUTPUT}")
print(f"   SFT LoRA 插件保存在：{SFT_LORA_PATH}")
print("   两套权重完全独立，可分别加载对比效果")

# ==========================================
# 7. 清理中间 checkpoint
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n🧹 已清理训练中间文件：{OUTPUT_DIR}")

