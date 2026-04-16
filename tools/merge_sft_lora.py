"""
SFT LoRA 合并脚本
==================
将 SFT 训练得到的 LoRA 权重"烧录"进基础模型，输出一个完整的独立模型。

合并前：
    原始模型（Qwen2.5-3B，~6GB）
        +
    SFT LoRA 插件（adapter_model.safetensors，~几十MB）
    （推理时两者叠加，需要同时加载两个目录）

合并后：
    合并模型（finance-qwen-3b-sft-merged，~6GB 完整权重）
    （单独一个目录，直接加载即可使用，无需再挂 LoRA）

适用场景：
    1. 作为 DPO/GRPO 串联训练的起点（本项目的串联脚本已内置此逻辑）
    2. 作为独立推理模型部署（比原始模型+LoRA 推理更方便）
    3. 对外分发：合并后的模型不暴露 LoRA 结构细节

运行方式：
    .venv/bin/python tools/merge_sft_lora.py

输出目录：
    sft/finance-qwen-3b-sft-merged/
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# ==========================================
# 路径配置
# ==========================================
_root = os.path.dirname(os.path.dirname(__file__))

# 模型选择（通过 .env 中的 SELECT_MODEL 控制）
SELECT_MODEL    = os.getenv("SELECT_MODEL", "Qwen2.5-3B")
_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"Qwen/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{BASE_MODEL_PATH}")

_new_models   = os.path.join(_root, "new_models")
SFT_LORA_PATH = os.path.join(_new_models, f"{SELECT_MODEL}-sft-lora-final")
OUTPUT_DIR    = os.path.join(_new_models, f"{SELECT_MODEL}-sft-merged")

# ==========================================
# 前置检查
# ==========================================
if not os.path.isdir(SFT_LORA_PATH):
    raise FileNotFoundError(
        f"找不到 SFT LoRA：{SFT_LORA_PATH}\n"
        "请先运行：.venv/bin/python sft/train_finance_mac.py"
    )

print("=" * 60)
print("SFT LoRA 合并脚本")
print("=" * 60)
print(f"  基础模型：{BASE_MODEL_PATH}")
print(f"  SFT LoRA：{SFT_LORA_PATH}")
print(f"  输出目录：{OUTPUT_DIR}")
print()

# ==========================================
# 1. 加载 tokenizer
# ==========================================
print("Step 1/4: 加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(SFT_LORA_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. 加载原始基础模型
# ==========================================
print("Step 2/4: 加载原始基础模型 ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="mps"
)

# ==========================================
# 3. 挂载 SFT LoRA → merge_and_unload
# ==========================================
print("Step 3/4: 挂载 SFT LoRA 并执行 merge_and_unload ...")
print("         （将 LoRA 权重数学叠加进基础模型，移除 LoRA 插件结构）")
model = PeftModel.from_pretrained(model, SFT_LORA_PATH)
model = model.merge_and_unload()
print("         ✅ merge 完成！模型现已内化 SFT 金融知识")

# ==========================================
# 4. 保存合并后的完整模型
# ==========================================
print(f"Step 4/4: 保存合并模型到 {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 60)
print("🎉 SFT LoRA 合并完成！")
print("=" * 60)
print(f"  合并后模型：{OUTPUT_DIR}")
print()
print("后续用法：")
print(f"  # 直接加载合并后的模型推理（无需再挂 LoRA）")
print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
print()
print(f"  # 或作为 DPO/GRPO 串联训练的起点（脚本会自动检测并加载）")
print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
print(f"  trainer = DPOTrainer(model=model, peft_config=new_lora_config, ...)")

