"""
GRPO LoRA 合并脚本
==================
将 GRPO 训练得到的 LoRA 权重合并成一个完整的独立模型。

GRPO 有两种训练方案，合并时的基底不同：

  ┌─────────────────────────────────────────────────────┐
  │ 并联 GRPO（train_grpo.py）                           │
  │   训练起点：原始 Qwen2.5-3B                           │
  │   合并公式：原始模型 + GRPO LoRA = 合并模型            │
  │   合并基底：BASE_MODEL（原始模型）                     │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │ 串联 GRPO（train_grpo_merged.py）                    │
  │   训练起点：SFT 合并模型（原始模型已内化 SFT 知识）       │
  │   合并公式：SFT 合并模型 + GRPO LoRA = 合并模型         │
  │   合并基底：SFT_MERGED_MODEL（SFT 合并后的模型）         │
  └─────────────────────────────────────────────────────┘

三阶段串联完整链路（最终效果最佳）：
  原始模型
    → [SFT] → SFT LoRA → merge → SFT合并模型
    → [DPO] → DPO LoRA → merge → DPO合并模型
    → [GRPO] → GRPO LoRA → merge → 最终合并模型

运行方式（通过命令行参数选择模式）：

    # 并联模式：合并并联 GRPO LoRA
    .venv/bin/python tools/merge_grpo_lora.py --mode parallel

    # 串联模式：合并串联 GRPO LoRA（基于 SFT 合并模型）
    .venv/bin/python tools/merge_grpo_lora.py --mode serial

    # 串联模式（如果还没有 SFT 合并模型，自动先合并 SFT）
    .venv/bin/python tools/merge_grpo_lora.py --mode serial --auto-merge-sft

输出目录：
    并联：gdpo/finance-qwen-3b-grpo-parallel-merged/
    串联：gdpo/finance-qwen-3b-grpo-serial-merged/
"""

import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

SELECT_MODEL = os.getenv("SELECT_MODEL", "Qwen2.5-3B")
HF_MODEL_ORG = os.getenv("HF_MODEL_ORG", "Qwen")               # HF 组织名，gemma 系列填 google
print(f"[SELECT_MODEL={SELECT_MODEL}]")

# ==========================================
# 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="GRPO LoRA 合并工具")
parser.add_argument(
    "--mode",
    choices=["parallel", "serial"],
    default="serial",
    help="并联模式(parallel)：合并并联GRPO LoRA；串联模式(serial)：合并串联GRPO LoRA（默认serial）"
)
parser.add_argument(
    "--auto-merge-sft",
    action="store_true",
    help="串联模式下，若 SFT 合并模型不存在，自动先执行 SFT 合并"
)
args = parser.parse_args()

# ==========================================
# 路径配置
# ==========================================
_root     = os.path.dirname(os.path.dirname(__file__))
_grpo_dir = os.path.join(_root, "gdpo")

_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"{HF_MODEL_ORG}/{SELECT_MODEL}"
_new_models     = os.path.join(_root, "new_models")    # LoRA 插件来源目录
_merge_models   = os.path.join(_root, "merge_models")  # 合并后完整模型输出目录
SFT_LORA_PATH   = os.path.join(_new_models, f"{SELECT_MODEL}-sft-lora-final")
SFT_MERGED_PATH = os.path.join(_merge_models, f"{SELECT_MODEL}-sft-merged")  # 从 merge_models 读取

# 并联：GRPO LoRA 基于原始模型训练
GRPO_PARALLEL_LORA   = os.path.join(_new_models, f"{SELECT_MODEL}-grpo-lora-final")
GRPO_PARALLEL_OUTPUT = os.path.join(_merge_models, f"{SELECT_MODEL}-grpo-parallel-merged")

# 串联：GRPO LoRA 基于 SFT 合并模型训练
GRPO_SERIAL_LORA     = os.path.join(_new_models, f"{SELECT_MODEL}-grpo-merged-final")
GRPO_SERIAL_OUTPUT   = os.path.join(_merge_models, f"{SELECT_MODEL}-grpo-serial-merged")

# ==========================================
# 选择模式
# ==========================================
if args.mode == "parallel":
    MODE_NAME  = "并联"
    LORA_PATH  = GRPO_PARALLEL_LORA
    BASE_PATH  = BASE_MODEL_PATH
    OUTPUT_DIR = GRPO_PARALLEL_OUTPUT
else:
    MODE_NAME  = "串联"
    LORA_PATH  = GRPO_SERIAL_LORA
    BASE_PATH  = SFT_MERGED_PATH
    OUTPUT_DIR = GRPO_SERIAL_OUTPUT

# ==========================================
# 前置检查
# ==========================================
if not os.path.isdir(LORA_PATH):
    if args.mode == "parallel":
        raise FileNotFoundError(
            f"找不到并联 GRPO LoRA：{LORA_PATH}\n"
            "请先运行：.venv/bin/python gdpo/train_grpo.py"
        )
    else:
        raise FileNotFoundError(
            f"找不到串联 GRPO LoRA：{LORA_PATH}\n"
            "请先运行：.venv/bin/python gdpo/train_grpo_merged.py"
        )

# 串联模式：检查 SFT 合并模型，必要时自动合并
if args.mode == "serial" and not os.path.isdir(BASE_PATH):
    if args.auto_merge_sft:
        print(f"⚠️  SFT 合并模型不存在：{BASE_PATH}")
        print("    --auto-merge-sft 已启用，自动先合并 SFT LoRA ...\n")
        if not os.path.isdir(SFT_LORA_PATH):
            raise FileNotFoundError(
                f"找不到 SFT LoRA：{SFT_LORA_PATH}\n"
                "请先运行：.venv/bin/python sft/train_finance_mac.py"
            )
        # 内嵌执行 SFT 合并逻辑
        print("  加载原始模型进行 SFT 合并 ...")
        _tok = AutoTokenizer.from_pretrained(SFT_LORA_PATH)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, dtype=torch.bfloat16, device_map="mps"
        )
        _m = PeftModel.from_pretrained(_m, SFT_LORA_PATH)
        _m = _m.merge_and_unload()
        os.makedirs(SFT_MERGED_PATH, exist_ok=True)
        _m.save_pretrained(SFT_MERGED_PATH)
        _tok.save_pretrained(SFT_MERGED_PATH)
        del _m, _tok
        print(f"  ✅ SFT 合并完成：{SFT_MERGED_PATH}\n")
    else:
        raise FileNotFoundError(
            f"串联模式需要 SFT 合并模型：{BASE_PATH}\n"
            "方案1：先运行 .venv/bin/python tools/merge_sft_lora.py\n"
            "方案2：添加 --auto-merge-sft 参数自动合并"
        )

print("=" * 60)
print(f"GRPO LoRA 合并脚本（{MODE_NAME}模式）")
print("=" * 60)
print(f"  GRPO LoRA：{LORA_PATH}")
print(f"  合并基底：{BASE_PATH}")
print(f"  输出目录：{OUTPUT_DIR}")
print()

# ==========================================
# 1. 加载 tokenizer
# ==========================================
print("Step 1/4: 加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. 加载基底模型
# ==========================================
print(f"Step 2/4: 加载基底模型（{MODE_NAME}基底）...")
if args.mode == "parallel":
    print(f"         并联基底 = 原始 {SELECT_MODEL}（不含 SFT 知识）")
else:
    print(f"         串联基底 = SFT 合并模型（已内化 SFT 金融知识）")

model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    dtype=torch.bfloat16,
    device_map="mps"
)

# ==========================================
# 3. 挂载 GRPO LoRA → merge_and_unload
# ==========================================
print(f"Step 3/4: 挂载 {MODE_NAME} GRPO LoRA 并执行 merge_and_unload ...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
print(f"         ✅ merge 完成！")

# ==========================================
# 4. 保存
# ==========================================
print(f"Step 4/4: 保存合并模型到 {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 60)
print(f"🎉 {MODE_NAME} GRPO LoRA 合并完成！")
print("=" * 60)
print(f"  合并后模型：{OUTPUT_DIR}")
print()

if args.mode == "parallel":
    print("知识来源：原始模型 + GRPO强化学习（奖励函数优化）")
    print("（不含 SFT 金融专业知识，只优化了回答行为风格）")
else:
    print("知识来源：原始模型 + SFT金融知识 + GRPO强化学习")
    print("（完整串联链路：专业知识 × 强化学习行为优化 双重增强）")
    print()
    print("三阶段完整串联链路回顾：")
    print("  原始模型 → SFT合并 → GRPO合并 = 本模型")
    print("  若同时跑了 DPO 串联，完整四阶段：")
    print("  原始模型 → SFT合并 → DPO合并 → GRPO合并")

