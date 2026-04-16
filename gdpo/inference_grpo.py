"""
四模型对比推理脚本
==================
同时加载四个版本，对同一问题给出回答，直观对比效果差异：
  1. 🤖 原始 Qwen2.5-3B（无微调）
  2. 📚 SFT 微调版（学会金融知识）
  3. 🎯 DPO 微调版（偏好对齐，回答更专业）
  4. 🧠 GRPO 微调版（规则强化，结构更清晰）

运行方式：
  .venv/bin/python gdpo/inference_grpo.py
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
_root     = os.path.dirname(os.path.dirname(__file__))
_grpo_dir = os.path.dirname(__file__)
_dpo_dir  = os.path.join(_root, "dpo")
_sft_dir  = os.path.join(_root, "sft")

BASE_MODEL_PATH  = os.path.join(_root, "models/Qwen2.5-3B")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "Qwen/Qwen2.5-3B"

SFT_LORA_PATH  = os.path.join(_sft_dir,  "finance-qwen-3b-lora-final")
DPO_LORA_PATH  = os.path.join(_dpo_dir,  "finance-qwen-3b-dpo-final")
GRPO_LORA_PATH = os.path.join(_grpo_dir, "finance-qwen-3b-grpo-final")

MAX_NEW_TOKENS = 300

# ==========================================
# 模型加载工具
# ==========================================
def load_model(lora_path: str | None, label: str):
    print(f"  [{label}] 加载中...", end="", flush=True)
    tok = AutoTokenizer.from_pretrained(
        lora_path if lora_path and os.path.isdir(lora_path) else BASE_MODEL_PATH
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="mps"
    )
    if lora_path and os.path.isdir(lora_path):
        mdl = PeftModel.from_pretrained(mdl, lora_path)
        print(f" ✅（LoRA: {os.path.basename(lora_path)}）")
    else:
        print(" ✅（无 LoRA）")

    mdl.eval()
    return tok, mdl


def generate(tokenizer, model, instruction: str, input_text: str = "") -> str:
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ==========================================
# 加载所有可用模型
# ==========================================
print("=" * 60)
print("加载模型中...")
print("=" * 60)

models: dict[str, tuple] = {}

tok_base, mdl_base = load_model(None, "原始 Qwen2.5-3B")
models["🤖 原始模型"] = (tok_base, mdl_base)

if os.path.isdir(SFT_LORA_PATH):
    tok_sft, mdl_sft = load_model(SFT_LORA_PATH, "SFT 微调")
    models["📚 SFT 微调"] = (tok_sft, mdl_sft)
else:
    print(f"  ⚠️  SFT LoRA 未找到，跳过")

if os.path.isdir(DPO_LORA_PATH):
    tok_dpo, mdl_dpo = load_model(DPO_LORA_PATH, "DPO 微调")
    models["🎯 DPO 微调"] = (tok_dpo, mdl_dpo)
else:
    print(f"  ⚠️  DPO LoRA 未找到，跳过（请先运行 dpo/train_dpo.py）")

if os.path.isdir(GRPO_LORA_PATH):
    tok_grpo, mdl_grpo = load_model(GRPO_LORA_PATH, "GRPO 微调")
    models["🧠 GRPO 微调"] = (tok_grpo, mdl_grpo)
else:
    print(f"  ⚠️  GRPO LoRA 未找到，跳过（请先运行 gdpo/train_grpo.py）")

print(f"\n✅ 已加载 {len(models)} 个模型\n")


# ==========================================
# 对比问答
# ==========================================
def compare(instruction: str, input_text: str = ""):
    print("\n" + "=" * 60)
    print(f"❓ 问题：{instruction}")
    if input_text:
        print(f"   补充：{input_text}")
    print("=" * 60)

    for label, (tok, mdl) in models.items():
        answer = generate(tok, mdl, instruction, input_text)
        # 简单统计：字数 + 是否含数字
        import re
        has_number = bool(re.search(r'\d', answer))
        print(f"\n{label}（{len(answer)}字，{'含数字✓' if has_number else '无数字'}）：")
        print(f"  {answer}")
    print("\n" + "-" * 60)


# ==========================================
# 示例演示
# ==========================================
EXAMPLES = [
    ("什么是市盈率（P/E ratio）？它如何帮助投资决策？", ""),
    ("如何评估一只股票是否被低估？", ""),
    ("量化宽松政策对普通投资者有什么影响？", ""),
    ("解释一下什么是资产配置，为什么重要？", "我是刚入门的投资者"),
]

print("【四模型对比演示】")
for inst, inp in EXAMPLES:
    compare(inst, inp)

# ==========================================
# 交互模式
# ==========================================
print("\n进入交互对比模式（输入 q 退出）：")
while True:
    try:
        user_input = input("\n❓ 你的问题：").strip()
        if user_input.lower() in ("q", "quit", "exit", ""):
            print("再见！")
            break
        extra = input("   补充信息（没有直接回车）：").strip()
        compare(user_input, extra)
    except KeyboardInterrupt:
        print("\n再见！")
        break

