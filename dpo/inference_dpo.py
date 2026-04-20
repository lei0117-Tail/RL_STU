"""
DPO 推理对比脚本
================
同时加载三个版本进行对比：
  1. 原始 Qwen2.5-3B（无微调）
  2. SFT 微调版（Qwen2.5-3B-sft-lora-final）
  3. DPO 微调版（Qwen2.5-3B-dpo-final 或 hh 版）

运行方式：
  .venv/bin/python dpo/inference_dpo.py
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
_root    = os.path.dirname(os.path.dirname(__file__))
_dpo_dir = os.path.dirname(__file__)
_new_models = os.path.join(_root, "new_models")

SELECT_MODEL = os.getenv("SELECT_MODEL", "Qwen2.5-3B")
HF_MODEL_ORG = os.getenv("HF_MODEL_ORG", "Qwen")               # HF 组织名，gemma 系列填 google

_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"{HF_MODEL_ORG}/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{BASE_MODEL_PATH}")

SFT_LORA_PATH = os.path.join(_new_models, f"{SELECT_MODEL}-sft-lora-final")
DPO_LORA_PATH = os.path.join(_new_models, f"{SELECT_MODEL}-dpo-merged-final")   # 串联 DPO（自生成数据）
DPO_HH_PATH   = os.path.join(_new_models, f"{SELECT_MODEL}-dpo-hh-merged-final")  # 串联 DPO（hh-rlhf）

MAX_NEW_TOKENS = 300

# ==========================================
# 工具函数
# ==========================================
def load_model(lora_path: str | None = None, label: str = ""):
    """加载基础模型，可选挂载 LoRA"""
    print(f"  加载 {label}...")
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

    mdl.eval()
    return tok, mdl


def generate(tokenizer, model, instruction: str, input_text: str = "") -> str:
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("mps")

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
print("加载模型中（共需约 1-2 分钟）...")
print("=" * 60)

models = {}

# 原始模型（无 LoRA）
tok_base, mdl_base = load_model(None, "原始 Qwen2.5-3B（无微调）")
models["🤖 原始模型"] = (tok_base, mdl_base)

# SFT 模型
if os.path.isdir(SFT_LORA_PATH):
    tok_sft, mdl_sft = load_model(SFT_LORA_PATH, "SFT 微调版")
    models["📚 SFT 微调"] = (tok_sft, mdl_sft)
else:
    print(f"  ⚠️  SFT LoRA 未找到：{SFT_LORA_PATH}，跳过")

# DPO 模型（方案二：自生成数据）
if os.path.isdir(DPO_LORA_PATH):
    tok_dpo, mdl_dpo = load_model(DPO_LORA_PATH, "DPO 微调版（自生成数据）")
    models["🎯 DPO（自生成）"] = (tok_dpo, mdl_dpo)
else:
    print(f"  ⚠️  DPO LoRA 未找到：{DPO_LORA_PATH}，跳过（请先运行 train_dpo.py）")

# DPO 模型（方案一：hh-rlhf）
if os.path.isdir(DPO_HH_PATH):
    tok_hh, mdl_hh = load_model(DPO_HH_PATH, "DPO 微调版（hh-rlhf）")
    models["💬 DPO（hh-rlhf）"] = (tok_hh, mdl_hh)
else:
    print(f"  ⚠️  DPO-HH LoRA 未找到：{DPO_HH_PATH}，跳过（请先运行 train_dpo_hh.py）")

print(f"\n✅ 已加载 {len(models)} 个模型：{list(models.keys())}\n")


# ==========================================
# 对比问答函数
# ==========================================
def compare(instruction: str, input_text: str = ""):
    print("\n" + "=" * 60)
    print(f"❓ 问题：{instruction}")
    if input_text:
        print(f"   补充：{input_text}")
    print("=" * 60)

    for label, (tok, mdl) in models.items():
        answer = generate(tok, mdl, instruction, input_text)
        print(f"\n{label}：")
        print(f"  {answer}")
        print("-" * 60)


# ==========================================
# 示例演示
# ==========================================
EXAMPLES = [
    ("什么是市盈率（P/E ratio）？它如何帮助投资决策？", ""),
    ("如何评估一只股票是否被低估？", ""),
    ("解释一下量化宽松政策的利与弊", ""),
]

print("【示例对比演示】")
for inst, inp in EXAMPLES:
    compare(inst, inp)

# ==========================================
# 交互模式
# ==========================================
print("\n\n进入交互对比模式（输入 q 退出）：")
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

