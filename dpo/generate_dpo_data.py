"""
生成 DPO 训练数据（方案二）
==============================
思路：
  chosen  = finance-alpaca 数据集里的标准答案（高质量）
  rejected = 原始 Qwen2.5-3B（未微调）生成的回答（较差）

生成完成后保存为 dpo/dpo_finance_data.jsonl，供 train_dpo.py 使用。

运行方式：
  .venv/bin/python dpo/generate_dpo_data.py
"""

import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

load_dotenv()

# ==========================================
# 配置
# ==========================================
_root = os.path.dirname(os.path.dirname(__file__))
BASE_MODEL_PATH = os.path.join(_root, "models/Qwen2.5-3B")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = "Qwen/Qwen2.5-3B"

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "dpo_finance_data.jsonl")

# 生成多少条 DPO 数据（每条需要跑一次推理，越多越慢）
NUM_SAMPLES = 500   # 用 500 条做 DPO 训练，可根据时间调整
MAX_NEW_TOKENS = 200

# ==========================================
# 加载原始模型（未微调，用来生成"差回答" rejected）
# ==========================================
print(f"加载原始模型：{BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="mps"
)
model.eval()
print("✅ 原始模型加载完毕")

# ==========================================
# 加载 finance-alpaca 数据集
# ==========================================
print("加载 finance-alpaca 数据集...")
dataset = load_dataset("gbharti/finance-alpaca", split=f"train[:{NUM_SAMPLES}]")
print(f"共 {len(dataset)} 条数据")

# ==========================================
# 用原始模型生成 rejected 回答
# ==========================================
def generate_rejected(instruction: str, input_text: str = "") -> str:
    """让原始模型（未微调）生成一个回答，作为 DPO 的 rejected"""
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.9,       # 稍高温度，让回答更随机（更"差"）
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ==========================================
# 构建 DPO 数据集并保存
# ==========================================
print(f"\n开始生成 {NUM_SAMPLES} 条 DPO 数据，每条需要模型推理一次...")
print("（大约需要 20-40 分钟，请耐心等待）\n")

results = []
skipped = 0

for item in tqdm(dataset, desc="生成 rejected 回答"):
    instruction = item["instruction"].strip()
    input_text  = item.get("input", "").strip()
    chosen      = item["output"].strip()

    # 跳过 chosen 太短的（质量差）
    if len(chosen) < 20:
        skipped += 1
        continue

    rejected = generate_rejected(instruction, input_text)

    # DPO 要求 chosen != rejected，且 rejected 不能完全为空
    if not rejected or rejected == chosen:
        skipped += 1
        continue

    # 构造 DPO 标准格式：prompt / chosen / rejected
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    results.append({
        "prompt":   prompt,
        "chosen":   chosen,
        "rejected": rejected,
    })

# 保存为 jsonl
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ 生成完毕！")
print(f"   有效数据：{len(results)} 条")
print(f"   跳过数据：{skipped} 条")
print(f"   已保存到：{OUTPUT_FILE}")
print("\n接下来运行 train_dpo.py 开始 DPO 训练")

