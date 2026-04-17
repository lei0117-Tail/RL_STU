"""
生成 DPO 训练数据
==============================
思路：
  chosen   = finance-alpaca 数据集里的标准答案（高质量）
  rejected = SFT merged 模型生成的回答（已经较好但不够精准）

为什么用 SFT 模型生成 rejected 而非原始模型：
  - 原始模型的 rejected 差得太多（废话、重复），DPO 只能学到粗粒度信号
  - SFT 模型的 rejected 已经不错但还不够专业，DPO 能学到精细对齐

运行方式：
  .venv/bin/python dpo/generate_dpo_data.py

前置条件：必须先跑完 merge_models/Qwen2.5-3B-sft-merged（通过 tools/merge_sft_lora.py 生成）
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
_root        = os.path.dirname(os.path.dirname(__file__))
SELECT_MODEL = os.getenv("SELECT_MODEL", "Qwen2.5-3B")   # 通过 .env 中的 SELECT_MODEL 控制

# rejected 由 SFT merged 模型生成，而非原始模型
SFT_MERGED_PATH = os.path.join(_root, "merge_models", f"{SELECT_MODEL}-sft-merged")
if not os.path.isdir(SFT_MERGED_PATH):
    raise FileNotFoundError(
        f"SFT merged 模型不存在：{SFT_MERGED_PATH}\n"
        f"请先跑 tools/merge_sft_lora.py 生成合并模型"
    )
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载 SFT merged 模型：{SFT_MERGED_PATH}")

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "dpo_finance_data.jsonl")

# 生成多少条 DPO 数据（每条需要跑一次推理，越多越慢）
NUM_SAMPLES    = 1000   # 扬展到 1000 条
MAX_NEW_TOKENS = 200

# ==========================================
# 加载 SFT merged 模型（用来生成“已经较好但不够精准”的 rejected）
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True
)
model.eval()
print(f"✅ SFT merged 模型加载完毕（{DEVICE}）")

# ==========================================
# 加载 finance-alpaca 数据集
# ==========================================
print("加载 finance-alpaca 数据集...")
# 多取一些预留给跳过逻辑，保证最终能出够 1000 条有效数据
dataset = load_dataset("gbharti/finance-alpaca", split=f"train[:{int(NUM_SAMPLES * 1.2)}]")
print(f"共 {len(dataset)} 条数据")

# ==========================================
# 用 SFT merged 模型生成 rejected 回答
# ==========================================
def generate_rejected(instruction: str, input_text: str = "") -> str:
    """用 SFT 已对齐的模型生成回答作为 DPO 的 rejected
    温度略高，让回答有一定随机性，与 chosen 形成恒定差异
    """
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

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
print(f"\n开始生成 {NUM_SAMPLES} 条 DPO 数据（rejected 由 SFT merged 模型生成）...")
print("（大约需要 40-80 分钟，请耐心等待）\n")

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

    # 已生成足够条数则提前退出
    if len(results) >= NUM_SAMPLES:
        break

    rejected = generate_rejected(instruction, input_text)

    # DPO 要求 chosen != rejected，且 rejected 不能完全为空
    # 同时过滤掉 rejected 质量太差的（过短）
    if not rejected or rejected == chosen or len(rejected) < 10:
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

