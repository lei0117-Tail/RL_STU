"""
GRPO 训练脚本（串联方案：SFT merge → GRPO）
=============================================
与 train_grpo.py 的区别：
  ❌ 并联：原始模型 → GRPO 新 LoRA
  ✅ 串联：原始模型 → SFT LoRA → merge → GRPO 新 LoRA

串联的优势：
  GRPO 的起点是"已经懂金融知识的 SFT 模型"，奖励函数训练会更有效
  因为模型已具备基础能力，强化学习只需要微调"回答风格"而非从头学知识

训练结果保存到：gdpo/finance-qwen-3b-grpo-merged-final/

运行方式：
  .venv/bin/python gdpo/train_grpo_merged.py
"""

import os
import re
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import GRPOConfig, GRPOTrainer

load_dotenv()

# ==========================================
# 路径配置
# ==========================================
_root     = os.path.dirname(os.path.dirname(__file__))
_grpo_dir = os.path.dirname(__file__)

# 模型选择（通过 .env 中的 SELECT_MODEL 控制）
SELECT_MODEL    = os.getenv("SELECT_MODEL", "Qwen2.5-3B")
_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"Qwen/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{BASE_MODEL_PATH}")

SFT_LORA_PATH   = os.path.join(_root, f"new_models/{SELECT_MODEL}-sft-lora-final")
_new_models     = os.path.join(_root, "new_models")

_merge_model_path = os.path.join(_root, "merge_models")

SFT_MERGED_PATH = os.path.join(_merge_model_path, f"{SELECT_MODEL}-sft-merged")  # SFT merge 后的完整模型（可选）

OUTPUT_DIR      = os.path.join(_new_models, f"checkpoints/{SELECT_MODEL}-grpo-merged-lora")
FINAL_OUTPUT    = os.path.join(_new_models, f"{SELECT_MODEL}-grpo-merged-final")

# ==========================================
# 前置检查
# ==========================================
if not os.path.isdir(SFT_LORA_PATH):
    raise FileNotFoundError(
        f"找不到 SFT LoRA：{SFT_LORA_PATH}\n"
        "请先运行：.venv/bin/python sft/train_finance_mac.py"
    )

# ==========================================
# 1. 加载数据集（GRPO 只需要 prompt）
# ==========================================
print("加载 finance-alpaca 数据集...")
raw_dataset = load_dataset("gbharti/finaetao nce-alpaca", split="train[:1000]")

def format_prompt(example):
    instruction = example["instruction"].strip()
    input_text  = example.get("input", "").strip()
    return {"prompt": f"指令: {instruction}\n输入: {input_text}\n回答:"}

dataset = raw_dataset.map(format_prompt, remove_columns=raw_dataset.column_names)
dataset = dataset.filter(lambda x: len(x["prompt"]) <= 300)
print(f"有效数据：{len(dataset)} 条")

# ==========================================
# 2. 奖励函数（与 train_grpo.py 完全相同）
# ==========================================

def length_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for text in completions:
        n = len(text.strip())
        if n < 10:
            rewards.append(-1.0)
        elif n < 50:
            rewards.append(0.0)
        elif n <= 500:
            rewards.append(1.0)
        elif n <= 800:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def number_reward(completions: list[str], **kwargs) -> list[float]:
    pattern = re.compile(r'\d+\.?\d*\s*[%倍元万亿美元$€]|\d{2,}|[1-9]\d*\.\d+')
    rewards = []
    for text in completions:
        matches = pattern.findall(text)
        if len(matches) >= 3:
            rewards.append(1.0)
        elif len(matches) >= 1:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def no_repeat_reward(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, prompt in zip(completions, prompts):
        prompt_words = set(re.findall(r'[\u4e00-\u9fff]+|\w+', prompt))
        comp_words   = set(re.findall(r'[\u4e00-\u9fff]+|\w+', completion))
        if not comp_words:
            rewards.append(-1.0)
            continue
        overlap = len(prompt_words & comp_words) / len(comp_words)
        if overlap > 0.8:
            rewards.append(-0.5)
        elif overlap > 0.6:
            rewards.append(0.0)
        else:
            rewards.append(0.5)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for text in completions:
        text = text.strip()
        if not text:
            rewards.append(-1.0)
        elif text[-1] in "。！？.!?":
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    r_length = length_reward(completions)
    r_number = number_reward(completions)
    r_repeat = no_repeat_reward(completions, prompts)
    r_format = format_reward(completions)
    return [
        0.4 * l + 0.3 * n + 0.2 * r + 0.1 * f
        for l, n, r, f in zip(r_length, r_number, r_repeat, r_format)
    ]

# ==========================================
# 3. 加载 SFT 合并模型（串联起点）
# ==========================================
if os.path.isdir(SFT_MERGED_PATH):
    print(f"\n[串联方案] 发现已有 SFT 合并模型，直接加载：{SFT_MERGED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )
    print(f"  ✅ 加载完成！模型 = {SELECT_MODEL} + SFT 金融知识（已合并）\n")
else:
    print(f"\n[串联方案] 未发现 SFT 合并模型，现场执行 merge ...")
    print(f"  提示：可先运行 tools/merge_sft_lora.py 生成合并模型，下次直接加载更快")

    tokenizer = AutoTokenizer.from_pretrained(SFT_LORA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Step 1/3: 加载原始模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )

    print(f"  Step 2/3: 挂载 SFT LoRA ...")
    model = PeftModel.from_pretrained(model, SFT_LORA_PATH)

    print(f"  Step 3/3: merge_and_unload ...")
    model = model.merge_and_unload()
    print(f"  ✅ 合并完成！GRPO 将在懂金融的 SFT 模型基础上做强化学习\n")

# ==========================================
# 4. 新 LoRA 配置
# ==========================================
# 检测模型架构
try:
    _cfg  = AutoConfig.from_pretrained(BASE_MODEL_PATH)
    _arch = getattr(_cfg, 'model_type', '')
except Exception:
    _arch = ''
_is_multimodal = _arch in ("gemma4", "paligemma", "llava", "idefics", "mllama")

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
# 5. GRPO 训练参数
# ==========================================
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_steps=200,
    logging_steps=10,
    save_steps=100,
    optim="adamw_torch",
    report_to="none",
    num_generations=4,
    max_completion_length=200,  # TRL>=1.0: 原 max_new_tokens 改名为 max_completion_length
    temperature=0.9,
    beta=0.01,
)

# ==========================================
# 6. 启动训练
# ==========================================
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=combined_reward,
    peft_config=peft_config,
)

# ==========================================
# 6b. 断点续训检测
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

print("🚀 串联 GRPO 训练开始！")
print("   起点 = SFT 金融知识已融合，强化学习专注于优化回答质量")
print("   奖励函数：长度(40%) + 数字专业性(30%) + 不重复(20%) + 格式(10%)\n")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 7. 保存
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"\n🎉 串联 GRPO 训练完成！")
print(f"   串联 GRPO LoRA 保存在：{FINAL_OUTPUT}")
print(f"\n完整串联链路：")
print(f"  原始模型 → SFT({os.path.join(_root, 'sft/Qwen2.5-3B-sft-lora-final')})")
print(f"           → merge → GRPO LoRA({FINAL_OUTPUT})")

# ==========================================
# 8. 清理中间 checkpoint
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n🧹 已清理训练中间文件：{OUTPUT_DIR}")

