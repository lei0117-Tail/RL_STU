"""
GRPO 训练脚本（Group Relative Policy Optimization）
=====================================================
DeepSeek-R1 同款方法，无需 chosen/rejected 标注数据。

核心思路：
  1. 同一个金融问题，让模型生成 4 个不同回答（一组）
  2. 用规则函数对每个回答打分（0~1 之间）
  3. 组内归一化：得分高于平均的鼓励，低于平均的惩罚
  4. 模型逐渐学会生成高分回答

奖励函数设计（金融场景）：
  - length_reward    : 回答长度适中（50~500字），过短或过长扣分
  - number_reward    : 包含具体数字/百分比，体现专业性
  - no_repeat_reward : 回答不大段重复 prompt，避免无效复读
  - format_reward    : 回答结构清晰（包含句号等标点）

训练结果保存到：gdpo/finance-qwen-3b-grpo-final/

运行方式：
  .venv/bin/python gdpo/train_grpo.py
"""

import os
import re
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig
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

_new_models  = os.path.join(_root, "new_models")
OUTPUT_DIR   = os.path.join(_new_models, f"checkpoints/{SELECT_MODEL}-grpo-lora")
FINAL_OUTPUT = os.path.join(_new_models, f"{SELECT_MODEL}-grpo-lora-final")

# ==========================================
# 1. 加载并格式化数据集
# ==========================================
# GRPO 只需要 prompt（问题），不需要答案！
# 模型自己生成回答，奖励函数来评判好坏
print("加载 finance-alpaca 数据集...")
raw_dataset = load_dataset("gbharti/finance-alpaca", split="train[:1000]")

def format_prompt(example):
    """把数据集里的 instruction/input 拼成 prompt"""
    instruction = example["instruction"].strip()
    input_text  = example.get("input", "").strip()
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    # GRPO 要求数据集有 "prompt" 字段
    return {"prompt": prompt}

dataset = raw_dataset.map(format_prompt, remove_columns=raw_dataset.column_names)
# 过滤掉 prompt 太长的（超过 300 字符），避免内存溢出
dataset = dataset.filter(lambda x: len(x["prompt"]) <= 300)
print(f"有效数据：{len(dataset)} 条")

# ==========================================
# 2. 奖励函数设计（GRPO 的核心！）
# ==========================================

def length_reward(completions: list[str], **kwargs) -> list[float]:
    """
    长度奖励：鼓励回答适中长度（50~500 个字符）
    太短说明模型在敷衍，太长可能在复读或废话
    """
    rewards = []
    for text in completions:
        n = len(text.strip())
        if n < 10:
            rewards.append(-1.0)        # 极短，严惩
        elif n < 50:
            rewards.append(0.0)         # 偏短，中性
        elif n <= 500:
            rewards.append(1.0)         # 理想长度，满分
        elif n <= 800:
            rewards.append(0.5)         # 稍长，小扣分
        else:
            rewards.append(0.0)         # 过长，废话太多
    return rewards


def number_reward(completions: list[str], **kwargs) -> list[float]:
    """
    数字奖励：金融回答里包含具体数字/百分比更专业
    例如："市盈率通常在 15-25 之间" 比 "市盈率有高有低" 更有价值
    """
    # 匹配数字、百分比、比率（如 3.5%、1:10、$100）
    pattern = re.compile(r'\d+\.?\d*\s*[%倍元万亿美元$€]|\d{2,}|[1-9]\d*\.\d+')
    rewards = []
    for text in completions:
        matches = pattern.findall(text)
        if len(matches) >= 3:
            rewards.append(1.0)         # 包含 3 个以上数字，很专业
        elif len(matches) >= 1:
            rewards.append(0.5)         # 有数字，加分
        else:
            rewards.append(0.0)         # 没有数字，不加不减
    return rewards


def no_repeat_reward(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    不重复奖励：回答不应该大量复读 prompt 里的内容
    如果回答和 prompt 重叠度太高，说明模型在偷懒
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # 提取 prompt 里的关键词（去掉"指令:""输入:""回答:"标签）
        prompt_words = set(re.findall(r'[\u4e00-\u9fff]+|\w+', prompt))
        comp_words   = set(re.findall(r'[\u4e00-\u9fff]+|\w+', completion))

        if not comp_words:
            rewards.append(-1.0)
            continue

        # 计算重叠比例
        overlap = len(prompt_words & comp_words) / len(comp_words)
        if overlap > 0.8:
            rewards.append(-0.5)        # 大量复读，惩罚
        elif overlap > 0.6:
            rewards.append(0.0)
        else:
            rewards.append(0.5)         # 有自己的内容，奖励
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    格式奖励：回答有完整的句子结构（以句号/感叹号结尾，不是突然截断）
    """
    rewards = []
    for text in completions:
        text = text.strip()
        if not text:
            rewards.append(-1.0)
        elif text[-1] in "。！？.!?":
            rewards.append(0.5)         # 结尾完整，加分
        else:
            rewards.append(0.0)         # 截断了，不加不减
    return rewards


def combined_reward(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    综合奖励：把所有子奖励加权合并
    权重设计：长度最重要，其次是内容质量（数字），再次是不重复，最后是格式
    """
    w_length  = 0.4
    w_number  = 0.3
    w_repeat  = 0.2
    w_format  = 0.1

    r_length = length_reward(completions)
    r_number = number_reward(completions)
    r_repeat = no_repeat_reward(completions, prompts)
    r_format = format_reward(completions)

    return [
        w_length * l + w_number * n + w_repeat * r + w_format * f
        for l, n, r, f in zip(r_length, r_number, r_repeat, r_format)
    ]

# ==========================================
# 3. 加载基础模型
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
# 4. LoRA 配置（全新，独立于 SFT 和 DPO）
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
# 5. GRPO 训练参数
# ==========================================
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,             # GRPO 学习率要更小，策略更新要谨慎
    max_steps=200,
    logging_steps=10,
    save_steps=100,
    optim="adamw_torch",
    report_to="none",
    # GRPO 专属参数
    num_generations=4,              # 每个 prompt 生成 4 个回答，组内比较
    max_completion_length=200,      # 每个回答最多 200 个 token（原 max_new_tokens）
    temperature=0.9,                # 生成时的温度，适当高一些增加多样性
    beta=0.01,                      # KL 散度惩罚系数，防止模型偏离太远
)

# ==========================================
# 6. 启动 GRPO 训练
# ==========================================
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=combined_reward,   # 传入综合奖励函数
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

print("🚀 GRPO 训练开始！")
print("   每个问题会生成 4 个回答，组内打分对比")
print("   奖励函数：长度(40%) + 数字专业性(30%) + 不重复(20%) + 格式(10%)\n")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 7. 保存（独立目录）
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"\n🎉 GRPO 训练完成！")
print(f"   GRPO LoRA 插件保存在：{FINAL_OUTPUT}")
print(f"   SFT  LoRA 插件保存在：{os.path.join(_root, 'sft/finance-qwen-3b-lora-final')}")
print(f"   DPO  LoRA 插件保存在：{os.path.join(_root, 'dpo/finance-qwen-3b-dpo-final')}")

# ==========================================
# 8. 清理中间 checkpoint
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"\n🧹 已清理训练中间文件：{OUTPUT_DIR}")

