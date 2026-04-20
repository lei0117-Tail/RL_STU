import os
import shutil

# ⚠️ 必须在所有 huggingface 库 import 之前设置，否则不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 国内镜像
os.environ["HF_HUB_DISABLE_XET"] = "1"                # 禁用 XET 协议，防止走 cas-bridge.xethub.hf.co 超时

import torch
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载其余环境变量（HF_TOKEN 等）

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTConfig
from trl import SFTTrainer  # noqa: E402

# ==========================================
# 0. 模型选择（通过 .env 中的 SELECT_MODEL 控制）
# ==========================================
_root        = os.path.dirname(os.path.dirname(__file__))
SELECT_MODEL  = os.getenv("SELECT_MODEL", "Qwen2.5-3B")          # 默认 Qwen2.5-3B
HF_MODEL_ORG  = os.getenv("HF_MODEL_ORG", "Qwen")                # HF 组织名，gemma 系列填 google
_local_model  = os.path.join(_root, "models", SELECT_MODEL)
model_id      = _local_model if os.path.isdir(_local_model) else f"{HF_MODEL_ORG}/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{model_id}")

# ==========================================
# 1. 加载金融数据集（全量 finance-alpaca + FinGPT 补充）
# ==========================================
print("正在加载金融数据集...")

# ── finance-alpaca：全量，标准金融问答（约 68912 条）
# 控制数量避免单数据集主导，取 4000 条
ds_alpaca = load_dataset("gbharti/finance-alpaca", split="train[:4000]")
print(f"  finance-alpaca: {len(ds_alpaca)} 条")

# ── FinGPT sentiment：金融情感分析补充（约 76772 条），取 1000 条保持比例 4:1
ds_fingpt = load_dataset("FinGPT/fingpt-sentiment-train", split="train[:1000]")
print(f"  FinGPT/fingpt-sentiment-train: {len(ds_fingpt)} 条")

from datasets import concatenate_datasets


# FinGPT 格式：{input, output} → 统一成 {instruction, input, output}
def normalize_fingpt(example):
    return {
        "instruction": example["input"],   # FinGPT 的 input 就是指令
        "input":       "",
        "output":      example["output"],
    }

ds_fingpt = ds_fingpt.map(normalize_fingpt, remove_columns=ds_fingpt.column_names)

# 合并两个数据集，shuffle 打散
dataset = concatenate_datasets([ds_alpaca, ds_fingpt]).shuffle(seed=42)
print(f"  合并后总计: {len(dataset)} 条")

# ==========================================
# 2. 加载模型与分词器 (M4 专属优化)
# ==========================================
print(f"正在加载原模型，来源：{model_id} ...")

# 检测模型架构，多模态模型（如 gemma4）需要特殊处理
try:
    _cfg = AutoConfig.from_pretrained(model_id)
    _arch = _cfg.model_type if hasattr(_cfg, 'model_type') else ""
except Exception:
    _arch = ""
_is_multimodal = _arch in ("gemma4", "paligemma", "llava", "idefics", "mllama")
if _is_multimodal:
    print(f"  检测到多模态模型 (model_type={_arch})，使用 AutoModelForCausalLM 尝试加载文本层")

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 核心优化：使用 bfloat16 精度加载模型，M4 芯片原生支持
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)

# ==========================================
# 3. 配置 LoRA 插件 (省内存神器)
# ==========================================
# gemma4 是多模态模型，视觉塔和音频塔中的模块是 Gemma4ClippableLinear，PEFT 不支持。
# 必须用正则表达式只匹配 language_model 路径下的标准 nn.Linear 模块，跳过 vision/audio tower。
if _is_multimodal and _arch == "gemma4":
    # 正则精确匹配：只注入 language_model 解码器层的注意力投影
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
# 新版 trl 1.x：不需要手动 get_peft_model，SFTTrainer 会通过 peft_config 自动处理

# ==========================================
# 4. 数据格式化函数
# ==========================================
# 新版 trl 1.x 的 formatting_func 每次传入单条样本（非批量），直接返回字符串即可
def formatting_prompts_func(example):
    instruction = example.get('instruction', '').strip()
    inp         = example.get('input', '').strip()
    output      = example.get('output', '').strip()
    return f"指令: {instruction}\n输入: {inp}\n回答: {output}"

# ==========================================
# 5. 设置训练参数 (严格防爆内存)
# ==========================================
# 新版 trl 1.x 使用 SFTConfig 替代 TrainingArguments，max_seq_length 移到这里
# LoRA 输出目录携带模型名，方便区分不同基础模型训练的结果
_new_models  = os.path.join(_root, "new_models")
OUTPUT_DIR   = os.path.join(_new_models, f"checkpoints/{SELECT_MODEL}-sft-lora")
FINAL_OUTPUT = os.path.join(_new_models, f"{SELECT_MODEL}-sft-lora-final")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # 【关键】每次只吃 1 条数据，绝不撑爆内存
    gradient_accumulation_steps=4,      # 攒够 4 条数据再更新一次参数，变相实现 batch_size=4
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,                      # 全量数据，训练步数提升到 500
    save_steps=200,
    optim="adamw_torch",
    report_to="none",
    max_length=512,
)

# ==========================================
# 6. 启动 SFT (监督微调) 训练
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
)

# ==========================================
# 6. 断点续训检测：自动找到最新 checkpoint
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

print("🚀 准备就绪，M4 芯片开始炼丹！")
trainer.train(resume_from_checkpoint=resume_from)

# ==========================================
# 7. 保存最终的 LoRA 插件
# ==========================================
trainer.save_model(FINAL_OUTPUT)
print(f"🎉 训练完成！你的金融大模型插件已保存在 {FINAL_OUTPUT} 目录下")

# ==========================================
# 8. 清理中间 checkpoint（节省磁盘空间）
# ==========================================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"🧹 已清理训练中间文件：{OUTPUT_DIR}")
    print(f"   （最终 LoRA 权重已保存在 {FINAL_OUTPUT}，checkpoint 不再需要）")

# ==========================================
# 9. 自动 merge：将 LoRA 烧录进基础模型
# ==========================================
import gc

print()
print("=" * 60)
print("Step merge: 将 SFT LoRA 合并进基础模型...")
print("=" * 60)

MERGE_OUTPUT = os.path.join(_root, "merge_models", f"{SELECT_MODEL}-sft-merged")

print(f"  基础模型：{model_id}")
print(f"  SFT LoRA：{FINAL_OUTPUT}")
print(f"  输出目录：{MERGE_OUTPUT}")
print()

# ── 先释放训练占用的显存 ──────────────────────────────────────────────
# 训练结束后 trainer.model 是 PeftModel，仍占着显存。
# 必须先彻底释放，再加载 base_model，否则两份模型同时在显存里会 OOM。
print("  释放训练模型显存 ...")
del trainer
del model
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
print("  ✅ 显存已释放")
print()

# 重新加载原始基础模型（干净状态，无 LoRA）
print("  加载基础模型 ...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)

print("  挂载 SFT LoRA ...")
base_model = PeftModel.from_pretrained(base_model, FINAL_OUTPUT)

print("  执行 merge_and_unload ...")
merged_model = base_model.merge_and_unload()
print("  ✅ merge 完成！")

# 保存合并后模型
print(f"  保存合并模型到 {MERGE_OUTPUT} ...")
os.makedirs(MERGE_OUTPUT, exist_ok=True)
merged_model.save_pretrained(MERGE_OUTPUT)

# tokenizer 从原始模型路径加载并保存（避免 save_pretrained 生成 expanded tokenizer.json）
_tok_merge = AutoTokenizer.from_pretrained(model_id)
if _tok_merge.pad_token is None:
    _tok_merge.pad_token = _tok_merge.eos_token
_tok_merge.save_pretrained(MERGE_OUTPUT)

print()
print("=" * 60)
print("🎉 SFT 训练 + Merge 全部完成！")
print("=" * 60)
print(f"  LoRA 权重：{FINAL_OUTPUT}")
print(f"  合并模型：{MERGE_OUTPUT}")
print()
print("下一步：")
print("  # 生成 DPO 数据（用合并后的 SFT 模型生成 rejected）")
print("  .venv/bin/python3 dpo/generate_dpo_data.py")

