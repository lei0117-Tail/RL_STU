import os
import shutil

# ⚠️ 必须在所有 huggingface 库 import 之前设置，否则不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 国内镜像
os.environ["HF_HUB_DISABLE_XET"] = "1"                # 禁用 XET 协议，防止走 cas-bridge.xethub.hf.co 超时

import torch
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载其余环境变量（HF_TOKEN 等）

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTConfig
from trl import SFTTrainer  # noqa: E402

# ==========================================
# 0. 模型选择（通过 .env 中的 SELECT_MODEL 控制）
# ==========================================
_root        = os.path.dirname(os.path.dirname(__file__))
SELECT_MODEL = os.getenv("SELECT_MODEL", "Qwen2.5-3B")          # 默认 Qwen2.5-3B
_local_model = os.path.join(_root, "models", SELECT_MODEL)
model_id     = _local_model if os.path.isdir(_local_model) else f"Qwen/{SELECT_MODEL}"
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{model_id}")

# ==========================================
# 1. 加载金融数据集
# ==========================================
print("正在从 Hugging Face 下载金融数据集...")
# 我们只取前 2000 条数据来做快速测试，跑通后再用全量数据
dataset = load_dataset("gbharti/finance-alpaca", split="train[:2000]")

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
    return f"指令: {example['instruction']}\n输入: {example['input']}\n回答: {example['output']}"

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
    per_device_train_batch_size=1,      # 【关键】每次只吃 1 条数据，绝不撑爆 24G 内存
    gradient_accumulation_steps=4,      # 攒够 4 条数据再更新一次参数，变相实现 batch_size=4
    learning_rate=2e-4,
    logging_steps=10,                   # 每 10 步在屏幕上打印一次日志
    max_steps=200,                      # 为了演示，我们只训练 200 步就停止
    save_steps=100,                     # 每 100 步保存一次插件
    optim="adamw_torch",
    report_to="none",                   # 关闭第三方日志记录
    max_length=512,                     # 【关键】限制每条数据的文本长度，太长会导致内存溢出
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

