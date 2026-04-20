import os

# 镜像 & 禁用 XET（如果需要联网下载模型时生效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# ==========================================
# 配置路径
# ==========================================
_root        = os.path.join(os.path.dirname(__file__), "..")
_new_models  = os.path.join(_root, "new_models")
SELECT_MODEL = os.getenv("SELECT_MODEL", "Qwen2.5-3B")
HF_MODEL_ORG = os.getenv("HF_MODEL_ORG", "Qwen")               # HF 组织名，gemma 系列填 google

_local_model    = os.path.join(_root, "models", SELECT_MODEL)
BASE_MODEL_PATH = _local_model if os.path.isdir(_local_model) else f"{HF_MODEL_ORG}/{SELECT_MODEL}"
LORA_PATH       = os.path.join(_new_models, f"{SELECT_MODEL}-sft-lora-final")
print(f"[SELECT_MODEL={SELECT_MODEL}] 加载模型：{BASE_MODEL_PATH}")

# ==========================================
# 加载原模型 + LoRA 插件
# ==========================================
print(f"加载基础模型：{BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)  # tokenizer 从 LoRA 目录读，保证 pad_token 一致

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="mps"
)

print(f"挂载 LoRA 插件：{LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()  # 推理模式，关闭 dropout
print("✅ 模型加载完毕，开始对话！\n")

# ==========================================
# 推理函数
# ==========================================
def ask(instruction: str, input_text: str = "") -> str:
    prompt = f"指令: {instruction}\n输入: {input_text}\n回答:"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,       # 最多生成 300 个 token
            temperature=0.7,          # 适当随机性，回答更自然
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分，去掉输入的 prompt
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ==========================================
# 交互式问答（Ctrl+C 退出）
# ==========================================
EXAMPLES = [
    ("什么是市盈率（P/E ratio）？", ""),
    ("解释一下量化宽松政策", ""),
    ("如何分散投资风险？", "我有10万元想投资"),
]

print("=" * 50)
print("示例问题演示：")
print("=" * 50)
for instruction, input_text in EXAMPLES:
    print(f"\n❓ 问：{instruction}")
    if input_text:
        print(f"   补充：{input_text}")
    answer = ask(instruction, input_text)
    print(f"💬 答：{answer}")
    print("-" * 50)

print("\n进入交互模式（输入 q 退出）：")
while True:
    try:
        user_input = input("\n❓ 你的问题：").strip()
        if user_input.lower() in ("q", "quit", "exit", ""):
            print("再见！")
            break
        extra = input("   补充信息（没有直接回车）：").strip()
        print("💬 回答中...")
        answer = ask(user_input, extra)
        print(f"💬 {answer}")
    except KeyboardInterrupt:
        print("\n再见！")
        break

