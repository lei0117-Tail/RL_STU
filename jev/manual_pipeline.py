"""
手动实现 OpenJev 的完整推理流水线（教学版）
================================================
不使用 OpenJev 类，只用 transformers 裸加载模型，
把"输入 → 决策结果"的每一步都摊开手写出来，
让你清楚看到 transformers 原生返回的张量是如何一步步变成 JSON 决策结果的。

流程总览（对应代码中的 6 个步骤）：
  步骤1  拼输入：把 state/questions 拼成 [CLS][STATE]xx[Q]xx[OPT]xx...[SEP] 并分词
  步骤2  主干推理：transformers 的 AutoModel 前向传播 → 语义向量 (L, 1024)
  步骤3  均值池化：把"每个token一个向量"压缩成"每个问题/选项一个向量"
  步骤4  决策头打分：3个向量拼接 → MLP → 每个选项一个分数(logit)
  步骤5  概率化：logit / 校准温度 → softmax → 概率分布
  步骤6  格式化：组装成 choice/score/noul 的 JSON 形态

运行：
  .venv/bin/python jev/manual_pipeline.py
"""

import json
import os

import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_root, "jev", "models", "open-jev-deberta-v3-large")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 准备：加载三样东西（主干、分词器、决策头权重、配置）
# ============================================================
from transformers import AutoModel, AutoTokenizer          # noqa: E402
from safetensors.torch import load_file                    # noqa: E402

print("准备阶段：加载模型三件套")
from typing import Any

tok: Any = AutoTokenizer.from_pretrained(MODEL_DIR)             # 分词器（含 [STATE]/[Q]/[OPT] 标记）
backbone = AutoModel.from_pretrained(                      # ← 这就是"transformers 加载模型"
    MODEL_DIR, attn_implementation="eager").to(device).eval()
head_weights = load_file(os.path.join(MODEL_DIR, "head.safetensors"))  # 决策头权重（transformers 不认识它！）
cfg = json.load(open(os.path.join(MODEL_DIR, "open_jev_config.json")))
TEMPERATURE = cfg["temperature"]                           # 校准温度 1.05

# 三个特殊标记的 token id（已内置在模型的 added_tokens.json 里）
M_STATE = tok.convert_tokens_to_ids("[STATE]")  # type: ignore
M_Q     = tok.convert_tokens_to_ids("[Q]")      # type: ignore
M_OPT   = tok.convert_tokens_to_ids("[OPT]")    # type: ignore

# ============================================================
# 测试输入
# ============================================================
state = "I was charged twice for the same order and nobody answers my emails. I want my money back now."
questions = [
    {"type": "noul",   "instructions": "The customer is asking for a refund."},
    {"type": "choice", "instructions": "Which product area is the message about?",
     "options": ["fees & charges", "pin & security", "refund & dispute", "card", "other"]},
]

# ============================================================
# 步骤 1：手动拼接并分词（OpenJev 的 Collator 干的活）
# ============================================================
print("\n" + "=" * 64)
print("步骤 1：拼接输入格式（原本由 OpenJev 内部的 Collator 完成）")
print("=" * 64)

def encode(text: str) -> list[int]:
    """分词（不加特殊符号）"""
    return tok(text, add_special_tokens=False)["input_ids"]  # type: ignore

ids = [tok.cls_token_id, M_STATE] + encode(state)[:256]    # state 截断到 256 token  # type: ignore

# 记录每个问题、每个选项的"文本 token"位置范围（用于步骤3的均值池化）
q_spans   = []   # [(问题文本起始, 结束)]
opt_spans = []   # [[(选项1起始,结束), (选项2起始,结束), ...], ...]

for q in questions:
    ids.append(M_Q)
    t = encode(q["instructions"])
    q_spans.append((len(ids), len(ids) + len(t)))
    ids += t
    spans = []
    options = ["no", "yes"] if q["type"] == "noul" else q["options"]
    for opt in options:
        ids.append(M_OPT)
        t = encode(opt)
        spans.append((len(ids), len(ids) + len(t)))
        ids += t
    opt_spans.append(spans)

ids.append(tok.sep_token_id)  # type: ignore

# 展示拼出来的句子长什么样
print("拼接后的输入（把 token id 翻译回文字）：")
print(" ", tok.decode(ids)[:200], "...")  # type: ignore

# ============================================================
# 步骤 2：主干前向传播 —— 这一步才是 transformers 负责的部分
# ============================================================
print("\n" + "=" * 64)
print("步骤 2：transformers 主干 forward（原生能力的全部范围）")
print("=" * 64)

input_ids = torch.tensor([ids], device=device)
attn = torch.ones_like(input_ids)
with torch.no_grad():
    out = backbone(input_ids=input_ids, attention_mask=attn)

hidden = out.last_hidden_state[0]        # 去掉 batch 维 → (L, 1024)
print(f"transformers 原生返回：{tuple(hidden.shape)} 的浮点张量")
print(f"  → 形状含义：{hidden.shape[0]} 个 token，每个 token 一个 {hidden.shape[1]} 维向量")
print(f"  → 此时模型已'读懂'了文本，但还没有任何判断/概率！")

# ============================================================
# 步骤 3：均值池化 —— 把 token 级向量压缩成 问题级/选项级 向量
# ============================================================
print("\n" + "=" * 64)
print("步骤 3：均值池化（每个选项的文本 token 向量 → 求平均 → 一个向量）")
print("=" * 64)

def mean_pool(start: int, end: int) -> torch.Tensor:
    """对 [start, end) 范围的 token 向量求平均"""
    return hidden[start:end].mean(dim=0)                  # → (1024,)

q_vecs   = [mean_pool(s, e) for s, e in q_spans]          # 每个问题一个向量
opt_vecs = [[mean_pool(s, e) for s, e in spans] for spans in opt_spans]
print(f"池化前：{hidden.shape[0]} 个 token 向量（每个 {hidden.shape[1]} 维）")
print(f"池化后：{sum(len(v) for v in opt_vecs)} 个选项向量 + {len(q_vecs)} 个问题向量")
print("  → 就像把一篇文章的每个字的注解，汇总成'每个段落的大意'")

# ============================================================
# 步骤 4：决策头打分 —— 语义向量 → 选项分数
# ============================================================
print("\n" + "=" * 64)
print("步骤 4：决策头 forward（head.safetensors，transformers 加载不到的部分）")
print("=" * 64)

# 决策头结构：cat[问题向量, 选项向量, 二者逐元素相乘] → Linear(3072→1024) → GELU → Linear(1024→1)
W1, b1 = head_weights["0.weight"].to(device), head_weights["0.bias"].to(device)
W2, b2 = head_weights["2.weight"].to(device), head_weights["2.bias"].to(device)

all_probs = []
for qi, q in enumerate(questions):
    logits = []
    for ov in opt_vecs[qi]:
        g = torch.cat([q_vecs[qi], ov, q_vecs[qi] * ov])  # (3072,) 问题、选项、交互三视图
        h = torch.nn.functional.gelu(g @ W1.T + b1)       # 隐层 (1024,)
        logits.append((h @ W2.T + b2).item())             # 一个标量分数
    print(f"  问题{qi + 1} 原始 logits（未归一化的分数）：{[round(l, 3) for l in logits]}")
    all_probs.append(logits)

# ============================================================
# 步骤 5：概率化 —— 除以校准温度 + softmax
# ============================================================
print("\n" + "=" * 64)
print("步骤 5：logit ÷ 校准温度(1.05) → softmax → 概率")
print("=" * 64)

results = []
for qi, q in enumerate(questions):
    lg = torch.tensor(all_probs[qi])
    probs = (lg / TEMPERATURE).softmax(dim=0).tolist()    # 温度让过度自信变柔和
    options = ["no", "yes"] if q["type"] == "noul" else q["options"]
    print(f"  问题{qi + 1} 概率：" + "  ".join(f"{o}:{p:.3f}" for o, p in zip(options, probs)))

    # 步骤 6：格式化输出（和 OpenJev.decide 返回的 JSON 完全同构）
    if q["type"] == "noul":
        results.append({"noul": probs[1]})                # p(yes)
    else:
        best = probs.index(max(probs))
        results.append({"choice": options[best],
                        "probabilities": dict(zip(options, [round(p, 4) for p in probs])),
                        "confidence": round(probs[best], 4)})

print("\n" + "=" * 64)
print("步骤 6：最终结果（手动流水线）")
print("=" * 64)
print(json.dumps(results, ensure_ascii=False, indent=2))

# ============================================================
# 验证：和 OpenJev 官方封装的结果对比，必须完全一致
# ============================================================
print("\n" + "=" * 64)
print("验证：与 OpenJev 官方封装对比")
print("=" * 64)

import importlib
import sys

sys.path.insert(0, MODEL_DIR)
OpenJev = importlib.import_module("typed_decisions.open_jev").OpenJev

m = OpenJev.from_pretrained(MODEL_DIR)
official = m.decide(state, questions)
print("官方封装结果：")
print(json.dumps(official, ensure_ascii=False, indent=2))

match = all(
    (abs(a["noul"] - b["noul"]) < 1e-4 if "noul" in a else a["choice"] == b["choice"])
    for a, b in zip(results, official)
)
print(f"\n{'✅ 手动流水线与官方封装结果一致！' if match else '❌ 结果不一致，请检查实现'}")
print("说明：OpenJev 类做的事 = 本脚本步骤1~6 的封装，没有更多魔法。")

