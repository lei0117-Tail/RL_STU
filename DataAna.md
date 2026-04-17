# 训练数据集分析报告

## 概述

本项目目标是训练一个**金融领域专用语言模型**，训练流水线为：

```
原始模型 → SFT → DPO → GRPO → 最终模型
```

各阶段对数据集的需求不同，以下对候选数据集逐一分析。

---

## 数据集分析

### 1. `gbharti/finance-alpaca`

**数据格式：**
```json
{
  "instruction": "What is a P/E ratio?",
  "input": "",
  "output": "P/E ratio is price divided by earnings..."
}
```

| 用途 | 适合度 | 理由 |
|------|--------|------|
| **SFT** | ✅✅✅ 最适合 | 标准 instruction-output 格式，直接可用 |
| **DPO 直接用** | ❌ | 只有 chosen，没有 rejected，无法直接做 DPO |
| **SFT merge 后生成 rejected** | ✅✅✅ 最推荐 | chosen=数据集 output，rejected=SFT 模型生成，差距精准 |

**结论：** 这个是 **SFT 主力数据集**，同时也是生成 DPO 数据的原料。
当前项目已使用此数据集做 SFT 训练（取前 2000 条）。

---

### 2. `winddude/reddit_finance_43_250k`

**数据格式：**
```json
{
  "title": "Should I invest in index funds?",
  "selftext": "...",
  "comments": [
    {"body": "Yes, index funds are great...", "score": 1523},
    {"body": "It depends on your risk...", "score": 89},
    {"body": "No way, pick stocks", "score": -12}
  ]
}
```

| 用途 | 适合度 | 理由 |
|------|--------|------|
| **SFT** | ⚠️ 一般 | 需要大量清洗，Reddit 口语化，质量参差不齐 |
| **DPO 直接用** | ✅✅ 适合 | 高赞评论=chosen，低赞/负赞=rejected，天然偏好对，无需额外推理 |
| **SFT merge 后生成 rejected** | ❌ 不必要 | 自带 chosen/rejected，不需要额外生成 |

**结论：** 这个是**现成的 DPO 数据集**，按 score 排序直接拆 chosen/rejected，省去推理生成步骤，适合快速构建 DPO 训练数据。

---

### 3. `FinGPT/fingpt-sentiment-train`

**数据格式：**
```json
{
  "input": "Analyze sentiment: 'Apple reports record profits'",
  "output": "Positive. The news indicates strong financial performance..."
}
```

| 用途 | 适合度 | 理由 |
|------|--------|------|
| **SFT** | ✅✅ 适合 | 金融情感分析任务，可补充 finance-alpaca 的能力维度 |
| **DPO 直接用** | ❌ 不适合 | 情感分析是分类任务，没有偏好对的概念 |
| **SFT merge 后生成 rejected** | ⚠️ 价值有限 | 情感分析答案比较单一，chosen/rejected 差异会很小 |

**结论：** 适合作为 **SFT 补充数据集**，增强金融情感理解能力，不适合做 DPO。

---

### 4. `Open-Orca/OpenOrca`（金融子集）

**数据格式：**
```json
{
  "system_prompt": "You are a helpful assistant...",
  "question": "Explain quantitative easing",
  "response": "Quantitative easing (QE) is a monetary policy..."
}
```
> response 由 GPT-4 生成，质量极高。

| 用途 | 适合度 | 理由 |
|------|--------|------|
| **SFT** | ✅✅✅ 极好 | GPT-4 标注，质量最高，需要过滤金融相关子集 |
| **DPO 直接用** | ❌ | 只有 chosen，无 rejected |
| **SFT merge 后生成 rejected** | ✅✅✅ 质量天花板最高 | GPT-4 回答作 chosen，SFT 模型生成作 rejected，差距最能体现能力边界 |

**结论：** GPT-4 标注的 chosen 质量最高，用作 DPO 的 chosen 天花板最高，是**最优的 SFT→DPO 原料**，但需要手动过滤金融子集。

---

## 汇总对比表

| 数据集 | SFT | DPO 直接用 | SFT merge 后生成 rejected |
|--------|-----|-----------|--------------------------|
| `gbharti/finance-alpaca` | ✅✅✅ | ❌ | ✅✅✅ ← 当前使用方案 |
| `winddude/reddit_finance_43_250k` | ⚠️ | ✅✅ | ❌（已自带） |
| `FinGPT/fingpt-sentiment-train` | ✅✅ | ❌ | ⚠️ |
| `Open-Orca/OpenOrca` 金融子集 | ✅✅✅ | ❌ | ✅✅✅ ← 质量天花板最高 |

---

## 最优训练组合建议

### SFT 阶段
```
主力 → gbharti/finance-alpaca（2000条，当前已用）
补充 → FinGPT/fingpt-sentiment-train（增强情感分析能力）
可选 → OpenOrca 金融子集（GPT-4 质量，需要过滤）
```

### DPO 阶段（两条路选一）

**路线 A（快速）**
```
数据集：winddude/reddit_finance_43_250k
方式：直接按 score 高低拆 chosen/rejected，无需推理
优点：不需要跑模型推理，省时省力
```

**路线 B（精准，推荐）**
```
数据集：gbharti/finance-alpaca 或 OpenOrca 金融子集
chosen：数据集原始高质量回答
rejected：用 SFT merged 模型生成（而非原始模型）
优点：chosen/rejected 差距更精准，DPO 能学到更细粒度的偏好
```

> 路线 B 中使用 SFT 模型生成 rejected 的理由：
> - 原始模型生成的 rejected 差异太大（废话、重复），DPO 只学到"别废话"这种粗粒度信号
> - SFT 模型生成的 rejected 已经"比较好"但还不够专业，DPO 能学到更精细的金融表达方式

### GRPO 阶段
```
奖励函数：长度(40%) + 数字专业性(30%) + 不重复(20%) + 格式(10%)
基础模型：DPO merged（在 DPO 对齐后的模型上做强化学习）
```

---

## 完整流水线

```
原始模型
  ↓ SFT（finance-alpaca + FinGPT）
SFT LoRA → merge_and_unload → SFT merged
  ↓ 用 SFT merged 生成 DPO rejected（路线 B）
DPO LoRA 训练（在 SFT merged 基础上）→ merge_and_unload → DPO merged
  ↓ GRPO 强化学习（在 DPO merged 基础上）
最终模型（金融专用，对齐充分）
```

---

## 当前项目状态

| 阶段 | 状态 | 数据集 | 输出 |
|------|------|--------|------|
| SFT | ✅ 完成 | `gbharti/finance-alpaca` 2000条 | `merge_models/Qwen2.5-3B-sft-merged` |
| DPO | ✅ 完成 | `gbharti/finance-alpaca` 500条，rejected=原始模型 | `merge_models/Qwen2.5-3B-dpo-serial-merged` |
| GRPO | 🔄 进行中 | 奖励函数驱动 | `new_models/checkpoints/` |

**下一步优化方向：** 将 DPO 的 rejected 生成模型从原始模型换成 SFT merged，重新生成 `dpo_finance_data.jsonl`，再跑一轮 DPO 训练。

