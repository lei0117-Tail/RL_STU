# 大模型微调与强化学习：核心原理总结

> 本文档系统梳理 SFT、DPO、GRPO、PPO、模型蒸馏、LoRA 等核心概念及其内在联系。

---

## 目录

1. [训练方法全景](#1-训练方法全景)
2. [SFT / DPO / GRPO / PPO 对比](#2-sft--dpo--grpo--ppo-对比)
3. [模型蒸馏（Knowledge Distillation）](#3-模型蒸馏knowledge-distillation)
4. [PPO 与 DPO 的本质关系](#4-ppo-与-dpo-的本质关系)
5. [Reward Model 的角色](#5-reward-model-的角色)
6. [LoRA 原理详解](#6-lora-原理解释)
7. [训练到底更新了什么？](#7-训练到底更新了什么)

---

## 1. 训练方法全景

```
┌─────────────────────────────────────────────────────────────┐
│                    大模型微调技术栈                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐             │
│   │  SFT    │ → │   DPO    │ → │  GRPO    │   串联方案     │
│   │(监督微调)│   │(偏好优化) │   │(强化学习) │              │
│   └─────────┘    └──────────┘    └──────────┘             │
│        ↓               ↓               ↓                  │
│   学"怎么答"      学"哪个更好"      学"探索最优"            │
│                                                             │
│   ┌─────────────────────────────────────────────┐          │
│   │         模型蒸馏（Knowledge Distillation）    │          │
│   │   大模型(Teacher) → 数据/信号 → 小模型(Student) │         │
│   └─────────────────────────────────────────────┘          │
│                                                             │
│   ┌─────────────────────────────────────────────┐          │
│   │            LoRA（低秩适配器）                   │          │
│   │   不更新全量参数，只训练少量增量矩阵 A × B       │          │
│   └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. SFT / DPO / GRPO / PPO 对比

### 一句话区分

| 方法 | 类比 | 核心机制 | 有反馈循环？ |
|------|------|----------|-------------|
| **SFT** | 给学生发标准答案，让他背 | `(prompt, response)` 监督学习 | ❌ |
| **DPO** | 给学生两份作文，说"这篇比那篇好"，让他自己悟 | `(prompt, chosen, rejected)` 偏好优化 | ❌ 离线静态 |
| **GRPO/PPO** | 让学生反复答题，答得好加分、答得差扣分，自己探索 | 强化学习 + 规则/RM 打分 | ✅ 在线动态 |

### 详细对比

| 维度 | SFT | DPO | PPO (含 GRPO) |
|------|-----|-----|---------------|
| **数据格式** | `(prompt, response)` | `(prompt, chosen, rejected)` | `prompt` + 实时生成 + 打分 |
| **训练信号来源** | 人工标注的标准答案 | 人工标注的偏好对比 | Reward Model 或规则函数 |
| **是否需要 RM** | ❌ | ❌（隐式等价） | ✅ PPO 需要 / GRPO 用规则替代 |
| **学习方式** | 离线监督学习 | 离线偏好优化 | 在线强化学习 |
| **稳定性** | ⭐⭐⭐ 最稳定 | ⭐⭐⭐ 很稳定 | ⭐⭐ 较难调参 |
| **效果上限** | 中等 | 较高 | 最高 |
| **计算成本** | 低 | 中 | 高 |
| **代表应用** | 所有基础模型的预训练后微调 | LLaMA 3 Chat、Mistral | ChatGPT (RLHF)、DeepSeek-R1 |

### 完整 RLHF 流程 vs DPO

```
传统 RLHF（PPO 路线）:
  SFT → 训练 RM → PPO 训练（Actor + Critic + KL 惩罚）

DPO（简化路线）:
  SFT → DPO 直接优化（数学上等价于 PPO + RM，但不需要显式训练 RM）

DPO 论文标题：
《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》
```

---

## 3. 模型蒸馏（Knowledge Distillation）

### 核心思想

用大模型（Teacher）的知识来指导小模型（Student），本质上都是"大模型指导小模型"。

### 方式一：数据蒸馏（最常见）

```
大模型(Teacher) → 生成高质量回答 → 小模型(Student) SFT 学习
```

- DeepSeek-R1-Distill 就是这个思路：用 R1 生成的 CoT 推理链来训练 Qwen/Llama
- 也叫 **Distillation via Data**
- 你项目中的 `generate_dpo_data.py` 就是轻量版：用当前模型生成 rejected，结合更好的 chosen 做 DPO

### 方式二：软标签蒸馏（Logit Distillation）

```python
# Teacher 输出的是概率分布（软标签），不是硬的 0/1
teacher_logits = teacher_model(input)  # [0.7, 0.2, 0.1, ...]
student_logits = student_model(input)

# 损失函数 = KL 散度，让 Student 的分布逼近 Teacher
loss = KLDivLoss(student_logits / T, teacher_logits / T)
# T 是温度参数，T 越大分布越平滑（包含更多信息）
```

小模型不只学"正确答案是什么"，还学大模型的"思维倾向"和"不确定性"。

### 方式三：过程蒸馏（Process Distillation / CoT 蒸馏）

```
大模型生成带 逐步推理的回答（Chain-of-Thought）
    ↓
小模型学整个推理过程，而不只是最终答案
```

### 蒸馏 vs PPO vs DPO 的知识传递方式

| 方法 | 知识如何从"大"传递到"小" |
|------|------------------------|
| 数据蒸馏 | 大模型的**输出文本**直接变成训练数据 |
| 软标签蒸馏 | 大模型的**概率分布（logits）**直接对齐 |
| PPO | 大模型(RM)的**评分标量**通过 RL 梯度传递 |
| DPO | 人类偏好的**排序信息**编码进损失函数 |

---

## 4. PPO 与 DPO 的本质关系

### 为什么你觉得它们像？

因为核心循环看起来相似：

```
PPO:  生成回答 → 打分 → 反馈更新
DPO:  准备好坏对 → 对比 → 直接更新
```

关键区别在于**反馈的实时性**：

```
PPO（在线 / on-policy）:
  每一步都用当前模型自己生成的数据
  模型更新 → 数据分布变 → 继续更新
  ↑ 动态循环，不稳定但上限高

DPO（离线 / off-policy）:
  数据提前准备好，训练时不变
  一次性训练完
  ↑ 静态稳定，简单高效
```

### 数学上的联系

DPO 证明了：**不需要显式的 Reward Model 和 PPO，可以直接从偏好数据推导出等价的优化目标**。

$$L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

其中 $\pi_{ref}$ 是参考模型（通常是 SFT 后的模型），$y_w$ 是 chosen，$y_l$ 是 rejected。

---

## 5. Reward Model 的角色

### PPO 中的 RM 就是一个被微调过的语言模型

```
人类标注 (prompt, chosen, rejected)
    ↓
训练 Reward Model（RM）
    ↓ RM 本质上是一个语言模型，只是最后一层输出标量分数
Student 模型生成回答 → RM 打分 → PPO 更新 Student
```

RM 通常的结构：
- 底层：一个语言模型（如 GPT-2、RoBERTa 等）
- 最后一层：替换为单个线性层，输出一个标量分数

### 不同方法的打分来源

| 方法 | 谁来打分？ | 打分方式 |
|------|-----------|---------|
| 传统 PPO | Reward Model（神经网络） | 学出来的评分函数 |
| GRPO（本项目） | 规则函数 | 可验证的结果（数学对不对、代码能不能跑通） |
| DPO | 不需要显式打分 | 偏好排序隐含在数据中 |
| 数据蒸馏 | 无打分 | 直接用大模型的输出作为 ground truth |

### DeepSeek-R1 的创新

DeepSeek-R1 用**规则函数完全替代了 RM**：

```python
# 不需要另一个神经网络来打分
def reward(response, question):
    if is_math(question):
        return 1.0 if check_answer(response, question) else 0.0
    elif is_code(question):
        return 1.0 if code_runs(response) else 0.0
    # ...
```

这就是你项目中 **GRPO** 的思路——更稳定、更省资源、不需要额外训练 RM。

---

## 6. LoRA 原理解释

### 为什么需要 LoRA？

全量微调一个大语言模型（如 Qwen2.5-3B，30亿参数）需要巨大的 GPU 内存。LoRA 的核心思想：

> **不直接更新原始的大权重矩阵 W，而是训练两个极小的矩阵 A 和 B，让它们的乘积 ΔW = B×A 近似表达权重的变化量。**

### LoRA 的结构

以 FFN 层的 `gate_proj` 为例（Qwen2.5-3B 实际尺寸：4096×14336）：

```
输入 x (4096维)
    │
    ├──→ W (冻结不动, 4096×14336) ──────────┐
    │                                        │ (+)
    ├──→ A (4096×16, 随机初始化) ──→ B (16×14336, 从0初始化) ─┘
    │                                        │
    ↓                                        ↓
                                    合并后的输出
```

- **A 矩阵**：降维投影，将高维输入映射到低秩空间（随机初始化，通常固定不动或用较小学习率）
- **B 矩阵**：升维投影，将低秩空间映射回原维度（从零初始化，主要学习对象）
- **ΔW = B × A**：学到的权重增量，维度与原始 W 相同

### 参数量对比

$$W_{original}: 4096 \times 14336 \approx 58,700,000 \text{ 参数}$$
$$A: 4096 \times 16 = 65,536 \text{ 参数}$$
$$B: 16 \times 14336 = 229,376 \text{ 参数}$$
$$\text{LoRA 总计}: \approx 294,000 \text{ 参数} \approx \mathbf{0.5\%}$$

### LoRA 作用范围——不只是 FFN

本项目配置的 `target_modules`：

```
Transformer 每一层（共 36 层 for Qwen2.5-3B）:
├── Self-Attention
│   ├── q_proj:  W_q + B_q × A_q     ← 一对 LoRA
│   ├── k_proj:  W_k + B_k × A_k     ← 一对 LoRA
│   ├── v_proj:  W_v + B_v × A_v     ← 一对 LoRA
│   └── o_proj:  W_o + B_o × A_o     ← 一对 LoRA
│
└── MLP (FFN)
    ├── gate_proj: W_gate + B_g × A_g  ← 一对 LoRA
    ├── up_proj:   W_up + B_u × A_u    ← 一对 LoRA
    └── down_proj: W_down + B_d × A_d  ← 一对 LoRA

总计: 36 层 × 7 个模块 = 252 对 A、B 矩阵
```

### merge_and_unload() 做了什么？

```python
# 合并前（推理时运行）
output = x @ W + x @ (B @ A)           # 原始权重 + LoRA 增量（两次矩阵乘法）

# 合并后（merge_and_unload 之后）
W_new = W + (B @ A)                    # 永久写入新权重（一次矩阵乘法）
output = x @ W_new                     # 直接使用，无额外开销
```

合并后的 `model.safetensors` 文件中存储的是完整的 `W_new`，不再区分哪些来自原始模型、哪些来自 LoRA。

### 关键超参数

| 参数 | 含义 | 常见值 | 说明 |
|------|------|--------|------|
| `r`（Rank） | LoRA 秩，决定表达能力 | 8 / 16 | 越大越强但参数越多 |
| `lora_alpha` | 缩放系数 | 2×r | 实际影响 = alpha/r × BA |
| `target_modules` | 注入哪些层 | q/k/v/o + FFN | 越多效果越好但越慢 |
| `lora_dropout` | Dropout 正则化 | 0.05 | 防止过拟合 |

---

## 7. 训练到底更新了什么？

### 全量微调时——所有参数都更新

```
Transformer 模型参数结构:
├── Embedding（词向量表）        ← ✅ 更新：词的新语义
├── Attention 层
│   ├── Q 投影矩阵              ← ✅ 更新：关注什么
│   ├── K 投影矩阵              ← ✅ 更新：如何关联
│   ├── V 投影矩阵              ← ✅ 更新：提取什么信息
│   └── Output 投影             ← ✅ 更新：整合结果
├── FFN 层
│   ├── gate_proj (第一层)      ← ✅ 更新：门控
│   ├── up_proj (第二层)        ← ✅ 更新：扩展
│   └── down_proj (第三层)      ← ✅ 更新：压缩
├── LayerNorm                   ← ✅ 更新：数值稳定
├── LM Head（输出层）            ← ✅ 更新：概率分布
└── Position Embedding          ← ✅ 更新（如果有）
```

### 各层学到什么？

| 模块 | 功能 | 学到的影响程度 |
|------|------|--------------|
| **Embedding 词向量** | 词的语义表示 | 🟡 中等 —— 金融语境下词义变化 |
| **Q 投影** | Query：关注哪些 token | 🔴🔴🔴 最关键 —— 注意力焦点 |
| **K 投影** | Key：如何匹配关联 | 🔴🔴🔴 最关键 —— 注意力模式 |
| **V 投影** | Value：提取什么信息 | 🔴🔴🔴 最关键 —— 信息选择 |
| **O 投影** | Output：整合多头注意力 | 🔴 重要 —— 结果整合 |
| **FFN gate/up/down** | 知识存储与推理 | 🔴🔴🔴 最关键 —— 主要知识载体 |
| **LayerNorm** | 归一化稳定 | 🟢 微调 |
| **LM Head** | 输出 token 概率 | 🔴 重要 —— 最终预测 |

### 直观理解

- **QKV（注意力层）**：决定模型"看什么"——学会关注金融术语、数字、时间等关键信息
- **FFN（前馈网络）**：决定模型"知道什么"——存储金融领域知识和推理逻辑
- **Embedding（词向量）**：决定"词是什么意思"——金融语境下同一个词的含义可能不同

> **研究结论**：Transformer 的知识主要存储在 FFN 层中，但 QKV 的注意力模式决定了如何调用这些知识。两者缺一不可。

### LoRA 微调时——只更新 A 和 B

使用 LoRA 时，上述所有模块的**原始权重全部冻结**，只有每个目标模块上的 A 和 B 矩阵参与训练。merge 之后，这些增量被永久写入对应模块的权重中。

---

## 附录：各方法的关系图谱

```
                        ┌──────────────┐
                        │  预训练模型   │
                        │  (Base LLM)  │
                        └──────┬───────┘
                               │
                          ┌────▼────┐
                          │   SFT   │  ← 学会基本任务能力
                          └────┬────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
          ┌──────▼──┐   ┌─────▼────┐  ┌────▼─────┐
          │   DPO   │   │   PPO    │  │  蒸馏     │
          │(离线偏好)│   │(在线RL)  │  │(Teacher) │
          └────┬─────┘   └─────┬────┘  └────┬─────┘
               │               │            │
               ▼               ▼            ▼
          对齐模型          对齐+探索      能力迁移
          (稳定可控)        (效果最强)    (快速复制)

          DPO 变体:
          ├─ IPO (Identity PO)     — 改善 DPO 的过拟合问题
          ├─ KTO (Kahneman-Tversky) — 只需"好/坏"标签，不需配对
          └─ ORPO (Odds Ratio PO)  — 简化 DPO 目标函数

          PPO 变体:
          ├─ GRPO (Group Relative) — 本项目使用，无需 Critic
          ├─ ReMax                — 改善奖励分布
          └─ PPO-Max              — 多目标优化

