# 大模型微调实战：SFT → DPO → GRPO 完整流程

基于 Qwen2.5-3B + LoRA，在 Apple M 芯片（MPS）上实现从监督微调到强化学习对齐的完整链路。

---

## 核心概念

### 三种训练方法

| 方法 | 数据格式 | 训练信号 | 代表模型 |
|---|---|---|---|
| **SFT**（监督微调） | 问题 + 标准答案 | 直接模仿答案 | 所有基础模型 |
| **DPO**（直接偏好优化） | 问题 + 好答案 + 坏答案 | 好坏答案对比 | LLaMA 3、Mistral |
| **GRPO**（组相对策略优化） | 只需问题 | 规则函数打分 | DeepSeek-R1 |

### 一句话区分

- **SFT** = 给学生发标准答案，让他背
- **DPO** = 给学生两份作文，说"这篇比那篇好"，让他自己悟
- **GRPO** = 让学生反复答题，答得好加分、答得差扣分，自己探索

---

## LoRA 是什么

LoRA（Low-Rank Adaptation）是一种**插件式微调**方法：

```
原始模型（冻结，6GB）
    +
LoRA 插件（可训练，几十MB）
    ↓
推理时两者叠加生效
```

只训练极少量参数（< 0.1%），大幅降低内存需求，训练完得到一个小文件，可随时挂载/卸载。

---

## 串联 vs 并联

### 并联方案（各自独立，适合学习对比）

```
原始模型 ──→ SFT LoRA
原始模型 ──→ DPO LoRA
原始模型 ──→ GRPO LoRA
```

三套 LoRA 互相独立，可对比各方法效果差异。

### 串联方案（知识累积，适合生产）

```
原始模型 → [SFT] → SFT 模型 → [DPO] → DPO 模型 → [GRPO] → 最终模型
```

每一步都继承上一步的成果，效果更好。工业界标准流程（ChatGPT、LLaMA 等均采用）。

### 串联的三种实现方式

**方式一：直接在 SFT LoRA 上继续训练（同一个 LoRA）**
```python
model = PeftModel.from_pretrained(model, SFT_LORA_PATH, is_trainable=True)
trainer = DPOTrainer(model=model, peft_config=None)  # 复用已有 LoRA
```
- ✅ 推理最简单：原始模型 + 一个 LoRA
- ⚠️ SFT 知识可能被后续训练覆盖（灾难性遗忘）

**方式二：merge 后套新 LoRA（本项目串联脚本采用）**
```python
model = PeftModel.from_pretrained(model, SFT_LORA_PATH)
model = model.merge_and_unload()   # 将 SFT 权重烧录进模型
trainer = DPOTrainer(model=model, peft_config=new_lora_config)
```
- ✅ SFT 知识永久保留，不会被覆盖
- ⚠️ 推理时需要用"merge 后的模型"作为基底，而非原始模型

**方式三：两个独立 LoRA 叠加**
```python
model = PeftModel.from_pretrained(model, SFT_LORA_PATH, adapter_name="sft")
model.add_adapter("grpo", LoraConfig(...))
model.set_adapter("grpo")  # 只训练 grpo
```
- ✅ 两套 LoRA 均保留，可随时切换
- ✅ 推理：原始模型同时加载 SFT + GRPO 两个 LoRA

---

## 项目结构

```
RL_STU/
├── models/Qwen2.5-3B/              基础模型（所有训练的起点）
│
├── sft/                            监督微调
│   ├── train_finance_mac.py        SFT 训练（finance-alpaca 数据集）
│   ├── inference_finance.py        SFT 推理
│   └── finance-qwen-3b-lora-final/ SFT LoRA 权重
│
├── dpo/                            直接偏好优化
│   ├── generate_dpo_data.py        生成 chosen/rejected 数据
│   ├── train_dpo.py                并联 DPO（自生成数据）
│   ├── train_dpo_hh.py             并联 DPO（hh-rlhf 数据）
│   ├── train_dpo_merged.py         串联 DPO（SFT merge → DPO，自生成数据）
│   ├── train_dpo_hh_merged.py      串联 DPO（SFT merge → DPO，hh-rlhf 数据）
│   └── inference_dpo.py            三模型对比推理
│
├── gdpo/                           GRPO 强化学习
│   ├── train_grpo.py               并联 GRPO
│   ├── train_grpo_merged.py        串联 GRPO（SFT merge → GRPO）
│   └── inference_grpo.py           四模型对比推理
│
└── tools/
    ├── doc_to_sft.py               文档 → SFT 数据 自动生成工具
    ├── merge_sft_lora.py           合并 SFT LoRA → 完整独立模型
    ├── merge_dpo_lora.py           合并 DPO LoRA（并联/串联两种模式）
    └── merge_grpo_lora.py          合并 GRPO LoRA（并联/串联两种模式）
```

---

## 运行顺序

### 并联方案（学习用，可任意顺序）

```bash
# 1. SFT 训练（约 15 分钟）
.venv/bin/python sft/train_finance_mac.py

# 2. 生成 DPO 数据（约 30-40 分钟）
.venv/bin/python dpo/generate_dpo_data.py

# 3. DPO 训练（约 15 分钟）
.venv/bin/python dpo/train_dpo.py
# 或使用 hh-rlhf 数据
.venv/bin/python dpo/train_dpo_hh.py

# 4. GRPO 训练（约 30 分钟，比 DPO 慢因为要实时生成4个回答）
.venv/bin/python gdpo/train_grpo.py

# 5. 推理对比
.venv/bin/python gdpo/inference_grpo.py
```

### 串联方案（生产用，必须按顺序）

```bash
# 1. SFT 训练（必须第一步）
.venv/bin/python sft/train_finance_mac.py

# 2. 串联 DPO（依赖 SFT 结果）
.venv/bin/python dpo/generate_dpo_data.py
.venv/bin/python dpo/train_dpo_merged.py

# 3. 串联 GRPO（依赖 SFT 结果）
.venv/bin/python gdpo/train_grpo_merged.py
```

---

## LoRA 合并工具

训练完成后，LoRA 是以"插件"形式保存的（需要搭配基础模型才能推理）。
合并工具可以把 LoRA 权重"烧录"进基础模型，输出一个**完整独立模型**，部署和分发更方便。

### 三个合并脚本的区别

| 脚本 | 作用 | 合并基底 | 输出 |
|---|---|---|---|
| `merge_sft_lora.py` | 合并 SFT LoRA | 原始 Qwen2.5-3B | `sft/finance-qwen-3b-sft-merged/` |
| `merge_dpo_lora.py --mode parallel` | 合并并联 DPO LoRA | 原始 Qwen2.5-3B | `dpo/finance-qwen-3b-dpo-parallel-merged/` |
| `merge_dpo_lora.py --mode serial` | 合并串联 DPO LoRA | SFT 合并模型 | `dpo/finance-qwen-3b-dpo-serial-merged/` |
| `merge_grpo_lora.py --mode parallel` | 合并并联 GRPO LoRA | 原始 Qwen2.5-3B | `gdpo/finance-qwen-3b-grpo-parallel-merged/` |
| `merge_grpo_lora.py --mode serial` | 合并串联 GRPO LoRA | SFT 合并模型 | `gdpo/finance-qwen-3b-grpo-serial-merged/` |

> **为什么串联合并的基底是 SFT 合并模型？**
> 串联训练脚本（`train_dpo_merged.py` / `train_grpo_merged.py`）的出发点是"原始模型先 merge SFT LoRA 后"的模型，
> 所以训练出的 DPO/GRPO LoRA 必须叠加在同一个基底上，才能还原出完整效果。

### 合并脚本运行命令

```bash
# ── 合并 SFT LoRA（无参数，只有一种模式）──────────────────────────
.venv/bin/python tools/merge_sft_lora.py

# ── 合并 DPO LoRA ────────────────────────────────────────────────
# 并联 DPO：基底 = 原始模型
.venv/bin/python tools/merge_dpo_lora.py --mode parallel

# 串联 DPO：基底 = SFT合并模型（需先跑 merge_sft_lora.py）
.venv/bin/python tools/merge_dpo_lora.py --mode serial

# 串联 DPO（SFT合并模型不存在时自动先合并）
.venv/bin/python tools/merge_dpo_lora.py --mode serial --auto-merge-sft

# ── 合并 GRPO LoRA ───────────────────────────────────────────────
# 并联 GRPO：基底 = 原始模型
.venv/bin/python tools/merge_grpo_lora.py --mode parallel

# 串联 GRPO：基底 = SFT合并模型（需先跑 merge_sft_lora.py）
.venv/bin/python tools/merge_grpo_lora.py --mode serial

# 串联 GRPO（SFT合并模型不存在时自动先合并）
.venv/bin/python tools/merge_grpo_lora.py --mode serial --auto-merge-sft
```

### 合并前 vs 合并后的使用方式

```python
# ── 合并前：推理需要同时加载基础模型 + LoRA ──
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-3B")
model = PeftModel.from_pretrained(base, "new_models/finance-qwen-3b-lora-final")

# ── 合并后：直接加载单个目录即可 ──
model = AutoModelForCausalLM.from_pretrained("sft/finance-qwen-3b-sft-merged")
```

### 各 LoRA 所包含的知识

```
并联 SFT LoRA   = 原始模型 + 金融问答知识
并联 DPO LoRA   = 原始模型 + 偏好对齐风格
并联 GRPO LoRA  = 原始模型 + 强化学习行为优化

串联 DPO LoRA   = 原始模型 + 金融知识(SFT) + 偏好对齐(DPO)
串联 GRPO LoRA  = 原始模型 + 金融知识(SFT) + 强化学习优化(GRPO)
```

---

## 公司内部数据接入

将公司文档自动转换为 SFT 训练数据：

```bash
# 本地模式（免费，用已下载的 Qwen 生成问答）
.venv/bin/python tools/doc_to_sft.py --input 你的文档.txt --mode local

# API 模式（质量更高，需要 API Key）
.venv/bin/python tools/doc_to_sft.py --input 你的文档.txt --mode api --api-key sk-xxx
```

生成的 jsonl 文件替换 `sft/train_finance_mac.py` 中的数据集加载：

```python
dataset = load_dataset('json', data_files='你的文档_sft_data.jsonl', split='train')
```

---

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# .env 文件配置
HF_TOKEN=你的HuggingFace_Token
HF_ENDPOINT=https://hf-mirror.com    # 国内镜像
HF_HUB_DISABLE_XET=1                 # 禁用 XET 协议

# 下载基础模型
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  .venv/bin/hf download Qwen/Qwen2.5-3B --local-dir ./models/Qwen2.5-3B
