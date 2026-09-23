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

#### 串联全链路详解（以 Qwen2.5-3B 为例）

```
① SFT 训练（sft/train_finance_mac.py）
   原始模型 models/Qwen2.5-3B（冻结）
        + LoRA（学习金融知识）
   → 产出 ② new_models/Qwen2.5-3B-sft-lora-final   （LoRA 插件，仅 ~14MB）
   → 脚本末尾自动 merge → ③ merge_models/Qwen2.5-3B-sft-merged
                              （带领域知识的完整模型，~6GB）

④ DPO/GRPO 后训练（train_dpo_merged.py / train_grpo_merged.py）
   基底 = ③ SFT合并模型（知识已烧进权重，不再是插件）
        + 一个全新的 LoRA（学习偏好 / 强化信号）
   → 产出 new_models/{模型}-dpo-merged-final / -grpo-merged-final

⑤ 再 merge（tools/merge_dpo_lora.py --mode serial）
   ③ SFT合并模型 + ④ 新 LoRA 烧录
   → merge_models/Qwen2.5-3B-dpo-serial-merged  （最终成品，可部署/Ollama）
```

链路要点：

- ③ 的 merge 让 SFT 知识**永久写进权重**，④ 的训练梯度碰不到它（防灾难性遗忘）
- ④ 新 LoRA 的 B 矩阵从零初始化，训练起点行为恰好等于 SFT 模型，无缝衔接
- **只要保留 `models/`（原始模型）和 `new_models/`（全部 LoRA），`merge_models/` 里的任何成品都可以随时重新生成**（这也是磁盘清理的依据，见文末指南）

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

### 挂载 LoRA vs 提前 merge：有什么区别？

**从推理结果看：两者数学等价**

```python
# 挂载方式（运行时）
output = x @ W  +  x @ (B @ A)      # 原始权重 + LoRA 增量，两路计算
# merge 后
output = x @ (W + B @ A)            # 增量已烧进权重，一路计算
```

两者输出的 logits 理论上相同（仅有 bf16 舍入顺序带来的极微差异）。**如果只是推理，挂载和 merge 没有区别。**

**从后续训练看：真正的分水岭**

| | 挂载旧 LoRA 继续训练（方式一） | 先 merge 再挂新 LoRA（方式二，本项目串联） |
|---|---|---|
| 梯度流向 | 直接更新旧 A、B 矩阵 | 只更新全新 LoRA，烧录的权重冻结 |
| SFT 知识 | 可能被 DPO/GRPO 梯度覆盖（灾难性遗忘） | 物理隔离，不可能被遗忘 |
| 资产形态 | 一个 LoRA，知识+对齐混在一起 | 两个独立 LoRA，可分别分析/替换/组合 |
| DPO ref model | 处理很绕（ref 要不要挂同一 LoRA？） | 基底即 SFT 模型，ref 干净利落 |

**从工程上看：各有取舍**

| 维度 | 挂载（base + LoRA） | 提前 merge |
|---|---|---|
| 磁盘占用 | 几十 MB（只有 A、B） | 完整模型 ~6GB |
| 依赖 | 需要 peft，两步加载 | 任何框架直接加载（vLLM/Ollama） |
| 推理速度 | 每层多两次小矩阵乘 | 单次矩阵乘，略快 |
| 可逆性 | ✅ 卸载插件即回到原始模型 | ❌ 单向操作，需保留原始模型备份 |

> 一句话：**merge 不改变模型的行为，改变的是知识的存储形态**——从"可被梯度修改的插件"变成"冻结在权重里的固定资产"。串联方案要的正是这个效果；代价是每个阶段多存一份完整模型。

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
model = PeftModel.from_pretrained(base, "new_models/Qwen2.5-3B-sft-lora-final")

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
```

---

## 配置参数说明

### `.env` 环境变量

| 变量 | 示例值 | 说明 |
|------|--------|------|
| `HF_TOKEN` | `hf_xxx...` | HuggingFace 访问令牌，下载模型/数据集时需要 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 镜像站地址，国内访问 HF 必须设置 |
| `HF_HUB_DISABLE_XET` | `1` | 禁用 XET 分块协议，防止连接 `cas-bridge.xethub.hf.co` 超时 |
| `SELECT_MODEL` | `Qwen2.5-3B` | **选择训练使用的基础模型**，对应 `models/` 目录下的子目录名 |

#### `SELECT_MODEL` 可选值

```
SELECT_MODEL=Qwen2.5-3B     # 纯文本模型（默认，训练速度快）
SELECT_MODEL=gemma-4-E2B    # 多模态模型（含视觉+音频编码器，参数更多）
```

切换后，所有训练脚本自动加载对应模型，LoRA 输出目录也会携带模型名加以区分，例如：
- `new_models/Qwen2.5-3B-sft-lora-final/`
- `new_models/gemma-4-E2B-sft-lora-final/`

---

### LoRA 配置参数（`LoraConfig`）

所有训练脚本使用统一的 LoRA 配置，下面逐个说明含义：

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

#### `r` — 秩（Rank）

LoRA 的核心思想是将大矩阵的更新分解为两个小矩阵相乘：

```
原始权重 W (4096×4096)  ← 冻结，不动
LoRA 更新 = A (4096×r) × B (r×4096)
```

`r` 就是中间维度的大小，**决定 LoRA 能表达多少信息**：

| r 值 | 参数量 | 建议场景 |
|------|--------|---------|
| 4 | 最少 | 内存极紧、只需微调口吻 |
| **8** | 少（本项目默认） | 一般微调，效果与效率均衡 |
| 16 | 中等 | 需要更强的任务适应能力 |
| 32+ | 较多 | 大幅改变模型知识/行为风格 |

> 对于 24GB M4 Mac，`r=8` 是安全选择；若效果不足，可提升到 `r=16`。

#### `lora_alpha` — 缩放系数

LoRA 更新生效时会乘以一个缩放因子：

```
实际更新量 = (lora_alpha / r) × A × B
```

当 `lora_alpha=16, r=8` 时，缩放因子 = **2.0**，即 LoRA 对原模型的影响力翻倍。

> **常见约定**：`lora_alpha = 2 × r`（如 r=8 → alpha=16），这是最稳定的默认配置。
> 增大 alpha 相当于提高 LoRA 的"学习强度"。

#### `target_modules` — 注入哪些层

指定把 LoRA 插件挂载到哪些线性层。不同模型架构的可选模块：

| 模块名 | 位置 | 说明 |
|--------|------|------|
| `q_proj` | 注意力层 Query 投影 | 控制"关注什么" |
| `k_proj` | 注意力层 Key 投影 | 配合 Q 计算注意力分数 |
| `v_proj` | 注意力层 Value 投影 | 控制"提取什么信息" |
| `o_proj` | 注意力层输出投影 | 整合多头注意力结果 |
| `gate_proj` / `up_proj` / `down_proj` | FFN 前馈网络 | 控制知识存储 |

**本项目默认配置**：
- 纯文本模型（Qwen 等）：`["q_proj", "v_proj"]`，最小有效集合，省内存
- gemma-4-E2B（多模态）：正则 `model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj`，精确匹配文本解码器，跳过视觉/音频塔

> 模块越多效果越好，但训练时间和内存也线性增加。推荐从 `q_proj + v_proj` 起步。

#### `lora_dropout` — Dropout 正则化

训练时随机关闭 `lora_dropout` 比例的 LoRA 权重连接，防止过拟合。

| 值 | 适用场景 |
|----|---------|
| `0` | 数据量大（>10万条），不需要正则 |
| `0.05` | 数据量小（本项目 2000 条），轻度正则 |
| `0.1` | 数据量极少，重度正则 |

#### `bias` — 偏置项处理

| 值 | 说明 | 推荐 |
|----|------|------|
| `"none"` | 偏置项完全不训练，最省内存 | ✅ 默认 |
| `"all"` | 所有层的偏置都参与训练 | 内存允许时可尝试 |
| `"lora_only"` | 只训练有 LoRA 的层的偏置 | 折中选项 |

#### `task_type` — 任务类型

告诉 PEFT 如何包装模型：

| 值 | 对应模型类型 |
|----|-------------|
| `CAUSAL_LM` | 自回归语言模型，如 GPT/Qwen/Gemma（**本项目使用**） |
| `SEQ_2_SEQ_LM` | 编码器-解码器，如 T5/BART |
| `SEQ_CLS` | 文本分类 |
| `TOKEN_CLS` | Token 级分类（NER 等） |

---

### 训练超参数

各训练脚本中通用的训练配置说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `per_device_train_batch_size` | `1` | 每步处理 1 条数据，防止 24GB 内存溢出 |
| `gradient_accumulation_steps` | `4` | 攒 4 步再更新参数，等效 batch_size=4 |
| `learning_rate` | SFT:`2e-4` / DPO:`5e-5` / GRPO:`1e-5` | LoRA 学习率，后续阶段要逐步减小 |
| `max_steps` | `200` | 训练总步数，演示用；生产建议 500~2000 |
| `save_steps` | `100` | 每 100 步保存一次 checkpoint（用于断点续训） |
| `max_length` | `512` | 单条数据最大 token 数，超过截断，防内存溢出 |
| `optim` | `adamw_torch` | 优化器，AdamW 是 LLM 微调标准选择 |
| `report_to` | `"none"` | 关闭 wandb/tensorboard 等第三方日志 |

#### DPO 专属参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | `0.1` | 偏好强度系数，越大越保守（变化越小），越小越激进 |

#### GRPO 专属参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_completion_length` | `200` | 每次生成回答的最大长度，控制内存 |
| `num_generations` | `4` | 每个 prompt 生成几个回答用于比较，越多越稳定但越慢 |

---

### 断点续训

所有训练脚本均支持断点续训，**中断后直接重新运行同一命令**即可：

```bash
# 第一次运行，训练到 step 100 时中断
.venv/bin/python sft/train_finance_mac.py

# 重新运行，自动从最新 checkpoint 继续
.venv/bin/python sft/train_finance_mac.py
# 输出：⏩ 发现断点，从 .../checkpoint-100 继续训练
```

训练完成后，脚本会自动清理中间 checkpoint（只保留 `FINAL_OUTPUT` 中的最终 LoRA 权重）。

---

## 目录作用与磁盘清理指南

### 三层模型存储结构

| 目录 | 存的是什么 | 角色定位 |
|---|---|---|
| `models/` | HuggingFace 下载的**原始预训练模型**（Qwen2.5-3B、gemma-4-E2B-it 等） | 原材料，训练起点，只读不写 |
| `new_models/` | 训练产出的 **LoRA 插件**（`adapter_model.safetensors`，每个仅几十 MB） | 半成品，需挂载到基底模型才能使用 |
| `new_models/checkpoints/` | 训练中间 checkpoint，用于断点续训 | 临时产物，训练完自动清理 |
| `merge_models/` | `merge_and_unload()` 后的**完整独立模型** | 成品，可直接部署；同时是串联训练的基底 |
| `jev/models/` | open-jev-deberta 决策模型（Jev 调研用） | 独立实验资产 |
| `.venv/` | Python 虚拟环境 | 运行环境 |

数据流：`models/`（冻结基底）→ 训练 → `new_models/`（LoRA 插件）→ merge → `merge_models/`（完整模型）→ 作为下一轮串联训练的基底 → 循环累积。

### 当前磁盘占用明细（总计约 55GB，2026-09 统计）

```
models/Qwen2.5-3B                              5.8G   ❌ 保留（一切训练的基底）
models/gemma-4-E2B-it                           9.6G   ❌ 保留（训练基底）
new_models/（全部 LoRA 插件）                   ~0.1G  ❌ 保留（核心资产，删了只能重训）
new_models/checkpoints/                          0B    ✅ 空的（脚本已自动清理）
merge_models/Qwen2.5-3B-sft-merged              5.8G   ⚠️ 可删（可随时重建）
merge_models/Qwen2.5-3B-dpo-serial-merged      12G     ⚠️ 可删（safetensors 5.7G + gguf 5.8G）
merge_models/gemma-4-E2B-it-sft-merged          9.5G   ⚠️ 可删（可随时重建）
merge_models/gemma-4-E2B-it-dpo-serial-merged   9.5G   ⚠️ 可删（gemma 最终成品，不用可删）
jev/models/open-jev-deberta-v3-large            1.6G   看需求（Jev 实验还在用就留）
.venv/                                          1G     ❌ 保留
```

### 清理建议（按优先级，删前请确认对应 LoRA 都在 new_models/ 下）

1. **`merge_models/gemma-4-E2B-it-sft-merged`（9.5G）**——串联训练的中间基底，当前没有正在进行的 gemma 串联训练即可删
2. **`merge_models/Qwen2.5-3B-dpo-serial-merged/qwen2.5-3b-dpo.gguf`（5.8G）**——`ollama create` 导入时已复制进 Ollama 自己的存储（`~/.ollama`），删除源 gguf 不影响已导入的 `qwen2.5-3b-dpo-finance` 模型
3. **`merge_models/gemma-4-E2B-it-dpo-serial-merged`（9.5G）**——gemma 的 DPO 最终成品，暂时不用 gemma 推理可删
4. **`merge_models/Qwen2.5-3B-dpo-serial-merged/` 的 safetensors 部分（5.7G）**——如果只通过 Ollama 的 gguf 版使用，safetensors 也可删（但今后重新转换 gguf 需要先重建它）

> 以上 2~4 项全删可释放约 **30G**；加上第 1 项共约 **39G**。

### 删除后如何找回

只要 `models/`（原始模型）和 `new_models/`（LoRA 插件）还在，`merge_models/` 里的任何成品都能一条命令重建：

```bash
# 重建 SFT 合并模型
.venv/bin/python tools/merge_sft_lora.py

# 重建串联 DPO 成品（sft-merged 不存在时自动先合并）
.venv/bin/python tools/merge_dpo_lora.py --mode serial --auto-merge-sft

# 重建串联 GRPO 成品
.venv/bin/python tools/merge_grpo_lora.py --mode serial --auto-merge-sft
```

> ⚠️ 唯一不可再生的是 `new_models/` 下的 LoRA 权重和 `models/` 下的原始模型——**这两个目录无论多缺空间都不要删**。
> gguf 的重建需要先用上面的命令恢复 safetensors，再按 `change_ollama.md` 的流程重新转换。
