# Jev：System One 决策模型 介绍与本地实践

> 本目录用于跟踪和实践 **Jev / System One Model** 这一新模型范式。
> 当前状态：调研完成，模型选型已定，训练计划待启动。

---

## 一、Jev 是什么

**Jev** 是 TypeSafe AI（创始人 Diogo Almeida，前 OpenAI 员工，ChatGPT 指令跟随研究的参与者之一）发布的首个 **System One Model（系统一模型）**。

- **System One** 出自卡尼曼《思考，快与慢》：快、直觉、无需深思的判断
- **Jev** 纪念经济学家 William Stanley Jevons（杰文斯悖论：智能成本每降一个数量级，就解锁一个数量级的新用例）

### 与传统 LLM 的根本区别

| 维度 | 传统 LLM（Chat / System Two） | Jev（System One） |
|---|---|---|
| 优化目标 | 人类偏好 RLHF / 可验证奖励 RLVR | **校准决策 RLCD**（Reinforcement Learning for Calibrated Decisions） |
| 输出 | 自由生成的字符串（可能是回答，也可能是幻觉） | **类型安全的结构化值**：noul（是/否）、choice（选择）、score（评分），附带校准概率 |
| 采样方式 | 逐 token 串行生成 | **并行**：一次查询出全部概率，无输出 token |
| 速度 | 3 ~ 329 秒 | **70 ~ 500 毫秒** |
| 价格 | 输入 $0.2~10/M token，输出约 5 倍贵 | 输入 $0.042/M token，**输出免费** |
| 幻觉 | 无法根除 | **结构性不可能**（输出空间被预定义约束，类型错误数学上不成立） |
| 置信度 | 普遍过度自信 | 每个决策自带校准概率，"高置信 = 高准确" |

官方宣称在 System One 型工作流任务上比 LLM **快 193.6 倍、便宜 444.6 倍**。

### 三种类型化决策（typed decision）

```
noul   →  问题是否成立          →  p(yes)
choice →  从 N 个选项中选一个    →  每个选项的概率（支持高基数，最多 255+）
score  →  在有序等级上评分       →  各等级分布 + 期望值
```

---

## 二、Jev 与 LLM 的组合生态

核心分工：**LLM 负责生成（慢思考），Jev 负责判断（快直觉）**。

```
                        ┌─────────────────────────────┐
 用户/事件/数据 ──────→ │  Jev（决策层，毫秒级）         │
                        │  分类 · 路由 · 打分 · 门控     │
                        └──────┬───────────┬──────────┘
                     高置信·简单  │           │ 低置信·复杂
                               ↓           ↓
                        直接执行分支    ┌─────────────────────┐
                        （不调用LLM）    │  LLM（生成层，秒级）   │
                                       │  写作 · 推理 · 代码    │
                                       └──────────┬──────────┘
                                                  ↓
                                       LLM 输出 → 再由 Jev 校验/守门
```

### 六种典型组合模式

| 模式 | Jev 做什么 | LLM 做什么 |
|---|---|---|
| **智能路由** | 判断用户问题属于哪类（客服意图/工单分类），毫秒级分流 | 只处理真正需要生成的部分 |
| **Agent 决策环** | 每步从 N 个工具/动作中选一个（官方 Doom demo：每秒 10 次决策，~$7/小时） | 执行被选中的工具、生成内容 |
| **高基数选择** | Wikiracing：上千个链接里挑下一步，零幻觉 | 做不了（生成编号会出错） |
| **守门/校验** | 给 LLM 输出打分、检测越狱/幻觉，低置信转人工 | 生成内容 |
| **海量批处理** | map-reduce 亿级数据打标签（$42/B tokens） | 只抽小样本精处理 |
| **成本止损线** | "置信度 > 0.9 自动执行，< 0.9 升级人工" | — |

### 为什么是互补而非替代

- Jev **放弃了字符串生成**：写不了文章、编不了代码，只会在给定选项空间里输出校准概率
- LLM 虽全能，但串行生成导致**慢、贵、过度自信**，无法承诺延迟和类型安全
- 类比：CPU（逻辑判断快）与 GPU（大规模计算）的分工

---

## 三、本地可跑的 Jev 复刻模型（HuggingFace 调研）

Jev 本体闭源（仅 API），但发布后社区涌现出一批开源复刻。以下为筛选后**适合 M4 Mac 本地跑 demo** 的模型：

### 3.1 ⭐ Jev-Style-Qwen3.5-2B-Decision（首推，有 Apple Silicon 原生版）

- 仓库：`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16`（MLX 版）
  / `chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF`（GGUF 版）
- 底座：Qwen3.5-2B + LoRA r16 + 单 token logprobs 决策（温度已折叠进 RMSNorm，推理零后处理）
- **M1 Max 实测 77ms/次**，3.9GB（bf16）/ 1.3GB（Q4）
- 效果：5 个决策任务准确率 82.3%（base 零样本 65.9%），**ECE 0.017 接近完美校准**
- 支持 choice / bool / score 三种类型
- GGUF 版可直接跑 **llama.cpp / LM Studio**，与本项目 Ollama 工作流衔接

```bash
# llama.cpp 方式
llama-server -hf chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF:Q8_0 --port 8080
python jev_style_client.py --url http://localhost:8080

# 注意：必须用它规定的 prompt 格式（以 "Answer:" 结尾），
# 普通 chat 消息会退化为无意义的文本续写
```

```python
# MLX 版（Apple Silicon 原生）
# decide(state, question, options) → [("Business", 0.68), ("Science/Technology", 0.31), ...]
```

### 3.2 ⭐ open-jev-deberta-v3-large（纯 CPU 可跑，接口最完整）

- 仓库：`com-kotobalabs/open-jev-deberta-v3-large`（0.4B，社区最火，1.3k 下载）
- 底座：DeBERTa-v3-large 编码器 + span 打分头，**一次 forward 回答任意多个问题**
- **M1 Max CPU fp32：1.8 秒回答 4 个问题**；H100 上 518 questions/s
- 完整实现 choice(≤255 选项) / noul / score 三种类型
- 训练只用公开金标数据（banking77 / sst5 / boolq），ECE 0.022
- 代码：`github.com/kotoba-lang/typed-decisions`

```text
# pip install git+https://github.com/kotoba-lang/typed-decisions
from typed_decisions.open_jev import OpenJev

m = OpenJev.from_pretrained("com-kotobalabs/open-jev-deberta-v3-large")
m.decide(
    "I was charged twice for the same order and nobody answers my emails.",
    [
        {"type": "choice", "instructions": "Which product area?",
         "options": ["fees & charges", "refund & dispute", "card", ...]},
        {"type": "score",  "instructions": "How positive is the sentiment?",
         "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
        {"type": "noul",   "instructions": "The customer is asking for a refund."},
    ])
# → [{'choice': 'fees & charges', 'confidence': 0.733}, {'score': 0.846}, {'noul': 0.892}]
```

局限：英文 only、512 token 上下文、OOD（没见过的问题类型）准确率降至 0.69。

### 3.3 mini-Jev（最小巧，Agent 工具选择专用）

- 仓库：`samatv256/mini-Jev`（决策头仅 **1.1MB** + 冻结 Qwen3-0.6B，共约 0.6B）
- 架构：冻结 Qwen3-0.6B → 候选描述 mean pooling → Linear(1024,256) → GELU → Linear(256,1) → grouped softmax
- 场景：输入 agent 状态 + 候选动作列表，直接排序选动作
- 合成数据评测：语义选择 72.97%，GH200 上 76-85ms（3-16 个候选）
- 适合学习"冻结主干 + 小决策头"的最小实现

### 3.4 其他（备选/参考）

| 模型 | 规模 | 说明 |
|---|---|---|
| `akhilaaa3/Jev-Omni` | 12B | Gemma 4 12B 多模态版（文本/图/音/视频），FP32 50GB 需 CUDA，**Mac 跑不动**，作参考实现 |
| `ZefanCai/Open-Jev-2B` / `Open-Jev-9B` | 2B/9B | Qwen3.5 + LoRA，测试集 94.7% 准确率；但依赖专用 CUDA loader（Transformers 5.10.2），MPS 不友好 |
| `mogita/jev-decider-qwen3-4b` | 4B | Qwen3-4B 决策版 |
| `lostargon/Tiny-Jev` | 0.6B | 轻量版 |
| `Meanblock/JEV-CPU` | — | 零样本分类 CPU 版 |

### 选型结论

| 需求 | 推荐 |
|---|---|
| 在 Mac 上体验毫秒级 Jev 风格决策 | **Jev-Style-Qwen3.5-2B-Decision-MLX-bf16** |
| 不想占 GPU、快速验证接口 | **open-jev-deberta-v3-large**（纯 CPU） |
| 学习最小实现 / 接入 agent 工具选择 | **mini-Jev** |
| 研究多模态决策的参考架构 | Jev-Omni（读代码，不跑） |

---

## 四、与本项目（RL_STU）的衔接计划

本项目已有资产可直接复用于自训练一个 Jev 风格决策模型（详见根目录讨论）：

```
jev/
├── README.md            # 本文档
├── build_jev_data.py    # 计划：finance-alpaca → 选择题（用 SFT 模型生成干扰项）
├── train_jev.py         # 计划：冻结 gemma-4-E2B-it + LoRA + 分类头，CE loss 自定义训练循环
└── inference_jev.py     # 计划：单次 forward 出概率，对比 base 模型 few-shot
```

复现要点（参考 Jev-Omni 的 `decision_config.json`）：

1. **架构**：主干取最后一个 token 的 hidden state → 标准化(mu/sd) → Linear(hidden → N) → 组内 mask + softmax
2. **训练**：LoRA r=8~16 + 分类头联合训练，交叉熵；hidden 尺度漂移需统计 mu/sd 或用 LayerNorm
3. **数据**：2~5 千条 (state, question, options, 正确index)，干扰项可用已训练的 SFT 模型批量生成（复用 DPO 数据生成套路）
4. **内存**：E2B bf16 ~4-5GB + LoRA + head，24GB 统一内存富余，MPS 可训

