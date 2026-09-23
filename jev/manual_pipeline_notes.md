# OpenJev 手动推理流水线：操作记录与技术说明

> 本文记录 `jev/manual_pipeline.py` 的来龙去脉：为什么要写它、六个步骤分别做什么、
> 以及如何验证手动实现的正确性。适合想彻底搞懂"决策模型内部发生了什么"的读者。
>
> 关联文档：本目录 `README.md`（Jev 生态与模型调研）、`demo_open_jev.py`（官方封装的体验 demo）。

---

## 一、背景：这个脚本回答了什么问题

在体验 `demo_open_jev.py`（官方 `OpenJev` 封装）时产生了一个疑问：

> "transformers 加载模型返回的数据，是不是基本就是 `decide()` 返回的那个 JSON 串，
> 只不过需要转换处理一下？"

**答案是：不是。** transformers 原生只能加载主干（DeBERTa 编码器），返回的是
`(L, 1024)` 的语义向量张量——模型已"读懂"文本，但没有任何判断和概率。
从向量到决策 JSON 之间还隔着三段实质计算：**池化、决策头打分、softmax 概率化**。

`OpenJev` 类本质上就是这六步的封装。为了证明"没有隐藏魔法"，`manual_pipeline.py`
**完全不用 `OpenJev` 类**，只用 transformers 裸加载 + 手写每一步，最后与官方封装
的结果逐位对比验证。

另一个相关实验：`pipeline("text-classification", model=...)` 对这个模型**不可用**——
pipeline 发现权重里没有标准分类头，会随机初始化一个（LOAD REPORT 打印 MISSING 警告），
输出无意义的 `LABEL_0 / 0.5487`。教训：**非标模型（主干+外挂头）必须用作者提供的加载器**。

## 二、准备工作：模型的三件套

`jev/models/open-jev-deberta-v3-large/` 目录下：

| 文件 | 作用 | 谁来加载 |
|------|------|---------|
| `model.safetensors` (1.7GB) | DeBERTa 主干权重（434M 参数） | transformers `AutoModel` ✅ |
| `head.safetensors` (几MB) | 决策头权重（3 层 MLP） | ❌ transformers 不认识，需手动 `load_file` |
| `open_jev_config.json` | 推理配方：温度 1.05、pool 方式、截断限制 | 手动读取 |
| `tokenizer.json` / `spm.model` | 分词器（含 `[STATE]`/`[Q]`/`[OPT]` 三个自造标记） | transformers `AutoTokenizer` ✅ |
| `typed_decisions/` | 官方封装代码（`OpenJev`/`encoder`/`schema`） | `sys.path.insert` 后 importlib 导入 |

加载方式（脚本“准备阶段”）：

```text
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
backbone = AutoModel.from_pretrained(MODEL_DIR, attn_implementation="eager").to(device).eval()
head_weights = load_file(MODEL_DIR + "/head.safetensors")   # 手动加载决策头
cfg = json.load(open(MODEL_DIR + "/open_jev_config.json"))  # 读温度等配置
```

注意 `attn_implementation="eager"` 来自配置文件的显式要求，不是随意选择。

## 三、六个步骤详解（脚本的主体）

### 步骤 1：拼接输入格式 + 分词

原本由 `OpenJev` 内部的 `Collator` 完成。把结构化输入拼成模型认识的单一序列：

```
[CLS][STATE] 投诉内容 [Q] 问题文本 [OPT] 选项1文本 [OPT] 选项2文本 ... [SEP]
```

```text
ids = [tok.cls_token_id, M_STATE] + encode(state)[:256]   # state 截断到 256 token
# 每遇到一个问题追加 [Q]+问题文本；每个选项追加 [OPT]+选项文本
# 同时记录 q_spans / opt_spans：每段文本 token 的起止位置（步骤 3 要用）
```

两个易错点：
- `state` 超长时**静默截断到前 256 token**（超限内容模型根本看不到，不报错）
- 必须记录每段文本的**位置范围**，否则步骤 3 不知道对哪些 token 做均值

### 步骤 2：主干 forward —— transformers 的能力边界

```text
with torch.no_grad():
    out = backbone(input_ids=input_ids, attention_mask=attn)

hidden = out.last_hidden_state[0]     # (62, 1024)：62 个 token，每个一个 1024 维向量
```

**这就是 transformers 原生返回的全部内容**——语义向量矩阵，无判断、无概率。
实测打印：`(62, 1024)`，数值形如 `[-0.5009, -0.1219, 0.2306, ...]`。

### 步骤 3：span 均值池化

把"每个 token 一个向量"压缩成"每个问题/选项一个向量"：

```text
def mean_pool(start, end):
    return hidden[start:end].mean(dim=0)    # 该文本段所有 token 向量取平均 → (1024,)
```

- 池化前：62 个 token 向量
- 池化后：7 个选项向量（2 个 noul 选项 + 5 个 choice 选项）+ 2 个问题向量

> 为什么用"选项文本 token 的均值"而不是 `[OPT]` 标记本身？仓库作者做过消融实验：
> 给新造的 `[OPT]` 标记 token 打分**学不动**（始终停在标签先验），因为新 token 的
> 注意力模式没有任何预训练结构可用；而选项**文本**的向量是预训练就有的"成熟语义"。
> 这就是配置里 `pool: "span"` 的含义。

### 步骤 4：决策头打分 —— head.safetensors 的 forward

决策头结构（`DecisionEncoder.head`，一个 `nn.Sequential`，权重键名 `0.*` 和 `2.*`）：

```
输入：cat[问题向量, 选项向量, 二者逐元素乘]   (3072,)  —— 问题、选项、交互三个视角
  → Linear(3072 → 1024)  (权重 0.weight / 0.bias)
  → GELU
  → Linear(1024 → 1)     (权重 2.weight / 2.bias)
  → 一个标量 logit（该选项的未归一化分数）
```

手动实现（绕过 nn.Module，直接用矩阵乘）：

```text
g = torch.cat([q_vecs[qi], ov, q_vecs[qi] * ov])       # (3072,)
h = torch.nn.functional.gelu(g @ W1.T + b1)            # 隐层 (1024,)
logit = (h @ W2.T + b2).item()
```

实测输出（客服投诉样例）：
- 问题1（noul 是否要求退款）：logits `[-2.38, -0.464]`
- 问题2（choice 业务归类）：logits `[1.73, -0.244, 0.863, -2.637, -1.93]`

### 步骤 5：softmax 概率化 + 温度校准

```text
probs = (logits / TEMPERATURE).softmax(dim=0)   # TEMPERATURE = 1.05（来自配置文件）
```

- logit 只是相对分数，softmax 归一化成概率分布
- 除以温度 1.05 让过度自信的分布稍微"降温"——这个温度是作者在验证集上**拟合出来**的
  后验校准参数（训练用 CE + Brier loss 内建校准压力，温度是最后微调）

实测概率：`no:0.139 yes:0.861`、`fees & charges:0.611 ...`

### 步骤 6：readout 格式化

按问题类型组装结果（与 `schema.py` 的 `readout()` 逻辑一致）：

- `noul` → `{"noul": p_yes}`（取 yes 的概率）
- `choice` → `{"choice": 最高概率选项, "probabilities": 全分布, "confidence": 最高概率}`
- `score` → `{"score": Σ(i×p_i) 期望等级, ...}`（可落在等级之间，如 0.845）

## 四、验证：手动流水线 vs 官方封装

脚本末尾加载 `OpenJev`（importlib 动态导入，因代码包在模型目录内、未 pip install），
用**完全相同的输入**跑一遍，逐位对比：

```
手动流水线：  noul = 0.8611112236976624
官方封装：    noul = 0.8611112236976624   ← 逐位一致 ✅
choice      = 'fees & charges'，probabilities 各项小数点后 4 位一致 ✅
```

结论：**`OpenJev.decide()` = 步骤 1~6 的封装，没有任何隐藏魔法。**

## 五、操作过程踩坑记录（复现时注意）

| 坑 | 现象 | 修复 |
|----|------|------|
| 设备不一致 | `RuntimeError: Tensor for argument #2 'mat2' is on CPU, but expected it to be on GPU` | 决策头权重 `load_file` 后停留在 CPU，需 `.to(device)` 与主干对齐 |
| pipeline 误用 | 返回 `LABEL_0 / 0.5487` 无意义结果，LOAD REPORT 打印 classifier 权重 MISSING | 该模型是"主干+外挂头"非标结构，标准 pipeline 会随机初始化缺失头；必须走本脚本的中层路线 |
| 静默截断 | state 很长时模型只看前 256 token，无任何提示 | 调用方自行做摘要/截断，不依赖内部截断（详见 `README.md` 长输入评估节） |
| IDE 误报 | `Unresolved reference 'typed_decisions'` | 代码包在运行时 `sys.path` 注入，静态分析看不到；用 `importlib.import_module(...)` 动态导入 + 必要处 `# type: ignore` |

## 六、运行方式

```bash
.venv/bin/python jev/manual_pipeline.py
```

输出全程带中文注释：拼接后的输入原文 → 向量形状 → 池化前后对比 → 原始 logits →
概率 → 最终 JSON → 与官方封装的对比验证。适合作为"决策模型原理"的教学演示。

## 七、一句话总结

> transformers 给你的是"考官读完卷子后的脑内理解"（向量）；
> 决策头是"考官手里的评分表"（外挂权重）；
> `decide()` 返回的 JSON 是"录用通知单"（格式化结果）。
> 从理解到通知单，隔着池化、打分、概率化三道工序——`OpenJev` 类就是把这三道工序
> 连同前后的拼接与格式化，共六步封装成的一个函数。**没有魔法，只有流水线。**

---

## 参考路径

- 手动流水线：`jev/manual_pipeline.py`
- 官方封装源码：`jev/models/open-jev-deberta-v3-large/typed_decisions/open_jev.py`（含 `Collator`/`DecisionEncoder`）
- 训练管线（同仓库的 `encoder.py`/`schema.py`）：`jev/typed-decisions/src/typed_decisions/`
- 概念延伸（主干与头范式）：`/Users/liulei/aiS/transformerS/backbone_and_head.md`

