# Kev — 可自己训练的 Jev 风格决策模型

> Jared Palmer（Formik 作者）的开源项目：小型「System One」决策模型，一次前向传播输出类型化问题的校准概率，不生成任何文本。

- GitHub 仓库：<https://github.com/jaredpalmer/kev>
- HF 在线 Demo（本目录复刻的对象）：<https://huggingface.co/spaces/jaredpalmer/kev>
- 模型合集：<https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd>
- 冻结评测集：<https://huggingface.co/datasets/jaredpalmer/kev-suites>
- 架构解读（Kev 的理论基础）：<https://archerhume.com/posts/jevs-architecture-unmasked>
- TypeSafe System One API：<https://docs.typesafe.ai/api>
- License: Apache-2.0

## 1. Kev 是什么

Kev 是一组 **决策专用小模型**，架构思想来自 TypeSafe 的闭源模型 Jev（本项目 `jev/` 目录已体验过 DeBERTa 版复刻）：

```text
输入:  state（一份文档：工单/评论/文章/JSON） + 若干类型化问题
输出:  每个问题一个概率分布（一次 forward，零文本生成）
```

三种问题类型（与 Jev / TypeSafe `/v1/systemone` 契约完全一致）：

| 类型 | 含义 | 输出 |
|---|---|---|
| `choice` | 多选一（2~255 个无序命名选项） | 各选项概率 + argmax + confidence |
| `noul` | 是非题 | p(yes) |
| `score` | 有序等级打分（如 1~5 星） | 期望等级（可落在等级之间）+ 各等级概率 |

关键特性：**问题之间相互隔离** —— 所有问题共享同一份 state，但彼此不可见（防止一个问题的内容泄漏答案给另一个问题，Space 里的 "Isolation probe" 示例专门演示这一点）。

## 2. 架构

```text
Qwen3.5-Base（冻结）
   │
   ├── LoRA adapter（r=16，约 11M 可训练参数）     ← Kev 发布的就是这个
   │
   └── PointerHead 决策头（fp32，含校准温度 T）    ← 打分时用 <decide> token 的隐状态
                                                    去和各选项 span 的隐状态做匹配
```

- 基座是 **Qwen3.5**（混合架构：Gated DeltaNet + Attention），不是纯 encoder；
- 每个问题占一行「state + question 分支」的因果 token 行，分支间互相不可读；
- 发布物 = LoRA adapter + `head.pt`（决策头权重 + 校准温度），基座从 HF 另行下载；
- **校准**：每个 checkpoint 在自家开发集上拟合一个温度（Kev-4B T=2.14，Kev-0.8B T=2.41），除以温度后概率更可信（OOD ECE 0.12 → 0.04），argmax 不变。

## 3. 模型列表

| 模型 | 链接 | in-dist | out-of-domain | 备注 |
|---|---|---|---|---|
| kev-0.8b | <https://huggingface.co/jaredpalmer/kev-0.8b> | 0.825 | 0.652 | 内存紧张时用；OOD 弱 |
| kev-4b | <https://huggingface.co/jaredpalmer/kev-4b> | 0.872 | 0.797 | **推荐**，最热门 |
| kev-9b | <https://huggingface.co/jaredpalmer/kev-9b> | 0.872 | 0.822 | 精度最高 |
| （对照）open-jev-deberta | 见 `jev/` 目录 | — | 0.857 | 本项目已跑通的 DeBERTa 复刻 |

## 4. 本目录内容

```text
jev/kev/
├── README.md        # 本文档
├── app_local.py     # HF Space 的本地复刻（Gradio Web UI，功能完整版）
├── client_sdk.py    # TypeSafe SDK 客户端（GitHub README 同款简单用法）
└── kev-repo/        # 官方仓库克隆（git clone --depth 1）
    └── kev/         # 官方 Python 包：model / api / checkpoint / mlx_model / train ...
```

## 两种本地使用方式

**方式 A：简单 SDK 调用（推荐日常使用）** —— 先起服务，再用 `client_sdk.py` 打电话：

```bash
# 终端 1：启动决策服务（首次下载约 1.6GB；Mac 自动走 MLX 后端）
cd jev/kev/kev-repo
.venv/bin/python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009

# 终端 2：SDK 调用（同 GitHub README 的 Quick Start）
.venv/bin/python ../client_sdk.py
```

**方式 B：网页演示（复刻 HF Space）** —— `app_local.py`，带进度条渲染、
6 个预置示例、选项顺序稳定性检查、双模型对比：

`app_local.py` 是 [HF Space](https://huggingface.co/spaces/jaredpalmer/kev) 的本地版，去掉了 ZeroGPU 依赖，模型按需懒加载，
并通过 `LoadOptions(backend="auto")` 自动走 **MLX Metal 后端**（`kev/mlx_model.py`，Mac 上比 torch 路径快约 3 倍）：

```bash
cd jev/kev/kev-repo
uv sync --extra serve && uv pip install --python .venv/bin/python gradio   # 首次

.venv/bin/python ../app_local.py
# 浏览器打开 http://127.0.0.1:7860
```

注意：不要用 `uv run` 跑 `app_local.py`，它会按 lockfile 重新同步环境、
把 gradio 卸掉；直接用 `.venv/bin/python` 即可。

首次运行会从 HF 下载 adapter（小）+ Qwen3.5 基座（0.8B ≈ 1.6GB / 4B ≈ 8GB），已配置 hf-mirror 镜像。

UI 功能与 Space 一致：

- 预置 6 个示例（客服分诊 / 新闻分类 / 评论打分 / 退货政策 / 隔离探针 / 边界伪造）
- **Calibrated** 开关：应用 checkpoint 自带的校准温度
- **date_facts**：自动补充 state 中日期对的间隔天数（模型自己不会算日期减法）
- **Option-order check**：打乱选项顺序重跑，检验 argmax 是否翻转（顺序稳定性）
- **Both**：同一请求同时跑 Kev-4B 和 Kev-0.8B 对比
- 原始 `/v1/systemone` JSON 响应查看

### 也可以不起 UI，直接起本地 API 服务

```bash
cd jev/kev/kev-repo
.venv/bin/python -m kev.serve --run jaredpalmer/kev-4b --port 8009

curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size.",
  "model": "kev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Refunds and wrong items", "billing": "Charges and invoices"}}
  }}'
```

## 5. 与本仓库其他实验的关系

- `jev/`：Jev 生态调研 —— open-jev-deberta（纯 encoder + 决策头，手工 pipeline 已实现）
- `jev/kev/`：Kev —— Qwen3.5 基座 + LoRA + 决策头，**带完整训练代码**，是未来自训决策模型的首选起点
- 训练入口：`kev-repo/kev/train.py`，支持 `--init_from` 在已发布 checkpoint 上做 delta 微调（小数据 + 回放即可提升 OOD，作者自己就用 1,425 条数据 +9 分钟训练拿到 +2.2pp）

## 6. 已知限制（来自官方模型卡）

- **Mac 上 torch 路径慢**：Qwen3.5 的 DeltaNet 内核没有 MPS 实现（4B 约 0.8s/请求），务必用 MLX 后端；
- kev-0.8b 的 OOD 能力有限（MMLU 0.41、PAWS 0.59），追求精度用 kev-4b/9b；
- 原始概率 OOD 校准不完美（Kev-4B raw ECE 0.12），生产使用先在自己的数据上测量。

