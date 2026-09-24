"""Kev 简单用法：TypeSafe SDK 客户端（对应 GitHub README 的 Quick Start）
======================================================================
这是"打电话"的那一侧：把请求发给一个已经跑起来的 kev.serve 服务。
（app_local.py 是复刻 HF Space 的完整网页演示，属于另一个定位。）

使用步骤（两个终端）：

终端 1 —— 启动本地决策服务（首次会下载 adapter + Qwen3.5 基座，约 1.6GB）：
  cd jev/kev/kev-repo
  uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
  # Mac 上 backend="auto" 自动走 MLX Metal；换 4B 把参数改成 jaredpalmer/kev-4b

终端 2 —— 跑本脚本：
  cd jev/kev/kev-repo
  uv run --extra serve python ../client_sdk.py
"""

import os

# 国内镜像（服务端下载模型时用；客户端本身不下载，但保持一致无害）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import importlib  # typesafe_sdk 装在 kev-repo 的 uv venv 里，运行时导入（静态分析无法解析）

typesafe_sdk = importlib.import_module("typesafe_sdk")
Choice = typesafe_sdk.Choice
Noul = typesafe_sdk.Noul
Score = typesafe_sdk.Score
TypeSafeClient = typesafe_sdk.TypeSafeClient

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8009",
    model="kev-latest",
)

# 一次请求：state + 三个不同类型的问题，一次前向全部出结果
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    },
)

print("billing (noul)  p(yes) =", response.nouls["billing"].noul)
print("tone   (choice)  picked =", response.choices["tone"].choice)
print("urgency (score)  level  =", response.scores["urgency"].score)

# 再来一个中文例子：验证跨语言表现（模型是英文训练的，预期会偏保守）
zh = client.system_one(
    state="鞋子晚了两个星期才到，尺码还不对。而且我看到卡上被扣了两次钱，麻烦尽快处理！",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"returns": "Exchanges, refunds, wrong items",
                      "shipping": "Delivery delays, lost packages",
                      "billing": "Charges, invoices, payment problems"},
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    },
)
print("中文工单 department =", zh.choices["department"].choice)
print("中文工单 frustration =", zh.scores["frustration"].score)

