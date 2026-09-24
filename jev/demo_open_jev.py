"""

https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large

open-jev-deberta-v3-large 本地体验 Demo
==========================================
Jev 风格的类型化决策模型（第三方开源复刻，com-kotobalabs 出品）：
  - 底座：DeBERTa-v3-large 编码器（434M 参数）+ span 匹配决策头
  - 输入：state（程序状态/上下文）+ 任意多个类型化问题
  - 输出：每个问题的校准概率分布，一次 forward 全部出结果，不生成任何文本

三种问题类型（与 TypeSafe Jev 的 API 同构）：
  choice → 从 2~255 个无序选项中选一个，返回每个选项概率
  score  → 在 2~10 个有序等级上打分，返回期望分数（可落在等级之间）
  noul   → 是非题，返回 p(yes)

运行方式：
  .venv/bin/python jev/demo_open_jev.py

前置条件：
  .venv/bin/pip install sentencepiece protobuf
  模型已下载到 jev/models/open-jev-deberta-v3-large（约 1.7GB）
"""

import os
import sys
import time

# 国内镜像 + 禁用 XET（与项目其他脚本保持一致）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_root, "jev", "models", "open-jev-deberta-v3-large")

# HF 仓库自带 typed_decisions 代码包，直接加入 sys.path，无需 pip install。
# 用 importlib 动态导入（路径是运行时拼接的，静态分析无法解析）
import importlib

sys.path.insert(0, MODEL_DIR)
OpenJev = importlib.import_module("typed_decisions.open_jev").OpenJev


def pretty(result: dict, label: str) -> None:
    """打印单个问题的决策结果"""
    print(f"  [{label}]")
    if "choice" in result:
        print(f"    choice      = {result['choice']}")
        print(f"    confidence  = {result['confidence']:.3f}")
        probs = "  ".join(f"{k}:{v:.2f}" for k, v in result["probabilities"].items())
        print(f"    probabilities = {probs}")
    elif "score" in result:
        print(f"    score       = {result['score']:.3f}  (期望等级，可落在等级之间)")
        print(f"    confidence  = {result['confidence']:.3f}")
        probs = "  ".join(f"{k}:{v:.2f}" for k, v in result["probabilities"].items())
        print(f"    probabilities = {probs}")
    elif "noul" in result:
        print(f"    p(yes)      = {result['noul']:.3f}")


def main() -> None:
    if not os.path.isdir(MODEL_DIR):
        print(f"❌ 模型目录不存在：{MODEL_DIR}")
        print("   请先运行下载命令（见本文件头部注释）")
        sys.exit(1)

    print(f"加载模型：{MODEL_DIR}")
    print("  （首次加载约 10-20 秒；自动使用 MPS，无则回退 CPU）\n")
    t0 = time.time()
    m = OpenJev.from_pretrained(MODEL_DIR)
    print(f"  ✅ 加载完成（{time.time() - t0:.1f}s），设备：{m.device}\n")

    # ==========================================
    # 场景一：客服工单分类（choice + score + noul 组合）
    # ==========================================
    print("=" * 70)
    print("场景一：客服工单 —— 一次 forward 回答 4 个类型化问题")
    print("=" * 70)
    state = ("I was charged twice for the same order and nobody answers my emails. "
             "I want my money back now.")
    questions = [
        {"type": "choice",
         "instructions": "Which product area is the message about?",
         "options": ["fees & charges", "pin & security", "refund & dispute",
                     "top-up", "exchange & fiat", "atm & cash", "transfer",
                     "card", "account & identity", "other"]},
        {"type": "score",
         "instructions": "How positive is the sentiment of this message?",
         "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
        {"type": "noul",
         "instructions": "The customer is asking for a refund."},
        {"type": "noul",
         "instructions": "The customer mentions a card problem."},
    ]

    t0 = time.time()
    results = m.decide(state, questions)
    dt = (time.time() - t0) * 1000

    for q, r in zip(questions, results):
        pretty(r, q["type"])
    print(f"\n  ⏱  4 个问题共耗时 {dt:.0f} ms（单次 forward，无文本生成）\n")

    # ==========================================
    # 场景二：金融文本（贴近本项目领域）
    # ==========================================
    print("=" * 70)
    print("场景二：金融新闻流 —— 高频路由分流（Jev 的核心使用场景）")
    print("=" * 70)
    state = ("Shares of the chipmaker jumped 8% after it raised its revenue "
             "forecast for the third quarter, citing strong data-center demand.")
    questions = [
        {"type": "choice",
         "instructions": "Which news section does this article belong to?",
         "options": ["World", "Sports", "Business", "Science/Technology"]},
        {"type": "noul",
         "instructions": "The stock price went up."},
        {"type": "score",
         "instructions": "How bullish is the tone of this news?",
         "options": ["very bearish", "bearish", "neutral", "bullish", "very bullish"]},
    ]

    t0 = time.time()
    results = m.decide(state, questions)
    dt = (time.time() - t0) * 1000

    for q, r in zip(questions, results):
        pretty(r, q["type"])
    print(f"\n  ⏱  3 个问题共耗时 {dt:.0f} ms\n")

    # ==========================================
    # 场景三：置信度门控演示（自动化止损线）
    # ==========================================
    print("=" * 70)
    print("场景三：置信度门控 —— 模拟 '高置信自动执行，低置信转人工'")
    print("=" * 70)
    state = "The quarterly report shows revenue up 12% year over year."
    q = {"type": "noul", "instructions": "Revenue increased year over year."}
    r = m.decide(state, [q])[0]
    p_yes = r["noul"]
    action = "✅ 自动执行" if p_yes > 0.9 else "🔔 转人工审核"
    print(f"  p(yes) = {p_yes:.3f}  →  {action}")

    state = "The meeting notes were inconclusive about next steps."
    q = {"type": "noul", "instructions": "A concrete decision was made in the meeting."}
    r = m.decide(state, [q])[0]
    p_yes = r["noul"]
    action = "✅ 自动执行" if p_yes > 0.9 else "🔔 转人工审核"
    print(f"  p(yes) = {p_yes:.3f}  →  {action}")
    print("\n  （第二个例子故意模糊——观察校准概率如何'诚实地不确定'）")

    # ==========================================
    # 场景四：中文测试（模型为纯英文训练，观察跨语言表现）
    # ==========================================
    print()
    print("=" * 70)
    print("场景四：中文 state —— 跨语言能力实测（模型仅用英文数据训练）")
    print("=" * 70)
    print("  ⚠️ 预期提示：训练数据为 banking77/sst5/boolq（全英文），")
    print("     中文属于 OOD（分布外）输入，概率可能不够可靠，重点观察其'诚实性'\n")

    # ── 4a：中文 state + 英文问题/选项 ──
    print("  ── 4a. 中文 state + 英文 question（部分跨语言）──")
    state = "客户投诉：同一笔订单被重复扣款两次，发邮件客服一直不回复，要求立即退款。"
    questions = [
        {"type": "choice",
         "instructions": "Which product area is the message about?",
         "options": ["fees & charges", "pin & security", "refund & dispute",
                     "top-up", "exchange & fiat", "atm & cash", "transfer",
                     "card", "account & identity", "other"]},
        {"type": "score",
         "instructions": "How positive is the sentiment of this message?",
         "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
        {"type": "noul",
         "instructions": "The customer is asking for a refund."},
    ]
    results = m.decide(state, questions)
    for q, r in zip(questions, results):
        pretty(r, q["type"])
    print()

    # ── 4b：全中文（state + 问题 + 选项都中文）──
    print("  ── 4b. 全中文（state / instructions / options 全部中文）──")
    state = ("美股收盘：芯片股大涨，某芯片制造商上调三季度营收指引 8%，"
             "数据中心需求强劲，股价单日上涨 6.2%。")
    questions = [
        {"type": "choice",
         "instructions": "这条新闻属于哪个版块？",
         "options": ["国际", "体育", "财经", "科技", "娱乐", "其他"]},
        {"type": "noul",
         "instructions": "股价上涨了。"},
        {"type": "score",
         "instructions": "这条新闻的情绪倾向如何？",
         "options": ["非常悲观", "悲观", "中性", "乐观", "非常乐观"]},
    ]
    results = m.decide(state, questions)
    for q, r in zip(questions, results):
        pretty(r, q["type"])
    print()

    # ── 4c：同一内容的英文对照（量化跨语言性能差异）──
    print("  ── 4c. 同内容英文版对照（量化跨语言差异）──")
    state_en = ("US stocks closed: chip stocks rallied. A chipmaker raised its Q3 revenue "
                "guidance by 8% on strong data-center demand; shares rose 6.2% in a single day.")
    questions_en = [
        {"type": "choice",
         "instructions": "Which news section does this article belong to?",
         "options": ["World", "Sports", "Business", "Technology", "Entertainment", "Other"]},
        {"type": "noul",
         "instructions": "The stock price went up."},
        {"type": "score",
         "instructions": "How bullish is the tone of this news?",
         "options": ["very bearish", "bearish", "neutral", "bullish", "very bullish"]},
    ]
    results = m.decide(state_en, questions_en)
    for q, r in zip(questions_en, results):
        pretty(r, q["type"])
    print("\n  💡 对比观察：若 4b 中文置信度显著低于 4c 英文版（或选项概率更平），")
    print("     即模型'知道自己不懂中文'——校准概率在 OOD 上变保守，这正是可取的性质。")
    print("     生产建议：中文场景先翻译成英文再决策，或用中文数据微调一版。")


if __name__ == "__main__":
    main()

