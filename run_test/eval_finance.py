"""
金融模型领域评测脚本
自动扫描 merge_models/ 目录下的所有合并模型，全量横向对比。

评测指标：
  - ROUGE-L（与 chosen 标准答案的相似度，越高越好）
  - 平均回答长度（words）
  - Chosen 胜率（模型回答 ROUGE 是否高于 rejected 的 ROUGE，越高越好）
  - 推理耗时（秒/条）

使用方式：
  # 全量评测 merge_models/ 下所有模型
  python3 run_test/eval_finance.py

  # 只评测指定模型（子目录名，空格分隔）
  python3 run_test/eval_finance.py Qwen2.5-3B-sft-merged Qwen2.5-3B-dpo-serial-merged
"""

import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

# ==========================================
# 配置
# ==========================================
ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGE_MODELS_DIR = os.path.join(ROOT, "merge_models")

DATA_PATH      = os.path.join(ROOT, "dpo", "dpo_finance_data.jsonl")
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_SAMPLES    = 30      # 评测样本数（调大更准，调小更快）
MAX_NEW_TOKENS = 200     # 生成最大 token 数
SEED           = 42


# ==========================================
# 自动发现模型
# ==========================================
def discover_models(filter_names: list[str] | None = None) -> dict[str, str]:
    """
    自动扫描 merge_models/ 目录，返回 {目录名: 完整路径} 字典。
    filter_names 非空时，只保留指定名称的子目录。
    判断是否为合法模型目录：包含 config.json 文件。
    """
    if not os.path.isdir(MERGE_MODELS_DIR):
        print(f"❌ merge_models 目录不存在：{MERGE_MODELS_DIR}")
        sys.exit(1)

    models = {}
    for name in sorted(os.listdir(MERGE_MODELS_DIR)):
        path = os.path.join(MERGE_MODELS_DIR, name)
        if not os.path.isdir(path):
            continue
        if not os.path.isfile(os.path.join(path, "config.json")):
            continue  # 不含 config.json，跳过（非模型目录）
        if filter_names and name not in filter_names:
            continue
        models[name] = path

    return models


# ==========================================
# 工具函数
# ==========================================
def load_test_data(path: str, n: int, seed: int) -> list[dict]:
    """从 jsonl 文件中随机采样 n 条作为测试集"""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    random.seed(seed)
    random.shuffle(data)
    return data[:n]


def generate(model, tokenizer, prompt: str) -> tuple[str, float]:
    """生成回答，返回 (回答文本, 耗时秒)"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed


def compute_rouge_l(prediction: str, reference: str) -> float:
    """计算 ROUGE-L F1"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def evaluate_model(model_name: str, model_path: str, test_data: list[dict]) -> dict:
    """评测单个模型，返回汇总指标"""
    print(f"\n{'='*60}")
    print(f"  评测模型：{model_name}")
    print(f"  路径：{model_path}")
    print(f"{'='*60}")

    # 加载
    print("  加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  加载模型（{DEVICE}）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    print("  ✅ 模型加载完毕，开始推理...\n")

    rouge_l_scores = []
    win_flags      = []   # 模型回答 ROUGE > rejected ROUGE → win
    answer_lengths = []
    latencies      = []
    total = len(test_data)

    for i, sample in enumerate(test_data):
        prompt   = sample["prompt"]
        chosen   = sample["chosen"]
        rejected = sample["rejected"]

        answer, elapsed = generate(model, tokenizer, prompt)

        r_pred_chosen   = compute_rouge_l(answer, chosen)
        r_pred_rejected = compute_rouge_l(answer, rejected)

        rouge_l_scores.append(r_pred_chosen)
        win_flags.append(1 if r_pred_chosen > r_pred_rejected else 0)
        answer_lengths.append(len(answer.split()))
        latencies.append(elapsed)

        # 进度
        marker = "✓" if win_flags[-1] else "✗"
        print(f"  [{i+1:>2}/{total}] {marker}  "
              f"ROUGE-L={r_pred_chosen:.3f}  "
              f"长度={answer_lengths[-1]}词  "
              f"耗时={elapsed:.1f}s")

        # 打印第 1 条样例供参考
        if i == 0:
            print(f"\n  ── 样例展示 ──")
            print(f"  Prompt  : {prompt[:120].replace(chr(10), ' ')}...")
            print(f"  Chosen  : {chosen[:100].replace(chr(10), ' ')}...")
            print(f"  预测结果: {answer[:100].replace(chr(10), ' ')}...")
            print()

    # 释放显存
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "rouge_l_mean": sum(rouge_l_scores) / len(rouge_l_scores),
        "rouge_l_max":  max(rouge_l_scores),
        "win_rate":     sum(win_flags) / len(win_flags),
        "avg_length":   sum(answer_lengths) / len(answer_lengths),
        "avg_latency":  sum(latencies) / len(latencies),
        "total_time":   sum(latencies),
    }


def print_summary(results: dict[str, dict]):
    """打印最终对比表格"""
    print(f"\n{'='*70}")
    print("  📊  评测结果汇总")
    print(f"{'='*70}")

    header = f"  {'模型':<36} {'ROUGE-L均值':>12} {'Chosen胜率':>10} {'平均长度':>10} {'推理速度':>10}"
    print(header)
    print(f"  {'-'*72}")

    for name, r in results.items():
        print(
            f"  {name:<36} "
            f"{r['rouge_l_mean']:>12.4f} "
            f"{r['win_rate']:>9.1%} "
            f"{r['avg_length']:>9.1f}词 "
            f"{r['avg_latency']:>8.1f}s/条"
        )

    print(f"{'='*70}")

    # 找出最优
    best_rouge = max(results, key=lambda k: results[k]["rouge_l_mean"])
    best_win   = max(results, key=lambda k: results[k]["win_rate"])
    print(f"\n  🏆 ROUGE-L 最高：{best_rouge}  ({results[best_rouge]['rouge_l_mean']:.4f})")
    print(f"  🏆 Chosen 胜率最高：{best_win}  ({results[best_win]['win_rate']:.1%})")

    # 动态两两对比：按发现顺序，相邻模型之间的提升幅度
    names = list(results.keys())
    if len(names) >= 2:
        print(f"\n  📈 相邻模型提升对比（按发现顺序）：")
        for i in range(1, len(names)):
            base_name = names[i - 1]
            curr_name = names[i]
            base = results[base_name]
            curr = results[curr_name]
            rouge_delta = (curr["rouge_l_mean"] - base["rouge_l_mean"]) / max(base["rouge_l_mean"], 1e-8) * 100
            win_delta   = curr["win_rate"] - base["win_rate"]
            print(f"     {base_name} → {curr_name}")
            print(f"       ROUGE-L  : {rouge_delta:+.1f}%")
            print(f"       Chosen胜率: {win_delta:+.1%}")
    print()


# ==========================================
# 主流程
# ==========================================
def main():
    # 命令行参数：可选指定模型子目录名，不填则全量扫描
    filter_names = sys.argv[1:] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  金融模型领域评测（方案一：领域专属评测）")
    print(f"  设备：{DEVICE}")
    print(f"  样本数：{NUM_SAMPLES}  |  最大生成 Token：{MAX_NEW_TOKENS}")
    if filter_names:
        print(f"  过滤模型：{filter_names}")
    else:
        print(f"  扫描目录：{MERGE_MODELS_DIR}（全量）")
    print("=" * 60)

    # 自动发现模型
    MODELS_VALID = discover_models(filter_names)
    if not MODELS_VALID:
        print(f"❌ 未在 {MERGE_MODELS_DIR} 下找到任何合法模型目录，退出")
        sys.exit(1)

    print(f"\n发现 {len(MODELS_VALID)} 个模型：")
    for name in MODELS_VALID:
        print(f"  • {name}")

    # 加载测试数据
    print(f"\n加载测试数据：{DATA_PATH}")
    test_data = load_test_data(DATA_PATH, NUM_SAMPLES, SEED)
    print(f"共采样 {len(test_data)} 条\n")

    # 逐个评测
    results = {}
    for name, path in MODELS_VALID.items():
        results[name] = evaluate_model(name, path, test_data)

    # 汇总输出
    print_summary(results)

    # 保存到文件
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  📁 详细结果已保存至：{out_path}\n")


if __name__ == "__main__":
    main()

