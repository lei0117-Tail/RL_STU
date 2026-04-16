"""
文档 → SFT 训练数据 自动生成工具
====================================
将公司内部文档（txt/md/pdf）自动转换为 SFT 格式的训练数据。

支持两种模式：
  1. 本地模式：用已训练好的 Qwen 模型生成问答（免费，质量一般）
  2. API 模式：调用 OpenAI/Qwen-Max API 生成（需要费用，质量高）

生成的数据格式：
  {"instruction": "问题", "input": "补充信息", "output": "答案"}

运行方式：
  # 本地模式（用项目里已下载的 Qwen 模型）
  .venv/bin/python tools/doc_to_sft.py --input 你的文档.txt --mode local

  # API 模式（需要设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY）
  .venv/bin/python tools/doc_to_sft.py --input 你的文档.txt --mode api
"""

import argparse
import json
import os
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 工具函数
# ==========================================

def load_document(file_path: str) -> str:
    """加载文档，支持 txt / md / 简单 pdf"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import pdfplumber
            text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text)
        except ImportError:
            raise ImportError("读取 PDF 需要安装 pdfplumber：pip install pdfplumber")

    else:
        raise ValueError(f"不支持的文件格式：{ext}，请使用 txt/md/pdf")


def split_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    将长文档切分为小块（Chunking）
    - chunk_size: 每块大约多少字
    - overlap: 相邻块之间的重叠字数（保证上下文连贯）
    """
    # 优先按段落切分
    paragraphs = re.split(r'\n{2,}', text.strip())

    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) <= chunk_size:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            # 加入 overlap：保留上一块的最后 overlap 个字
            current = current[-overlap:] + para + "\n" if overlap > 0 else para + "\n"

    if current.strip():
        chunks.append(current.strip())

    # 过滤太短的块（少于 50 字，内容不够）
    return [c for c in chunks if len(c) >= 50]


# ==========================================
# 生成问答的 Prompt 模板
# ==========================================

QA_PROMPT = """你是一个专业的数据标注员。请根据下面这段文档内容，生成 3 个高质量的问答对。

要求：
1. 问题要自然，像真实用户会问的
2. 答案要基于文档内容，准确简洁
3. 覆盖不同角度（事实查询、操作指引、概念解释）
4. 每个问答对用 JSON 格式，如下：

[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}},
  {{"question": "问题3", "answer": "答案3"}}
]

文档内容：
{chunk}

请直接输出 JSON 数组，不要其他内容："""


def extract_qa_from_response(response: str) -> list[dict]:
    """从模型回复中提取 JSON 问答对"""
    # 尝试找到 JSON 数组
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if not match:
        return []

    try:
        pairs = json.loads(match.group())
        result = []
        for p in pairs:
            q = p.get("question", "").strip()
            a = p.get("answer", "").strip()
            if q and a and len(a) >= 10:
                result.append({"instruction": q, "input": "", "output": a})
        return result
    except json.JSONDecodeError:
        return []


# ==========================================
# 本地模式：用 Qwen 模型生成
# ==========================================

def generate_qa_local(chunks: list[str], base_model_path: str) -> list[dict]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    print(f"加载本地模型：{base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map="mps"
    )
    model.eval()

    all_qa = []
    for chunk in tqdm(chunks, desc="生成问答对"):
        prompt = QA_PROMPT.format(chunk=chunk)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("mps")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        qa_pairs = extract_qa_from_response(response)
        all_qa.extend(qa_pairs)

    return all_qa


# ==========================================
# API 模式：调用 OpenAI 兼容接口
# ==========================================

def generate_qa_api(chunks: list[str], api_base: str, api_key: str, model_name: str) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("API 模式需要安装 openai：pip install openai")

    from tqdm import tqdm

    client = OpenAI(api_key=api_key, base_url=api_base)
    all_qa = []

    for chunk in tqdm(chunks, desc="调用 API 生成问答对"):
        prompt = QA_PROMPT.format(chunk=chunk)
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            response = resp.choices[0].message.content
            qa_pairs = extract_qa_from_response(response)
            all_qa.extend(qa_pairs)
        except Exception as e:
            print(f"  API 调用失败：{e}，跳过该块")

    return all_qa


# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="文档 → SFT 训练数据生成工具")
    parser.add_argument("--input",       required=True,  help="输入文档路径（txt/md/pdf）")
    parser.add_argument("--output",      default="",     help="输出 jsonl 文件路径（默认与输入同目录）")
    parser.add_argument("--mode",        default="local", choices=["local", "api"], help="生成模式")
    parser.add_argument("--chunk-size",  type=int, default=500,  help="每块文档的字数（默认 500）")
    parser.add_argument("--overlap",     type=int, default=50,   help="相邻块重叠字数（默认 50）")
    # API 模式参数
    parser.add_argument("--api-base",    default="https://api.openai.com/v1", help="API 地址")
    parser.add_argument("--api-key",     default="",     help="API Key（也可在 .env 里设置 OPENAI_API_KEY）")
    parser.add_argument("--model-name",  default="gpt-4o-mini", help="API 模型名称")
    args = parser.parse_args()

    # 输出路径
    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = base + "_sft_data.jsonl"

    # 1. 加载文档
    print(f"加载文档：{args.input}")
    text = load_document(args.input)
    print(f"文档总字数：{len(text)}")

    # 2. 切块
    chunks = split_chunks(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"切分为 {len(chunks)} 个块（每块约 {args.chunk_size} 字）")

    # 3. 生成问答
    if args.mode == "local":
        _root = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(_root, "models/Qwen2.5-3B")
        if not os.path.isdir(model_path):
            model_path = "Qwen/Qwen2.5-3B"
        all_qa = generate_qa_local(chunks, model_path)

    else:  # api
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("API 模式需要提供 --api-key 或在 .env 中设置 OPENAI_API_KEY")
        all_qa = generate_qa_api(chunks, args.api_base, api_key, args.model_name)

    # 4. 保存
    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成！")
    print(f"   生成问答对：{len(all_qa)} 条")
    print(f"   已保存到：{args.output}")
    print(f"\n接下来可以用生成的数据做 SFT 训练：")
    print(f"   修改 sft/train_finance_mac.py 中的 load_dataset，改为：")
    print(f"   dataset = load_dataset('json', data_files='{args.output}', split='train')")


if __name__ == "__main__":
    main()

