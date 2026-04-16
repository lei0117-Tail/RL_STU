# 将 HuggingFace 模型转换为 Ollama 使用指南

## 概述

HuggingFace 上的模型无法直接被 Ollama 加载，需要先转换为 **GGUF 格式**。

---

## 三种情况判断

```
HF 上的模型
    ├── 官方 Ollama 库有？→ ollama pull xxx（最简单）
    ├── HF 上有 GGUF 文件？→ 直接写 Modelfile FROM 指向 gguf
    └── 只有原始权重？→ convert_hf_to_gguf.py 转换 → Modelfile → ollama create
```

| 情况 | 处理方式 |
|------|---------|
| Ollama 官方库收录的模型 | `ollama pull qwen2.5:3b` 直接拉取 |
| HF 上已有 GGUF 版本（如 TheBloke） | 下载 `.gguf` 文件，直接写 Modelfile |
| 本地训练/只有 safetensors 原始权重 | 需要 llama.cpp 转换（见下方完整流程） |

---

## 完整转换流程（针对本地训练模型）

### 第一步：安装并编译 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Mac 使用 Metal GPU 加速编译
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)

# 安装 Python 依赖
pip install -r requirements.txt
```

### 第二步：将 SafeTensors 转换为 GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py \
  /path/to/your/model \
  --outfile /path/to/output/model.gguf \
  --outtype q8_0
```

**量化类型选择：**

| 类型 | 文件大小 | 精度 | 推荐场景 |
|------|---------|------|---------|
| `bf16` / `f16` | ~6GB | 最高，无损失 | 精度优先 |
| `q8_0` | ~3GB | 极小损失 | ✅ 日常推荐 |
| `q4_k_m` | ~1.8GB | 有一定损失 | 内存受限时 |

### 第三步：创建 Modelfile

```
FROM /path/to/output/model.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM "你是一个专业的金融分析助手，擅长回答金融、投资、理财相关问题。"
```

> ⚠️ `TEMPLATE` 中的聊天模板需要与模型匹配。Qwen2.5 系列使用 `<|im_start|>` / `<|im_end|>` 格式。

### 第四步：导入 Ollama

```bash
ollama create my-model-name -f /path/to/Modelfile
```

### 第五步：运行模型

```bash
# 交互式对话
ollama run my-model-name

# 单次提问
ollama run my-model-name "什么是市盈率？"

# API 调用（兼容 OpenAI 格式）
curl http://localhost:11434/api/chat -d '{
  "model": "my-model-name",
  "messages": [
    {"role": "user", "content": "什么是市盈率？"}
  ]
}'
```

---

## 本项目示例：Qwen2.5-3B-dpo-finance

```bash
# 转换为 q8_0 量化 GGUF
python llama.cpp/convert_hf_to_gguf.py \
  /Users/liulei/aiS/tranning_model/RL_STU/merge_models/Qwen2.5-3B-dpo-serial-merged \
  --outfile /Users/liulei/aiS/tranning_model/RL_STU/merge_models/Qwen2.5-3B-dpo-serial-merged/qwen2.5-3b-dpo-q8.gguf \
  --outtype q8_0

# 导入 Ollama
ollama create qwen2.5-3b-dpo-finance \
  -f /Users/liulei/aiS/tranning_model/RL_STU/merge_models/Qwen2.5-3B-dpo-serial-merged/Modelfile

# 运行
ollama run qwen2.5-3b-dpo-finance
```

---

## llama.cpp 支持的模型架构

| 支持 ✅ | 不支持或有限支持 ❌ |
|--------|----------------|
| Qwen2 / Qwen2.5 | 部分 MoE 模型 |
| LLaMA 1 / 2 / 3 | 多模态模型（图像部分）|
| Mistral / Mixtral | Encoder-only（BERT 系列）|
| Gemma / Gemma2 | Embedding 专用模型 |
| Phi-2 / Phi-3 | — |

---

## 常用 Ollama 命令

```bash
ollama list              # 查看已安装的模型
ollama show my-model     # 查看模型详情
ollama rm my-model       # 删除模型
ollama ps                # 查看正在运行的模型
ollama serve             # 启动 Ollama 服务（默认端口 11434）

