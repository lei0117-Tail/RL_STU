# 模型上传到 HuggingFace Hub

## 前置条件

`.env` 文件中需要配置上传专用 Token：

```bash
HF_UPLOAD_TOKEN=hf_xxx...    # 在 https://huggingface.co/settings/tokens 创建，需要 write 权限
HF_ENDPOINT=https://hf-mirror.com   # 国内镜像加速
```

---

## 上传命令

### 基本格式

```bash
cd /path/to/RL_STU

HF_UPLOAD_TOKEN=$(grep '^HF_UPLOAD_TOKEN' .env | cut -d= -f2 | tr -d ' \r')

HF_ENDPOINT=https://hf-mirror.com \
HUGGING_FACE_HUB_TOKEN=$HF_UPLOAD_TOKEN \
.venv/bin/hf upload <用户名>/<仓库名> <本地目录> . --repo-type model
```

### 排除特定文件（推荐）

上传时可以用 `--exclude` 跳过不需要的文件（如 `.gguf`、`Modelfile` 等本地专用格式）：

```bash
HF_UPLOAD_TOKEN=$(grep '^HF_UPLOAD_TOKEN' .env | cut -d= -f2 | tr -d ' \r')

HF_ENDPOINT=https://hf-mirror.com \
HUGGING_FACE_HUB_TOKEN=$HF_UPLOAD_TOKEN \
.venv/bin/hf upload <用户名>/<仓库名> <本地目录> . \
  --repo-type model \
  --exclude "*.gguf" \
  --exclude "Modelfile"
```

### 本项目实际示例

```bash
cd /Users/liulei/aiS/tranning_model/RL_STU

HF_UPLOAD_TOKEN=$(grep '^HF_UPLOAD_TOKEN' .env | cut -d= -f2 | tr -d ' \r')

# 上传 Qwen2.5-3B DPO 串联合并模型（排除 gguf 和 Modelfile）
HF_ENDPOINT=https://hf-mirror.com \
HUGGING_FACE_HUB_TOKEN=$HF_UPLOAD_TOKEN \
.venv/bin/hf upload Tail-LS/Qwen2.5-3B-dpo-finance \
  merge_models/Qwen2.5-3B-dpo-serial-merged . \
  --repo-type model \
  --exclude "*.gguf" \
  --exclude "Modelfile"
```

---

## 注意事项

### 上传机制
- HuggingFace 的上传是**先传文件、再一次性 commit**
- 在所有文件传完之前，仓库页面**不会显示任何变化**
- 传完后刷新页面，所有文件和时间戳才会更新

### 查看上传进度

方式一：**实时查看日志**（推荐，可以看到进度条）

上传命令加 `&` 放后台运行时，用 `tail -f` 查看日志文件：
```bash
tail -f <log文件路径>
# Ctrl+C 退出查看（不会中断上传）
```

方式二：**查看仓库已有文件**（间接判断）

```bash
HF_UPLOAD_TOKEN=$(grep '^HF_UPLOAD_TOKEN' .env | cut -d= -f2 | tr -d ' \r')
HUGGING_FACE_HUB_TOKEN=$HF_UPLOAD_TOKEN \
hf models files <用户名>/<仓库名>
```

### 仓库不存在时会自动创建
首次上传时，如果仓库不存在，`hf upload` 会自动创建（默认公开仓库）。
如需创建私有仓库：
```bash
... --private
```

### 常用 `--exclude` 模式

| 模式 | 说明 |
|------|------|
| `"*.gguf"` | 排除 Ollama 量化格式 |
| `"Modelfile"` | 排除 Ollama 配置文件 |
| `"*.bin"` | 排除旧版 PyTorch 格式（已被 safetensors 替代） |
| `"checkpoint-*"` | 排除训练中间 checkpoint |

---

## 本项目可上传的模型产物

| 本地路径 | 建议仓库名 | 说明 |
|---------|-----------|------|
| `merge_models/Qwen2.5-3B-sft-merged/` | `用户名/Qwen2.5-3B-sft-finance` | SFT 合并模型 |
| `merge_models/Qwen2.5-3B-dpo-serial-merged/` | `用户名/Qwen2.5-3B-dpo-finance` | SFT+DPO 串联合并模型 |
| `merge_models/gemma-4-E2B-it-sft-merged/` | `用户名/gemma-4-E2B-sft-finance` | Gemma SFT 合并模型 |
| `merge_models/gemma-4-E2B-it-dpo-serial-merged/` | `用户名/gemma-4-E2B-dpo-finance` | Gemma SFT+DPO 串联合并模型 |
| `new_models/Qwen2.5-3B-sft-lora-final/` | `用户名/Qwen2.5-3B-sft-lora` | SFT LoRA 插件（轻量，需搭配基础模型） |
| `new_models/Qwen2.5-3B-dpo-merged-final/` | `用户名/Qwen2.5-3B-dpo-lora` | DPO LoRA 插件（轻量，需搭配基础模型） |

