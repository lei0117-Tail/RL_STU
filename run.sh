#!/usr/bin/env bash
# =============================================================================
#  RL_STU 完整训练流水线
#  串联方案：原始模型 → SFT → merge → DPO → GRPO → 推理对比
#
#  用法：
#    ./run.sh                    # 默认串联方案（全流程）
#    ./run.sh --mode parallel    # 并联方案（各自独立，适合学习对比）
#    ./run.sh --step sft         # 只跑 SFT
#    ./run.sh --step dpo         # 只跑 DPO（需先跑 sft）
#    ./run.sh --step grpo        # 只跑 GRPO（需先跑 sft）
#    ./run.sh --step inference   # 只跑推理对比
#    ./run.sh --model gemma-4-E2B  # 指定模型（覆盖 .env 中的 SELECT_MODEL）
#    ./run.sh --dry-run          # 演习模式，只打印命令不执行
# =============================================================================

set -e   # 任何步骤失败即退出

# ──────────────────────────────────────────────
# 颜色输出
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'   # No Color

# ──────────────────────────────────────────────
# 默认参数
# ──────────────────────────────────────────────
MODE="serial"        # serial | parallel
STEP="all"           # all | sft | dpo | grpo | inference
MODEL_OVERRIDE=""    # 若非空则覆盖 .env 中的 SELECT_MODEL
DRY_RUN=false

# ──────────────────────────────────────────────
# 解析命令行参数
# ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";          shift 2 ;;
        --step)       STEP="$2";          shift 2 ;;
        --model)      MODEL_OVERRIDE="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;        shift   ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^#\s*//'
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数：$1${NC}"
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────
# 确定脚本根目录（无论从哪里调用都正确）
# ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ──────────────────────────────────────────────
# 加载 .env，并允许 --model 覆盖 SELECT_MODEL
# ──────────────────────────────────────────────
if [[ -f ".env" ]]; then
    # 只导出非注释、非空的 KEY=VALUE 行
    set -o allexport
    # shellcheck disable=SC2046
    eval $(grep -E '^[A-Z_]+=.+' .env | grep -v '^#')
    set +o allexport
fi

if [[ -n "$MODEL_OVERRIDE" ]]; then
    export SELECT_MODEL="$MODEL_OVERRIDE"
fi

SELECT_MODEL="${SELECT_MODEL:-Qwen2.5-3B}"
PYTHON=".venv/bin/python"

# ──────────────────────────────────────────────
# 检查 Python 环境
# ──────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
    echo -e "${RED}❌ 找不到 .venv/bin/python，请先创建虚拟环境：${NC}"
    echo "   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
log_step() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_info()    { echo -e "${GREEN}  ✅ $1${NC}"; }
log_warn()    { echo -e "${YELLOW}  ⚠️  $1${NC}"; }
log_skip()    { echo -e "${YELLOW}  ⏭️  跳过：$1${NC}"; }

run_cmd() {
    local label="$1"; shift
    echo -e "${CYAN}  → $*${NC}"
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}  [dry-run] 跳过执行${NC}"
        return 0
    fi
    if ! "$@"; then
        echo -e "${RED}❌ 步骤失败：${label}${NC}"
        exit 1
    fi
}

check_dir() {
    local path="$1"
    local name="$2"
    if [[ -d "$path" ]]; then
        log_info "已存在 $name → $path，跳过重新训练"
        return 0   # 0 = 已存在，可跳过
    fi
    return 1       # 1 = 不存在，需要训练
}

# ──────────────────────────────────────────────
# 路径定义（与各训练脚本保持一致）
# ──────────────────────────────────────────────
SFT_LORA_FINAL="new_models/${SELECT_MODEL}-sft-lora-final"
SFT_MERGED="merge_models/${SELECT_MODEL}-sft-merged"
DPO_DATA="dpo/dpo_finance_data_${SELECT_MODEL}.jsonl"
DPO_FINAL="new_models/${SELECT_MODEL}-dpo-merged-final"       # 串联
DPO_PARALLEL_FINAL="new_models/${SELECT_MODEL}-dpo-final"     # 并联
GRPO_FINAL="new_models/${SELECT_MODEL}-grpo-merged-final"     # 串联
GRPO_PARALLEL_FINAL="new_models/${SELECT_MODEL}-grpo-final"   # 并联

# ──────────────────────────────────────────────
# 打印运行信息
# ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║         RL_STU 大模型微调流水线                              ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "  ${BOLD}模型：${NC} ${SELECT_MODEL}"
echo -e "  ${BOLD}模式：${NC} ${MODE}"
echo -e "  ${BOLD}步骤：${NC} ${STEP}"
[[ "$DRY_RUN" == true ]] && echo -e "  ${YELLOW}${BOLD}[演习模式，不执行实际命令]${NC}"
echo ""

# ══════════════════════════════════════════════
#  STEP 1 ── SFT 监督微调
#  输出：new_models/{MODEL}-sft-lora-final/
#        merge_models/{MODEL}-sft-merged/   （脚本内自动 merge）
# ══════════════════════════════════════════════
run_sft() {
    log_step "Step 1 / SFT 监督微调 (finance-alpaca + FinGPT)"

    if check_dir "$SFT_LORA_FINAL" "SFT LoRA"; then
        return 0
    fi

    echo -e "  预计耗时：~30 分钟（500 steps，M4 Max）"
    run_cmd "SFT 训练" "$PYTHON" sft/train_finance_mac.py
    log_info "SFT 完成 → ${SFT_LORA_FINAL}"
    log_info "SFT 合并模型 → ${SFT_MERGED}"
}

# ══════════════════════════════════════════════
#  STEP 2 ── 生成 DPO 数据（串联方案专用）
#  输出：dpo/dpo_finance_data.jsonl
# ══════════════════════════════════════════════
run_gen_dpo_data() {
    log_step "Step 2 / 生成 DPO 训练数据（SFT 合并模型生成 rejected）"

    if [[ -f "$DPO_DATA" ]]; then
        local lines
        lines=$(wc -l < "$DPO_DATA" 2>/dev/null || echo 0)
        log_info "已存在 DPO 数据（${lines} 条）→ ${DPO_DATA}，跳过生成"
        return 0
    fi

    if [[ ! -d "$SFT_MERGED" ]]; then
        log_warn "SFT 合并模型不存在（${SFT_MERGED}），generate_dpo_data.py 需要它"
        log_warn "请先完成 SFT 步骤（sft/train_finance_mac.py 末尾会自动 merge）"
        exit 1
    fi

    echo -e "  预计耗时：~40-80 分钟（生成 1000 条 rejected 回答）"
    run_cmd "生成 DPO 数据" "$PYTHON" dpo/generate_dpo_data.py
    log_info "DPO 数据 → ${DPO_DATA}"
}

# ══════════════════════════════════════════════
#  STEP 3a ── 串联 DPO 训练
#  输入：merge_models/{MODEL}-sft-merged/
#        dpo/dpo_finance_data.jsonl
#  输出：new_models/{MODEL}-dpo-merged-final/
# ══════════════════════════════════════════════
run_dpo_serial() {
    log_step "Step 3 / 串联 DPO 训练（SFT merged → 新 LoRA）"

    if check_dir "$DPO_FINAL" "串联 DPO LoRA"; then
        return 0
    fi

    echo -e "  预计耗时：~15 分钟（200 steps）"
    run_cmd "串联 DPO 训练" "$PYTHON" dpo/train_dpo_merged.py
    log_info "串联 DPO 完成 → ${DPO_FINAL}"
}

# ══════════════════════════════════════════════
#  STEP 3b ── 并联 DPO 训练（hh-rlhf 数据）
#  输出：new_models/{MODEL}-dpo-final/
# ══════════════════════════════════════════════
run_dpo_parallel() {
    log_step "Step 3 / 并联 DPO 训练（原始模型 → 新 LoRA，hh-rlhf 数据）"

    if check_dir "$DPO_PARALLEL_FINAL" "并联 DPO LoRA"; then
        return 0
    fi

    echo -e "  预计耗时：~15 分钟（200 steps）"
    run_cmd "并联 DPO 训练" "$PYTHON" dpo/train_dpo_hh.py
    log_info "并联 DPO 完成 → ${DPO_PARALLEL_FINAL}"
}

# ══════════════════════════════════════════════
#  STEP 4a ── 串联 GRPO 训练
#  输入：merge_models/{MODEL}-sft-merged/
#  输出：new_models/{MODEL}-grpo-merged-final/
# ══════════════════════════════════════════════
run_grpo_serial() {
    log_step "Step 4 / 串联 GRPO 强化学习（SFT merged → 新 LoRA）"

    if check_dir "$GRPO_FINAL" "串联 GRPO LoRA"; then
        return 0
    fi

    echo -e "  预计耗时：~30 分钟（200 steps，每 prompt 生成 4 个候选）"
    run_cmd "串联 GRPO 训练" "$PYTHON" gdpo/train_grpo_merged.py
    log_info "串联 GRPO 完成 → ${GRPO_FINAL}"
}

# ══════════════════════════════════════════════
#  STEP 4b ── 并联 GRPO 训练
#  输出：new_models/{MODEL}-grpo-final/
# ══════════════════════════════════════════════
run_grpo_parallel() {
    log_step "Step 4 / 并联 GRPO 强化学习（原始模型 → 新 LoRA）"

    if check_dir "$GRPO_PARALLEL_FINAL" "并联 GRPO LoRA"; then
        return 0
    fi

    echo -e "  预计耗时：~30 分钟（200 steps，每 prompt 生成 4 个候选）"
    run_cmd "并联 GRPO 训练" "$PYTHON" gdpo/train_grpo.py
    log_info "并联 GRPO 完成 → ${GRPO_PARALLEL_FINAL}"
}

# ══════════════════════════════════════════════
#  STEP 5 ── 推理对比
# ══════════════════════════════════════════════
run_inference() {
    log_step "Step 5 / 四模型推理对比"
    echo -e "  加载：原始模型 / SFT / DPO / GRPO，依次回答示例问题"
    run_cmd "推理对比" "$PYTHON" gdpo/inference_grpo.py
}

# ══════════════════════════════════════════════
#  主流程调度
# ══════════════════════════════════════════════
case "$STEP" in
    # ── 只跑单个步骤 ──────────────────────────
    sft)
        run_sft
        ;;
    dpo)
        if [[ "$MODE" == "serial" ]]; then
            run_gen_dpo_data
            run_dpo_serial
        else
            run_dpo_parallel
        fi
        ;;
    grpo)
        if [[ "$MODE" == "serial" ]]; then
            run_grpo_serial
        else
            run_grpo_parallel
        fi
        ;;
    inference)
        run_inference
        ;;

    # ── 全流程 ────────────────────────────────
    all)
        if [[ "$MODE" == "serial" ]]; then
            # ═══════ 串联完整流程 ═══════
            echo -e "  ${BOLD}流程：原始模型 → SFT → merge → DPO → GRPO → 推理${NC}"
            run_sft
            run_gen_dpo_data
            run_dpo_serial
            run_grpo_serial
            run_inference
        else
            # ═══════ 并联完整流程 ═══════
            echo -e "  ${BOLD}流程：原始模型 → SFT / DPO / GRPO（互相独立）→ 推理${NC}"
            run_sft
            run_dpo_parallel
            run_grpo_parallel
            run_inference
        fi
        ;;
    *)
        echo -e "${RED}❌ 未知 --step 值：${STEP}（可选：all / sft / dpo / grpo / inference）${NC}"
        exit 1
        ;;
esac

# ──────────────────────────────────────────────
# 完成总结
# ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  🎉 所有步骤完成！                                           ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  模型产物："
[[ -d "$SFT_LORA_FINAL"       ]] && echo -e "  ${GREEN}✅${NC} SFT  LoRA   → ${SFT_LORA_FINAL}"
[[ -d "$SFT_MERGED"           ]] && echo -e "  ${GREEN}✅${NC} SFT  合并   → ${SFT_MERGED}"
[[ -d "$DPO_FINAL"            ]] && echo -e "  ${GREEN}✅${NC} DPO  LoRA   → ${DPO_FINAL}"
[[ -d "$DPO_PARALLEL_FINAL"   ]] && echo -e "  ${GREEN}✅${NC} DPO  LoRA(并)→ ${DPO_PARALLEL_FINAL}"
[[ -d "$GRPO_FINAL"           ]] && echo -e "  ${GREEN}✅${NC} GRPO LoRA   → ${GRPO_FINAL}"
[[ -d "$GRPO_PARALLEL_FINAL"  ]] && echo -e "  ${GREEN}✅${NC} GRPO LoRA(并)→ ${GRPO_PARALLEL_FINAL}"
echo ""

