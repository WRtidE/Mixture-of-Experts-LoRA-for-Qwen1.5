#!/bin/bash
# ============================================================================
# MoE-LoRA 消融实验脚本：探索专家数量对模型性能的影响
#
# 实验变量: num_experts ∈ {1, 2, 4, 8, 16}
# 固定变量: r=16, top_k=2, lora_alpha=32, epochs=3
# 模型: Qwen1.5-0.5B (轻量级，适合快速消融实验)
# 数据集: tatsu-lab/alpaca
#
# 用法:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh              # 运行全部实验
#   ./run_ablation.sh --dry-run    # 仅打印命令，不执行
#   ./run_ablation.sh --expert 8   # 只运行 num_experts=8
# ============================================================================

set -euo pipefail

# ---- 固定超参数 ----
MODEL="Qwen/Qwen1.5-0.5B"
LORA_R=16
TOP_K=2
LORA_ALPHA=32
EPOCHS=3
BATCH_SIZE=4
MAX_LENGTH=512
LEARNING_RATE=2e-4
BASE_OUTPUT_DIR="./ablation_results"

# ---- 消融实验变量: 专家数量 ----
# 1 个专家 = 退化到标准 LoRA
# 2/4/8/16 个专家 = MoE-LoRA
EXPERT_COUNTS=(1 2 4 8 16)

# ---- 解析命令行参数 ----
DRY_RUN=false
TARGET_EXPERT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --expert)
            TARGET_EXPERT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--dry-run] [--expert N]"
            exit 1
            ;;
    esac
done

# ---- 确保环境和依赖 ----
echo "============================================"
echo "  MoE-LoRA 专家数量消融实验"
echo "============================================"
echo "模型:  $MODEL"
echo "数据集: tatsu-lab/alpaca"
echo "LoRA r=$LORA_R, top_k=$TOP_K, alpha=$LORA_ALPHA"
echo "训练: ${EPOCHS} epochs, lr=$LEARNING_RATE"
echo ""

# 检测 Python 环境：优先使用 conda 激活的，回退到 python3/python
if command -v conda &> /dev/null && conda env list | grep -q "qwen"; then
    PYTHON="conda run -n qwen python"
    echo "Python: (conda env: qwen)"
else
    if command -v python3 &> /dev/null; then
        PYTHON="python3"
    elif command -v python &> /dev/null; then
        PYTHON="python"
    else
        echo "[错误] 未找到 Python，请先安装 Python 或激活 conda 环境"
        exit 1
    fi
    echo "Python: $($PYTHON --version)"
fi

echo "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)')" 2>/dev/null || echo "PyTorch: (检测失败)"
echo "Transformers: $($PYTHON -c 'import transformers; print(transformers.__version__)')" 2>/dev/null || echo "Transformers: (检测失败)"
echo ""

# ---- 运行消融实验 ----
TOTAL=${#EXPERT_COUNTS[@]}
CURRENT=0

for N_EXPERTS in "${EXPERT_COUNTS[@]}"; do
    # 如果指定了单个 expert，只运行那个
    if [[ -n "$TARGET_EXPERT" ]] && [[ "$N_EXPERTS" != "$TARGET_EXPERT" ]]; then
        continue
    fi

    CURRENT=$((CURRENT + 1))
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/experts_${N_EXPERTS}"

    echo "============================================"
    echo "[$CURRENT/$([ -n "$TARGET_EXPERT" ] && echo 1 || echo $TOTAL)] 实验: num_experts=$N_EXPERTS"
    echo "输出目录: $OUTPUT_DIR"
    echo "============================================"

    CMD="$PYTHON scripts/train.py \
        --model_name $MODEL \
        --num_experts $N_EXPERTS \
        --top_k $TOP_K \
        --r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --learning_rate $LEARNING_RATE \
        --output_dir $OUTPUT_DIR \
        --no_4bit"

    echo "[命令] $CMD"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] 跳过实际执行"
        echo ""
    else
        echo "开始训练..."
        eval $CMD

        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            echo "[完成] num_experts=$N_EXPERTS 训练成功"
            echo "  模型保存于: $OUTPUT_DIR"
        else
            echo "[失败] num_experts=$N_EXPERTS 训练异常退出"
        fi
        echo ""
    fi
done

echo "============================================"
echo "  消融实验全部完成！"
echo "  结果目录: $BASE_OUTPUT_DIR/"
echo ""
echo "下一步："
echo "  1. 运行评估: $PYTHON scripts/evaluate.py --results_dir $BASE_OUTPUT_DIR"
echo "  2. 生成图表: $PYTHON scripts/visualize.py --results_dir $BASE_OUTPUT_DIR"
echo "============================================"
