#!/bin/bash
# Per-category SegFormer binary 모델의 Best mIoU checkpoint로 시각화 이미지 생성
#
# Usage:
#     bash tools/visualize_per_category.sh

set -e

PYTHON_BIN="/workspace/sangrak/anaconda3/envs/mmseg/bin/python"
CONFIG_DIR="configs/segformer/binary"
WORK_DIR_ROOT="work_dirs"

# 각 카테고리의 Best mIoU 시점 (이전 학습 로그 분석 결과)
declare -A BEST_ITERS=(
    ["crack"]="24000"
    ["corrosion"]="8000"
    ["efflorescence"]="12000"
    ["peeling"]="36000"
)

CATEGORIES=("crack" "corrosion" "efflorescence" "peeling")

echo "============================================================"
echo "  Per-category SegFormer Visualization"
echo "============================================================"

for cat in "${CATEGORIES[@]}"; do
    iter=${BEST_ITERS[$cat]}
    config="${CONFIG_DIR}/segformer_mit-b0_8xb2-40k_bridge-${cat}-binary-512x512.py"
    checkpoint="${WORK_DIR_ROOT}/segformer_b0_bridge_${cat}_binary/iter_${iter}.pth"
    show_dir="vis_${cat}_best_${iter}"

    echo ""
    echo "============================================================"
    echo "  Category: ${cat} | Best Iter: ${iter}"
    echo "============================================================"

    if [ ! -f "$checkpoint" ]; then
        echo "  [SKIP] Checkpoint not found: $checkpoint"
        continue
    fi

    if [ ! -f "$config" ]; then
        echo "  [SKIP] Config not found: $config"
        continue
    fi

    $PYTHON_BIN tools/test.py \
        "$config" \
        "$checkpoint" \
        --show-dir "$show_dir" \
        --cfg-options val_evaluator.type=TPRFPRMetric \
                      val_evaluator.iou_metrics="['mIoU','mFscore']" \
                      test_evaluator.type=TPRFPRMetric \
                      test_evaluator.iou_metrics="['mIoU','mFscore']"

    echo ""
    echo "  [DONE] ${cat} → saved to ${WORK_DIR_ROOT}/segformer_b0_bridge_${cat}_binary/<timestamp>/${show_dir}/"
done

echo ""
echo "============================================================"
echo "  All visualizations completed!"
echo "============================================================"

# 결과 저장 위치 안내
echo ""
echo "Result locations:"
for cat in "${CATEGORIES[@]}"; do
    iter=${BEST_ITERS[$cat]}
    latest=$(ls -td "${WORK_DIR_ROOT}/segformer_b0_bridge_${cat}_binary"/2026* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "  [${cat}] ${latest}/vis_${cat}_best_${iter}/"
    fi
done
