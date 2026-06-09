#!/bin/bash
# Train one SegFormer model per category (binary classification).
# Sequential training to avoid GPU contention.
#
# Usage:
#     bash tools/train_per_category.sh
#
# Resume:
#     bash tools/train_per_category.sh --resume

set -e

PYTHON=/workspace/sangrak/anaconda3/envs/mmseg/bin/python
RESUME_FLAG=""
if [ "$1" == "--resume" ]; then
    RESUME_FLAG="--resume"
fi

CATEGORIES=("crack" "corrosion" "efflorescence" "peeling")

for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Training: ${cat} (binary)"
    echo "  Started: $(date)"
    echo "================================================================"

    CONFIG="configs/segformer/binary/segformer_mit-b0_8xb2-40k_bridge-${cat}-binary-512x512.py"
    WORK_DIR="work_dirs/segformer_b0_bridge_${cat}_binary"

    $PYTHON tools/train.py "$CONFIG" --work-dir "$WORK_DIR" $RESUME_FLAG

    echo ""
    echo "  ${cat} finished: $(date)"
done

echo ""
echo "================================================================"
echo "  All 4 per-category models trained."
echo "================================================================"
