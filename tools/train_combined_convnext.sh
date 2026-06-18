#!/bin/bash
# ConvNeXt-B → ConvNeXt-L 통합 데이터셋(dacl + S2DS) 순차 학습 + 평가
#
# Usage:
#   nohup bash tools/train_combined_convnext.sh > logs_combined_train.log 2>&1 &
#   disown
#
# 모니터링:
#   tail -f logs_combined_train.log

set -e

WORKDIR="/workspace/minkyung/Dron/mmsegmentation"
PYTHON="/workspace/sangrak/anaconda3/envs/mmseg/bin/python"

# GPU 여유 최소 요구사항 (ConvNeXt는 가벼움)
MIN_FREE_MB=8000

# 평가 데이터셋 경로
DACL_TEST_IMG="/workspace/nas_200/minkyung/dacl_s2ds_combined/test/images"
DACL_TEST_GT="/workspace/nas_200/minkyung/dacl_s2ds_combined/test/masks"
S2DS_TEST_IMG="/workspace/nas_200/minkyung/dacl_s2ds_combined/s2ds_test/images"
S2DS_TEST_GT="/workspace/nas_200/minkyung/dacl_s2ds_combined/s2ds_test/masks"

cd "$WORKDIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_gpu() {
    local need_mb=$1
    local label=$2
    log "Waiting for GPU... need ${need_mb} MB free for $label"
    while true; do
        local free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$free_mb" ] && [ "$free_mb" -ge "$need_mb" ]; then
            log "  GPU free: ${free_mb} MB >= ${need_mb} MB. Starting!"
            return 0
        fi
        sleep 300
    done
}

train_and_eval() {
    local name=$1
    local config=$2
    local workdir_name=$3

    log "========================================"
    log "  $name 학습 시작"
    log "========================================"

    wait_for_gpu $MIN_FREE_MB "$name"

    # 학습
    $PYTHON tools/train.py "$config" \
        --work-dir "work_dirs/${workdir_name}" \
        2>&1 | tee -a "logs_${workdir_name}_train.log"

    log "  $name 학습 완료"

    # 마지막 체크포인트 찾기 (best 우선, 없으면 last)
    local ckpt
    ckpt=$(ls "work_dirs/${workdir_name}"/best_mIoU_iter_*.pth 2>/dev/null | tail -1)
    if [ -z "$ckpt" ]; then
        ckpt="work_dirs/${workdir_name}/iter_40000.pth"
    fi
    log "  사용할 checkpoint: $ckpt"

    # GPU 잠시 대기 (메모리 정리)
    sleep 30

    # === In-domain 평가 (dacl test) ===
    log "  -> dacl test 평가 (in-domain)"
    $PYTHON tools/inference_dacl_efflorescence.py \
        --config "$config" \
        --checkpoint "$ckpt" \
        --input "$DACL_TEST_IMG" \
        --gt-dir "$DACL_TEST_GT" \
        --output "results_${workdir_name}_dacl_test" \
        --compute-metrics 2>&1 | tee -a "logs_${workdir_name}_eval.log" | tail -20

    # === Cross-domain 평가 (S2DS test) ===
    log "  -> S2DS test 평가 (cross-domain)"
    $PYTHON tools/inference_dacl_efflorescence.py \
        --config "$config" \
        --checkpoint "$ckpt" \
        --input "$S2DS_TEST_IMG" \
        --gt-dir "$S2DS_TEST_GT" \
        --output "results_${workdir_name}_s2ds_test" \
        --compute-metrics 2>&1 | tee -a "logs_${workdir_name}_eval.log" | tail -20

    log "  $name 평가 완료"
}

# === Step 1: ConvNeXt-B (작고 빠름) ===
train_and_eval \
    "ConvNeXt-B (Combined)" \
    "configs/convnext/convnext-base_upernet_40k_combined-efflorescence-512x512.py" \
    "upernet_convnext-base_combined_efflorescence"

# === Step 2: ConvNeXt-L (큰 모델) ===
train_and_eval \
    "ConvNeXt-L (Combined)" \
    "configs/convnext/convnext-large_upernet_40k_combined-efflorescence-512x512.py" \
    "upernet_convnext-large_combined_efflorescence"

log "========================================"
log "  모든 학습 + 평가 완료"
log "========================================"

echo ""
echo "=== 최종 결과 요약 ==="
for name in convnext-base convnext-large; do
    for ds in dacl s2ds; do
        d="results_upernet_${name}_combined_efflorescence_${ds}_test"
        if [ -f "$d/metrics.txt" ]; then
            echo ""
            echo "### $name - ${ds} test ###"
            tail -10 "$d/metrics.txt"
        fi
    done
done
