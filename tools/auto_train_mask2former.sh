#!/bin/bash
# GPU 여유가 생기면 자동으로 Mask2Former + Swin-B → Swin-L 순차 학습
#
# Usage:
#     nohup bash tools/auto_train_mask2former.sh > logs_auto_train.log 2>&1 &
#     disown
#
# 모니터링:
#     tail -f logs_auto_train.log

set -e

WORKDIR="/workspace/minkyung/Dron/mmsegmentation"
PYTHON="/workspace/sangrak/anaconda3/envs/mmseg/bin/python"

# 실측치 (Swin-B ~23.5GB peak, Swin-L ~21.3GB peak)
# 24GB GPU에서는 다른 작업 없을 때만 가능
MIN_FREE_MB_SWIN_B=22000
MIN_FREE_MB_SWIN_L=22000

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
        # 5분마다 확인
        sleep 300
    done
}

# === Swin-B 학습 ===
log "====== Step 1: Mask2Former + Swin-B ======"
wait_for_gpu $MIN_FREE_MB_SWIN_B "Swin-B"

log "Starting Swin-B training..."
$PYTHON tools/train.py \
    configs/mask2former/mask2former_swin-b_40k_dacl-efflorescence-512x512.py \
    --work-dir work_dirs/mask2former_swin-b_dacl_efflorescence \
    2>&1 | tee -a logs_swin_b_train.log

SWIN_B_EXIT=${PIPESTATUS[0]}
if [ $SWIN_B_EXIT -eq 0 ]; then
    log "Swin-B training finished successfully."
else
    log "Swin-B training FAILED (exit code $SWIN_B_EXIT). Continuing to Swin-L anyway."
fi

# 잠시 대기 (GPU 메모리 정리 시간)
sleep 30

# === Swin-L 학습 ===
log "====== Step 2: Mask2Former + Swin-L ======"
wait_for_gpu $MIN_FREE_MB_SWIN_L "Swin-L"

log "Starting Swin-L training..."
$PYTHON tools/train.py \
    configs/mask2former/mask2former_swin-l_40k_dacl-efflorescence-512x512.py \
    --work-dir work_dirs/mask2former_swin-l_dacl_efflorescence \
    2>&1 | tee -a logs_swin_l_train.log

SWIN_L_EXIT=${PIPESTATUS[0]}
if [ $SWIN_L_EXIT -eq 0 ]; then
    log "Swin-L training finished successfully."
else
    log "Swin-L training FAILED (exit code $SWIN_L_EXIT)."
fi

log "====== All done ======"
