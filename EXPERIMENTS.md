# Experiments: Efflorescence Segmentation

dacl10k_v2_devphase 데이터셋의 **백태(Efflorescence)** 단일 클래스 segmentation 실험.

## 1. 개요

| 항목 | 내용 |
|---|---|
| **Task** | Binary semantic segmentation (background vs Efflorescence) |
| **Architectures** | UPerNet + ConvNeXt (B / L), Mask2Former + Swin (B / L) |
| **Framework** | mmsegmentation 1.2.2 + mmpretrain 1.2.0 + mmdet 3.3.0 |
| **GPU** | NVIDIA RTX A5000 (24GB) |

---

## 2. 데이터셋

### dacl10k_v2_devphase / efflorescence

dacl10k v2 dev phase의 efflorescence 단일 카테고리 데이터.

| Split | 이미지 수 | 출처 |
|---|---|---|
| **train** | **1,371** | dacl train의 90% |
| **val** | **152** | dacl train의 10% |
| **test** | **206** | dacl validation |

### 전처리

- 원본 마스크: 0/255 (binary)
- 변환: 0 (background) / 1 (defect)
- 변환 스크립트: [tools/convert_dacl_efflorescence_mask.py](tools/convert_dacl_efflorescence_mask.py)

### 이미지 특성

- 가변 해상도 (600×800 ~ 1600×1200 등 76 종류)
- Efflorescence 픽셀 비율: 평균 ~3-7% (클래스 불균형 존재)

---

## 3. 학습 설정

### 공통 설정

| 항목 | 값 |
|---|---|
| Decoder | UPerNet (with auxiliary head FCNHead) |
| Pretrain | ImageNet-22K (ConvNeXt 자체) |
| Optimizer | AdamW (lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05) |
| Mixed Precision | AmpOptimWrapper |
| LR Scheduler | LinearLR warmup (1,500 iter) → PolyLR (power=1.0) |
| Loss | CrossEntropyLoss (binary) |
| crop_size | 512 × 512 |
| Augmentation | RandomResize(0.5~2.0) + RandomCrop + RandomFlip + PhotoMetricDistortion |
| batch_size | 2 |
| Total Iterations | 40,000 |
| Val Interval | 4,000 iter |
| Test Inference Mode | Slide window (stride 341) |

### 모델별 차이

| 항목 | ConvNeXt-B | ConvNeXt-L |
|---|---|---|
| Params | 89M | 198M |
| embed_dims | [128, 256, 512, 1024] | [192, 384, 768, 1536] |
| decode_head channels | 512 | 768 |
| **VRAM (peak)** | ~5.0 GB | ~7.4 GB |
| **iter당 시간** | ~0.29s | ~0.39s |
| **총 학습 시간** | 약 3시간 | 약 3.5시간 |

---

## 4. 학습 결과 (Validation)

### Validation mIoU 추이

| iter | ConvNeXt-B | ConvNeXt-L |
|---|---|---|
| 4K | 60.79 | 65.18 |
| 8K | 66.76 | 66.08 |
| 12K | 67.96 | 67.13 |
| 16K | 69.41 | 68.82 |
| **20K** | 68.26 | **70.27** ⭐ |
| 24K | 67.96 | 68.31 |
| 28K | 68.36 | 68.49 |
| 32K | 70.07 | 69.30 |
| 36K | 70.06 | 68.97 |
| **40K** | **70.15** ⭐ | 70.30 |

### Best Checkpoint 선택

| Model | Best Iter | Best Val mIoU |
|---|---|---|
| **ConvNeXt-B** | iter_40000 | **70.15** |
| **ConvNeXt-L** | iter_20000 | **70.27** |

> Note: ConvNeXt-L은 20K iter에 빠르게 수렴, ConvNeXt-B는 학습 끝까지 꾸준한 향상.

---

## 5. 테스트 결과 (Test 206장)

### ConvNeXt-B (iter_40000)

| Class | IoU | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| background | 95.19 | 97.54 | 96.49 | 98.61 | - |
| **Efflorescence** | **55.80** | **71.63** | **81.91** | **63.64** | **1.39** |

- **mIoU**: **75.50**
- **mF1**: **84.58**

### ConvNeXt-L (iter_20000)

| Class | IoU | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| background | 94.73 | 97.29 | 96.05 | 98.58 | - |
| **Efflorescence** | **51.51** | **67.99** | **80.35** | **58.93** | **1.42** |

- **mIoU**: **73.12**
- **mF1**: **82.64**

---

## 6. 모델 비교

| 지표 | ConvNeXt-B | ConvNeXt-L | 차이 |
|---|---|---|---|
| Val mIoU (Best) | 70.15 | 70.27 | +0.12 (L) |
| **Test mIoU** | **75.50** | 73.12 | **+2.38 (B)** |
| **Test Efflorescence IoU** | **55.80** | 51.51 | **+4.29 (B)** |
| Test Recall | 63.64 | 58.93 | +4.71 (B) |
| Params | 89M | 198M | L is 2.2× larger |
| 학습 시간 | ~3시간 | ~3.5시간 | L is slower |

### 핵심 관찰

**ConvNeXt-B가 ConvNeXt-L보다 test 성능이 더 좋아요!**

- **Val에서는 두 모델이 거의 동등** (B: 70.15 / L: 70.27)
- **Test에서는 B가 우세** (B: 75.50 / L: 73.12)

추정 원인:
1. **데이터 부족**: 1,371장 train으로는 198M 파라미터 모델(L)이 일반화하기에 부족
2. **과적합 가능성**: ConvNeXt-L이 val에는 잘 맞지만 test 분포에 덜 일반화
3. **Best iter 차이**: L은 20K에서 best (early stop 효과), B는 40K까지 꾸준히 학습

### Val vs Test 차이

| Model | Val mIoU | Test mIoU | 차이 |
|---|---|---|---|
| ConvNeXt-B | 70.15 | 75.50 | **+5.35** |
| ConvNeXt-L | 70.27 | 73.12 | +2.85 |

→ Test 데이터(dacl validation)가 val(train의 10% split)보다 더 쉬운 케이스일 가능성. 또는 분포 차이.

---

## 7. 정성적 결과

각 이미지는 `[원본] [GT] [예측]` 가로 배치 시각화.

### ConvNeXt-B 샘플

**Sample 0012**
![ConvNeXt-B sample 0012](docs/images/convnext_b/dacl10k_v2_validation_0012_compare.jpg)

**Sample 0152**
![ConvNeXt-B sample 0152](docs/images/convnext_b/dacl10k_v2_validation_0152_compare.jpg)

**Sample 0268**
![ConvNeXt-B sample 0268](docs/images/convnext_b/dacl10k_v2_validation_0268_compare.jpg)

**Sample 0453**
![ConvNeXt-B sample 0453](docs/images/convnext_b/dacl10k_v2_validation_0453_compare.jpg)

**Sample 0810**
![ConvNeXt-B sample 0810](docs/images/convnext_b/dacl10k_v2_validation_0810_compare.jpg)

---

### ConvNeXt-L 샘플

**Sample 0012**
![ConvNeXt-L sample 0012](docs/images/convnext_l/dacl10k_v2_validation_0012_compare.jpg)

**Sample 0152**
![ConvNeXt-L sample 0152](docs/images/convnext_l/dacl10k_v2_validation_0152_compare.jpg)

**Sample 0268**
![ConvNeXt-L sample 0268](docs/images/convnext_l/dacl10k_v2_validation_0268_compare.jpg)

**Sample 0453**
![ConvNeXt-L sample 0453](docs/images/convnext_l/dacl10k_v2_validation_0453_compare.jpg)

**Sample 0810**
![ConvNeXt-L sample 0810](docs/images/convnext_l/dacl10k_v2_validation_0810_compare.jpg)

전체 206장 시각화 결과는 인퍼런스 스크립트로 재생성 가능 (아래 참고).

---

## 8. 재현 가이드

### 1) 마스크 변환

```bash
python tools/convert_dacl_efflorescence_mask.py \
    --data-root /path/to/efflorescence \
    --out-root /path/to/efflorescence_binary
```

### 2) ConvNeXt pretrained 다운로드

```bash
mkdir -p checkpoints
wget -P checkpoints/ \
    "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth"
wget -P checkpoints/ \
    "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth"
```

### 3) 학습

```bash
# ConvNeXt-B
python tools/train.py \
    configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py \
    --work-dir work_dirs/upernet_convnext-base_dacl_efflorescence

# ConvNeXt-L
python tools/train.py \
    configs/convnext/convnext-large_upernet_40k_dacl-efflorescence-512x512.py \
    --work-dir work_dirs/upernet_convnext-large_dacl_efflorescence
```

### 4) Test 평가 + 시각화

```bash
# ConvNeXt-B
python tools/inference_dacl_efflorescence.py \
    --config configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py \
    --checkpoint work_dirs/upernet_convnext-base_dacl_efflorescence/iter_40000.pth \
    --input /path/to/efflorescence_binary/test/images \
    --gt-dir /path/to/efflorescence_binary/test/masks \
    --output results_convnext_b_test/ \
    --compute-metrics

# ConvNeXt-L
python tools/inference_dacl_efflorescence.py \
    --config configs/convnext/convnext-large_upernet_40k_dacl-efflorescence-512x512.py \
    --checkpoint work_dirs/upernet_convnext-large_dacl_efflorescence/iter_20000.pth \
    --input /path/to/efflorescence_binary/test/images \
    --gt-dir /path/to/efflorescence_binary/test/masks \
    --output results_convnext_l_test/ \
    --compute-metrics
```

---

## 9. 결론

| 항목 | 추천 |
|---|---|
| **최종 모델** | **ConvNeXt-B (iter_40000)** |
| **이유** | 더 작은 모델로 test에서 더 좋은 성능, 학습 시간도 짧음 |
| **Test 성능** | mIoU 75.50, Defect IoU 55.80, F1 71.63 |

### 향후 개선 방향

| 방향 | 기대 효과 |
|---|---|
| **Class weight 적용** | Recall 향상 (현재 63.64%) |
| **Dice Loss 추가** | IoU 직접 최적화 |
| **더 강한 augmentation** | 작은 데이터셋 일반화 향상 |
| **다른 backbone 비교** | Swin-T/B, MiT-B5 등 |
| **Multi-scale inference** | Test 성능 추가 향상 가능 |

---

## 10. Mask2Former 추가 실험 (계획)

ConvNeXt 외에 **Mask2Former + Swin Transformer (B/L)** 추가 실험을 위한 config 작성.

### 설정 비교

| 항목 | UPerNet + ConvNeXt | Mask2Former + Swin |
|---|---|---|
| Decoder | UPerNet (FPN-like) | Mask2Former (Transformer decoder, 9 layers) |
| Loss | CrossEntropyLoss | CE + Dice + Mask (cls/mask/dice on 9 decoder outputs) |
| Inference | Standard | Slide window |

### 실측 자원 사용량

| 모델 | batch_size | VRAM (peak) | iter당 시간 | 40K iter 예상 |
|---|---|---|---|---|
| Mask2Former + Swin-B | 2 | ~23.5 GB | 2.21 s | ~25시간 |
| Mask2Former + Swin-L (with_cp) | 1 | ~21.3 GB | 3.05 s | ~34시간 |

> Swin-L은 메모리 제약으로 `with_cp=True` (gradient checkpointing) 적용.
> Mask2Former는 9개 decoder layer × 3개 loss로 매우 무거움 (ConvNeXt 대비 ~7배 느림).

### 추가 의존성

```bash
# mmdet 설치 필요
pip install mmdet
# 그 후 __init__.py 수정 (mmcv 2.2.0 호환 허용)

# mmcv는 우리 torch 환경(2.9.1+cu128)에 맞춰 source build 필요
git clone -b v2.2.0 https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.6' pip install -e .
```

### 자동 트리거 학습 스크립트

GPU 여유 생기면 Swin-B → Swin-L 순차 학습:
```bash
nohup bash tools/auto_train_mask2former.sh > logs_auto_train.log 2>&1 &
```

---

## 부록

### Config 파일 위치

- ConvNeXt-B: [configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py](configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py)
- ConvNeXt-L: [configs/convnext/convnext-large_upernet_40k_dacl-efflorescence-512x512.py](configs/convnext/convnext-large_upernet_40k_dacl-efflorescence-512x512.py)
- Mask2Former + Swin-B: [configs/mask2former/mask2former_swin-b_40k_dacl-efflorescence-512x512.py](configs/mask2former/mask2former_swin-b_40k_dacl-efflorescence-512x512.py)
- Mask2Former + Swin-L: [configs/mask2former/mask2former_swin-l_40k_dacl-efflorescence-512x512.py](configs/mask2former/mask2former_swin-l_40k_dacl-efflorescence-512x512.py)

### Dataset Config

- [configs/_base_/datasets/dacl_efflorescence.py](configs/_base_/datasets/dacl_efflorescence.py)
- [mmseg_custom/datasets/dacl_efflorescence.py](mmseg_custom/datasets/dacl_efflorescence.py)

### Raw Metrics

- [docs/metrics/convnext_b_test.txt](docs/metrics/convnext_b_test.txt)
- [docs/metrics/convnext_l_test.txt](docs/metrics/convnext_l_test.txt)

### Tools

- [tools/auto_train_mask2former.sh](tools/auto_train_mask2former.sh) — GPU 여유 자동 감지 + 순차 학습
- [tools/inference_dacl_efflorescence.py](tools/inference_dacl_efflorescence.py) — Inference + metric 계산
- [tools/combine_dacl_s2ds.py](tools/combine_dacl_s2ds.py) — dacl + S2DS 데이터 통합 (symlink)
- [tools/train_combined_convnext.sh](tools/train_combined_convnext.sh) — Combined 데이터셋으로 ConvNeXt-B/L 순차 학습

---

## 11. Mask2Former + InternImage-L 실험

다른 워크스테이션에서 학습한 결과 통합.

### 설정

| 항목 | 값 |
|---|---|
| Backbone | InternImage-L (`with_cp=True`) |
| Decoder | Mask2Former |
| batch_size | 1 |
| max_iters | 40,000 |
| Best Iter | **16K** (가장 빠른 수렴) |

### Test 결과 (dacl, 206장)

| Class | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| background | 94.87 | 97.37 | 96.80 | 97.94 |
| **Efflorescence** | **55.65** | **71.51** | **76.37** | **67.23** |

- mIoU: **75.26**
- mF1: **84.44**

### 학습 효율

- Best mIoU 도달이 16K iter로 다른 모델 대비 가장 빠름
- VRAM 사용: ~7.7 GB (batch=1, with_cp)
- DCNv3 컴파일 필요 (설치 복잡)

### Config

- [configs/mask2former/mask2former_internimage-l_40k_dacl-efflorescence-512x512.py](configs/mask2former/mask2former_internimage-l_40k_dacl-efflorescence-512x512.py)

---

## 12. S2DS 외부 데이터 평가 (Cross-domain Zero-shot)

학습에 안 쓴 외부 데이터셋(S2DS test 93장)에서 5개 모델 평가.

### S2DS 데이터셋

- 출처: Benz & Rodehorst 2022, DAGM GCPR
- 7-class (background + crack, spalling, corrosion, **efflorescence**, vegetation, control point)
- 평가용으로 efflorescence만 binary 변환 (Cyan BGR(255,255,0) → 1)

### Cross-domain 결과 (S2DS test, Efflorescence 클래스)

| 모델 | Eff IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| UPerNet + ConvNeXt-B | 53.21 | 69.46 | 53.32 | 99.61 |
| **UPerNet + ConvNeXt-L** | **58.60** ⭐ | **73.89** ⭐ | **58.70** ⭐ | 99.70 |
| Mask2Former + Swin-B | 46.19 | 63.19 | 46.28 | 99.59 |
| Mask2Former + Swin-L | 40.76 | 57.91 | 40.82 | 99.62 |
| Mask2Former + InternImage-L | 46.21 | 63.21 | 46.23 | **99.88** ⭐ |

### dacl vs S2DS 변화 (도메인 일반화)

| 모델 | dacl Eff IoU | S2DS Eff IoU | 변화 |
|---|---|---|---|
| ConvNeXt-B | 55.80 | 53.21 | -2.59 (안정) |
| **ConvNeXt-L** | 51.51 | **58.60** | **+7.09** 🚀 |
| Mask2Former Swin-B | 58.50 | 46.19 | -12.31 |
| Mask2Former Swin-L | 56.77 | 40.76 | -16.01 |
| Mask2Former InternImage-L | 55.65 | 46.21 | -9.44 |

### 관찰

- **ConvNeXt-L**: 유일하게 외부 도메인에서 **향상** — 일반화 능력 가장 우수
- **Mask2Former 계열**: 도메인 shift에 취약 (dacl에 일부 과적합 가능성)
- **모든 모델 Recall 99%+**: 백태 영역을 거의 다 잡아내지만 Precision 차이 큼

### 결론

| 평가 | 1위 |
|---|---|
| In-domain (dacl) | Mask2Former + Swin-B (mIoU 76.83) |
| **Cross-domain (S2DS)** | **UPerNet + ConvNeXt-L (mIoU 78.76)** ⭐ |
| **종합 (두 도메인 평균)** | **UPerNet + ConvNeXt-L** |

---

## 13. Combined 데이터셋 학습 (계획)

베이스라인 강화를 위해 **dacl + S2DS 통합 학습** 진행 예정 (실증 데이터 파인튜닝 전 단계).

### 데이터 통합

```
dacl_s2ds_combined/
├── train/        2,021장 (dacl train 1,371 + S2DS train 563 + S2DS val 87)
├── val/          152장 (dacl val - best checkpoint 선택용)
├── test/         206장 (dacl test - in-domain 평가)
└── s2ds_test/    93장 (S2DS test - cross-domain 평가, 학습 안 됨)
```

### 학습 계획

- Phase 1: UPerNet + ConvNeXt-B (~3시간)
- Phase 2: UPerNet + ConvNeXt-L (~5시간)
- 각자 학습 후 dacl test + S2DS test 평가

### Config

- [configs/convnext/convnext-base_upernet_40k_combined-efflorescence-512x512.py](configs/convnext/convnext-base_upernet_40k_combined-efflorescence-512x512.py)
- [configs/convnext/convnext-large_upernet_40k_combined-efflorescence-512x512.py](configs/convnext/convnext-large_upernet_40k_combined-efflorescence-512x512.py)

### Base dataset config

- [configs/_base_/datasets/combined_efflorescence.py](configs/_base_/datasets/combined_efflorescence.py)

### 실행 스크립트

```bash
nohup bash tools/train_combined_convnext.sh > logs_combined_train.log 2>&1 &
disown
```
