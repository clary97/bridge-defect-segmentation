# Bridge Defect Segmentation

mmsegmentation 기반 교량 결함 세그멘테이션 실험 프로젝트.

## 데이터셋

- **AI Hub 교량 결함 데이터**: ConcreteCrack, SteelDefect, Efflorescence, PaintDamage
- **dacl10k**: Crack, Rust, Efflorescence, Spalling
- **dacl10k v2 (efflorescence)**: 백태 단일 클래스

## 실험한 모델들

| 모델 | Backbone | 데이터셋 | 비고 |
|---|---|---|---|
| SegFormer | MIT-B0, B5 | AI Hub (multi-class) | Baseline / CW / Dice 조합 |
| SegFormer | MIT-B0 | AI Hub (binary, per-category) | 카테고리별 1개 모델 |
| SegFormer | MIT-B0 | dacl10k_4cate | Multi-class |
| DeepLabV3+ | ResNet101 | AI Hub | Baseline / CW / Dice |
| UPerNet | ConvNeXt-B/L | dacl10k v2 efflorescence | (예정) |

상세 결과는 [EXPERIMENTS.md](EXPERIMENTS.md) 참고.

## 설치

[INSTALL.md](INSTALL.md) 참고.

## 빠른 시작

### 1. mmsegmentation에 커스텀 파일 추가

```bash
# mmseg/datasets/에 추가
cp mmseg_custom/datasets/*.py /path/to/mmsegmentation/mmseg/datasets/

# mmseg/evaluation/metrics/에 추가
cp mmseg_custom/evaluation/metrics/*.py /path/to/mmsegmentation/mmseg/evaluation/metrics/

# __init__.py에 등록 (수동 또는 patch)
```

### 2. 데이터셋 mask 변환

```bash
# AI Hub: JSON polygon → 5-class mask
python tools/convert_json_to_mask.py --data-root /path/to/data --splits train val

# 5-class → binary mask (per category)
python tools/convert_to_binary_mask.py --data-root /path/to/data --splits train val

# dacl10k: JSON → mask
python tools/convert_dacl10k_to_mask.py --data-root /path/to/dacl10k_4cate --splits train validation

# dacl10k v2 efflorescence: 0/255 → 0/1
python tools/convert_dacl_efflorescence_mask.py --data-root /path/to/efflorescence --out-root /path/to/efflorescence_binary
```

### 3. 학습 실행

```bash
# SegFormer baseline
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-40k_bridge-defect-512x512.py

# UPerNet + ConvNeXt-B
python tools/train.py configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py
```

## 디렉토리 구조

```
configs/         # 학습 config들
mmseg_custom/    # mmsegmentation에 추가할 파일들
tools/           # 데이터 변환 / 평가 / 시각화 스크립트
```

## License

mmsegmentation 라이선스 (Apache 2.0)를 따릅니다.
