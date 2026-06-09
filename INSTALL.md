# 설치 가이드

## 환경

- Python 3.10
- PyTorch 2.x with CUDA
- NVIDIA GPU (24GB VRAM 권장, 일부 모델은 더 적게도 가능)

## 1. mmsegmentation 설치

```bash
# Conda 환경 생성
conda create -n mmseg python=3.10 -y
conda activate mmseg

# PyTorch (CUDA 12.x 기준)
pip install torch torchvision

# OpenMMLab
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html

# mmsegmentation
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .

# ConvNeXt 사용 시 추가
pip install mmpretrain
```

## 2. 이 레포의 커스텀 파일 추가

```bash
# 이 레포 클론
git clone https://github.com/<your-username>/bridge-defect-segmentation.git
cd bridge-defect-segmentation

# 환경변수 (mmsegmentation 경로)
export MMSEG_ROOT=/path/to/mmsegmentation

# Custom datasets 복사
cp mmseg_custom/datasets/*.py $MMSEG_ROOT/mmseg/datasets/

# Custom metric 복사
cp mmseg_custom/evaluation/metrics/tpr_fpr_metric.py $MMSEG_ROOT/mmseg/evaluation/metrics/

# Configs 복사
cp -r configs/_base_/datasets/*.py $MMSEG_ROOT/configs/_base_/datasets/
cp -r configs/segformer/* $MMSEG_ROOT/configs/segformer/
cp -r configs/deeplabv3plus/* $MMSEG_ROOT/configs/deeplabv3plus/
cp -r configs/convnext/* $MMSEG_ROOT/configs/convnext/

# Tools 복사
cp tools/* $MMSEG_ROOT/tools/
```

## 3. __init__.py 수정 (수동 필요)

### `mmsegmentation/mmseg/datasets/__init__.py` 에 다음 추가:

```python
from .bridge_defect import BridgeDefectDataset
from .bridge_defect_binary import BridgeDefectBinaryDataset
from .dacl10k import Dacl10kDataset
from .dacl_efflorescence import DaclEfflorescenceDataset

__all__ = [
    # ... 기존 항목들 ...
    'BridgeDefectDataset',
    'BridgeDefectBinaryDataset',
    'Dacl10kDataset',
    'DaclEfflorescenceDataset',
]
```

### `mmsegmentation/mmseg/evaluation/metrics/__init__.py` 에 다음 추가:

```python
from .tpr_fpr_metric import TPRFPRMetric

__all__ = [
    # ... 기존 항목들 ...
    'TPRFPRMetric',
]
```

## 4. (선택) 버그 패치

mmsegmentation의 `class_weight + ignore_index` 조합 버그 수정.

`mmsegmentation/mmseg/models/losses/cross_entropy_loss.py` line ~66 부분:

**기존:**
```python
label_weights = torch.stack([class_weight[cls] for cls in label
                             ]).to(device=class_weight.device)
```

**수정:**
```python
valid_mask = (label != ignore_index)
safe_label = label.clone()
safe_label[~valid_mask] = 0
label_weights = class_weight[safe_label].to(
    device=class_weight.device)
label_weights[~valid_mask] = 0
```

상세한 설명은 `mmseg_custom/PATCHES.md` 참고.

## 5. 동작 확인

```bash
cd $MMSEG_ROOT
python -c "from mmseg.datasets import BridgeDefectDataset; print('OK')"
python -c "from mmseg.evaluation.metrics import TPRFPRMetric; print('OK')"
```
