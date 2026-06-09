"""
Per-category 4 binary 모델을 앙상블해서 5-class 평가.

각 픽셀에 대해:
    1. 4개 binary 모델 모두 추론 → defect probability 추출
    2. max(p_defect_i) > threshold면 그 카테고리, 아니면 background
    3. 5-class GT (masks/val/{cat}/*.png)와 비교
    4. mIoU, F1, TPR, FPR 계산

Usage:
    python tools/per_category_ensemble_eval.py \
        --data-root /workspace/nas_192/datasets/public/AI_Hub_bridge_defect_data \
        --device cuda:0 \
        --threshold 0.5

Class mapping:
    0 = background
    1 = ConcreteCrack (crack)
    2 = SteelDefect (corrosion)
    3 = Efflorescene (efflorescence)
    4 = PaintDamage (peeling)
"""

import argparse
import os
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from prettytable import PrettyTable

from mmseg.apis import init_model, inference_model

# config_path, checkpoint_path, gt_class_index
MODELS = {
    1: {  # ConcreteCrack
        'name': 'crack',
        'config': 'configs/segformer/binary/segformer_mit-b0_8xb2-40k_bridge-crack-binary-512x512.py',
        'checkpoint': 'work_dirs/segformer_b0_bridge_crack_binary/iter_24000.pth',
    },
    2: {  # SteelDefect
        'name': 'corrosion',
        'config': 'configs/segformer/binary/segformer_mit-b0_8xb2-40k_bridge-corrosion-binary-512x512.py',
        'checkpoint': 'work_dirs/segformer_b0_bridge_corrosion_binary/iter_8000.pth',
    },
    3: {  # Efflorescence
        'name': 'efflorescence',
        'config': 'configs/segformer/binary/segformer_mit-b0_8xb2-40k_bridge-efflorescence-binary-512x512.py',
        'checkpoint': 'work_dirs/segformer_b0_bridge_efflorescence_binary/iter_12000.pth',
    },
    4: {  # PaintDamage
        'name': 'peeling',
        'config': 'configs/segformer/binary/segformer_mit-b0_8xb2-40k_bridge-peeling-binary-512x512.py',
        'checkpoint': 'work_dirs/segformer_b0_bridge_peeling_binary/iter_36000.pth',
    },
}

CLASS_NAMES = ('background', 'ConcreteCrack', 'SteelDefect',
               'Efflorescene', 'PaintDamage')
NUM_CLASSES = 5

# Val image folder name (under images/val/{folder})
CATEGORY_FOLDERS = ['crack', 'corrosion', 'efflorescence', 'peeling']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (contains images/val and masks/val)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Defect probability threshold (default: 0.5)')
    parser.add_argument('--output', default='ensemble_eval_results.txt',
                        help='Output file for results')
    return parser.parse_args()


def get_defect_probability(model, img_path, device):
    """Run binary model inference and return P(defect) at original image resolution.

    Returns:
        np.ndarray of shape (H, W): probability of defect (class 1)
    """
    result = inference_model(model, img_path)
    # seg_logits: (num_classes, H, W) - raw logits BEFORE softmax
    logits = result.seg_logits.data  # tensor (2, H, W)
    probs = F.softmax(logits, dim=0)  # (2, H, W), sums to 1 along dim=0
    defect_prob = probs[1].cpu().numpy()  # P(class=1, defect)
    return defect_prob


def ensemble_predict(models, img_path, device, threshold=0.5):
    """4개 모델 추론 → 5-class 예측 통합.

    Returns:
        pred: np.ndarray of shape (H, W), values in {0, 1, 2, 3, 4}
    """
    defect_probs = []
    for class_idx in [1, 2, 3, 4]:
        p = get_defect_probability(models[class_idx], img_path, device)
        defect_probs.append(p)

    # stack: (4, H, W) → 각 픽셀당 [p_crack, p_corrosion, p_effl, p_paint]
    stacked = np.stack(defect_probs, axis=0)
    max_prob = stacked.max(axis=0)  # (H, W)
    max_idx = stacked.argmax(axis=0)  # (H, W), values in {0,1,2,3} → classes {1,2,3,4}

    # threshold 넘으면 그 카테고리, 안 넘으면 background
    pred = np.where(max_prob > threshold, max_idx + 1, 0).astype(np.int64)
    return pred


def compute_intersection_union(pred, gt, num_classes, ignore_index=255):
    """각 클래스의 intersection, union, pred_total, gt_total 계산."""
    mask = (gt != ignore_index)
    pred = pred[mask]
    gt = gt[mask]

    intersect = pred[pred == gt]
    area_intersect = np.bincount(intersect, minlength=num_classes)
    area_pred = np.bincount(pred, minlength=num_classes)
    area_gt = np.bincount(gt, minlength=num_classes)
    area_union = area_pred + area_gt - area_intersect
    return area_intersect, area_union, area_pred, area_gt


def main():
    args = parse_args()

    print(f'Loading 4 models on {args.device}...')
    models = {}
    for class_idx, info in MODELS.items():
        ckpt = info['checkpoint']
        cfg = info['config']
        if not osp.exists(ckpt):
            print(f'  [SKIP] {info["name"]}: checkpoint not found at {ckpt}')
            continue
        print(f'  Loading {info["name"]} (class {class_idx}): {osp.basename(ckpt)}')
        models[class_idx] = init_model(cfg, ckpt, device=args.device)

    if len(models) != 4:
        raise SystemExit(f'Need all 4 models. Loaded: {list(models.keys())}')

    # Collect all val images + their GT mask paths
    print('\nCollecting val images...')
    items = []  # [(img_path, gt_path, source_category), ...]
    for cat in CATEGORY_FOLDERS:
        img_dir = osp.join(args.data_root, 'images/val', cat)
        gt_dir = osp.join(args.data_root, 'masks/val', cat)
        if not osp.isdir(img_dir):
            print(f'  [SKIP] {img_dir}')
            continue
        for f in sorted(os.listdir(img_dir)):
            stem = osp.splitext(f)[0]
            gt_path = osp.join(gt_dir, stem + '.png')
            if osp.exists(gt_path):
                items.append((osp.join(img_dir, f), gt_path, cat))
    print(f'  Total val images: {len(items)}')

    # Accumulate metrics
    total_intersect = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pred = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_gt = np.zeros(NUM_CLASSES, dtype=np.int64)

    print(f'\nRunning ensemble inference (threshold={args.threshold})...')
    for i, (img_path, gt_path, src_cat) in enumerate(items):
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(items)}')

        # Predict
        pred = ensemble_predict(models, img_path, args.device, args.threshold)

        # Load GT
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        # Accumulate
        ai, au, ap, ag = compute_intersection_union(pred, gt, NUM_CLASSES)
        total_intersect += ai
        total_union += au
        total_pred += ap
        total_gt += ag

    # Compute metrics
    print('\n' + '=' * 80)
    print(f'Per-Category Ensemble Evaluation (threshold={args.threshold})')
    print('=' * 80)

    # Per-class metrics
    iou = total_intersect / np.maximum(total_union, 1)
    precision = total_intersect / np.maximum(total_pred, 1)
    recall = total_intersect / np.maximum(total_gt, 1)  # = TPR = Acc
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-10)

    # FPR per class
    total_pixels = total_gt.sum()
    fp = total_pred - total_intersect  # False positives per class
    tn = total_pixels - total_gt - fp  # True negatives per class
    fpr = fp / np.maximum(fp + tn, 1)

    # Print per-class table
    table = PrettyTable()
    table.field_names = ['Class', 'IoU', 'F1', 'Prec', 'Recall(TPR)', 'FPR']
    for i, name in enumerate(CLASS_NAMES):
        table.add_row([
            name,
            f'{iou[i]*100:.2f}',
            f'{f1[i]*100:.2f}',
            f'{precision[i]*100:.2f}',
            f'{recall[i]*100:.2f}',
            f'{fpr[i]*100:.2f}'
        ])
    print(table)

    # Mean metrics
    miou = iou.mean()
    mf1 = f1.mean()
    mprec = precision.mean()
    mrec = recall.mean()
    mfpr = fpr.mean()
    aacc = total_intersect.sum() / max(total_gt.sum(), 1)

    # Defect-only means (exclude background)
    miou_def = iou[1:].mean()
    mf1_def = f1[1:].mean()
    mrec_def = recall[1:].mean()

    summary = (
        f'\n=== Summary (5 classes, background 포함) ===\n'
        f'  aAcc:  {aacc*100:.2f}\n'
        f'  mIoU:  {miou*100:.2f}\n'
        f'  mF1:   {mf1*100:.2f}\n'
        f'  mPrec: {mprec*100:.2f}\n'
        f'  mTPR:  {mrec*100:.2f}\n'
        f'  mFPR:  {mfpr*100:.2f}\n'
        f'\n=== Defect-only (background 제외, 4 classes) ===\n'
        f'  mIoU:  {miou_def*100:.2f}\n'
        f'  mF1:   {mf1_def*100:.2f}\n'
        f'  mTPR:  {mrec_def*100:.2f}\n'
    )
    print(summary)

    # Save to file
    with open(args.output, 'w') as f:
        f.write(f'Per-Category Ensemble Evaluation\n')
        f.write(f'Threshold: {args.threshold}\n')
        f.write(f'Total val images: {len(items)}\n\n')
        f.write(str(table))
        f.write(summary)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
