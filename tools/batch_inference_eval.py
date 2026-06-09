"""
Batch zero-shot inference + GT comparison for bridge defect dataset.

Usage:
    python tools/batch_inference_eval.py \
        --config configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
        --checkpoint checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth \
        --img-root /workspace/nas_192/datasets/public/AI_Hub_bridge_defect_data/images/val \
        --ann-root /workspace/nas_192/datasets/public/AI_Hub_bridge_defect_data/annotations/val \
        --out-dir output/segformer_b0_ade20k \
        --device cuda:0
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model

CATEGORIES = ['corrosion', 'crack', 'efflorescence', 'peeling']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-root', required=True)
    parser.add_argument('--ann-root', required=True)
    parser.add_argument('--out-dir', default='output/zero_shot')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opacity', type=float, default=0.5)
    return parser.parse_args()


def load_gt_mask(ann_path, height, width):
    """JSON polygon annotation -> binary mask (defect=1, background=0)."""
    with open(ann_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data['shapes']:
        pts = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def get_prediction_mask(result):
    """Get per-pixel class prediction from mmseg result."""
    seg = result.pred_sem_seg.data[0].cpu().numpy()  # (H, W)
    return seg.astype(np.int32)


def binary_prediction(seg_map):
    """Convert multi-class prediction to binary anomaly mask.

    Logic: the most frequent class = background, everything else = anomaly.
    """
    bg_class = np.bincount(seg_map.flatten()).argmax()
    anomaly = (seg_map != bg_class).astype(np.uint8)
    return anomaly, int(bg_class)


def compute_metrics(pred_binary, gt_binary):
    """Compute IoU, F1, Precision, Recall for binary masks."""
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    pixel_acc = (tp + tn) / (tp + fp + fn + tn)

    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pixel_acc': pixel_acc,
    }


def create_comparison_image(img, gt_mask, pred_binary, opacity=0.5):
    """Create side-by-side: [Original] [GT overlay] [Prediction overlay]."""
    h, w = img.shape[:2]

    # GT overlay (green)
    gt_overlay = img.copy()
    gt_overlay[gt_mask == 1] = (
        gt_overlay[gt_mask == 1] * (1 - opacity) +
        np.array([0, 255, 0]) * opacity
    ).astype(np.uint8)

    # Prediction overlay (red)
    pred_overlay = img.copy()
    pred_overlay[pred_binary == 1] = (
        pred_overlay[pred_binary == 1] * (1 - opacity) +
        np.array([0, 0, 255]) * opacity
    ).astype(np.uint8)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(gt_overlay, 'GT (green)', (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(pred_overlay, 'Pred (red)', (10, 30), font, 0.7, (0, 0, 255), 2)

    return np.concatenate([img, gt_overlay, pred_overlay], axis=1)


def main():
    args = parse_args()

    # Init model
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Results storage
    all_metrics = {}
    category_metrics = {cat: [] for cat in CATEGORIES}

    for cat in CATEGORIES:
        img_dir = os.path.join(args.img_root, cat)
        ann_dir = os.path.join(args.ann_root, cat)
        out_cat_dir = os.path.join(args.out_dir, cat)
        os.makedirs(out_cat_dir, exist_ok=True)

        if not os.path.isdir(img_dir):
            print(f'[SKIP] {img_dir} not found')
            continue

        img_files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f'\n=== {cat} ({len(img_files)} images) ===')

        for i, img_file in enumerate(img_files):
            img_path = os.path.join(img_dir, img_file)
            stem = os.path.splitext(img_file)[0]
            ann_path = os.path.join(ann_dir, stem + '.json')

            # Load image
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # Inference
            result = inference_model(model, img_path)
            seg_map = get_prediction_mask(result)

            # Binary masks
            pred_binary, bg_class = binary_prediction(seg_map)

            # GT mask
            if os.path.exists(ann_path):
                gt_mask = load_gt_mask(ann_path, h, w)
                metrics = compute_metrics(pred_binary, gt_mask)
                category_metrics[cat].append(metrics)

                # Save comparison image
                comparison = create_comparison_image(
                    img.copy(), gt_mask, pred_binary, args.opacity)
                cv2.imwrite(os.path.join(out_cat_dir, f'{stem}_compare.jpg'),
                            comparison)
            else:
                gt_mask = None
                metrics = None
                # Save prediction only
                pred_overlay = img.copy()
                pred_overlay[pred_binary == 1] = (
                    pred_overlay[pred_binary == 1] * 0.5 +
                    np.array([0, 0, 255]) * 0.5
                ).astype(np.uint8)
                cv2.imwrite(os.path.join(out_cat_dir, f'{stem}_pred.jpg'),
                            pred_overlay)

            if metrics:
                print(f'  [{i+1}/{len(img_files)}] {img_file} | '
                      f'IoU={metrics["iou"]:.4f} F1={metrics["f1"]:.4f} '
                      f'Prec={metrics["precision"]:.4f} Rec={metrics["recall"]:.4f} '
                      f'PixAcc={metrics["pixel_acc"]:.4f} (bg_class={bg_class})')
            else:
                print(f'  [{i+1}/{len(img_files)}] {img_file} | No GT annotation')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY (Binary: defect vs background)')
    print('=' * 70)
    print(f'{"Category":<18} {"Images":>6} {"IoU":>8} {"F1":>8} '
          f'{"Prec":>8} {"Recall":>8} {"PixAcc":>8}')
    print('-' * 70)

    total_metrics = []
    for cat in CATEGORIES:
        if not category_metrics[cat]:
            print(f'{cat:<18} {"0":>6} {"N/A":>8} {"N/A":>8} '
                  f'{"N/A":>8} {"N/A":>8} {"N/A":>8}')
            continue

        m_list = category_metrics[cat]
        total_metrics.extend(m_list)
        avg = {k: np.mean([m[k] for m in m_list]) for k in m_list[0]}
        print(f'{cat:<18} {len(m_list):>6} {avg["iou"]:>8.4f} {avg["f1"]:>8.4f} '
              f'{avg["precision"]:>8.4f} {avg["recall"]:>8.4f} {avg["pixel_acc"]:>8.4f}')

    if total_metrics:
        avg_all = {k: np.mean([m[k] for m in total_metrics]) for k in total_metrics[0]}
        print('-' * 70)
        print(f'{"TOTAL":<18} {len(total_metrics):>6} {avg_all["iou"]:>8.4f} '
              f'{avg_all["f1"]:>8.4f} {avg_all["precision"]:>8.4f} '
              f'{avg_all["recall"]:>8.4f} {avg_all["pixel_acc"]:>8.4f}')

    # Save metrics to file
    summary_path = os.path.join(args.out_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Config: {args.config}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n\n')
        for cat in CATEGORIES:
            if not category_metrics[cat]:
                continue
            m_list = category_metrics[cat]
            avg = {k: np.mean([m[k] for m in m_list]) for k in m_list[0]}
            f.write(f'{cat}: IoU={avg["iou"]:.4f} F1={avg["f1"]:.4f} '
                    f'Prec={avg["precision"]:.4f} Recall={avg["recall"]:.4f} '
                    f'PixAcc={avg["pixel_acc"]:.4f} (n={len(m_list)})\n')
        if total_metrics:
            avg_all = {k: np.mean([m[k] for m in total_metrics]) for k in total_metrics[0]}
            f.write(f'\nTOTAL: IoU={avg_all["iou"]:.4f} F1={avg_all["f1"]:.4f} '
                    f'Prec={avg_all["precision"]:.4f} Recall={avg_all["recall"]:.4f} '
                    f'PixAcc={avg_all["pixel_acc"]:.4f} (n={len(total_metrics)})\n')
    print(f'\nMetrics saved to: {summary_path}')
    print(f'Comparison images saved to: {args.out_dir}/')


if __name__ == '__main__':
    main()
