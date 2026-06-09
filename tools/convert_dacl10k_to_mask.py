"""
Convert dacl10k_4cate JSON polygon annotations to PNG segmentation masks.

Class mapping:
    0 = background
    1 = Crack
    2 = Rust
    3 = Efflorescence
    4 = Spalling

Usage:
    python tools/convert_dacl10k_to_mask.py \
        --data-root /workspace/nas_192/datasets/public/dacl10k/dacl10k_4cate \
        --splits train validation
"""

import argparse
import json
import os

import cv2
import numpy as np

# label name -> class index (0 = background)
LABEL_TO_CLASS = {
    'Crack': 1,
    'Rust': 2,
    'Efflorescence': 3,
    'Spalling': 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (contains images/, annotations/)')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation'],
                        help='Splits to convert')
    return parser.parse_args()


def convert_one(ann_path):
    """Convert a single JSON annotation to a class-index mask."""
    with open(ann_path, 'r') as f:
        data = json.load(f)

    h = data.get('imageHeight')
    w = data.get('imageWidth')
    mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background

    for shape in data.get('shapes', []):
        label = shape['label']
        cls_id = LABEL_TO_CLASS.get(label)
        if cls_id is None:
            print(f'  [WARN] Unknown label "{label}" in {ann_path}, skipped')
            continue
        pts = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], cls_id)

    return mask, h, w


def main():
    args = parse_args()

    for split in args.splits:
        ann_dir = os.path.join(args.data_root, 'annotations', split)
        img_dir = os.path.join(args.data_root, 'images', split)
        mask_dir = os.path.join(args.data_root, 'masks', split)
        os.makedirs(mask_dir, exist_ok=True)

        if not os.path.isdir(ann_dir):
            print(f'[SKIP] {ann_dir} not found')
            continue

        ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
        print(f'[{split}] Converting {len(ann_files)} annotations...')

        total = 0
        skipped = 0
        for ann_file in ann_files:
            stem = os.path.splitext(ann_file)[0]
            ann_path = os.path.join(ann_dir, ann_file)

            # Check image exists
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                print(f'  [SKIP] No image for {ann_file}')
                skipped += 1
                continue

            mask, h, w = convert_one(ann_path)
            mask_path = os.path.join(mask_dir, stem + '.png')
            cv2.imwrite(mask_path, mask)
            total += 1

        print(f'[{split}] Done: {total} masks created, {skipped} skipped')

    print('\n=== Class Mapping ===')
    print('  0: background')
    for label, idx in sorted(LABEL_TO_CLASS.items(), key=lambda x: x[1]):
        print(f'  {idx}: {label}')


if __name__ == '__main__':
    main()
