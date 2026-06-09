"""
Convert JSON polygon annotations to PNG segmentation masks.

Class mapping:
    0 = background
    1 = ConcreteCrack (crack)
    2 = SteelDefect (corrosion)
    3 = Efflorescence (백태, JSON label is 'Efflorescene' typo)
    4 = PaintDamage (peeling)

Usage:
    python tools/convert_json_to_mask.py \
        --data-root /workspace/nas_192/datasets/public/AI_Hub_bridge_defect_data \
        --splits train val
"""

import argparse
import json
import os

import cv2
import numpy as np

# label name -> class index (0 = background)
LABEL_TO_CLASS = {
    'ConcreteCrack': 1,
    'SteelDefect': 2,
    'Efflorescene': 3,
    'PaintDamage': 4,
}

CATEGORIES = ['corrosion', 'crack', 'efflorescence', 'peeling']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (contains images/, annotations/)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Splits to convert (default: train val)')
    return parser.parse_args()


def convert_one(ann_path, height, width):
    """Convert a single JSON annotation to a class-index mask."""
    with open(ann_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)  # 0 = background

    for shape in data['shapes']:
        label = shape['label']
        cls_id = LABEL_TO_CLASS.get(label)
        if cls_id is None:
            print(f'  [WARN] Unknown label "{label}" in {ann_path}, skipped')
            continue
        pts = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], cls_id)

    return mask


def main():
    args = parse_args()

    for split in args.splits:
        ann_root = os.path.join(args.data_root, 'annotations', split)
        img_root = os.path.join(args.data_root, 'images', split)
        mask_root = os.path.join(args.data_root, 'masks', split)

        total = 0
        skipped = 0

        for cat in CATEGORIES:
            ann_dir = os.path.join(ann_root, cat)
            img_dir = os.path.join(img_root, cat)
            mask_dir = os.path.join(mask_root, cat)
            os.makedirs(mask_dir, exist_ok=True)

            if not os.path.isdir(ann_dir):
                print(f'[SKIP] {ann_dir} not found')
                continue

            ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
            print(f'[{split}/{cat}] Converting {len(ann_files)} annotations...')

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

                # Get image size from annotation
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                h = data.get('imageHeight', 512)
                w = data.get('imageWidth', 512)

                # Convert and save
                mask = convert_one(ann_path, h, w)
                mask_path = os.path.join(mask_dir, stem + '.png')
                cv2.imwrite(mask_path, mask)
                total += 1

        print(f'\n[{split}] Done: {total} masks created, {skipped} skipped')

    # Print class mapping for reference
    print('\n=== Class Mapping ===')
    print('  0: background')
    for label, idx in sorted(LABEL_TO_CLASS.items(), key=lambda x: x[1]):
        print(f'  {idx}: {label}')


if __name__ == '__main__':
    main()
