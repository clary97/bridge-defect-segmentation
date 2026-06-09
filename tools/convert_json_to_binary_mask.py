"""
Convert JSON polygon annotations directly to binary masks.

For AI Hub full dataset where each folder contains ONE defect type only.
The JSON labels are converted to binary (defect=1, background=0).

Usage:
    python tools/convert_json_to_binary_mask.py \
        --img-dir /path/to/images/콘크리트_균열 \
        --ann-dir /path/to/annotations/콘크리트_균열 \
        --out-dir /path/to/binary_masks/train_full/crack \
        --label ConcreteCrack
"""

import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True,
                        help='Source images directory')
    parser.add_argument('--ann-dir', required=True,
                        help='Source annotations (JSON) directory')
    parser.add_argument('--out-dir', required=True,
                        help='Output binary mask directory')
    parser.add_argument('--label', required=True,
                        help='Target label name to convert (e.g., ConcreteCrack)')
    return parser.parse_args()


def convert_one(ann_path, target_label):
    """Convert a single JSON annotation to a binary mask."""
    with open(ann_path, 'r') as f:
        data = json.load(f)

    h = data.get('imageHeight')
    w = data.get('imageWidth')
    if h is None or w is None:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get('shapes', []):
        if shape.get('label') == target_label:
            pts = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(args.ann_dir):
        raise SystemExit(f'[ERROR] ann-dir not found: {args.ann_dir}')

    ann_files = sorted([f for f in os.listdir(args.ann_dir) if f.endswith('.json')])
    print(f'Converting {len(ann_files)} JSON files (label="{args.label}")...')

    total = 0
    skipped = 0
    existing = 0
    for i, ann_file in enumerate(ann_files):
        if (i + 1) % 5000 == 0:
            print(f'  {i + 1}/{len(ann_files)} (created={total}, existing={existing})')

        stem = os.path.splitext(ann_file)[0]
        ann_path = os.path.join(args.ann_dir, ann_file)
        out_path = os.path.join(args.out_dir, stem + '.png')

        # Skip if mask already exists
        if os.path.exists(out_path):
            existing += 1
            continue

        # Check image exists
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            cand = os.path.join(args.img_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            skipped += 1
            continue

        mask = convert_one(ann_path, args.label)
        if mask is None:
            skipped += 1
            continue

        cv2.imwrite(out_path, mask)
        total += 1

    print(f'\nDone: {total} binary masks created, {existing} already existed, {skipped} skipped')
    print(f'Output: {args.out_dir}')


if __name__ == '__main__':
    main()
