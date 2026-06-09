"""
Convert 5-class masks to per-category binary masks for AI Hub bridge defect.

For each category, creates binary masks where:
    0 = background (or other classes)
    1 = the target defect class

Output structure:
    binary_masks/{split}/{category}/xxx.png

Usage:
    python tools/convert_to_binary_mask.py \
        --data-root /workspace/nas_192/datasets/public/AI_Hub_bridge_defect_data \
        --splits train val
"""

import argparse
import os

import cv2
import numpy as np

# (folder_name, class_index_in_5class_mask)
CATEGORIES = [
    ('crack', 1),          # ConcreteCrack
    ('corrosion', 2),      # SteelDefect
    ('efflorescence', 3),  # Efflorescence
    ('peeling', 4),        # PaintDamage
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (contains masks/ subdir)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    return parser.parse_args()


def main():
    args = parse_args()

    for split in args.splits:
        for category, class_id in CATEGORIES:
            src_dir = os.path.join(args.data_root, 'masks', split, category)
            dst_dir = os.path.join(args.data_root, 'binary_masks', split, category)

            if not os.path.isdir(src_dir):
                print(f'[SKIP] {src_dir} not found')
                continue

            os.makedirs(dst_dir, exist_ok=True)
            files = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
            print(f'[{split}/{category}] Converting {len(files)} masks (class {class_id} -> 1)...')

            count = 0
            for f in files:
                mask = cv2.imread(os.path.join(src_dir, f), cv2.IMREAD_GRAYSCALE)
                # Convert: target class -> 1, everything else -> 0
                binary = (mask == class_id).astype(np.uint8)
                cv2.imwrite(os.path.join(dst_dir, f), binary)
                count += 1

            print(f'[{split}/{category}] Done: {count} binary masks')

    print('\n=== Binary Class Mapping (per model) ===')
    print('  0: background')
    print('  1: defect (target class)')


if __name__ == '__main__':
    main()
