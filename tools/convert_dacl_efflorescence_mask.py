"""
Convert dacl10k_v2_devphase/efflorescence masks (0/255) to mmseg format (0/1).

Usage:
    python tools/convert_dacl_efflorescence_mask.py \
        --data-root /workspace/nas_200/minkyung/dacl10k_v2_devphase/efflorescence \
        --out-root /workspace/nas_200/minkyung/dacl10k_v2_devphase/efflorescence_binary
"""

import argparse
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True,
                        help='Source dataset root (contains train/val/test)')
    parser.add_argument('--out-root', required=True,
                        help='Output dataset root')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()

    for split in args.splits:
        src_mask_dir = os.path.join(args.data_root, split, 'masks')
        dst_mask_dir = os.path.join(args.out_root, split, 'masks')

        if not os.path.isdir(src_mask_dir):
            print(f'[SKIP] {src_mask_dir}')
            continue

        os.makedirs(dst_mask_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(src_mask_dir) if f.endswith('.png')])
        print(f'[{split}] Converting {len(files)} masks...')

        for f in files:
            mask = cv2.imread(os.path.join(src_mask_dir, f), cv2.IMREAD_GRAYSCALE)
            binary = (mask > 0).astype(np.uint8)  # 0/255 -> 0/1
            cv2.imwrite(os.path.join(dst_mask_dir, f), binary)

        print(f'  Done: {len(files)} masks')

    # Symlink images (no copy needed)
    for split in args.splits:
        src_img = os.path.join(args.data_root, split, 'images')
        dst_img = os.path.join(args.out_root, split, 'images')
        if os.path.isdir(src_img) and not os.path.exists(dst_img):
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.symlink(src_img, dst_img)
            print(f'Linked: {dst_img} -> {src_img}')

    print('\n=== Class Mapping ===')
    print('  0: background')
    print('  1: Efflorescence')


if __name__ == '__main__':
    main()
