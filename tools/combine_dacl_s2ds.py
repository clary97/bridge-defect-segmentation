"""
dacl10k_v2 + S2DS 데이터셋을 efflorescence binary 학습용으로 통합.

전략:
    - train: dacl train + S2DS (train + val)  -> 학습 데이터 확장
    - val:   dacl val                          -> best checkpoint 선택용
    - test:  dacl test                         -> in-domain 평가
    - s2ds_test: S2DS test                     -> cross-domain 평가 (별도)

Symlink 사용 (디스크 절약).

Usage:
    python tools/combine_dacl_s2ds.py
"""

import os

DACL_ROOT = '/workspace/nas_200/minkyung/dacl10k_v2_devphase/efflorescence_binary'
S2DS_ROOT = '/workspace/nas_200/minkyung/s2ds_binary_efflorescence'
DST = '/workspace/nas_200/minkyung/dacl_s2ds_combined'


def symlink_dir(src_img_dir, src_msk_dir, dst_img_dir, dst_msk_dir,
                prefix=''):
    """src의 이미지/마스크를 dst에 symlink로 복사."""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_msk_dir, exist_ok=True)

    count = 0
    for f in sorted(os.listdir(src_img_dir)):
        stem, ext = os.path.splitext(f)
        # 이미지
        src_img = os.path.join(src_img_dir, f)
        dst_img = os.path.join(dst_img_dir, prefix + stem + ext)
        if not os.path.exists(dst_img):
            # symlink: src 파일이 이미 symlink면 원본을 찾아서 link
            real_src = os.path.realpath(src_img)
            os.symlink(real_src, dst_img)
        # 마스크
        src_msk = os.path.join(src_msk_dir, stem + '.png')
        dst_msk = os.path.join(dst_msk_dir, prefix + stem + '.png')
        if os.path.exists(src_msk) and not os.path.exists(dst_msk):
            real_src = os.path.realpath(src_msk)
            os.symlink(real_src, dst_msk)
            count += 1
    return count


def main():
    print(f'Creating combined dataset at: {DST}')

    # === train: dacl train + S2DS (train + val) ===
    print('\n[train] dacl train + S2DS train + S2DS val')
    n_dacl = symlink_dir(
        f'{DACL_ROOT}/train/images', f'{DACL_ROOT}/train/masks',
        f'{DST}/train/images', f'{DST}/train/masks', prefix='dacl_')
    n_s2ds_train = symlink_dir(
        f'{S2DS_ROOT}/train/images', f'{S2DS_ROOT}/train/masks',
        f'{DST}/train/images', f'{DST}/train/masks', prefix='s2ds_train_')
    n_s2ds_val = symlink_dir(
        f'{S2DS_ROOT}/val/images', f'{S2DS_ROOT}/val/masks',
        f'{DST}/train/images', f'{DST}/train/masks', prefix='s2ds_val_')
    print(f'  dacl train:    {n_dacl}')
    print(f'  s2ds train:    {n_s2ds_train}')
    print(f'  s2ds val:      {n_s2ds_val}')
    print(f'  TOTAL train:   {n_dacl + n_s2ds_train + n_s2ds_val}')

    # === val: dacl val (best checkpoint 선택용) ===
    print('\n[val] dacl val (152장)')
    n = symlink_dir(
        f'{DACL_ROOT}/val/images', f'{DACL_ROOT}/val/masks',
        f'{DST}/val/images', f'{DST}/val/masks', prefix='dacl_')
    print(f'  TOTAL val:     {n}')

    # === test: dacl test (in-domain 평가) ===
    print('\n[test] dacl test (206장, in-domain 평가)')
    n = symlink_dir(
        f'{DACL_ROOT}/test/images', f'{DACL_ROOT}/test/masks',
        f'{DST}/test/images', f'{DST}/test/masks', prefix='dacl_')
    print(f'  TOTAL test:    {n}')

    # === s2ds_test: 학습에 안 쓴 외부 데이터 (cross-domain 평가) ===
    print('\n[s2ds_test] S2DS test (93장, cross-domain 평가용 - 학습에 안 씀)')
    n = symlink_dir(
        f'{S2DS_ROOT}/test/images', f'{S2DS_ROOT}/test/masks',
        f'{DST}/s2ds_test/images', f'{DST}/s2ds_test/masks', prefix='s2ds_')
    print(f'  TOTAL s2ds_test: {n}')

    print('\n=== Done ===')
    print(f'\nFinal counts:')
    for split in ['train', 'val', 'test', 's2ds_test']:
        n = len(os.listdir(f'{DST}/{split}/images'))
        print(f'  {split}: {n}')


if __name__ == '__main__':
    main()
