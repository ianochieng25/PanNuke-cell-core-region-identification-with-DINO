import os
import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="評估分割結果")
    parser.add_argument(
        '--pred_dir', required=True,
        help="模型預測輸出檔案所在的目錄"
    )
    parser.add_argument(
        '--gt_dir', required=True,
        help="Ground truth mask 檔案所在的目錄"
    )
    return parser.parse_args()

def load_masks(paths):
    masks = []
    for p in paths:
        if p.lower().endswith('.npy'):
            masks.append(np.load(p))
        else:
            masks.append(np.array(Image.open(p).convert('L')))
    return masks

def main():
    args = parse_args()

    # 1) 如果預測目錄不存在，直接跳過
    if not os.path.isdir(args.pred_dir):
        print(f"Warning: 找不到預測目錄，跳過評估：{args.pred_dir}", file=sys.stderr)
        return

    # 2) 如果 GT 目錄不存在，也跳過
    if not os.path.isdir(args.gt_dir):
        print(f"Warning: 找不到 Ground Truth 目錄，跳過評估：{args.gt_dir}", file=sys.stderr)
        return

    # 3) 讀取並排序所有檔案
    pred_files = sorted([
        os.path.join(args.pred_dir, f)
        for f in os.listdir(args.pred_dir)
        if f.lower().endswith(('.npy', '.png', '.jpg', '.jpeg'))
    ])
    gt_files = sorted([
        os.path.join(args.gt_dir, f)
        for f in os.listdir(args.gt_dir)
        if f.lower().endswith(('.npy', '.png', '.jpg', '.jpeg'))
    ])

    # 4) 如果檔案數量不符，跳過並警告
    if len(pred_files) != len(gt_files):
        print(
            f"Warning: 預測檔案數 ({len(pred_files)}) 與 GT 檔案數 ({len(gt_files)}) 不符，跳過評估",
            file=sys.stderr
        )
        return

    # 5) 載入所有 masks
    preds = load_masks(pred_files)
    gts   = load_masks(gt_files)

    # 6) 計算指標
    flat_pred = np.concatenate([p.flatten() for p in preds])
    flat_gt   = np.concatenate([g.flatten() for g in gts])

    acc = accuracy_score(flat_gt, flat_pred)
    f1  = f1_score(flat_gt, flat_pred)
    iou = jaccard_score(flat_gt, flat_pred)

    # 7) 輸出結果
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

if __name__ == "__main__":
    main()
