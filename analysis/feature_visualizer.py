# File: analysis/feature_visualizer.py

import os
import sys
import argparse
import numpy as np

# 嘗試匯入 matplotlib，若失敗則跳過可視化
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: 沒有安裝 matplotlib，跳過 feature_visualizer 可視化步驟", file=sys.stderr)
    sys.exit(0)

from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser(description="TSNE 可視化 ViT 特徵")
    parser.add_argument('--features_dir', required=True,
                        help="儲存 cls_tokens.npy、patch_embeddings.npy 的資料夾")
    parser.add_argument('--output_png', required=True,
                        help="輸出 tsne 圖的檔案路徑")
    return parser.parse_args()

def main():
    args = parse_args()

    cls_path   = os.path.join(args.features_dir, 'cls_tokens.npy')
    patch_path = os.path.join(args.features_dir, 'patch_embeddings.npy')

    if not os.path.isfile(cls_path) or not os.path.isfile(patch_path):
        print(f"Warning: 找不到特徵檔案，跳過可視化：{cls_path}, {patch_path}", file=sys.stderr)
        return

    # 載入 CLS token 特徵
    cls_tokens = np.load(cls_path)  # shape [N, D]
    # 只取前 1000 個 sample 做 TSNE，加速
    subsample = min(1000, cls_tokens.shape[0])
    data = cls_tokens[:subsample]

    # TSNE 降到 2 維
    tsne = TSNE(n_components=2, random_state=0)
    emb = tsne.fit_transform(data)

    # 畫圖
    plt.figure(figsize=(6,6))
    plt.scatter(emb[:,0], emb[:,1], s=5, alpha=0.7)
    plt.title('TSNE of ViT CLS Tokens')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, dpi=300)
    print(f"TSNE 圖已儲存到：{args.output_png}")

if __name__ == "__main__":
    main()
