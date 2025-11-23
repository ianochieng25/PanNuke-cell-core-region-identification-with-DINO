"""
視覺化預測結果：將預測遮罩以透明顏色疊加在原始影像上
使用方式：python visualize_predictions.py --split train --num_samples 10
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# 專案路徑設定
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'preprocess'))

PANNUKE_ROOT = os.path.join(PROJECT_ROOT, "data", "pannuke")
PRED_ROOT = os.path.join(PROJECT_ROOT, "predictions")
VIZ_ROOT = os.path.join(PROJECT_ROOT, "visualizations")

FOLDS = {
    'train': os.path.join(PANNUKE_ROOT, 'train'),
    'test':  os.path.join(PANNUKE_ROOT, 'test'),
}


def create_overlay(image, gt_mask, pred_mask, alpha=0.4):
    """
    創建疊加圖
    Args:
        image: (H, W, 3) RGB 影像
        gt_mask: (H, W) Ground Truth 遮罩 (0=背景, 1=細胞)
        pred_mask: (H, W) 預測遮罩 (0=背景, 1=細胞)
        alpha: 透明度
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 確保影像在 [0, 255] 範圍
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Row 1: Ground Truth
    # 1. 原始影像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. GT 遮罩
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. GT 疊加圖 (綠色=細胞, 紅色=背景)
    overlay_gt = image.copy()
    # 為 GT 創建彩色遮罩
    color_mask_gt = np.zeros_like(image)
    color_mask_gt[gt_mask == 1] = [0, 255, 0]  # 綠色：細胞
    color_mask_gt[gt_mask == 0] = [255, 0, 0]  # 紅色：背景
    
    overlay_gt = (overlay_gt * (1 - alpha) + color_mask_gt * alpha).astype(np.uint8)
    axes[0, 2].imshow(overlay_gt)
    axes[0, 2].set_title('GT Overlay (Green=Cells, Red=Bg)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Prediction
    # 1. 原始影像
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 2. Pred 遮罩
    axes[1, 1].imshow(pred_mask, cmap='gray')
    axes[1, 1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 3. Pred 疊加圖 (藍色=細胞, 黃色=背景)
    overlay_pred = image.copy()
    color_mask_pred = np.zeros_like(image)
    color_mask_pred[pred_mask == 1] = [0, 0, 255]  # 藍色：預測的細胞
    color_mask_pred[pred_mask == 0] = [255, 255, 0]  # 黃色：預測的背景
    
    overlay_pred = (overlay_pred * (1 - alpha) + color_mask_pred * alpha).astype(np.uint8)
    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title('Pred Overlay (Blue=Cells, Yellow=Bg)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_split(split_name, num_samples=10, alpha=0.4, seed=42):
    """
    視覺化指定 split 的預測結果
    """
    print(f"\n{'='*60}")
    print(f"  視覺化 {split_name.upper()} 集預測結果")
    print(f"{'='*60}\n")
    
    # 讀取資料
    data_folder = FOLDS[split_name]
    images_path = os.path.join(data_folder, "images.npy")
    masks_path = os.path.join(data_folder, "masks.npy")
    pred_path = os.path.join(PRED_ROOT, f"{split_name}_pred_masks.npy")
    
    if not os.path.exists(images_path):
        print(f"❌ 找不到影像檔案: {images_path}")
        return
    if not os.path.exists(masks_path):
        print(f"❌ 找不到 GT 遮罩: {masks_path}")
        return
    if not os.path.exists(pred_path):
        print(f"❌ 找不到預測遮罩: {pred_path}")
        return
    
    print(f"Loading files...")
    print(f"   Image: {images_path}")
    print(f"   GT Mask: {masks_path}")
    print(f"   Pred Mask: {pred_path}\n")
    
    # 使用 mmap 讀取
    images = np.load(images_path, mmap_mode='r')
    gt_masks = np.load(masks_path, mmap_mode='r')
    pred_masks = np.load(pred_path, mmap_mode='r')
    
    total_samples = images.shape[0]
    print(f"Total samples: {total_samples}")
    print(f"Image shape: {images.shape}")
    print(f"GT shape: {gt_masks.shape}")
    print(f"Pred shape: {pred_masks.shape}\n")
    
    # 處理 GT 和 Pred 的格式
    # GT 可能是 (N, H, W, 6) 多通道格式，需要轉為 (N, H, W)
    if gt_masks.ndim == 4:
        print("  GT is multi-channel, converting to single channel...")
        gt_masks_processed = np.argmax(gt_masks, axis=-1)
    else:
        gt_masks_processed = gt_masks
    
    # Pred 可能也是多通道
    if pred_masks.ndim == 4:
        print("  Pred is multi-channel, converting to single channel...")
        pred_masks_processed = np.argmax(pred_masks, axis=-1)
    else:
        pred_masks_processed = pred_masks
    
    # 轉換為二元遮罩 (0=背景, 1=細胞)
    # 原始標籤: 0:Neoplastic, 1:Inflammatory, 2:Connective, 3:Dead, 4:Epithelial, 5:Background
    # 二元: 背景(0)←3,5 ; 細胞(1)←0,1,2,4
    print("Converting to binary mask (0=Bg, 1=Cells)...\n")
    
    def to_binary(mask):
        binary = np.zeros_like(mask, dtype=np.uint8)
        is_cell = (mask != 3) & (mask != 5)
        binary[is_cell] = 1
        return binary
    
    # 隨機選擇樣本
    random.seed(seed)
    if num_samples > total_samples:
        num_samples = total_samples
    indices = random.sample(range(total_samples), num_samples)
    
    # 創建輸出資料夾
    output_dir = os.path.join(VIZ_ROOT, split_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} visualizations...")
    for i, idx in enumerate(indices):
        print(f"   Processing {i+1}/{num_samples}: Sample Index {idx}...", end=' ')
        
        # 取得資料
        img = np.array(images[idx])
        gt = to_binary(np.array(gt_masks_processed[idx]))
        pred = np.array(pred_masks_processed[idx])
        
        # 如果預測結果不是二元，也轉換
        if pred.max() > 1:
            pred = to_binary(pred)
        
        # 創建疊加圖
        fig = create_overlay(img, gt, pred, alpha=alpha)
        
        # 儲存
        output_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved to {output_path}")
    
    print(f"\nDone! Visualizations saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="視覺化分割預測結果")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='要視覺化的資料集 (train/test)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要生成的樣本數量')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='遮罩透明度 (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    
    args = parser.parse_args()
    
    visualize_split(args.split, args.num_samples, args.alpha, args.seed)


if __name__ == "__main__":
    main()
