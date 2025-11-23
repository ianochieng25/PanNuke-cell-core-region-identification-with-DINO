# File: preprocess/patch_exporter.py

import os
import sys
# 確保能找到 src 資料夾
CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '..', 'src', 'dataset')))

import numpy as np
from dataset.pannuke_dataset import PannukeDataset
from PIL import Image

def export_patches(fold_dir, output_dir):
    """
    將指定 fold 資料夾中的所有影像拆分成單張 patch，
    並存到 output_dir/class0 底下，符合 ImageFolder 結構。
    """
    # 1) 讀取資料集
    ds = PannukeDataset(fold_dir)

    # 2) 準備 class0 子資料夾
    class_dir = os.path.join(output_dir, 'class0')
    os.makedirs(class_dir, exist_ok=True)

    # 3) 輸出每張 patch
    # 3) 輸出每張 patch
    import concurrent.futures
    
    def save_patch(idx):
        try:
            img, _, _ = ds[idx]  # img: Tensor CxHxW in [-1,1] or numpy
            # Check if img is Tensor or Numpy
            if hasattr(img, 'permute'):
                # Assuming Tensor from PannukeDataset might be normalized if aug was passed, 
                # but here we init PannukeDataset without aug, so it returns numpy usually?
                # Wait, the new PannukeDataset returns numpy if no aug.
                # Let's check PannukeDataset implementation again.
                # It returns img (numpy) if no aug.
                pass
            
            # If it's numpy (H,W,C)
            if isinstance(img, np.ndarray):
                # Assuming it's uint8 [0,255] or float [0,1]?
                # Original code assumed: ((img * 0.5 + 0.5) * 255).permute(1, 2, 0)
                # This implies original dataset returned Tensor in [-1, 1].
                # My new PannukeDataset returns numpy directly from .npy
                # The .npy usually contains uint8 0-255 or float.
                # Let's assume it's the raw data from .npy.
                # If raw data is float 0-1, we scale. If uint8, we save directly.
                
                if img.dtype == np.float32 or img.dtype == np.float64:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                    
                # If shape is (C, H, W), transpose to (H, W, C)
                if img.shape[0] <= 3 and img.ndim == 3: 
                     # Heuristic check, but usually npy is (H,W,C) or (N,H,W,C)
                     # The dataset returns (H,W,C)
                     pass
            
            # If it is a Tensor (from previous logic)
            elif hasattr(img, 'cpu'):
                # This path is for safety if dataset changes
                img = img.cpu().numpy()
                if img.ndim == 3 and img.shape[0] <= 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            filename = os.path.join(class_dir, f"patch_{idx:05d}.png")
            Image.fromarray(img).save(filename)
        except Exception as e:
            print(f"Error saving patch {idx}: {e}")

    # Use ThreadPoolExecutor for I/O bound task
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_patch, range(len(ds))))

    print(f"Exported {len(ds)} patches into {class_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export patches from .npy to PNG')
    parser.add_argument('--fold_dir',   type=str, required=True,
                        help="Path to Pannuke fold (e.g. data/pannuke/Fold 1)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save patches; will create output_dir/class0")
    args = parser.parse_args()
    export_patches(args.fold_dir, args.output_dir)
