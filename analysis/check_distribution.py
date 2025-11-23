import numpy as np
from pathlib import Path
import os

def check_distribution(data_root):
    print(f"Checking distribution in {data_root}")
    mask_path = Path(data_root) / "masks" / "masks.npy"
    
    if not mask_path.exists():
        print(f"File not found: {mask_path}")
        # Try to find subfiles if merged file doesn't exist
        mask_files = sorted(Path(data_root).rglob("*_masks.npy"))
        if not mask_files:
            print("No mask files found.")
            return
        print(f"Found {len(mask_files)} mask files. Checking the first one...")
        mask_path = mask_files[0]

    print(f"Loading {mask_path}...")
    masks = np.load(mask_path, mmap_mode='r')
    print(f"Shape: {masks.shape}")
    
    # Sample a subset if too large
    if len(masks) > 100:
        sample_indices = np.random.choice(len(masks), 100, replace=False)
        masks_sample = masks[sample_indices]
    else:
        masks_sample = masks

    if masks_sample.ndim == 4: # (N, H, W, C)
        print("Detected 4D masks (One-hot or Multi-channel). Converting to indices via argmax...")
        masks_indices = np.argmax(masks_sample, axis=-1)
    else:
        masks_indices = masks_sample

    unique, counts = np.unique(masks_indices, return_counts=True)
    total_pixels = masks_indices.size
    
    print("\nClass Distribution (in sample):")
    for cls, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        print(f"Class {cls}: {count} pixels ({percentage:.2f}%)")

if __name__ == "__main__":
    # Adjust path as needed
    data_root = r"d:\pannuke project 改進\pannuke_project\data" 
    check_distribution(data_root)
