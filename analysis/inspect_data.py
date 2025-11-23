import numpy as np
import os
import sys

def inspect_masks(npy_path):
    print(f"Inspecting: {npy_path}")
    if not os.path.exists(npy_path):
        print("File not found.")
        return

    try:
        # Use mmap_mode to avoid loading entire file
        arr = np.load(npy_path, mmap_mode='r')
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")

        if arr.ndim == 4:
            # (N, H, W, C) - likely one-hot or multi-channel
            print("Detected 4D array (Multi-channel masks).")
            n_samples, h, w, n_channels = arr.shape
            print(f"Number of channels (classes): {n_channels}")
            
            # Calculate class distribution on a subset to save time
            subset_size = min(100, n_samples)
            print(f"Calculating distribution on first {subset_size} samples...")
            subset = arr[:subset_size]
            
            # Argmax to get class indices
            indices = np.argmax(subset, axis=-1)
            unique, counts = np.unique(indices, return_counts=True)
            total_pixels = indices.size
            
            print("Class Distribution (Pixel-wise):")
            for u, c in zip(unique, counts):
                print(f"  Class {u}: {c:10d} ({c/total_pixels*100:.2f}%)")
                
        elif arr.ndim == 3:
            # (N, H, W) - integer labels
            print("Detected 3D array (Integer labels).")
            subset_size = min(100, arr.shape[0])
            subset = arr[:subset_size]
            unique, counts = np.unique(subset, return_counts=True)
            total_pixels = subset.size
            
            print("Class Distribution (Pixel-wise):")
            for u, c in zip(unique, counts):
                print(f"  Class {u}: {c:10d} ({c/total_pixels*100:.2f}%)")
                
        else:
            print("Unknown dimensions.")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Check train and test masks
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_masks = os.path.join(project_root, "data", "pannuke", "train", "masks.npy")
    test_masks = os.path.join(project_root, "data", "pannuke", "test", "masks.npy")
    
    print("--- Train Masks ---")
    inspect_masks(train_masks)
    print("\n--- Test Masks ---")
    inspect_masks(test_masks)
