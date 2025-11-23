import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class SimpleImageDataset(Dataset):
    """
    Dataset for loading images from a folder (PNG, JPG, TIF).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Support common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(sorted(self.root_dir.rglob(ext)))
            
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")
            
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Basic normalization if no transform provided (0-1 float)
        if self.transform:
            # Albumentations expects numpy
            augmented = self.transform(image=img_np)
            img_tensor = augmented['image']
        else:
            # Default to simple tensor conversion
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            
        return img_tensor, str(img_path)

class PanNukeDataset(Dataset):
    """
    Simplified PanNuke Dataset for Inference.
    Reads *_images.npy files.
    """
    def __init__(self, data_root, transform=None):
        self.transform = transform
        data_root = Path(data_root)
        
        # Find images.npy or *_images.npy
        if (data_root / "images.npy").exists():
            self.img_files = [data_root / "images.npy"]
        else:
            self.img_files = sorted(data_root.rglob("*_images.npy"))
            
        if not self.img_files:
            raise RuntimeError(f"No *_images.npy found in {data_root}")

        self.samples = []
        for img_path in self.img_files:
            # Read metadata only
            frames = np.load(img_path, mmap_mode='r')
            for idx in range(frames.shape[0]):
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, frame_idx = self.samples[idx]
        
        # Load specific frame
        # Use mmap to avoid loading full file
        mmap_img = np.load(img_path, mmap_mode='r')
        img_np = np.array(mmap_img[frame_idx], copy=True) # (H, W, C)
        
        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            
        # Generate a dummy name
        name = f"{img_path.stem}_{frame_idx:05d}"
        
        return img_tensor, name
