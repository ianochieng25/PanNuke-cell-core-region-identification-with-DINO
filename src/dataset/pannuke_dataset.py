import os
import glob
import numpy as np
from torch.utils.data import Dataset

class PannukeDataset(Dataset):
    def __init__(self, data_root, aug=None):
        self.aug = aug
        self.samples = []  # List of (img_path, mask_path, type_path, local_idx)

        # 1. Find all shards
        # We look for *_images.npy, *_types.npy, *_masks.npy
        # Or images.npy, types.npy, masks.npy
        
        # Helper to find files
        def find_files(pattern):
            files = sorted(glob.glob(os.path.join(data_root, pattern)))
            if not files:
                # Try single file without prefix
                single = os.path.join(data_root, pattern.replace("*_", ""))
                if os.path.isfile(single):
                    files = [single]
            return files

        img_files = find_files("*_images.npy")
        type_files = find_files("*_types.npy")
        mask_files = find_files("*_masks.npy")

        if not img_files:
            print(f"[Warning] No images found in {data_root}")
            return

        # Create a mapping from prefix to files
        # Prefix is usually the fold number or 'images' if it's a merged file
        file_map = {}

        def get_prefix(path):
            name = os.path.basename(path)
            if name in ['images.npy', 'types.npy', 'masks.npy']:
                return 'merged'
            return name.split('_')[0]

        for p in img_files:
            file_map.setdefault(get_prefix(p), {})['img'] = p
        for p in type_files:
            file_map.setdefault(get_prefix(p), {})['type'] = p
        for p in mask_files:
            file_map.setdefault(get_prefix(p), {})['mask'] = p

        # Build sample index
        for prefix, paths in file_map.items():
            img_path = paths.get('img')
            mask_path = paths.get('mask')
            type_path = paths.get('type')

            if not img_path: continue
            
            # Use mmap to get length without loading data
            try:
                # mmap_mode='r' is enough to read shape
                img_arr = np.load(img_path, mmap_mode='r')
                length = img_arr.shape[0]
                
                # Verify other files exist and have same length (optional but good for safety)
                # We allow missing masks/types if they are not strictly required, 
                # but for this dataset they usually go together.
                # If missing, we can pass None and handle in __getitem__
                
                for i in range(length):
                    self.samples.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'type_path': type_path,
                        'idx': i
                    })
            except Exception as e:
                print(f"[Error] Failed to read metadata for {img_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # [Fix] Cache mmap objects to avoid repeated open/close and WinError 8
        if not hasattr(self, 'mmap_cache'):
            self.mmap_cache = {}
            
        def get_mmap(path):
            path_str = str(path)
            if path_str not in self.mmap_cache:
                self.mmap_cache[path_str] = np.load(path, mmap_mode='r')
            return self.mmap_cache[path_str]
        
        # Lazy load using cache
        # Note: np.load with mmap_mode returns a memmap object, slicing it reads the data
        img = get_mmap(sample['img_path'])[sample['idx']]
        
        if sample['mask_path']:
            msk = get_mmap(sample['mask_path'])[sample['idx']]
        else:
            msk = np.zeros(img.shape[:2], dtype=np.int32) # Default mask
            
        if sample['type_path']:
            lbl = get_mmap(sample['type_path'])[sample['idx']]
        else:
            lbl = 0 # Default label

        # Ensure data is contiguous and writable (mmap is read-only)
        img = np.array(img, copy=True)
        msk = np.array(msk, copy=True)
        # lbl is usually a scalar or small array, copy is cheap
        lbl = np.array(lbl, copy=True)

        # [Fix] Convert to float32 [0, 1] BEFORE transform to avoid OpenCV type errors
        # Enforce uint8 first if it's float > 1.0 (just in case)
        if img.dtype.kind == 'f':
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        # Then convert to float32 [0, 1] for Albumentations
        img = img.astype(np.float32) / 255.0

        if self.aug:
            aug = self.aug(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]
            
        # Handle string labels
        if lbl.dtype.kind in ('U', 'S'):
            # Simple mapping based on known PanNuke types if not already mapped
            # Assuming standard PanNuke types: Neoplastic, Inflammatory, Connective, Dead, Epithelial
            # But here we might just need a consistent mapping.
            # Let's try to use the same mapping logic as data_loader.py if possible, 
            # or just hash it/index it if we built the index in __init__.
            # Since we didn't build type2idx in __init__ yet, let's do a safe fallback or simple hash for now
            # OR better, let's just return 0 if it's a string and we don't use it for DINO training (which uses images only usually).
            # However, extract_vit_features might use it? No, it uses images.
            # But the collate function will still try to stack them.
            # Let's map known types to ints.
            
            type_map = {
                'Neoplastic': 0,
                'Inflammatory': 1,
                'Connective': 2,
                'Dead': 3,
                'Epithelial': 4,
                'Background': 5
            }
            lbl_str = str(lbl)
            lbl = type_map.get(lbl_str, 0) # Default to 0 if unknown
        
        lbl = np.array(lbl, dtype=np.int64)

        return img, msk, lbl

