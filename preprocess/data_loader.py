import warnings
warnings.filterwarnings(
    "ignore",
    message=".*The given NumPy array is not writable.*",
    category=UserWarning
)

import numpy as np
import torch
from pathlib import Path
from PIL import Image                 # 新增：匯入 Image
from torch.utils.data import Dataset, DataLoader  # 修改：同時匯入 DataLoader
import logging

class PanNuke(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        data_root = Path(data_root)

        # 判斷資料夾結構：有 images/ 才切子資料夾，否則直接在根目錄找
        if (data_root/"images").exists():
            img_dir = data_root/"images"
        else:
            img_dir = data_root
        if (data_root/"masks").exists():
            mask_dir = data_root/"masks"
        else:
            mask_dir = data_root
        if (data_root/"types").exists():
            type_dir = data_root/"types"
        else:
            type_dir = data_root

        # 先搜尋所有檔案
        # 優先尋找 images.npy (合併後的檔案)
        if (img_dir / "images.npy").exists():
            all_img_files = [img_dir / "images.npy"]
        else:
            # 否則尋找所有 *_images.npy (例如 Fold_1_images.npy)
            all_img_files  = sorted(list(img_dir.rglob("*_images.npy")))
            
        if (mask_dir / "masks.npy").exists():
            all_mask_files = [mask_dir / "masks.npy"]
        else:
            all_mask_files = sorted(list(mask_dir.rglob("*_masks.npy")))
            
        if (type_dir / "types.npy").exists():
            all_type_files = [type_dir / "types.npy"]
        else:
            all_type_files = sorted(list(type_dir.rglob("*_types.npy")))

        # Debug info
        if not all_img_files:
            print(f"[Debug] Searching in {img_dir}")
            print(f"[Debug] Found files: {list(img_dir.glob('*'))}")
            raise RuntimeError(f"在 {img_dir} 找不到任何 *_images.npy 或 images.npy")
        if not all_mask_files:
            raise RuntimeError(f"在 {mask_dir} 找不到任何 *_masks.npy 或 masks.npy")
        if not all_type_files:
            raise RuntimeError(f"在 {type_dir} 找不到任何 *_types.npy 或 types.npy")

        # 建立字串標籤對應整數的映射表
        unique_types = set()
        for tfile in all_type_files:
            # 使用 mmap_mode='r' 減少記憶體消耗
            arr = np.load(tfile, mmap_mode='r')
            for v in np.unique(arr):
                unique_types.add(v)
        self.type2idx = {name: idx for idx, name in enumerate(sorted(unique_types))}

        # 建立 prefix -> (mask_path, type_path) 對照
        mask_lookup = {}
        for m in all_mask_files:
            if m.name == 'masks.npy':
                prefix = 'merged'
            else:
                prefix = m.stem.split('_')[0]
            mask_lookup.setdefault(prefix, {})['mask'] = m
        for t in all_type_files:
            if t.name == 'types.npy':
                prefix = 'merged'
            else:
                prefix = t.stem.split('_')[0]
            mask_lookup.setdefault(prefix, {})['type'] = t

        # 配對並展開 samples
        self.samples = []
        for img_path in all_img_files:
            if img_path.name == 'images.npy':
                prefix = 'merged'
            else:
                prefix = img_path.stem.split('_')[0]
            
            info = mask_lookup.get(prefix, {})
            mask_path = info.get('mask')
            type_path = info.get('type')
            if mask_path is None or type_path is None:
                logging.warning(f"[PanNuke] 跳過無對應配對：{img_path.name}")
                continue

            # 使用 mmap_mode='r' 只讀取 metadata (shape)，不讀取實際資料
            frames = np.load(img_path, mmap_mode='r')  # shape = (N, H, W, C)
            for idx in range(frames.shape[0]):
                self.samples.append((img_path, idx, mask_path, type_path))

        if not self.samples:
            raise RuntimeError("沒有任何配對成功的樣本，請檢查檔名是否正確。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, frame_idx, mask_path, type_path = self.samples[idx]
        
        # [Fix] Cache mmap objects to avoid repeated open/close and WinError 8
        if not hasattr(self, 'mmap_cache'):
            self.mmap_cache = {}
            
        def get_mmap(path):
            path_str = str(path)
            if path_str not in self.mmap_cache:
                self.mmap_cache[path_str] = np.load(path)
            return self.mmap_cache[path_str]

        # Use cached mmap
        mmap_img = get_mmap(img_path)
        img_np = np.array(mmap_img[frame_idx], copy=True)

        mmap_mask = get_mmap(mask_path)
        mask_np = np.array(mmap_mask[frame_idx], copy=True)

        mmap_type = get_mmap(type_path)
        types_raw = np.array(mmap_type[frame_idx], copy=True)

        # 如果是多通道 mask (H, W, C)，轉為單通道索引 (H, W)
        if mask_np.ndim == 3:
            mask_np = np.argmax(mask_np, axis=-1)

        # [Modified] Binary Segmentation: Cells (1) vs Background (0)
        # Original: 0:Neoplastic, 1:Inflammatory, 2:Connective, 3:Dead, 4:Epithelial, 5:Background
        # New:
        #   Background (0) <- Original 5 (Background) + 3 (Dead)
        #   Cells (1)      <- Original 0, 1, 2, 4
        
        # 先複製一份以免改壞
        new_mask = np.zeros_like(mask_np)
        
        # 設定前景 (Cells)
        # 只要不是 3 (Dead) 和 5 (Background)，就是前景
        is_cell = (mask_np != 3) & (mask_np != 5)
        new_mask[is_cell] = 1
        
        # 背景維持 0 (初始化就是 0)，包含原來的 3 和 5
        # (其實 new_mask 初始化全 0，所以只要把前景設為 1 即可)
        
        mask_np = new_mask

        # [Debug] Check mask values
        unique_vals = np.unique(mask_np)
        if len(unique_vals) > 2 or not np.all(np.isin(unique_vals, [0, 1])):
             print(f"[Warning] Mask values unexpected: {unique_vals} in {mask_path}")
        # Only print once to avoid spam
        if not hasattr(self, '_debug_printed'):
             print(f"[Debug] First mask unique values: {unique_vals}")
             self._debug_printed = True

        # 如果是字串陣列，透過對照表映射
        if types_raw.dtype.kind in ('U', 'S'):
            mapper = np.vectorize(lambda x: self.type2idx[x])
            types_np = mapper(types_raw).astype(np.int64)
        else:
            types_np = types_raw.astype(np.int64)
        
        # 確保是 copy 且 writable (mmap 返回的是 read-only)
        img_np = np.array(img_np, copy=True)
        mask_np = np.array(mask_np, copy=True)
        types_np = np.array(types_np, copy=True)

        # Apply transforms if provided
        if self.transform:
            # Albumentations expects image in (H, W, C) and mask in (H, W)
            # img_np is already (H, W, C) uint8 or float
            # mask_np is (H, W)
            img_np = np.asarray(img_np)
            mask_np = np.asarray(mask_np)

            # Enforce uint8 for OpenCV compatibility first (to handle float64 inputs)
            if img_np.dtype.kind == 'f':
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            elif img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            # [Fix] Convert to float32 BEFORE transform because A.Normalize expects float
            # and we disabled other augmentations that might need uint8.
            img_np = img_np.astype(np.float32) / 255.0

            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented["image"]
            mask_np = augmented["mask"]

        # 轉成 Tensor
        # Check if img_np is already tensor (some transforms might convert it)
        if not isinstance(img_np, torch.Tensor):
             img = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        else:
             img = img_np

        if not isinstance(mask_np, torch.Tensor):
            mask = torch.from_numpy(mask_np).long()
        else:
            mask = mask_np
            
        types = torch.from_numpy(types_np).long()
        return img, mask, types
class TCIApng(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(Path(root).rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_np = np.array(Image.open(self.paths[idx]).convert("RGB"), dtype=np.float32)
        arr = np.ascontiguousarray(img_np)
        img = torch.from_numpy(arr).permute(2, 0, 1) / 255.0

        if self.transform:
            img = self.transform(img)
        return img


class SRpair(Dataset):
    def __init__(self, lr_root, hr_root, transform=None):
        self.lr_paths = sorted(Path(lr_root).rglob("*.png"))
        self.hr_paths = sorted(Path(hr_root).rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_np = np.array(Image.open(self.lr_paths[idx]).convert("RGB"), dtype=np.float32)
        hr_np = np.array(Image.open(self.hr_paths[idx]).convert("RGB"), dtype=np.float32)

        lr_arr = np.ascontiguousarray(lr_np)
        hr_arr = np.ascontiguousarray(hr_np)

        lr = torch.from_numpy(lr_arr).permute(2, 0, 1) / 255.0
        hr = torch.from_numpy(hr_arr).permute(2, 0, 1) / 255.0

        if self.transform:
            return self.transform(lr, hr)
        return lr, hr


def _detect_pannuke(root):
    # 只要資料夾底下有 *_images.npy 就視為 PanNuke
    return any(Path(root).rglob("*_images.npy"))


def make_loader(root,
                batch_size=8,
                shuffle=True,
                num_workers=4,
                transform=None,
                sr=False,
                hr_root=None,
                pin_memory=True):
    if sr:
        dataset = SRpair(root, hr_root, transform)
    elif _detect_pannuke(root):
        dataset = PanNuke(root)
    else:
        dataset = TCIApng(root, transform)

    if len(dataset) == 0:
        raise FileNotFoundError(f"No usable data found under: {root}")

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)



