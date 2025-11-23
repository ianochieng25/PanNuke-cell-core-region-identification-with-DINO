import requests
import argparse
import os
import sys
import glob
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dino.main_dino import train_dino
import random
import logging
from tqdm import tqdm
import cv2
import visualize_predictions  # Integrated visualization module
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

PANNUKE_ROOT = os.path.join(PROJECT_ROOT, 'data', 'pannuke')

ORIG_FOLD_IDX = {'train': 1, 'test': 2}

# 設定模組搜尋路徑，確保自訂套件可被找到
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'preprocess'))


# 清除 checkpoints 中的訓練產物，但保留預訓練權重

def reset_predictions():
    if os.path.exists(PRED_ROOT):
        shutil.rmtree(PRED_ROOT)
    os.makedirs(PRED_ROOT)
    print(f"🧹 已清空預測結果資料夾：{PRED_ROOT}")

def clean_checkpoints_but_keep_pretrained():
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints')
    keep_filename = 'dino_vitsmall16_pretrain.pth'

    if not os.path.exists(ckpt_path):
        print(f"[ℹ️] checkpoints 資料夾不存在，略過清理。")
        return

    # 1. 收集要保留的檔案完整路徑
    keep_files = []
    for root, dirs, files in os.walk(ckpt_path):
        for f in files:
            if f == keep_filename:
                keep_files.append(os.path.join(root, f))

    # 2. 收集這些檔案的所有上層資料夾（包含自己）
    keep_dirs = set()
    for fpath in keep_files:
        dirpath = fpath
        while True:
            dirpath = os.path.dirname(dirpath)
            keep_dirs.add(dirpath)
            if dirpath == ckpt_path:
                break

    # 3. 遍歷 checkpoints，刪掉不在保留名單的檔案與資料夾
    for root, dirs, files in os.walk(ckpt_path, topdown=False):
        for f in files:
            fpath = os.path.join(root, f)
            if fpath in keep_files:
                print(f"[✓] 保留預訓練模型：{fpath}")
                continue
            try:
                os.remove(fpath)
                print(f"[🧹] 已刪除檔案：{fpath}")
            except Exception as e:
                print(f"[⚠️] 無法刪除 {fpath}：{e}")
        for d in dirs:
            dirpath = os.path.join(root, d)
            if dirpath in keep_dirs:
                print(f"[✓] 保留資料夾（含預訓練模型）：{dirpath}")
                continue
            try:
                shutil.rmtree(dirpath)
                print(f"[🧹] 已刪除資料夾：{dirpath}")
            except Exception as e:
                print(f"[⚠️] 無法刪除 {dirpath}：{e}")



import shutil

def prompt_reset():
    choice = input("是否重置所有訓練過程？輸入 y 重置，其他鍵跳過：").strip().lower()
    if choice == 'y':
        print("正在重置訓練資料...")
        # reset_split_dirs()  # Removed manual split reset
        # shuffle_and_split_pannuke() # Removed manual split logic
        clean_checkpoints_but_keep_pretrained()
        reset_predictions()
        clean_temp_files()
        print("✅ 重置完成，將從頭開始訓練。\n")
    else:
        print("⏭️ 跳過重置，保留現有訓練資料。\n")


from dataset.pannuke_dataset import PannukeDataset
from augment import get_training_augmentation, get_validation_augmentation

# 顯示 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 資料夾設定
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PANNUKE_ROOT = os.path.join(PROJECT_ROOT, "data", "pannuke")
PRED_ROOT = os.path.join(PROJECT_ROOT, "predictions")

ORIG_FOLD_IDX = {'train': 1, 'test': 2}

# 官方預訓練權重路徑與下載連結
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dino", "dino_deitsmall16_pretrain_full_checkpoint.pth")
PRETRAINED_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth"

# 原始 fold 路徑：Fold 1, 2, 3
RAW_FOLDS = [
    os.path.join(PANNUKE_ROOT, 'Fold 1'),
    os.path.join(PANNUKE_ROOT, 'Fold 2'),
    os.path.join(PANNUKE_ROOT, 'Fold 3'),
]

# shuffle 後要儲存的新位置
FOLDS = {
    'train': os.path.join(PANNUKE_ROOT, 'train'),
    'test':  os.path.join(PANNUKE_ROOT, 'test'),
}




def setup_train_test_split():
    """
    Automatically organize Fold 1, 2, 3 into train and test directories.
    Train: Fold 1, Fold 2
    Test: Fold 3
    Renames files to {Fold}_images.npy etc. to avoid conflicts.
    """
    print("Checking dataset structure...")
    
    # Define source folds and their destination
    # You can adjust which fold goes where
    split_config = {
        'Fold 1': 'train',
        'Fold 2': 'train',
        'Fold 3': 'test'
    }
    
    for fold_name, split_name in split_config.items():
        src_fold = os.path.join(PANNUKE_ROOT, fold_name)
        dst_split = FOLDS[split_name]
        
        if not os.path.exists(src_fold):
            print(f"[Warn] Source {src_fold} does not exist. Skipping.")
            continue
            
        os.makedirs(dst_split, exist_ok=True)
        
        # Files to copy - Handle both "images.npy" and "1_images.npy"
        # We look for *images.npy, *masks.npy, *types.npy
        for suffix in ['images.npy', 'masks.npy', 'types.npy']:
            # Find files matching the suffix
            candidates = glob.glob(os.path.join(src_fold, f"*{suffix}"))
            
            for src_file in candidates:
                filename = os.path.basename(src_file)
                
                # Construct new name: Fold_1_images.npy or similar
                # If filename already has fold info (e.g. 1_images.npy), we can keep it or normalize it.
                # Let's normalize to Fold_X_images.npy to be safe and consistent.
                
                # Simple normalization:
                # If it starts with a digit (1_images.npy), prepend Fold_
                if filename[0].isdigit():
                     new_name = f"Fold_{filename}"
                else:
                     new_name = f"{fold_name.replace(' ', '_')}_{filename}"
                     
                dst_file = os.path.join(dst_split, new_name)
                
                if not os.path.exists(dst_file):
                    print(f"[Setup] Copying {src_file} -> {dst_file}")
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        print(f"[Error] Failed to copy {src_file}: {e}")
                else:
                    # print(f"[Setup] {dst_file} already exists.")
                    pass

def merge_split_npy():
    """合併 train/val/test 資料夾裡的 *_images.npy、*_types.npy、*_masks.npy 為單一檔案"""
    for split, folder in FOLDS.items():
        # 定義要合併的目標
        targets = [
            ('images.npy', '*_images.npy'),
            ('types.npy',  '*_types.npy'),
            ('masks.npy',  '*_masks.npy')
        ]
        
        for out_name, pattern in targets:
            files = sorted(glob.glob(os.path.join(folder, pattern)))
            # 排除已經合併好的檔案自己 (避免重複讀取)
            files = [f for f in files if os.path.basename(f) != out_name]
            
            if not files:
                continue
                
            out_path = os.path.join(folder, out_name)
            print(f"[merge] 正在合併 {out_name} 到 {out_path} ...")

            # 1. 計算總長度與形狀
            # 1. 計算總長度與形狀
            total_len = 0
            shapes = []
            dtype = None
            
            # 先讀取 metadata（不使用 mmap，避免句柄耗盡）
            for f in files:
                arr = np.load(f)  # 直接載入完整檔案
                if dtype is None:
                    dtype = arr.dtype
                total_len += arr.shape[0]
                shapes.append(arr.shape)
                del arr  # 釋放記憶體
            
            if total_len == 0:
                continue
            
            # 2. 建立輸出的 memmap 檔案
            # 使用第一個檔案的 shape[1:] 作為特徵維度
            final_shape = (total_len,) + shapes[0][1:]
            
            # 如果檔案已存在先刪除
            if os.path.exists(out_path):
                os.remove(out_path)
            
            # 建立 memmap
            merged = np.lib.format.open_memmap(out_path, mode='w+', dtype=dtype, shape=final_shape)
            
            # 3. 分批寫入
            current_idx = 0
            for f in files:
                arr = np.load(f)
                n = arr.shape[0]
                merged[current_idx : current_idx + n] = arr[:]
                current_idx += n
                del arr  # 釋放記憶體
            
            # 確保寫入磁碟
            merged.flush()
            del merged
            print(f"  [OK] {out_name} 合併完成 (shape={final_shape})")





def clean_temp_files():
    temp_dirs = ['__pycache__', 'tmp', 'cache']
    for d in temp_dirs:
        dir_path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            print(f"🧹 清除暫存資料夾：{dir_path}")

def find_types_npy(fold_dir):
    matches = glob.glob(os.path.join(fold_dir, "**", "*type*.npy"), recursive=True)
    return matches[0] if len(matches) == 1 else None

def count_labels(types_path):
    arr = np.load(types_path, allow_pickle=True).flatten()
    unique, counts = np.unique(arr, return_counts=True)
    return {str(u): int(c) for u, c in zip(unique, counts)}, arr.size

def count_distribution():
    print("=== PanNuke 各 Fold Patch 分類標籤分佈 ===\n")
    for split, folder in FOLDS.items():
        print(f"--- {split.upper():5s} ({folder}) ---")
        # 先試單一 types.npy，若不存在再找所有 *_types.npy
        types_paths = glob.glob(os.path.join(folder, "types.npy"))
        if not types_paths:
            types_paths = glob.glob(os.path.join(folder, "*_types.npy"))
        if not types_paths:
            print(f"  [Error] 找不到任何 types.npy 或 *_types.npy: {folder}", file=sys.stderr)
            continue

        # 載入並合併
        arrs = []
        for p in sorted(types_paths):
            try:
                arrs.append(np.load(p).flatten())
            except Exception as e:
                print(f"  [Warning] 無法讀取 {p}：{e}", file=sys.stderr)
        if not arrs:
            print(f"  [Error] 無可用的 label 資料", file=sys.stderr)
            continue

        all_labels = np.concatenate(arrs)
        unique, counts = np.unique(all_labels, return_counts=True)
        total = all_labels.size

        print(f"  總樣本數：{total}")
        print(f"  {'Label':>20s}{'Count':>10s}{'Percent':>10s}")
        print("  " + "-"*42)
        for lbl, cnt in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            print(f"  {str(lbl):>20s}{cnt:>10d}{cnt/total*100:>10.2f}%")
        print()

def find_pred_npy(split):
    cand = glob.glob(os.path.join(PRED_ROOT, f"{split}*_pred*.npy"))
    if len(cand) == 1:
        return cand[0]
    all_preds = glob.glob(os.path.join(PRED_ROOT, "*.npy"))
    if len(all_preds) == 1:
        return all_preds[0]
    pred_files = [p for p in all_preds if os.path.basename(p).startswith('pred')]
    return pred_files[0] if len(pred_files) == 1 else None

def evaluate_classification():
    for split, folder in FOLDS.items():
        print(f"=== {split.upper():5s} ({folder}) ===")
        types_path = find_types_npy(os.path.join(PANNUKE_ROOT, folder))
        if not types_path:
            print("  [skip] 找不到 types.npy，跳過評估", file=sys.stderr)
            continue
        y_true = np.load(types_path).flatten()
        pred_path = find_pred_npy(split)
        if not pred_path or not os.path.isfile(pred_path):
            print(f"  [skip] 找不到預測檔案 for {split}，跳過評估", file=sys.stderr)
            continue
        y_pred = np.load(pred_path).flatten()
        if y_true.shape != y_pred.shape:
            print(f"  [skip] true {y_true.shape} vs pred {y_pred.shape} 不符，跳過評估", file=sys.stderr)
            continue
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(f"  Accuracy: {acc:.4f}\n")
        print("  Classification Report:"); [print(f"    {line}") for line in report.splitlines()]
        print("  Confusion Matrix:")
        print("   " + " ".join(f"{i:>4d}" for i in range(cm.shape[0])))
        for i, row in enumerate(cm): print(f"{i:>2d} " + " ".join(f"{c:>4d}" for c in row))
        print()

def step_done_dir(path: str) -> bool:
    return os.path.isdir(path) and len(os.listdir(path)) > 0

def step_done_file(path: str) -> bool:
    return os.path.isfile(path)

def run_script(rel_path, args=None, cwd=None):
    script = os.path.join(PROJECT_ROOT, rel_path)
    workdir = os.path.join(PROJECT_ROOT, cwd) if cwd else PROJECT_ROOT
    cmd = [sys.executable, script] + (args or [])
    logging.info(f">>> Running: {' '.join(cmd)} (cwd={workdir})")
    # 改為不 capture_output，讓子程序的 stdout/stderr 直接顯示在終端機 (包含 tqdm 進度條)
    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}"); sys.exit(result.returncode)

def extract_vit_features(checkpoint_path, data_root, output_dir, batch_size=16):
    os.makedirs(output_dir, exist_ok=True)
    ds = PannukeDataset(data_root, aug=get_training_augmentation())  # ★加強化
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = create_model('vit_small_patch8_224', pretrained=False, num_classes=0)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k,v in ckpt.get('student', ckpt).items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    all_cls, all_patches = [], []
    model.eval().cuda()
    all_cls, all_patches = [], []
    with torch.no_grad():
        for imgs, _, _ in tqdm(loader, desc='Extracting Features'):
            imgs = F.interpolate(imgs.cuda(non_blocking=True), size=(224,224), mode='bicubic', align_corners=False)
            tokens = model.forward_features(imgs)
            all_cls.append(tokens[:,0,:].cpu().numpy()); all_patches.append(tokens[:,1:,:].cpu().numpy())
    np.save(os.path.join(output_dir,'cls_tokens.npy'),np.concatenate(all_cls,axis=0)); np.save(os.path.join(output_dir,'patch_embeddings.npy'),np.concatenate(all_patches,axis=0))
    print(f"Extracted features: CLS {all_cls[0].shape}, patches {all_patches[0].shape} → {output_dir}")

def run_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    # 1. 匯出 patches
    patches_dir = os.path.join(PROJECT_ROOT, 'preprocess', 'patches', 'train_patches')
    class0_dir = os.path.join(patches_dir, 'class0')
    if not step_done_dir(class0_dir):
        print(f"[Info] Patches missing or empty in {class0_dir}, running exporter...")
        # Clean up potential empty dir to be safe
        if os.path.exists(patches_dir):
            shutil.rmtree(patches_dir)
            
        run_script('preprocess/patch_exporter.py',
                   ['--fold_dir', os.path.join(PANNUKE_ROOT, 'train'),
                    '--output_dir', patches_dir])
    else:
        print(f"[skip] patches 已存在：{patches_dir}")

    # 2. DINO 自監督預訓練
    dino_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'dino', 'checkpoint.pth')
    if not step_done_file(dino_ckpt):
        dino_args = argparse.Namespace(
            data_path=patches_dir,
            output_dir=os.path.join(PROJECT_ROOT, 'checkpoints', 'dino'),
            arch="vit_small_patch16_224",
            patch_size=16,
            out_dim=1024,               # 安全先用 1024（或 2048/4096 都行）
            batch_size_per_gpu=2,       # 一定不要超過 4，先 2 最安全
            epochs=1,                   # 只訓練 1 epoch 確認 pipeline 通順
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=0,
            student_temp=0.1,
            local_crops_number=0,       # 只產生 global crop，安全
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            lr=0.0005,
            min_lr=1e-6,
            clip_grad=3.0,
            weight_decay=0.04,
            saveckp_freq=1,             # 每 epoch 存 checkpoint
            dist_url="tcp://127.0.0.1:29500",
            use_fp16=False,
            momentum_teacher=0.996,
            num_workers=2,
            freeze_last_layer=1               # 加這個，避免多線程搶記憶體（或直接設 0）
        )
        train_dino(dino_args)
    else:
        print(f"[skip] DINO checkpoint 已存在：{dino_ckpt}")

    # 3. 特徵提取
    features_dir = os.path.join(PROJECT_ROOT, 'features', 'train_features')
    if not step_done_dir(features_dir):
        extract_vit_features(checkpoint_path=dino_ckpt,
                             data_root=os.path.join(PANNUKE_ROOT, 'train'),
                             output_dir=features_dir,
                             batch_size=16)
    else:
        print(f"[skip] features 已存在：{features_dir}")

    # 4. Segmentation 訓練
    seg_ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentor')
    if not step_done_dir(seg_ckpt_dir):
        run_script('segmentor/train_segmentor.py',
                   ['--data_root', os.path.join(PANNUKE_ROOT),
                    '--checkpoint_dir', seg_ckpt_dir,
                    '--pretrained_ckpt', dino_ckpt,  # [Fix] Load DINO pretrained weights
                    '--batch_size', '4',
                    '--grad_accum_steps', '4',
                    '--num_workers', '0']) # [Fix] Set to 0 to avoid WinError 8
    else:
        print(f"[skip] segmentor checkpoints 已存在：{seg_ckpt_dir}")

    # 5. Segmentation 推論：單張 NPY/PNG + 單一大陣列 NPY
    pred_single_dir = os.path.join(PROJECT_ROOT, 'predictions', 'single_masks')
    os.makedirs(pred_single_dir, exist_ok=True)
    pred_combined_npy = os.path.join(PRED_ROOT, 'pred_masks.npy')
    for split, folder in FOLDS.items():
        pred_npy = os.path.join(PRED_ROOT, f"{split}_pred_masks.npy")
        run_script(
            'segmentor/predict_segmentor.py',
            [
                '--weights', os.path.join(seg_ckpt_dir, 'model_final.pth'),
                '--input_dir', os.path.join(PANNUKE_ROOT, folder),
                '--output_dir', pred_single_dir,
                '--output_npy', pred_npy,
                '--batch_size', '8'
            ],
            cwd='segmentor'
        )

    # 6. 特徵可視化
    viz_png = os.path.join(PROJECT_ROOT, 'results', 'tsne.png')
    if not step_done_file(viz_png):
        run_script('analysis/feature_visualizer.py',
                ['--features_dir', features_dir,
                    '--output_png', viz_png])
        clean_temp_files()
    else:
        print(f"[skip] 可視化結果已存在：{viz_png}")
        clean_temp_files()

    # 7. 預測結果可視化 (Overlay Masks)
    print("\n=== 生成預測結果視覺化圖 (Overlay) ===")
    for split in ['train', 'test']:
        visualize_predictions.visualize_split(split, num_samples=10)




def clean_mask(mask_binary):
    """
    Apply morphological opening to remove small noise.
    mask_binary: (N, H, W) or (H, W) numpy array, uint8
    """
    kernel = np.ones((3,3), np.uint8)
    # Ensure uint8
    if mask_binary.dtype != np.uint8:
        mask_binary = mask_binary.astype(np.uint8)
        
    if mask_binary.ndim == 2:
        return cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    
    cleaned = np.zeros_like(mask_binary)
    for i in range(mask_binary.shape[0]):
        cleaned[i] = cv2.morphologyEx(mask_binary[i], cv2.MORPH_OPEN, kernel)
    return cleaned

def evaluate_segmentation_for_fold(fold_name, fold_folder, num_classes=6):
    # 確保已合併 shards 為單一檔案
    merge_split_npy()

    print(f"=== {fold_name} ({fold_folder}) 分割結果評估 ===")
    # 1. 直接從 split 資料夾讀 GT masks.npy
    gt_path = os.path.join(fold_folder, "masks.npy")
    if not os.path.isfile(gt_path):
        print(f"[Warn] GT masks.npy 不存在，執行 merge_split_npy()", file=sys.stderr)
        merge_split_npy()
        if not os.path.isfile(gt_path):
            print(f"[Error] 仍找不到 GT masks.npy: {gt_path}", file=sys.stderr)
            return

    print(f"=== {fold_name} ({fold_folder}) 分割結果評估 ===")
    # 1. 直接從 split 資料夾讀 GT masks.npy
    gt_path = os.path.join(fold_folder, "masks.npy")
    if not os.path.isfile(gt_path):
        print(f"[Warn] GT masks.npy 不存在，執行 merge_split_npy()", file=sys.stderr)
        merge_split_npy()
        if not os.path.isfile(gt_path):
            print(f"[Error] 仍找不到 GT masks.npy: {gt_path}", file=sys.stderr)
            return

    # 2. 讀預測結果
    pred_path = os.path.join(PRED_ROOT, f"{fold_name}_pred_masks.npy")
    if not os.path.isfile(pred_path):
        print(f"[Error] 找不到預測檔: {pred_path}", file=sys.stderr)
        return

    # 使用 mmap 讀取
    gt_mmap = np.load(gt_path, mmap_mode='r')
    pred_mmap = np.load(pred_path, mmap_mode='r')

    if gt_mmap.shape[0] != pred_mmap.shape[0]:
        print(f"[Error] GT count {gt_mmap.shape[0]} vs Pred count {pred_mmap.shape[0]} 不符", file=sys.stderr)
        return

    # 分塊計算 IoU
    # Binary: Class 0 (Background), Class 1 (Cells)
    # Intersection & Union accumulators
    inter_acc = {0: 0, 1: 0}
    union_acc = {0: 0, 1: 0}
    
    chunk_size = 1000
    total = gt_mmap.shape[0]
    
    print(f"Evaluating Binary Segmentation (0=Bg, 1=Cells) in chunks...")
    
    for start_idx in range(0, total, chunk_size):
        end_idx = min(start_idx + chunk_size, total)
        
        # Load chunk
        gt_chunk = gt_mmap[start_idx:end_idx]
        pred_chunk = pred_mmap[start_idx:end_idx]
        
        # Handle GT format
        if gt_chunk.ndim == 4:
            gt_chunk = np.argmax(gt_chunk, axis=-1)
            
        # Handle Pred format
        if pred_chunk.ndim == 4:
            pred_chunk = np.argmax(pred_chunk, axis=-1)
            
        # Convert GT to Binary
        # Original 3 (Dead), 5 (Background) -> 0 (Background)
        # Original 0, 1, 2, 4 -> 1 (Cells)
        gt_binary = np.zeros_like(gt_chunk, dtype=np.uint8)
        is_cell_gt = (gt_chunk != 3) & (gt_chunk != 5)
        gt_binary[is_cell_gt] = 1
        
        # Convert Pred to Binary (if needed)
        pred_binary = pred_chunk
        if pred_chunk.max() > 1:
             pred_binary = np.zeros_like(pred_chunk, dtype=np.uint8)
             is_cell_pred = (pred_chunk != 3) & (pred_chunk != 5)
             pred_binary[is_cell_pred] = 1
        
        # Apply noise filtering
        pred_binary = clean_mask(pred_binary)
        
        # Accumulate
        for cls in [0, 1]:
            inter = np.logical_and(gt_binary == cls, pred_binary == cls).sum()
            union = np.logical_or(gt_binary == cls, pred_binary == cls).sum()
            inter_acc[cls] += inter
            union_acc[cls] += union

    # Calculate Final IoU and Dice
    class_ious = []
    class_dices = []
    
    # Also accumulate total pred and gt pixels for Dice
    pred_acc = {0: 0, 1: 0}
    gt_acc = {0: 0, 1: 0}
    
    # Recalculate to get pred and gt counts
    for start_idx in range(0, total, chunk_size):
        end_idx = min(start_idx + chunk_size, total)
        gt_chunk = gt_mmap[start_idx:end_idx]
        pred_chunk = pred_mmap[start_idx:end_idx]
        
        if gt_chunk.ndim == 4:
            gt_chunk = np.argmax(gt_chunk, axis=-1)
        if pred_chunk.ndim == 4:
            pred_chunk = np.argmax(pred_chunk, axis=-1)
            
        gt_binary = np.zeros_like(gt_chunk, dtype=np.uint8)
        is_cell_gt = (gt_chunk != 3) & (gt_chunk != 5)
        gt_binary[is_cell_gt] = 1
        
        pred_binary = pred_chunk
        if pred_chunk.max() > 1:
            pred_binary = np.zeros_like(pred_chunk, dtype=np.uint8)
            is_cell_pred = (pred_chunk != 3) & (pred_chunk != 5)
            pred_binary[is_cell_pred] = 1
            
        # Apply noise filtering
        pred_binary = clean_mask(pred_binary)
            
        for cls in [0, 1]:
            pred_acc[cls] += (pred_binary == cls).sum()
            gt_acc[cls] += (gt_binary == cls).sum()
    
    print(f"\n{'='*60}")
    print(f"  {fold_name.upper()} Segmentation Metrics")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'IoU':>10} {'Dice':>10}")
    print(f"{'-'*60}")
    
    for cls in [0, 1]:
        i = inter_acc[cls]
        u = union_acc[cls]
        p = pred_acc[cls]
        g = gt_acc[cls]
        
        iou = i / u if u > 0 else float('nan')
        dice = (2 * i) / (p + g) if (p + g) > 0 else float('nan')
        
        class_ious.append(iou)
        class_dices.append(dice)
        
        cls_name = "Background" if cls == 0 else "Cells"
        print(f"{cls_name:<15} {iou:>10.4f} {dice:>10.4f}")

    # 平均 IoU 和 Dice
    miou = np.nanmean(class_ious)
    mdice = np.nanmean(class_dices)
    print(f"{'-'*60}")
    print(f"{'Mean':<15} {miou:>10.4f} {mdice:>10.4f}")
    print(f"{'='*60}\n")


def evaluate_all_splits():
    for split, folder in FOLDS.items():
        evaluate_segmentation_for_fold(split, folder, num_classes=2)

    # Download code removed as it is unnecessary here and causing 403 errors
    pass




def need_seg_train() -> bool:
    """若 checkpoints/segmentor/model_final.pth 不存在則回傳 True"""
    seg_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentor', 'model_final.pth')
    return not os.path.isfile(seg_ckpt)

def finetune_with_pretrained(pretrained_ckpt):
    # 0. 詢問是否要進行 segmentation 微調訓練
    while True:
        ft_choice = input("[微調] 是否要進行 segmentation 微調訓練？(y/n): ").strip().lower()
        if ft_choice in ('y', 'n'):
            break
        print("請輸入 y 或 n。")

    seg_ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentor')
    os.makedirs(seg_ckpt_dir, exist_ok=True)

    if ft_choice == 'y':
        print("🗑️ 刪除舊模型並重新微調訓練...")
        shutil.rmtree(seg_ckpt_dir, ignore_errors=True)
        os.makedirs(seg_ckpt_dir, exist_ok=True)
        run_script(
            'segmentor/train_segmentor.py',
            [
                '--data_root', os.path.join(PANNUKE_ROOT),
                '--checkpoint_dir', seg_ckpt_dir,
                '--pretrained_ckpt', pretrained_ckpt,
                '--batch_size', '4',
                '--grad_accum_steps', '4'
            ]
        )
    else:
        if need_seg_train():
            print("[注意] 未找到 segmentation checkpoint，將自動啟動訓練。")
            train_segmentor_only()
        else:
            print("[微調] 跳過 segmentation 微調訓練。")

    # 1. 無論是否訓練，都要推二個 splits
    print("[微調] 開始推論二個 splits ...")
    pred_single_dir = os.path.join(PRED_ROOT, 'single_masks')
    os.makedirs(pred_single_dir, exist_ok=True)
    for split, folder in FOLDS.items():
        pred_npy = os.path.join(PRED_ROOT, f"{split}_pred_masks.npy")
        run_script(
            'segmentor/predict_segmentor.py',
            [
                '--weights', os.path.join(seg_ckpt_dir, 'model_final.pth'),
                '--input_dir', os.path.join(PANNUKE_ROOT, folder),
                '--output_dir', pred_single_dir,
                '--output_npy', pred_npy,
                '--batch_size', '8'
            ],
            cwd='segmentor'
        )

    # 2. 分割評估
    print("[微調] 評估分割結果 ...")
    evaluate_all_splits()

    # 3. 視覺化
    print("[微調] 生成視覺化結果 ...")
    for split in ['train', 'test']:
        visualize_predictions.visualize_split(split, num_samples=10)


def select_mode():
    """互動式選擇模式，返回 'train', 'finetune' 或 'inference'"""
    while True:
        print("\n請選擇模式：")
        print("  1. 自己訓練 DINO (train)")
        print("  2. 下載官方 DINO 預訓練權重微調分割 (finetune)")
        print("  3. 使用現有 checkpoint 進行推論 (inference)")
        choice = input("請輸入 1, 2 或 3：").strip()
        if choice == '1':
            return 'train'
        if choice == '2':
            return 'finetune'
        if choice == '3':
            return 'inference'
        print("輸入錯誤，請重新輸入。")

def ensure_official_pretrained():
    """確保官方預訓練權重已下載，否則提示下載"""
    if not os.path.isfile(PRETRAINED_PATH):
        print(f"未檢測到官方預訓練權重，請先下載\n{PRETRAINED_URL}\n並放到:\n {PRETRAINED_PATH}")
        input("下載完成後請按 Enter 繼續...")
    else:
        print(f"已發現預訓練權重：{PRETRAINED_PATH}")

def main_loop():

    while True:
        mode = select_mode()
        # 重置選項
        reset_choice = input("是否重置所有訓練過程？輸入 y 重置，其他鍵跳過：").strip().lower()
        if reset_choice == 'y':
            prompt_reset()
        else:
            # 未重置，確保已切分資料
            if not step_done_file(os.path.join(FOLDS['train'], 'images.npy')):
                print("ℹ️ 正在自動設定 Train/Test 資料夾...")
                setup_train_test_split()
                merge_split_npy()
        # 顯示分布
        count_distribution()

        if mode == 'train':
            print("【模式：自己訓練 DINO】")
            run_pipeline()
            evaluate_all_splits()

        elif mode == 'finetune':
            print("【模式：官方預訓練微調】")
            ensure_official_pretrained()
            finetune_with_pretrained(PRETRAINED_PATH)
            evaluate_all_splits()
        
        elif mode == 'inference':
            print("【模式：使用現有 checkpoint 推論】")
            default_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentor', 'model_final.pth')
            ckpt_path = input(f"請輸入 checkpoint 路徑 (預設: {default_ckpt}): ").strip()
            if not ckpt_path:
                ckpt_path = default_ckpt
            
            if not os.path.isfile(ckpt_path):
                print(f"❌ 找不到檔案：{ckpt_path}")
                continue
                
            print(f"使用 checkpoint: {ckpt_path}")
            
            # 推論
            pred_single_dir = os.path.join(PRED_ROOT, 'single_masks')
            os.makedirs(pred_single_dir, exist_ok=True)
            for split, folder in FOLDS.items():
                pred_npy = os.path.join(PRED_ROOT, f"{split}_pred_masks.npy")
                run_script(
                    'segmentor/predict_segmentor.py',
                    [
                        '--weights', ckpt_path,
                        '--input_dir', os.path.join(PANNUKE_ROOT, folder),
                        '--output_dir', pred_single_dir,
                        '--output_npy', pred_npy,
                        '--batch_size', '8'
                    ],
                    cwd='segmentor'
                )
            
            # 評估
            evaluate_all_splits()

            # 視覺化
            print("生成視覺化結果 ...")
            for split in ['train', 'test']:
                visualize_predictions.visualize_split(split, num_samples=10)

        # 是否再次執行
        again = input("是否要再次執行流程？(y/n): ").strip().lower()
        if again != 'y':
            print("流程結束。")
            break





def ensure_dino_checkpoint():
    """確保已存在訓練好的 DINO 權重；若無則停止執行。"""
    dino_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'dino')
    ckpt = os.path.join(dino_dir, 'dino_vitsmall16_pretrain.pth')
    if os.path.isfile(ckpt):
        print(f'[OK] DINO checkpoint found: {ckpt}')
        return ckpt
    raise FileNotFoundError('缺少 DINO 權重，請先完成 DINO 預訓練或下載官方權重。')

def train_segmentor_only():
    """僅執行 segmentation 訓練與驗證"""
    seg_ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentor')
    os.makedirs(seg_ckpt_dir, exist_ok=True)

    # 1) 訓練 segmentation
    dino_ckpt = ensure_dino_checkpoint()
    run_script(
        'segmentor/train_segmentor.py',
        [
            '--data_root', PANNUKE_ROOT,
            '--checkpoint_dir', seg_ckpt_dir,
            '--pretrained_ckpt', dino_ckpt,
            '--batch_size', '4',
            '--grad_accum_steps', '4'
        ],
        cwd='segmentor'
    )

    # 2) 推論 segmentation
    pred_single_dir = os.path.join(PRED_ROOT, 'single_masks')
    os.makedirs(pred_single_dir, exist_ok=True)
    for split, folder in FOLDS.items():
        pred_npy = os.path.join(PRED_ROOT, f"{split}_pred_masks.npy")
        run_script(
            'segmentor/predict_segmentor.py',
            [
                '--weights', os.path.join(seg_ckpt_dir, 'model_final.pth'),
                '--input_dir', os.path.join(PANNUKE_ROOT, folder),
                '--output_dir', pred_single_dir,
                '--output_npy', pred_npy,
                '--batch_size', '8'
            ],
            cwd='segmentor'
        )

    # 3) 評估
    evaluate_all_splits()

if __name__ == "__main__":
    main_loop()
