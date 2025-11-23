import os
import sys
import argparse
import importlib.util
from pathlib import Path
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import albumentations as A

# 加速
torch.backends.cudnn.benchmark = True

from transunet import TransUNet
from losses import ComboLoss

# ── 以下三行務必放在檔案開頭，import 之後 ──
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SCRIPT_DIR)
CURRENT_DIR = SCRIPT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------
# Dynamically import PanNuke dataset class
# --------------------------------------------------------------------
ds_spec = importlib.util.spec_from_file_location(
    'data_loader',
    os.path.join(PROJECT_ROOT, 'preprocess', 'data_loader.py')
)
ds_mod = importlib.util.module_from_spec(ds_spec)
ds_spec.loader.exec_module(ds_mod)
sys.modules['data_loader'] = ds_mod
PanNukeDataset = getattr(ds_mod, 'PanNuke')

# --------------------------------------------------------------------
def pad_collate(batch):
    imgs, masks, types = zip(*batch)

    # 找出最大的 H/W
    max_h = max(t.shape[-2] for t in imgs)
    max_w = max(t.shape[-1] for t in imgs)

    # pad imgs
    im = [F.pad(t, (0, max_w - t.shape[-1], 0, max_h - t.shape[-2])) for t in imgs]
    # pad masks
    ma = [F.pad(t, (0, max_w - t.shape[-1], 0, max_h - t.shape[-2])) for t in masks]
    
    # pad types （如果 types 是 2D map 才 pad，否則直接 stack）
    if types[0].ndim >= 2:
        ty = [F.pad(t, (0, max_w - t.shape[-1], 0, max_h - t.shape[-2])) for t in types]
        ty = torch.stack(ty)
    else:
        ty = torch.stack(types)

    return torch.stack(im), torch.stack(ma), ty

# --------------------------------------------------------------------
# Early Stopping Class
# --------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
import numpy as np

# --------------------------------------------------------------------
# TTA Helper
# --------------------------------------------------------------------
def tta_inference(model, img):
    """
    Simple Test Time Augmentation:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    """
    # 1. Original
    out = F.softmax(model(img), dim=1)
    
    # 2. HFlip
    img_hf = torch.flip(img, [-1])
    out_hf = F.softmax(model(img_hf), dim=1)
    out += torch.flip(out_hf, [-1])
    
    # 3. VFlip
    img_vf = torch.flip(img, [-2])
    out_vf = F.softmax(model(img_vf), dim=1)
    out += torch.flip(out_vf, [-2])
    
    return out / 3.0

# --------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='訓練 TransUNet')

    parser.add_argument(
        '--checkpoint_dir',
        default='../checkpoints',
        help='checkpoint 儲存路徑，預設 ../checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='總訓練 epoch 數，預設 50'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4, # Lower LR for AdamW
        help='學習率，預設 1e-4'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='如需續訓，填 checkpoint 檔名 (可省略)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32, # Increased default batch size for ~14.7GB VRAM usage
        help='批次大小，預設 32'
    )
    parser.add_argument(
        '--data_root',
        default='data/pannuke',
        help='Root directory of dataset (relative to project root if not absolute)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0, # [Fix] Set to 0 to avoid WinError 8 (multiprocessing handle limit)
        help='DataLoader 的工作緒數量'
    )

    # 補回之前可能用到的選項
    parser.add_argument(
        '--sr',
        action='store_true',
        help='啟用超解析度資料集 (SRpair)'
    )
    parser.add_argument(
        '--hr_root',
        default=None,
        help='超解析度的高解析度資料根目錄，配合 --sr'
    )
    parser.add_argument(
        '--transform',
        default=None,
        help='要套用到 Dataset / DataLoader 的 transform 名稱'
    )
    parser.add_argument(
        '--pin_memory',
        type=lambda x: (str(x).lower() in ['true','1','yes']),
        default=True,
        help='DataLoader pin_memory，預設 True'
    )
    parser.add_argument(
        '--pretrained_ckpt',
        default=None,
        help='DINO 預訓練權重路徑 (.pth)'
    )
    parser.add_argument(
        '--grad_accum_steps',
        type=int,
        default=1,
        help='梯度累積步數'
    )

    args, _ = parser.parse_known_args()
    return args


# --------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 資料集路徑設定
    if os.path.isabs(args.data_root):
        data_root = args.data_root
    else:
        data_root = os.path.join(PROJECT_ROOT, args.data_root)
    logging.info(f'Using data root: {data_root}')

    # Define Transforms
    # [Modified] Disabled complex augmentation as requested by user
    # [Modified] Re-enabled augmentation for better generalization
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
    ])

    # 建立 Dataset
    # 建立 Dataset
    # 顯示實際要搜尋的路徑
    print(f"Data root: {data_root}")
    
    # Check for train/test split
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    
    if os.path.isdir(train_root) and os.path.isdir(test_root):
        print(f"Found split directories: {train_root}, {test_root}")
        train_ds = PanNukeDataset(train_root, transform=train_transform)
        test_ds  = PanNukeDataset(test_root, transform=val_transform)
    else:
        print("No train/test subdirectories found, using data_root for both (WARNING: Data Leakage if not handled otherwise)")
        images_dir = os.path.join(data_root, 'images')
        masks_dir  = os.path.join(data_root, 'masks')
        types_dir  = os.path.join(data_root, 'types')
        print("Searching paths:")
        print(f"  images: {images_dir}")
        print(f"  masks:  {masks_dir}")
        print(f"  types:  {types_dir}")
        train_ds = PanNukeDataset(data_root, transform=train_transform)
        test_ds  = PanNukeDataset(data_root, transform=val_transform)

    # DataLoader（使用 pad_collate）
    # 若 num_workers > 0 且 persistent_workers=True 可減少每個 epoch 重建 worker 的開銷
    use_persistent_workers = (args.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = args.num_workers,
        pin_memory   = args.pin_memory,
        collate_fn   = pad_collate,
        persistent_workers = use_persistent_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size   = args.batch_size, # Use same batch size for val
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = args.pin_memory,
        collate_fn   = pad_collate,
        persistent_workers = use_persistent_workers
    )

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 若有指定 pretrained_ckpt，則先不讓 timm 下載 ImageNet 權重 (pretrained_vit=False)，我們自己載
    use_imagenet = (args.pretrained_ckpt is None)
    # [Modified] Binary Segmentation -> num_classes=2
    model = TransUNet(in_chans=3, num_classes=2, pretrained_vit=use_imagenet).to(device)

    # 載入 DINO 預訓練權重
    if args.pretrained_ckpt:
        if os.path.isfile(args.pretrained_ckpt):
            print(f"[Info] Loading DINO pretrained weights from: {args.pretrained_ckpt}")
            checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
            
            # DINO 存檔時通常在 'student' 或 'teacher' key 下，但也可能是直接 state_dict
            if 'student' in checkpoint:
                state_dict = checkpoint['student']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 處理 key 名稱：移除 "module." 前綴 (DDP 訓練會有)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # 過濾出 encoder (ViT) 的權重
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = "encoder." + k
                new_state_dict[new_key] = v
            
            # 載入權重，允許不匹配
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"[Info] DINO weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        else:
            print(f"[Warning] Pretrained checkpoint not found: {args.pretrained_ckpt}")

    # Loss & optimizer
    # [Modified] Use ComboLoss (Dice + Focal)
    criterion = ComboLoss(n_classes=2, alpha=0.5).to(device)
    print(f"[Info] Using ComboLoss (Dice + Focal)")
    
    # [Modified] AdamW Optimizer with Weight Decay (Improvement: Weight Decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    # [Modified] Cosine Annealing Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    scaler = GradScaler()  # 混合精度訓練

    # [Modified] Early Stopping (Improvement: Early Stopping)
    early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join(args.checkpoint_dir, 'model_best.pth'))

    # 續訓設定
    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume if os.path.isabs(args.resume) else os.path.join(args.checkpoint_dir, args.resume)
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt.get('epoch', 0)
            print(f'[Resume] Loaded {ckpt_path} (epoch {start_epoch})')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', unit='batch')

        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, masks, types) in enumerate(pbar, start=1):
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)
            # types = types.to(device, dtype=torch.long) # Not used in binary seg loss currently

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item() * args.grad_accum_steps
            running_loss += current_loss * imgs.size(0)

            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        avg_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{args.epochs} - Train loss: {avg_loss:.4f}')

        # [Fix] Clear VRAM before validation as requested
        del imgs, masks, outputs, loss
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]', unit='batch', leave=False)

        with torch.no_grad():
            for imgs, masks, types in val_pbar:
                imgs = imgs.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.long)
                
                with autocast():
                    # [Modified] TTA
                    outputs = tta_inference(model, imgs)
                    loss = criterion(outputs, masks)

                val_loss += loss.item() * imgs.size(0)
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val = val_loss / len(test_loader.dataset)
        print(f'          - Val loss:   {avg_val:.4f}')

        # Step Scheduler
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr}')

        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        ckpt_file = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }, ckpt_file)
        # print(f'[Save] {ckpt_file}') # Reduce spam

    # Final save
    final_path = os.path.join(args.checkpoint_dir, 'model_final.pth')
    if os.path.exists(os.path.join(args.checkpoint_dir, 'model_best.pth')):
        print(f'[Info] Loading best model from early stopping for final save...')
        best_state = torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth'))
        torch.save(best_state, final_path)
    else:
        torch.save(model.state_dict(), final_path)
        
    print(f'[Done] Final model saved to {final_path}')

if __name__ == '__main__':
    main()