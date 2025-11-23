"""
predict_segmentor_simple.py

簡化版推論腳本，跳過 PNG 儲存，只產生合併的 NPY 檔案
專門處理 Windows 平台可能的編碼和路徑問題
"""

import os
import sys
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 確保專案根目錄可供模組搜尋
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 匯入模型與資料集
# -----------------------------------------------------------------------------
from segmentor.transunet import TransUNet
from preprocess.data_loader import PanNuke

def parse_args():
    parser = argparse.ArgumentParser(description="TransUNet 簡化版推理（無 PNG 輸出）")
    parser.add_argument(
        '--weights',
        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'model_final.pth'),
        help='模型權重檔案 (.pth)'
    )
    parser.add_argument(
        '--input_dir',
        default=os.path.join(PROJECT_ROOT, 'data', 'pannuke'),
        help='輸入資料夾'
    )
    parser.add_argument(
        '--output_npy',
        default=os.path.join(PROJECT_ROOT, 'results', 'all_masks.npy'),
        help='合併後所有 mask 的 NPY 檔案路徑'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='推論批次大小'
    )
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    
    print("=" * 60)
    print("TransUNet 簡化版推理")
    print("=" * 60)
    
    # 建立 DataLoader
    dataset = PanNuke(args.input_dir)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 避免 Windows multiprocessing 問題
        pin_memory=False
    )
    
    total_samples = len(dataset)
    if total_samples == 0:
        print("[錯誤] 無資料可推論")
        return
    
    print(f"[資訊] 總樣本數: {total_samples}")
    print(f"[資訊] 批次大小: {args.batch_size}")
    
    # 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[推理設備] 使用: {device}")
    if torch.cuda.is_available():
        print(f"[GPU 資訊] 名稱: {torch.cuda.get_device_name(0)}")
        print(f"[GPU 資訊] 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = TransUNet(in_chans=3, num_classes=2).to(device)
    
    if not os.path.exists(args.weights):
        print(f"[錯誤] 模型權重不存在: {args.weights}")
        return
    
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[資訊] 模型已載入: {args.weights}")
    
    # 準備輸出檔案
    first_img, _, _ = dataset[0]
    h, w = first_img.shape[1], first_img.shape[2]
    output_shape = (total_samples, h, w)
    
    print(f"[資訊] 輸出形狀: {output_shape}")
    print(f"[資訊] 輸出檔案: {args.output_npy}")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
    
    # 建立 memmap
    if os.path.exists(args.output_npy):
        print(f"[警告] 輸出檔案已存在，將覆寫")
        os.remove(args.output_npy)
    
    pred_memmap = np.lib.format.open_memmap(
        args.output_npy, 
        mode='w+', 
        dtype=np.uint8, 
        shape=output_shape
    )
    
    # 推論迴圈
    print("\n開始推論...")
    print("-" * 60)
    
    with torch.no_grad():
        global_idx = 0
        
        for batch_idx, (imgs, _, _) in enumerate(loader):
            imgs = imgs.float().to(device)
            b_size = imgs.size(0)
            
            # 推論（不使用混合精度以避免潛在問題）
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            
            # 寫入 memmap
            pred_memmap[global_idx : global_idx + b_size] = preds[:]
            
            global_idx += b_size
            
            # 進度顯示
            if batch_idx % 5 == 0 or global_idx >= total_samples:
                progress = (global_idx / total_samples) * 100
                print(f"  已處理: {global_idx}/{total_samples} ({progress:.1f}%)")
            
            # 定期 flush
            if batch_idx % 10 == 0:
                pred_memmap.flush()
    
    # 最後 flush
    pred_memmap.flush()
    print("-" * 60)
    print(f"[完成] 合併 mask 已儲存至: {args.output_npy}")
    print(f"[完成] 檔案大小: {os.path.getsize(args.output_npy) / 1024**2:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[錯誤] 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
