
"""
predict_segmentor.py

使用 TransUNet 可變尺寸版進行推論
支援任意 H×W（只要能被 patch_size 整除）的影像
"""

import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm
from torch.cuda.amp import autocast
import cv2

# -----------------------------------------------------------------------------
# 確保專案根目錄可供模組搜尋（根目錄視 segmentor/ 上層為 project root）
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 匯入模型與資料集
# 假設已在 preprocess/ 下新增 __init__.py，並將 data_loader.py 放入其中
# Likewise, segmentor/transunet.py 內已有可變尺寸 TransUNet 定義
# -----------------------------------------------------------------------------
from segmentor.transunet import TransUNet
from preprocess.data_loader import PanNuke

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

def parse_args():
    parser = argparse.ArgumentParser(description="TransUNet 可變尺寸推理")
    parser.add_argument(
        '--weights',
        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'model_final.pth'),
        help='模型權重檔案 (.pth)，預設 checkpoints/model_final.pth'
    )
    parser.add_argument(
        '--input_dir',
        default=os.path.join(PROJECT_ROOT, 'data', 'pannuke'),
        help='輸入資料夾，會遞迴搜尋 *_images.npy'
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join(PROJECT_ROOT, 'results', 'single_masks'),
        help='單張預測結果輸出目錄 (PNG & NPY)'
    )
    parser.add_argument(
        '--output_npy',
        default=os.path.join(PROJECT_ROOT, 'results', 'all_masks.npy'),
        help='合併後所有 mask 的 NPY 檔案路徑'
    )
    parser.add_argument(
        '--output_pred_npy',
        default=None,
        help='可選：將合併的 mask 儲存到 predictions 資料夾的檔案路徑'
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
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # 建立 DataLoader
    # Windows 平台避免使用多進程以防止路徑編碼問題
    # -----------------------------------------------------------------------------
    import albumentations as A
    # [Fix] Apply same normalization as training
    val_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
    ])
    
    dataset = PanNuke(args.input_dir, transform=val_transform)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows 平台設為 0 避免 multiprocessing 問題
        pin_memory=False  # CPU 推理時不需要 pin_memory
    )

    # -----------------------------------------------------------------------------
    # 載入模型
    # 請確認 in_chans 與資料影像通道數一致 (此範例 RGB => in_chans=3)
    # -----------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # [Modified] Binary Segmentation -> num_classes=2
    model  = TransUNet(in_chans=3, num_classes=2).to(device)

    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # -----------------------------------------------------------------------------
    # 準備輸出檔案 (Memmap)
    # -----------------------------------------------------------------------------
    # 先讀取一個 batch 確定輸出形狀
    # 為了不浪費 iterator，我們重新建立 loader 或只是用 dataset 屬性
    # 但 dataset 可能是 lazy 的，所以我們假設輸出大小與輸入相同 (H,W)
    # 這裡我們假設所有影像大小一致 (PanNuke patches 都是 256x256)
    
    total_samples = len(dataset)
    if total_samples == 0:
        print("無資料可推論")
        return

    # 預先建立 memmap 檔案
    os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
    
    # 取得第一張圖的形狀
    # dataset[0] -> (img, mask, type)
    # img is Tensor (C, H, W)
    first_img, _, _ = dataset[0]
    h, w = first_img.shape[1], first_img.shape[2]
    
    output_shape = (total_samples, h, w)
    print(f"Output shape: {output_shape}")
    
    # 建立 memmap
    if os.path.exists(args.output_npy):
        os.remove(args.output_npy)
    
    # 使用 w+ 模式建立新檔案
    pred_memmap = np.lib.format.open_memmap(args.output_npy, mode='w+', dtype=np.uint8, shape=output_shape)

    # -----------------------------------------------------------------------------
    # 推論迴圈
    # -----------------------------------------------------------------------------
    print(f"[資訊] 開始推論，總樣本數：{total_samples}")
    print(f"[資訊] 輸出目錄：{args.output_dir}")
    print(f"[資訊] 輸出 NPY：{args.output_npy}")
    
    with torch.no_grad():
        global_idx = 0
        batch_idx = -1  # 初始化以避免 UnboundLocalError
        
        try:
            # 使用簡單的進度顯示避免 tqdm 編碼問題
            for batch_idx, (imgs, _, _) in enumerate(loader):
                # imgs: [B, C, H, W]
                imgs = imgs.float().to(device)
                b_size = imgs.size(0)
                
                # 混合精度推論
                with autocast():
                    logits = model(imgs)                    # [B, num_classes, H, W]
                
                preds  = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B,H,W]

                # Apply noise filtering
                preds = clean_mask(preds)

                # 寫入 memmap
                pred_memmap[global_idx : global_idx + b_size] = preds[:]
                
                # 單張存檔 (Optional: 可以註解掉以節省時間/空間)
                # 為了速度，這裡可以考慮只存 NPY 或跳過 PNG
                # 若必須存 PNG，建議使用 ThreadPool
                try:
                    for i, mask in enumerate(preds):
                        idx = global_idx + i
                        # npy_path = os.path.join(args.output_dir, f'pred_{idx:05d}.npy')
                        png_path = os.path.join(args.output_dir, f'pred_{idx:05d}.png')
                        
                        # 驗證路徑
                        png_path = os.path.abspath(png_path)
                        
                        # np.save(npy_path, mask)
                        # 以灰階乘40強化可視化
                        Image.fromarray((mask * 40).astype(np.uint8)).save(png_path)
                except Exception as png_error:
                    print(f"[警告] 無法儲存 PNG (Batch {batch_idx}, idx {global_idx}): {png_error}")
                    # 繼續執行，不中斷推論
                
                global_idx += b_size
                
                # 簡單進度顯示
                if batch_idx % 10 == 0 or batch_idx == 0:
                    progress = (global_idx / total_samples) * 100
                    print(f"[進度] {global_idx}/{total_samples} ({progress:.1f}%)")
                    pred_memmap.flush()
        
        except Exception as e:
            print(f"[錯誤] 推論過程中發生錯誤：{e}")
            print(f"  當前 batch: {batch_idx}")
            print(f"  當前索引: {global_idx}")
            print(f"  錯誤類型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # 提供更多診斷資訊
            if batch_idx == -1:
                print("\n[診斷] 錯誤發生在載入第一個 batch 之前")
                print("  可能原因：")
                print("    1. DataLoader 初始化問題")
                print("    2. 資料集路徑包含無效字符")
                print("    3. Windows 路徑編碼問題")
                print(f"    4. 資料集大小: {len(dataset)} 樣本")
            raise

    # 最後 flush
    pred_memmap.flush()
    print(f"[完成] 合併 mask 儲存至：{args.output_npy}")

    # 若提供了 output_pred_npy，複製一份 (或建立 symlink)
    if args.output_pred_npy and args.output_pred_npy != args.output_npy:
        import shutil
        try:
            # 正規化路徑以避免 Windows 路徑問題
            output_pred_npy = os.path.abspath(args.output_pred_npy)
            output_npy = os.path.abspath(args.output_npy)
            
            # 確保目標目錄存在
            pred_dir = os.path.dirname(output_pred_npy)
            if pred_dir:  # 只有當 dirname 不為空時才創建
                os.makedirs(pred_dir, exist_ok=True)
            
            # 檢查源文件是否存在
            if not os.path.exists(output_npy):
                print(f"[警告] 源文件不存在，無法複製：{output_npy}")
            else:
                shutil.copy(output_npy, output_pred_npy)
                print(f"[完成] 合併 mask 亦儲存至：{output_pred_npy}")
        except Exception as e:
            print(f"[錯誤] 複製檔案時發生錯誤：{e}")
            print(f"  源檔案：{args.output_npy}")
            print(f"  目標檔案：{args.output_pred_npy}")

if __name__ == "__main__":
    main()

