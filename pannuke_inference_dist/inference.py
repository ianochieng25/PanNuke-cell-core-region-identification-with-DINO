import os
import sys
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import local modules
from model import TransUNet
from dataset import SimpleImageDataset, PanNukeDataset

def parse_args():
    parser = argparse.ArgumentParser(description="PanNuke Segmentation Inference")
    parser.add_argument('--input', required=True, help='Input directory containing images or PanNuke .npy files')
    parser.add_argument('--output', default='results', help='Output directory for predictions')
    parser.add_argument('--weights', required=True, help='Path to trained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--pannuke_format', action='store_true', help='Set this flag if input is PanNuke dataset structure')
    parser.add_argument('--alpha', type=float, default=0.4, help='Overlay transparency')
    return parser.parse_args()

def clean_mask(mask_binary):
    """Apply morphological opening to remove small noise."""
    kernel = np.ones((3,3), np.uint8)
    if mask_binary.ndim == 2:
        return cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    cleaned = np.zeros_like(mask_binary)
    for i in range(mask_binary.shape[0]):
        cleaned[i] = cv2.morphologyEx(mask_binary[i], cv2.MORPH_OPEN, kernel)
    return cleaned

def create_overlay(image, pred_mask, alpha=0.4):
    """Create visualization overlay."""
    # Ensure image is uint8 [0, 255]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
    # Create colored mask (Blue for Cells)
    color_mask = np.zeros_like(image)
    color_mask[pred_mask == 1] = [0, 0, 255] # Blue
    
    # Blend
    overlay = image.copy()
    mask_indices = (pred_mask == 1)
    overlay[mask_indices] = (image[mask_indices] * (1 - alpha) + color_mask[mask_indices] * alpha).astype(np.uint8)
    
    return overlay

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")

    # 1. Setup Data
    # Normalization matching training
    transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    if args.pannuke_format:
        dataset = PanNukeDataset(args.input, transform=transform)
    else:
        dataset = SimpleImageDataset(args.input, transform=transform)
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Load Model
    print(f"Loading model from {args.weights}...")
    model = TransUNet(in_chans=3, num_classes=2).to(args.device)
    state = torch.load(args.weights, map_location=args.device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # 3. Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for imgs, paths in tqdm(loader):
            imgs = imgs.to(args.device)
            
            # Predict
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            
            # Post-process
            preds = clean_mask(preds)
            
            # Save results
            # Convert images back to numpy for visualization
            # Inverse normalize: x * 0.5 + 0.5 -> [0, 1] -> [0, 255]
            imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1) # (B, H, W, C)
            imgs_np = (imgs_np * 0.5 + 0.5) * 255
            imgs_np = np.clip(imgs_np, 0, 255).astype(np.uint8)
            
            for i in range(len(preds)):
                pred_mask = preds[i]
                orig_img = imgs_np[i]
                
                # Handle path/name
                if args.pannuke_format:
                    name = paths[i] # It's a tuple/list from dataloader, but here it's just a string batch
                else:
                    name = Path(paths[i]).stem
                
                # Save Mask
                mask_path = os.path.join(args.output, f"{name}_mask.png")
                # Save as 0/255 for visibility
                cv2.imwrite(mask_path, pred_mask * 255)
                
                # Save Overlay
                overlay = create_overlay(orig_img, pred_mask, args.alpha)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
                overlay_path = os.path.join(args.output, f"{name}_overlay.jpg")
                cv2.imwrite(overlay_path, overlay)

    print("Done!")

if __name__ == "__main__":
    main()
