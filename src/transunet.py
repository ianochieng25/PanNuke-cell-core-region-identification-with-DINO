# File: src/transunet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class TransUNet(nn.Module):
    """
    簡易版 TransUNet：使用 ViT 作為編碼器，再接簡單的解碼器。
    """
    def __init__(self, num_classes=1,
                 vit_name='vit_small_patch8_224',
                 pretrained=True):
        super().__init__()
        # ViT 編碼器，移除分類頭
        self.encoder = create_model(vit_name,
                                    pretrained=pretrained,
                                    num_classes=0)
        # 凍結編碼器參數
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 解碼器：從維度 384 放大到輸出維度
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # ViT 要求輸入 224×224
        x = F.interpolate(x, size=(224, 224),
                          mode='bilinear',
                          align_corners=False)
        # 透過 ViT 取得 tokens (包含 CLS token)
        tokens = self.encoder.forward_features(x)  # [B, 1+N, D]
        # 去除 CLS token，只保留 patch embeddings
        patch_tokens = tokens[:, 1:, :]           # [B, N, D]
        # 重塑為影像特徵圖 [B, D, G, G]
        N, D = patch_tokens.shape[1], patch_tokens.shape[2]
        G = int(N ** 0.5)
        feat = patch_tokens.permute(0, 2, 1) \
                           .view(B, D, G, G)
        # 解碼階段
        out = self.decoder(feat)                  # [B, num_classes, 4G, 4G]
        # 放大回原始尺寸
        out = F.interpolate(out, size=(H, W),
                            mode='bilinear',
                            align_corners=False)
        return out
