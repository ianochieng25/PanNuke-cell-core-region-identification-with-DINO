import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch8_224 as ViT
from timm.models.vision_transformer import PatchEmbed

class TransUNet(nn.Module):
    """
    TransUNet 可變輸入尺寸版本
    - ViT-Small/8 backbone (embed_dim=384, patch_size=8)
    - 支援任意 H, W (需能被 patch_size 整除)
    - 可設定 in_chans、num_classes
    """
    def __init__(self, in_chans=3, num_classes=6, pretrained_vit=True):
        super().__init__()
        # 1️⃣ ViT 編碼器
        self.encoder = ViT(pretrained=pretrained_vit)
        D  = self.encoder.embed_dim               # 384
        ps = self.encoder.patch_embed.patch_size  # (8,8)
        self.ps = ps[0] if isinstance(ps, tuple) else ps

        # 2️⃣ 重建 patch_embed：允許任意 img_size
        self.encoder.patch_embed = PatchEmbed(
            img_size=None,
            patch_size=ps,
            in_chans=in_chans,
            embed_dim=D,
            flatten=False  # 確保輸出為 [B, D, H, W]
        )

        # 保存位置嵌入相關屬性
        self.cls_token = self.encoder.cls_token
        self.pos_embed = self.encoder.pos_embed
        self.pos_drop  = self.encoder.pos_drop
        self.blocks    = self.encoder.blocks
        self.norm      = self.encoder.norm
        self.encoder.head = nn.Identity()

        # 3️⃣ U-Net 解碼器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(D,   D//2, 2, 2), nn.BatchNorm2d(D//2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(D//2, D//4, 2, 2), nn.BatchNorm2d(D//4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(D//4, D//8, 2, 2), nn.BatchNorm2d(D//8), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(D//8, D//16,2, 2), nn.BatchNorm2d(D//16),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(D//16,D//32,2, 2), nn.BatchNorm2d(D//32),nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(D//32, num_classes, kernel_size=1)

    def _interp_pos(self, B, Hp, Wp):
        """
        動態內插位置嵌入
        """
        cls = self.cls_token.expand(B, -1, -1)       # [B,1,D]
        pos = self.pos_embed[:, 1:]                  # [1,N0,D]
        orig = int((pos.shape[1])**0.5)
        # reshape & permute to [1,D,orig,orig]
        pos = pos.reshape(1, orig, orig, -1).permute(0, 3, 1, 2)
        # bicubic 內插到 (Hp, Wp)
        pos = F.interpolate(pos, size=(Hp, Wp), mode='bicubic', align_corners=False)
        # restore [1,Hp*Wp,D]
        pos = pos.permute(0, 2, 3, 1).reshape(1, Hp*Wp, -1)
        return torch.cat([cls, pos.expand(B, -1, -1)], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) embed
        patches = self.encoder.patch_embed(x)      # [B,D,Hp,Wp]
        Hp, Wp = patches.shape[-2:]
        # 2) flatten + pos
        y = patches.flatten(2).transpose(1,2)      # [B,N,D]
        
        # 加入 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y = torch.cat((cls_tokens, y), dim=1)      # [B, N+1, D]

        y = y + self._interp_pos(B, Hp, Wp)
        y = self.pos_drop(y)
        # 3) transformer
        for blk in self.blocks:
            y = blk(y)
        y = self.norm(y)
        # 4) reshape to feature map
        feat = y[:,1:,:].transpose(1,2).reshape(B, -1, Hp, Wp)
        # 5) decode
        seg = self.classifier(self.decoder(feat))
        # 6) upsample
        seg = T.Resize((H, W))(seg)
        return seg