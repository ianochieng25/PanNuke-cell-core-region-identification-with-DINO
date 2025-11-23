import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes=2, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, target):
        # inputs: [B, C, H, W] (logits)
        # target: [B, H, W] (class indices)
        
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice for each class
        intersection = (inputs * target_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average Dice Loss across classes (1 - Dice)
        # We can also weight classes here if needed, but usually average is fine
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, target):
        # inputs: [B, C, H, W] (logits)
        # target: [B, H, W]
        
        ce_loss = F.cross_entropy(inputs, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ComboLoss(nn.Module):
    def __init__(self, n_classes=2, alpha=0.5, ce_ratio=0.5):
        super(ComboLoss, self).__init__()
        self.dice = DiceLoss(n_classes=n_classes)
        self.focal = FocalLoss()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha # weight for dice vs focal/ce
        self.ce_ratio = ce_ratio # weight for ce in the non-dice part

    def forward(self, inputs, target):
        dice_loss = self.dice(inputs, target)
        focal_loss = self.focal(inputs, target)
        # ce_loss = self.ce(inputs, target) # Optional: Add pure CE if needed
        
        # Combine: Alpha * Dice + (1-Alpha) * Focal
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss
