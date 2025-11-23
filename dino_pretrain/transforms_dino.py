# transforms_dino.py
from torchvision import transforms
from PIL import Image
import numpy as np

class DinoTransform:
    """
    Generate multi-crop augmentations for DINO.
    Two global crops and several local crops.

    Usage:
        transform = DinoTransform()
        crops = transform(pil_img)
    """
    def __init__(self,
                 global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=8):
        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(0.2),
            transforms.GaussianBlur(23, sigma=(0.1,2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(0.2),
            transforms.GaussianBlur(23, sigma=(0.1,2.0)),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(0.5),
            transforms.GaussianBlur(7, sigma=(0.1,2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        crops = []
        # global views
        crops.append(self.global1(image))
        crops.append(self.global2(image))
        # local views
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops
