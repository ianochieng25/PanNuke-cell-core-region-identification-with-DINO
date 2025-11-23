import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=1.0),              # 隨機旋轉 -180~180度（等於0~360度）
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=0.5),
        A.GaussNoise(p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Blur(blur_limit=3),
        ], p=0.3),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1.0),
        ToTensorV2()
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1.0),
        ToTensorV2()
    ])
