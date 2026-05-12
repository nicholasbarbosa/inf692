import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transforms = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

def apply_train_transform(img: np.ndarray) -> np.ndarray:
    aug_only = A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.RandomScale(scale_limit=0.2, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
        A.Resize(224, 224),
    ])
    return aug_only(image=img)["image"]
