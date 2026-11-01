"""Data augmentation pipeline using albumentations."""

import albumentations as A


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
    ])


def get_val_transforms():
    return None
