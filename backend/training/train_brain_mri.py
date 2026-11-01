"""Train U-Net for brain MRI tumor segmentation.

Dataset: Kaggle LGG MRI Segmentation
Usage:
    python -m training.train_brain_mri --data_dir /path/to/lgg-mri-segmentation/kaggle_3m --epochs 100
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from training.unet import build_unet
from training.losses import bce_dice_loss, dice_metric

IMG_SIZE = 256


def load_dataset(data_dir: Path):
    images, masks = [], []

    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for mask_path in sorted(patient_dir.glob("*_mask.tif")):
            img_path = mask_path.parent / mask_path.name.replace("_mask", "")
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            masks.append((msk > 127).astype(np.float32))

    X = np.array(images, dtype=np.float32)[..., np.newaxis] / 255.0
    Y = np.array(masks, dtype=np.float32)[..., np.newaxis]
    return X, Y


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    X, Y = load_dataset(Path(args.data_dir))
    print(f"Dataset: {len(X)} images")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = train_ds.shuffle(512).map(augment).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=bce_dice_loss, metrics=[dice_metric])
    model.summary()

    out_path = Path(__file__).parent.parent / "app" / "models_registry" / "unet_brain_mri.h5"
    callbacks = [
        ModelCheckpoint(str(out_path), monitor="val_dice_metric", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_dice_metric", mode="max", patience=15, verbose=1),
        ReduceLROnPlateau(monitor="val_dice_metric", mode="max", factor=0.5, patience=5),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    print(f"Best model saved to {out_path}")


if __name__ == "__main__":
    main()
