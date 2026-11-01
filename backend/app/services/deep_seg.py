"""Deep learning segmentation with U-Net."""

import cv2
import numpy as np


def prepare_input(image: np.ndarray, model, target_size: int = 256) -> np.ndarray:
    """Resize and scale to [0, 1] with the channel count the model expects (1 = gray, 3 = RGB)."""
    channels = model.input_shape[-1]

    if channels == 3:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    if channels == 1:
        x = x[..., np.newaxis]
    return x[np.newaxis, ...]


def predict_unet(image: np.ndarray, model, target_size: int = 256) -> np.ndarray:
    """Run U-Net inference. Returns binary mask at original resolution."""
    orig_h, orig_w = image.shape[:2]

    x = prepare_input(image, model, target_size)
    pred = model.predict(x, verbose=0)[0, ..., 0]
    binary = (pred > 0.5).astype(np.uint8) * 255

    mask = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask
