"""MC Dropout inference for epistemic uncertainty quantification."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from app.services.deep_seg import prepare_input


@tf.function
def _stochastic_forward(model, x):
    return model(x, training=True)


def mc_dropout_predict(model, image: np.ndarray, n_passes: int = 30, target_size: int = 256):
    orig_h, orig_w = image.shape[:2]

    x_tensor = tf.constant(prepare_input(image, model, target_size))

    preds = np.stack(
        [_stochastic_forward(model, x_tensor).numpy()[0, ..., 0] for _ in range(n_passes)],
        axis=0,
    )

    mean_prob = preds.mean(axis=0)
    std_map = preds.std(axis=0)

    eps = 1e-7
    p = np.clip(mean_prob, eps, 1 - eps)
    entropy_map = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    fg = mean_prob >= 0.5
    if fg.sum() > 0:
        conf = float(1.0 - entropy_map[fg].mean() / np.log(2))
    else:
        conf = float(1.0 - entropy_map.mean() / np.log(2))

    mask = (mean_prob >= 0.5).astype(np.uint8) * 255
    mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    std_full = cv2.resize(std_map.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    entropy_full = cv2.resize(entropy_map.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return mask_full, std_full, entropy_full, conf


def colorize(arr: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
    colormap = plt.colormaps[cmap]
    return (colormap(norm)[..., :3] * 255).astype(np.uint8)
