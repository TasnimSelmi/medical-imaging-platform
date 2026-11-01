"""Image registration via optical flow (TP5)."""

import cv2
import numpy as np


def optical_flow_farneback(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, float]:
    gray1 = _ensure_gray(img1)
    gray2 = _ensure_gray(img2)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(mag))

    hsv = np.zeros((*gray1.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_viz = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow_viz, mean_mag


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
