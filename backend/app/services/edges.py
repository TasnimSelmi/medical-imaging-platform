"""Edge detection (TP2)."""

import cv2
import numpy as np


def sobel(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _ensure_gray(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)


def laplace(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _ensure_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return np.clip(np.abs(lap) / np.abs(lap).max() * 255, 0, 255).astype(np.uint8)


def canny(image: np.ndarray, low: float = 50, high: float = 150) -> np.ndarray:
    gray = _ensure_gray(image)
    return cv2.Canny(gray, low, high)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
