"""Morphological operations (TP2)."""

import cv2
import numpy as np


def erosion(image: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    gray = _ensure_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(gray, kernel, iterations=iterations)


def dilation(image: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    gray = _ensure_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(gray, kernel, iterations=iterations)


def opening(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _ensure_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


def closing(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _ensure_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def gradient(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _ensure_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(gray, kernel, iterations=1)
    eroded = cv2.erode(gray, kernel, iterations=1)
    return cv2.subtract(dilated, eroded)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
