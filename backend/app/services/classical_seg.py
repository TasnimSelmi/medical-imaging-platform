"""Classical segmentation methods."""

import cv2
import numpy as np
from skimage.segmentation import chan_vese, inverse_gaussian_gradient, morphological_geodesic_active_contour


def otsu(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu thresholding. Returns binary mask and threshold value."""
    gray = _ensure_gray(image)
    thresh_val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, float(thresh_val)


def adaptive_threshold(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Adaptive thresholding."""
    gray = _ensure_gray(image)
    block_size = block_size | 1
    if block_size < 3:
        block_size = 3
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)


def watershed(image: np.ndarray) -> np.ndarray:
    """Watershed segmentation using distance transform markers."""
    gray = _ensure_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if len(image.shape) == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = image.copy()

    markers = cv2.watershed(color, markers)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255
    return mask


def kmeans_seg(image: np.ndarray, k: int = 2) -> np.ndarray:
    """K-means clustering on pixel intensities."""
    gray = _ensure_gray(image)
    pixels = gray.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    fg_label = int(np.argmax(centers.flatten()))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[labels.reshape(gray.shape) == fg_label] = 255
    return mask


def meanshift(image: np.ndarray, spatial_radius: int = 10, color_radius: int = 20) -> np.ndarray:
    if len(image.shape) == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = image.copy()
    shifted = cv2.pyrMeanShiftFiltering(color, spatial_radius, color_radius)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def region_growing(image: np.ndarray, seed_x: int, seed_y: int, threshold: float = 10) -> np.ndarray:
    """Region growing from a seed point."""
    gray = _ensure_gray(image)
    h, w = gray.shape
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)

    seed_val = float(gray[seed_y, seed_x])
    stack = [(seed_y, seed_x)]

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x]:
            continue
        visited[y, x] = True
        if abs(float(gray[y, x]) - seed_val) <= threshold:
            mask[y, x] = 255
            stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])

    return mask


def chan_vese_seg(image: np.ndarray, mu: float = 0.25, lambda1: float = 1.0, lambda2: float = 1.0, max_iter: int = 200) -> np.ndarray:
    """Chan-Vese active contour segmentation."""
    gray = _ensure_gray(image)
    gray_f = gray.astype(np.float64) / 255.0
    result = chan_vese(gray_f, mu=mu, lambda1=lambda1, lambda2=lambda2, max_num_iter=max_iter)
    return (result.astype(np.uint8)) * 255


def geodesic_active_contour(image: np.ndarray, iterations: int = 200, balloon: int = 1, smoothing: int = 2) -> np.ndarray:
    """Morphological geodesic active contour (snake driven by image edges)."""
    gray = _ensure_gray(image)
    gray_f = gray.astype(np.float64) / 255.0
    gimage = inverse_gaussian_gradient(gray_f, alpha=100, sigma=2)
    h, w = gray.shape
    init = np.zeros((h, w), dtype=np.int8)
    init[h // 10 : h - h // 10, w // 10 : w - w // 10] = 1
    # balloon < 0 shrinks the initial contour inward until it locks onto edges
    result = morphological_geodesic_active_contour(gimage, iterations, init_level_set=init, smoothing=smoothing, balloon=-abs(balloon))
    return (result.astype(np.uint8)) * 255


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
