"""Evaluation metrics for segmentation."""

import numpy as np


def iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Intersection over Union."""
    t = (y_true > 127).astype(bool)
    p = (y_pred > 127).astype(bool)
    intersection = np.logical_and(t, p).sum()
    union = np.logical_or(t, p).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dice coefficient (F1 score for segmentation)."""
    t = (y_true > 127).astype(bool)
    p = (y_pred > 127).astype(bool)
    intersection = np.logical_and(t, p).sum()
    total = t.sum() + p.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def pearson_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pearson correlation coefficient between two images."""
    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64).flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean squared error."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def nmi(img1: np.ndarray, img2: np.ndarray, bins: int = 64) -> float:
    """Normalized mutual information: (H(X)+H(Y))/H(X,Y)."""
    a = img1.flatten().astype(np.float64)
    b = img2.flatten().astype(np.float64)

    hist_joint, _, _ = np.histogram2d(a, b, bins=bins)
    hist_joint = hist_joint / hist_joint.sum()

    hist_a = hist_joint.sum(axis=1)
    hist_b = hist_joint.sum(axis=0)

    ha = _entropy(hist_a)
    hb = _entropy(hist_b)
    hab = _entropy(hist_joint.flatten())

    if hab < 1e-10:
        return 1.0
    return float((ha + hb) / hab)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))
