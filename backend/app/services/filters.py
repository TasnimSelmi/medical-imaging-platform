"""Image preprocessing filters."""

import cv2
import numpy as np


def mean(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    ksize = ksize | 1
    gray = _ensure_gray(image)
    from skimage.filters.rank import mean as rank_mean
    from skimage.morphology import disk
    return rank_mean(gray, disk(ksize // 2))


def gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur."""
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply median filter."""
    ksize = ksize | 1
    return cv2.medianBlur(image, ksize)


def bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def nlm(image: np.ndarray, h: float = 10, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Non-local means denoising."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window, search_window)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    gray = _ensure_gray(image)
    cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return cl.apply(gray)


def histogram_eq(image: np.ndarray) -> np.ndarray:
    """Standard histogram equalization."""
    gray = _ensure_gray(image)
    return cv2.equalizeHist(gray)


def zscore(image: np.ndarray) -> np.ndarray:
    """Z-score normalization, rescaled to 0-255."""
    arr = image.astype(np.float64)
    mean, std = arr.mean(), arr.std()
    if std < 1e-8:
        return image.copy()
    normalized = (arr - mean) / std
    normalized = ((normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255)
    return normalized.astype(np.uint8)


def n4_bias(image: np.ndarray) -> np.ndarray:
    """N4 bias field correction via SimpleITK."""
    import SimpleITK as sitk
    gray = _ensure_gray(image)
    sitk_img = sitk.GetImageFromArray(gray.astype(np.float32))
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 4)
    corrected = corrector.Execute(sitk_img, mask)
    result = sitk.GetArrayFromImage(corrected)
    result = ((result - result.min()) / (result.max() - result.min() + 1e-8) * 255).astype(np.uint8)
    return result


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
