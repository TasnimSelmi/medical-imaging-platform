"""Image I/O: load PNG, JPG, DICOM, NIfTI into numpy arrays."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image file and return a grayscale or RGB numpy array."""
    suffix = path.suffix.lower()
    full_name = path.name.lower()

    if suffix in (".nii",) or full_name.endswith(".nii.gz"):
        return _load_nifti(path)
    elif suffix == ".dcm":
        return _load_dicom(path)
    else:
        return _load_standard(path)


def _load_standard(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _load_dicom(path: Path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float64)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    return arr


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    nii = nib.load(str(path))
    data = nii.get_fdata()
    if data.ndim == 3:
        mid = data.shape[2] // 2
        data = data[:, :, mid]
    data = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
    return data


def save_as_png(arr: np.ndarray, path: Path) -> None:
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), arr)


def make_thumbnail(arr: np.ndarray, max_size: int = 256) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return arr.copy()
