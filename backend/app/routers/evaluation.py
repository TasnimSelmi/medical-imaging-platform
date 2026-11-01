"""Evaluation and comparison endpoints."""

import time

import cv2

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store, metrics, classical_seg
from app.services.deep_seg import predict_unet
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api", tags=["evaluation"])


def _as_mask(img):
    """Masks saved as JPG/RGB come back with 3 channels; metrics need a single one."""
    if img.ndim == 3:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    return img


class EvaluateRequest(BaseModel):
    mask_pred_id: str
    mask_truth_id: str


class CompareRequest(BaseModel):
    image_id: str
    methods: list[str]
    ground_truth_id: str | None = None
    modality: str = "endoscopy"


@router.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """Compute metrics between predicted and ground-truth masks."""
    pred_rec = image_store.get(req.mask_pred_id)
    truth_rec = image_store.get(req.mask_truth_id)
    if not pred_rec or not truth_rec:
        raise HTTPException(404, "Mask not found")

    pred = _as_mask(load_image(pred_rec.path))
    truth = _as_mask(load_image(truth_rec.path))

    return {
        "iou": round(metrics.iou(truth, pred), 4),
        "dice": round(metrics.dice(truth, pred), 4),
        "pearson": round(metrics.pearson_correlation(truth, pred), 4),
        "mse": round(metrics.mse(truth, pred), 2),
        "nmi": round(metrics.nmi(truth, pred), 4),
    }


CLASSICAL_METHODS = {
    "otsu": lambda img, _: classical_seg.otsu(img)[0],
    "adaptive_threshold": lambda img, _: classical_seg.adaptive_threshold(img),
    "watershed": lambda img, _: classical_seg.watershed(img),
    "chan_vese": lambda img, _: classical_seg.chan_vese_seg(img),
    "active_contour": lambda img, _: classical_seg.geodesic_active_contour(img),
    "kmeans": lambda img, _: classical_seg.kmeans_seg(img),
    "meanshift": lambda img, _: classical_seg.meanshift(img),
    "region_growing": lambda img, _: classical_seg.region_growing(img, img.shape[1] // 2, img.shape[0] // 2, 10),
}


def _unet(img, modality):
    from app.main import MODELS
    model = MODELS.get(modality)
    if model is None:
        return None
    return predict_unet(img, model)


@router.post("/compare")
async def compare(req: CompareRequest):
    """Run multiple segmentation methods and optionally compare to ground truth."""
    rec = image_store.get(req.image_id)
    if not rec:
        raise HTTPException(404, "Image not found")

    img = load_image(rec.path)

    gt = None
    if req.ground_truth_id:
        gt_rec = image_store.get(req.ground_truth_id)
        if not gt_rec:
            raise HTTPException(404, "Ground truth not found")
        gt = _as_mask(load_image(gt_rec.path))

    results = []
    for method_name in req.methods:
        t0 = time.perf_counter()
        if method_name == "unet":
            mask = _unet(img, req.modality)
            if mask is None:
                continue
        elif method_name in CLASSICAL_METHODS:
            mask = CLASSICAL_METHODS[method_name](img, {})
        else:
            continue
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        mask_id, mask_path = image_store.allocate_path(".png")
        save_as_png(mask, mask_path)
        h, w = mask.shape[:2]
        image_store.save(mask_id, mask_path, w, h)

        entry = {"method": method_name, "mask_id": mask_id, "duration_ms": duration_ms}

        if gt is not None:
            entry["metrics"] = {
                "iou": round(metrics.iou(gt, mask), 4),
                "dice": round(metrics.dice(gt, mask), 4),
            }

        results.append(entry)

    return {"results": results}
