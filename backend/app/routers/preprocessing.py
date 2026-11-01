"""Preprocessing endpoints."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store, filters
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api", tags=["preprocessing"])

METHODS = {
    "gaussian": lambda img, p: filters.gaussian(img, sigma=p.get("sigma", 1.0)),
    "median": lambda img, p: filters.median(img, ksize=int(p.get("ksize", 5))),
    "mean": lambda img, p: filters.mean(img, ksize=int(p.get("ksize", 5))),
    "bilateral": lambda img, p: filters.bilateral(img, d=int(p.get("d", 9)), sigma_color=p.get("sigma_color", 75), sigma_space=p.get("sigma_space", 75)),
    "nlm": lambda img, p: filters.nlm(img, h=p.get("h", 10), template_window=int(p.get("template_window", 7)), search_window=int(p.get("search_window", 21))),
    "clahe": lambda img, p: filters.clahe(img, clip_limit=p.get("clip_limit", 2.0), tile_grid_size=int(p.get("tile_grid_size", 8))),
    "histogram_eq": lambda img, p: filters.histogram_eq(img),
    "zscore": lambda img, p: filters.zscore(img),
    "n4_bias": lambda img, p: filters.n4_bias(img),
}


class PreprocessRequest(BaseModel):
    image_id: str
    params: dict = {}


@router.post("/preprocess/{method}")
async def preprocess(method: str, req: PreprocessRequest):
    """Apply a preprocessing filter to an image."""
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    rec = image_store.get(req.image_id)
    if not rec:
        raise HTTPException(404, "Image not found")

    img = load_image(rec.path)
    t0 = time.perf_counter()
    result = METHODS[method](img, req.params)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    result_id, result_path = image_store.allocate_path(".png")
    save_as_png(result, result_path)
    h, w = result.shape[:2]
    image_store.save(result_id, result_path, w, h)

    return {"result_id": result_id, "duration_ms": duration_ms}


@router.get("/preprocess/histogram/{image_id}")
async def get_histogram(image_id: str):
    """Return histogram data for an image (TP1)."""
    rec = image_store.get(image_id)
    if not rec:
        raise HTTPException(404, "Image not found")

    import cv2
    import numpy as np
    img = load_image(rec.path)

    if len(img.shape) == 3 and img.shape[2] >= 3:
        channels = {"r": img[:, :, 0], "g": img[:, :, 1], "b": img[:, :, 2]}
    else:
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        channels = {"gray": gray}

    histograms = {}
    for name, ch in channels.items():
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        histograms[name] = hist.tolist()

    return {"histograms": histograms}
