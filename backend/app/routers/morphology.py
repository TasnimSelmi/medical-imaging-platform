"""Morphological operation endpoints (TP2)."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store, morphology
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api/morphology", tags=["morphology"])

METHODS = {
    "erosion": lambda img, p: morphology.erosion(img, ksize=int(p.get("ksize", 3)), iterations=int(p.get("iterations", 1))),
    "dilation": lambda img, p: morphology.dilation(img, ksize=int(p.get("ksize", 3)), iterations=int(p.get("iterations", 1))),
    "opening": lambda img, p: morphology.opening(img, ksize=int(p.get("ksize", 3))),
    "closing": lambda img, p: morphology.closing(img, ksize=int(p.get("ksize", 3))),
    "gradient": lambda img, p: morphology.gradient(img, ksize=int(p.get("ksize", 3))),
}


class MorphologyRequest(BaseModel):
    image_id: str
    params: dict = {}


@router.post("/{method}")
async def apply_morphology(method: str, req: MorphologyRequest):
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
