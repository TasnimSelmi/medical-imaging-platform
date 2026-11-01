"""Edge detection endpoints (TP2)."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store, edges
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api/edges", tags=["edges"])

METHODS = {
    "sobel": lambda img, p: edges.sobel(img, ksize=int(p.get("ksize", 3))),
    "laplace": lambda img, p: edges.laplace(img, ksize=int(p.get("ksize", 3))),
    "canny": lambda img, p: edges.canny(img, low=float(p.get("low", 50)), high=float(p.get("high", 150))),
}


class EdgeRequest(BaseModel):
    image_id: str
    params: dict = {}


@router.post("/{method}")
async def detect_edges(method: str, req: EdgeRequest):
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
