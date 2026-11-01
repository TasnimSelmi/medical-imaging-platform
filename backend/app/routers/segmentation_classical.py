"""Classical segmentation endpoints."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store, classical_seg
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api/segment/classical", tags=["segmentation-classical"])


class SegmentRequest(BaseModel):
    image_id: str
    params: dict = {}


@router.post("/{method}")
async def segment_classical(method: str, req: SegmentRequest):
    """Apply a classical segmentation method."""
    rec = image_store.get(req.image_id)
    if not rec:
        raise HTTPException(404, "Image not found")

    img = load_image(rec.path)
    t0 = time.perf_counter()
    extra = {}

    if method == "otsu":
        mask, thresh = classical_seg.otsu(img)
        extra["threshold"] = thresh
    elif method == "adaptive_threshold":
        mask = classical_seg.adaptive_threshold(img, block_size=int(req.params.get("block_size", 11)), C=int(req.params.get("C", 2)))
    elif method == "watershed":
        mask = classical_seg.watershed(img)
    elif method == "chan_vese":
        mask = classical_seg.chan_vese_seg(img, mu=req.params.get("mu", 0.25), lambda1=req.params.get("lambda1", 1.0), lambda2=req.params.get("lambda2", 1.0), max_iter=int(req.params.get("max_iter", 200)))
    elif method == "active_contour":
        mask = classical_seg.geodesic_active_contour(img, iterations=int(req.params.get("iterations", 200)))
    elif method == "kmeans":
        mask = classical_seg.kmeans_seg(img, k=int(req.params.get("k", 2)))
    elif method == "meanshift":
        mask = classical_seg.meanshift(img, spatial_radius=int(req.params.get("spatial_radius", 10)), color_radius=int(req.params.get("color_radius", 20)))
    elif method == "region_growing":
        seed_x = int(req.params.get("seed_x", img.shape[1] // 2))
        seed_y = int(req.params.get("seed_y", img.shape[0] // 2))
        threshold = float(req.params.get("threshold", 10))
        mask = classical_seg.region_growing(img, seed_x, seed_y, threshold)
    else:
        raise HTTPException(400, f"Unknown method: {method}")

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    result_id, result_path = image_store.allocate_path(".png")
    save_as_png(mask, result_path)
    h, w = mask.shape[:2]
    image_store.save(result_id, result_path, w, h)

    return {"mask_id": result_id, "duration_ms": duration_ms, **extra}
