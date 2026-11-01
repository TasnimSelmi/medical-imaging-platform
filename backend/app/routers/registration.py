"""Image registration endpoints (TP5)."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store, registration
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api/registration", tags=["registration"])


class OpticalFlowRequest(BaseModel):
    image_id_1: str
    image_id_2: str


@router.post("/optical_flow")
async def compute_optical_flow(req: OpticalFlowRequest):
    rec1 = image_store.get(req.image_id_1)
    rec2 = image_store.get(req.image_id_2)
    if not rec1 or not rec2:
        raise HTTPException(404, "Image not found")

    img1 = load_image(rec1.path)
    img2 = load_image(rec2.path)

    t0 = time.perf_counter()
    flow_viz, mean_mag = registration.optical_flow_farneback(img1, img2)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    viz_id, viz_path = image_store.allocate_path(".png")
    save_as_png(flow_viz, viz_path)
    h, w = flow_viz.shape[:2]
    image_store.save(viz_id, viz_path, w, h)

    return {
        "flow_viz_id": viz_id,
        "mean_magnitude": round(mean_mag, 4),
        "duration_ms": duration_ms,
    }
