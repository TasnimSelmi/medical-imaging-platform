"""Deep learning segmentation endpoints."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store
from app.services.deep_seg import predict_unet
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api/segment/deep", tags=["segmentation-deep"])


class DeepSegRequest(BaseModel):
    image_id: str
    modality: str  # "endoscopy" | "brain"


@router.post("/unet")
async def segment_unet(req: DeepSegRequest):
    """Run U-Net segmentation on an image."""
    from app.main import MODELS

    if req.modality not in ("endoscopy", "brain"):
        raise HTTPException(400, "Modality must be 'endoscopy' or 'brain'")

    model = MODELS.get(req.modality)
    if model is None:
        raise HTTPException(503, f"Model for '{req.modality}' is not loaded. Ensure the .h5 weights file is in models_registry/.")

    rec = image_store.get(req.image_id)
    if not rec:
        raise HTTPException(404, "Image not found")

    img = load_image(rec.path)
    t0 = time.perf_counter()
    mask = predict_unet(img, model)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    result_id, result_path = image_store.allocate_path(".png")
    save_as_png(mask, result_path)
    h, w = mask.shape[:2]
    image_store.save(result_id, result_path, w, h)

    return {"mask_id": result_id, "duration_ms": duration_ms}
