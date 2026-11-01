"""MC Dropout uncertainty quantification endpoints."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import image_store
from app.services.uncertainty import mc_dropout_predict, colorize
from app.services.io import load_image, save_as_png

router = APIRouter(prefix="/api/segment/deep", tags=["uncertainty"])


class MCDropoutRequest(BaseModel):
    image_id: str
    modality: str
    n_passes: int = 30


@router.post("/unet_mc")
async def segment_unet_mc(req: MCDropoutRequest):
    from app.main import MODELS

    if req.modality not in ("endoscopy", "brain"):
        raise HTTPException(400, "Modality must be 'endoscopy' or 'brain'")

    model = MODELS.get(req.modality)
    if model is None:
        raise HTTPException(503, f"Model for '{req.modality}' is not loaded.")

    rec = image_store.get(req.image_id)
    if not rec:
        raise HTTPException(404, "Image not found")

    img = load_image(rec.path)
    t0 = time.perf_counter()
    mask, std_map, entropy_map, confidence = mc_dropout_predict(model, img, n_passes=req.n_passes)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    mask_id, mask_path = image_store.allocate_path(".png")
    save_as_png(mask, mask_path)
    h, w = mask.shape[:2]
    image_store.save(mask_id, mask_path, w, h)

    std_colored = colorize(std_map, "inferno")
    std_id, std_path = image_store.allocate_path(".png")
    save_as_png(std_colored, std_path)
    image_store.save(std_id, std_path, w, h)

    entropy_colored = colorize(entropy_map, "viridis")
    ent_id, ent_path = image_store.allocate_path(".png")
    save_as_png(entropy_colored, ent_path)
    image_store.save(ent_id, ent_path, w, h)

    return {
        "mask_id": mask_id,
        "std_map_id": std_id,
        "entropy_map_id": ent_id,
        "confidence_score": round(confidence, 4),
        "n_passes": req.n_passes,
        "duration_ms": duration_ms,
    }
