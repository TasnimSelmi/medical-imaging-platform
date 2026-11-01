"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODEL_DIR

logger = logging.getLogger(__name__)

MODELS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prefer the ImageNet-pretrained U-Net; fall back to the from-scratch one
    unet_endo = MODEL_DIR / "unet_endoscopy_pretrained.keras"
    if not unet_endo.exists():
        unet_endo = MODEL_DIR / "unet_endoscopy.h5"
    unet_brain = MODEL_DIR / "unet_brain_mri.h5"
    if unet_endo.exists():
        from tensorflow.keras.models import load_model
        MODELS["endoscopy"] = load_model(str(unet_endo), compile=False)
        logger.info("Loaded endoscopy model from %s", unet_endo.name)
    else:
        logger.warning("Endoscopy model not found at %s — DL endpoint disabled", unet_endo)
    if unet_brain.exists():
        from tensorflow.keras.models import load_model
        MODELS["brain"] = load_model(str(unet_brain), compile=False)
        logger.info("Loaded brain MRI model")
    else:
        logger.warning("Brain MRI model not found at %s — DL endpoint disabled", unet_brain)
    yield
    MODELS.clear()


app = FastAPI(
    title="Medical Imaging Platform",
    description="Prétraitement, segmentation classique, apprentissage profond et quantification d'incertitude",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import (
    upload,
    preprocessing,
    morphology,
    edges,
    segmentation_classical,
    segmentation_deep,
    uncertainty,
    registration,
    evaluation,
)

app.include_router(upload.router)
app.include_router(preprocessing.router)
app.include_router(morphology.router)
app.include_router(edges.router)
app.include_router(segmentation_classical.router)
app.include_router(segmentation_deep.router)
app.include_router(uncertainty.router)
app.include_router(registration.router)
app.include_router(evaluation.router)
