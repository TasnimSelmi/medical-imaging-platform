"""Upload and image retrieval endpoints."""

from pathlib import Path

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import FileResponse

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR
from app.services import image_store
from app.services.io import load_image, save_as_png, make_thumbnail

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_image(file: UploadFile):
    """Upload a medical image (PNG, JPG, DICOM, NIfTI). Returns image_id, width, height."""
    ext = _get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}")

    image_id, dest_path = image_store.allocate_path(ext)
    content = await file.read()
    dest_path.write_bytes(content)

    try:
        arr = load_image(dest_path)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Cannot decode image: {e}")

    h, w = arr.shape[:2]

    png_id, png_path = image_store.allocate_path(".png")
    save_as_png(arr, png_path)
    image_store.save(image_id, png_path, w, h)

    thumb_id, thumb_path = image_store.allocate_path(".png")
    thumb = make_thumbnail(arr)
    save_as_png(thumb, thumb_path)
    image_store.save(f"{image_id}_thumb", thumb_path, thumb.shape[1], thumb.shape[0])

    if dest_path != png_path and dest_path.exists():
        dest_path.unlink(missing_ok=True)

    return {"image_id": image_id, "width": w, "height": h}


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve the full-size PNG for a given image ID."""
    rec = image_store.get(image_id)
    if not rec:
        raise HTTPException(404, "Image not found")
    return FileResponse(rec.path, media_type="image/png")


@router.get("/image/{image_id}/thumbnail")
async def get_thumbnail(image_id: str):
    """Retrieve the 256px thumbnail for a given image ID."""
    rec = image_store.get(f"{image_id}_thumb")
    if not rec:
        rec = image_store.get(image_id)
    if not rec:
        raise HTTPException(404, "Image not found")
    return FileResponse(rec.path, media_type="image/png")


@router.get("/download/{result_id}")
async def download_result(result_id: str, format: str = "png"):
    """Download a result image."""
    rec = image_store.get(result_id)
    if not rec:
        raise HTTPException(404, "Result not found")
    return FileResponse(rec.path, media_type="image/png", filename=f"{result_id}.png")


def _get_extension(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(name).suffix
