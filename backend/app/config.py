import os
from pathlib import Path

UPLOAD_DIR = Path(os.getenv("MEDIPLAT_UPLOAD_DIR", "/tmp/mediplat"))
SESSION_EXPIRY_SECONDS = 3600
MODEL_DIR = Path(__file__).parent / "models_registry"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz"}
MAX_UPLOAD_SIZE_MB = 100
