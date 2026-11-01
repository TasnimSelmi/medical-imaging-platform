"""In-memory registry mapping image IDs to file paths and metadata."""

import time
import uuid
import shutil
from pathlib import Path
from dataclasses import dataclass, field

from app.config import UPLOAD_DIR, SESSION_EXPIRY_SECONDS


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    width: int
    height: int
    created_at: float = field(default_factory=time.time)


_store: dict[str, ImageRecord] = {}


def _session_dir() -> Path:
    session_id = uuid.uuid4().hex[:12]
    d = UPLOAD_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_id() -> str:
    return uuid.uuid4().hex[:16]


def save(image_id: str, path: Path, width: int, height: int) -> ImageRecord:
    rec = ImageRecord(image_id=image_id, path=path, width=width, height=height)
    _store[image_id] = rec
    return rec


def get(image_id: str) -> ImageRecord | None:
    rec = _store.get(image_id)
    if rec and not rec.path.exists():
        _store.pop(image_id, None)
        return None
    return rec


def allocate_path(suffix: str = ".png") -> tuple[str, Path]:
    image_id = generate_id()
    d = UPLOAD_DIR / image_id[:4]
    d.mkdir(parents=True, exist_ok=True)
    return image_id, d / f"{image_id}{suffix}"


def cleanup_expired() -> int:
    now = time.time()
    expired = [k for k, v in _store.items() if now - v.created_at > SESSION_EXPIRY_SECONDS]
    for k in expired:
        rec = _store.pop(k)
        if rec.path.exists():
            rec.path.unlink(missing_ok=True)
    return len(expired)
