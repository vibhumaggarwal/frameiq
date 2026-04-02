"""
storage.py — File storage abstraction for FRAMEIQ.

V1: local filesystem. Swap class for S3 later without touching callers.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CACHE_DIR = BASE_DIR / "cache"


def _ensure_dirs():
    UPLOAD_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


_ensure_dirs()


class LocalStorage:
    """Local filesystem storage backend."""

    # ── Video file operations ─────────────────────────────────────────────────

    def save_upload(self, file_bytes: bytes, original_filename: str) -> tuple[str, str]:
        """
        Save uploaded bytes to uploads/. Returns (video_id, file_path).
        video_id = SHA256 of content (dedup-safe).
        """
        video_id = hashlib.sha256(file_bytes).hexdigest()[:16]
        ext = Path(original_filename).suffix or ".mp4"
        dest = UPLOAD_DIR / f"{video_id}{ext}"
        dest.write_bytes(file_bytes)
        return video_id, str(dest)

    def get_upload_path(self, video_id: str) -> str | None:
        """Find the upload file for a video_id."""
        for p in UPLOAD_DIR.iterdir():
            if p.stem == video_id:
                return str(p)
        return None

    def delete_upload(self, video_id: str):
        """Remove the raw upload after processing."""
        path = self.get_upload_path(video_id)
        if path and os.path.exists(path):
            os.unlink(path)

    # ── Output MKV ────────────────────────────────────────────────────────────

    def get_output_path(self, video_id: str) -> str | None:
        p = CACHE_DIR / f"{video_id}_frameiq.mkv"
        return str(p) if p.exists() else None

    def get_output_target_path(self, video_id: str, original_filename: str) -> str:
        stem = Path(original_filename).stem
        return str(CACHE_DIR / f"{video_id}_{stem}_frameiq.mkv")

    # ── SRT ───────────────────────────────────────────────────────────────────

    def get_srt_path(self, video_id: str) -> str | None:
        p = CACHE_DIR / f"{video_id}.srt"
        return str(p) if p.exists() else None

    def get_srt_target_path(self, video_id: str) -> str:
        return str(CACHE_DIR / f"{video_id}.srt")

    # ── Index + metadata ──────────────────────────────────────────────────────

    def index_path(self, video_id: str) -> str:
        return str(CACHE_DIR / f"{video_id}_index.faiss")

    def meta_path(self, video_id: str) -> str:
        return str(CACHE_DIR / f"{video_id}_meta.json")

    def desc_path(self, video_id: str) -> str:
        return str(CACHE_DIR / f"{video_id}_descriptions.json")

    def index_exists(self, video_id: str) -> bool:
        return Path(self.index_path(video_id)).exists()

    def desc_exists(self, video_id: str) -> bool:
        return Path(self.desc_path(video_id)).exists()

    # ── Video registry ─────────────────────────────────────────────────────────

    def _registry_path(self) -> Path:
        return CACHE_DIR / "registry.json"

    def register_video(self, video_id: str, meta: Dict[str, Any]):
        """Add or update an entry in the video registry."""
        registry = self.list_videos_raw()
        registry[video_id] = meta
        tmp = self._registry_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(registry, indent=2))
        tmp.replace(self._registry_path())

    def list_videos_raw(self) -> Dict[str, Any]:
        p = self._registry_path()
        if not p.exists():
            return {}
        return json.loads(p.read_text())

    def list_videos(self) -> List[Dict[str, Any]]:
        raw = self.list_videos_raw()
        return [{"video_id": vid, **meta} for vid, meta in raw.items()]

    def delete_video(self, video_id: str):
        """Remove all cache + registry entry for a video."""
        for p in CACHE_DIR.iterdir():
            if p.name.startswith(video_id):
                p.unlink(missing_ok=True)
        self.delete_upload(video_id)
        registry = self.list_videos_raw()
        registry.pop(video_id, None)
        self._registry_path().write_text(json.dumps(registry, indent=2))


# Singleton
storage = LocalStorage()
