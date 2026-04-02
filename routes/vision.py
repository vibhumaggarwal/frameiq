"""
routes/vision.py — Per-video vision query endpoints + frame image serving.
"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from storage import storage

router = APIRouter()

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _load_meta(video_id: str):
    meta_path = storage.meta_path(video_id)
    if not os.path.exists(meta_path):
        raise HTTPException(404, "Metadata not found. Process this video first.")
    with open(meta_path) as f:
        return json.load(f)


# ── Faces ──────────────────────────────────────────────────────────────────────

@router.get("/api/files/{video_id}/faces")
async def get_faces(video_id: str, name: Optional[str] = Query(None)):
    """All frames where faces (or a specific named person) appear."""
    metadata = _load_meta(video_id)
    results = []
    for frame in metadata:
        faces = frame.get("faces", [])
        if not faces:
            continue
        if name:
            matches = [f for f in faces if f["name"].lower() == name.lower()]
            if not matches:
                continue
            faces = matches
        results.append({
            "timestamp_ms": frame["timestamp_ms"],
            "timestamp_human": frame["timestamp_human"],
            "faces": faces,
        })
    return {"video_id": video_id, "results": results, "total": len(results)}


# ── Plates ─────────────────────────────────────────────────────────────────────

@router.get("/api/files/{video_id}/plates")
async def get_plates(video_id: str, partial: Optional[str] = Query(None)):
    """All license plates detected in a video, optionally filtered by partial match."""
    metadata = _load_meta(video_id)
    results = []
    seen = set()
    for frame in metadata:
        for plate in frame.get("plates", []):
            text = plate["plate_text"]
            if partial and partial.upper() not in text.upper():
                continue
            if text not in seen:
                seen.add(text)
            results.append({
                "timestamp_ms": frame["timestamp_ms"],
                "timestamp_human": frame["timestamp_human"],
                "plate_text": text,
                "confidence": plate["confidence"],
            })
    return {"video_id": video_id, "results": results, "total": len(results)}


# ── Objects ────────────────────────────────────────────────────────────────────

@router.get("/api/files/{video_id}/objects")
async def get_objects(video_id: str):
    """Aggregate object counts across all frames in a video."""
    metadata = _load_meta(video_id)
    total_counts: dict = {}
    per_frame = []
    for frame in metadata:
        objs = frame.get("objects", {})
        counts = objs.get("objects", {}) if isinstance(objs, dict) else {}
        for cls, cnt in counts.items():
            total_counts[cls] = total_counts.get(cls, 0) + cnt
        if counts:
            per_frame.append({
                "timestamp_human": frame["timestamp_human"],
                "objects": counts,
            })
    return {"video_id": video_id, "total": total_counts, "per_frame": per_frame}


# ── Frame image serving ────────────────────────────────────────────────────────

@router.get("/api/files/{video_id}/frame/{timestamp_ms}")
async def serve_frame(video_id: str, timestamp_ms: int):
    """Serve the extracted JPEG for a given frame timestamp."""
    frames_dir = CACHE_DIR / "frames" / video_id
    if not frames_dir.exists():
        raise HTTPException(404, "Frame directory not found")

    # Find frame file matching timestamp
    for f in frames_dir.iterdir():
        if f"_{timestamp_ms}ms" in f.name and f.suffix == ".jpg":
            return FileResponse(str(f), media_type="image/jpeg")

    raise HTTPException(404, f"Frame at {timestamp_ms}ms not found")
