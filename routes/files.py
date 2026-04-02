"""
routes/files.py — List, download, and delete processed videos.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from storage import storage

router = APIRouter()


@router.get("/api/files")
async def list_files():
    return {"files": storage.list_videos()}


@router.get("/api/files/{video_id}/download")
async def download_mkv(video_id: str):
    path = storage.get_output_path(video_id)
    if not path:
        raise HTTPException(404, "Output MKV not found")
    filename = Path(path).name
    return FileResponse(path, media_type="video/x-matroska",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/api/files/{video_id}/srt")
async def download_srt(video_id: str):
    path = storage.get_srt_path(video_id)
    if not path:
        raise HTTPException(404, "SRT not found")
    filename = Path(path).name
    return FileResponse(path, media_type="text/plain",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.delete("/api/files/{video_id}")
async def delete_video(video_id: str):
    videos = storage.list_videos_raw()
    if video_id not in videos:
        raise HTTPException(404, "Video not found")
    storage.delete_video(video_id)
    return {"deleted": video_id}


@router.post("/api/embed")
async def re_embed(video_id: str):
    """Re-embed subtitle track from existing cached descriptions."""
    if not storage.desc_exists(video_id):
        raise HTTPException(404, "No descriptions cached for this video.")

    import json
    desc_path = storage.desc_path(video_id)
    with open(desc_path) as f:
        raw = json.load(f)
    descriptions = sorted(
        [{"timestamp_ms": int(k), "description": v} for k, v in raw.items()],
        key=lambda x: x["timestamp_ms"],
    )

    upload_path = storage.get_upload_path(video_id)
    if not upload_path:
        raise HTTPException(400, "Original video upload not found. Re-upload first.")

    from embedder import embed_subtitles
    meta = storage.list_videos_raw().get(video_id, {})
    original_filename = meta.get("filename", "video.mp4")
    output_path = storage.get_output_target_path(video_id, original_filename)
    embed_subtitles(upload_path, descriptions, output_path=output_path)
    return {"output": output_path}
