"""
routes/process.py — POST /api/process

Accepts a video upload, saves it, enqueues the full pipeline job.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from storage import storage
from jobs import create_job, update_job, enqueue

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
FRAME_INTERVAL_DEFAULT = float(os.getenv("FRAME_INTERVAL_DEFAULT", "5"))


def _pipeline_worker(job_id: str, video_id: str, video_path: str,
                     original_filename: str, interval: float):
    """Full pipeline: extract → describe → embed → index. Runs in thread pool."""
    try:
        # ── 1. Extract + Vision ────────────────────────────────────────────────
        update_job(job_id, status="extracting", progress=5,
                   message="Extracting keyframes + running vision analysis...")
        from extractor import extract_keyframes
        keyframes = extract_keyframes(
            video_path,
            interval_seconds=interval,
            video_id=video_id,   # saves frames to cache/frames/{video_id}/
            run_vision=True,
        )

        if not keyframes:
            raise ValueError("No keyframes extracted — is this a valid video file?")

        face_count = sum(len(kf.get("faces", [])) for kf in keyframes)
        plate_count = sum(len(kf.get("plates", [])) for kf in keyframes)
        obj_frame_count = sum(1 for kf in keyframes if kf.get("objects", {}).get("objects"))
        update_job(job_id, progress=22,
                   message=f"Extracted {len(keyframes)} frames "
                           f"({face_count} faces, {plate_count} plates, "
                           f"{obj_frame_count} frames with objects)")

        # ── 2. Describe ───────────────────────────────────────────────────────
        update_job(job_id, status="describing", progress=27,
                   message=f"Describing frames with AI (0/{len(keyframes)})...")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        desc_path = storage.desc_path(video_id)

        import json
        import anthropic

        cache = {}
        if os.path.exists(desc_path):
            with open(desc_path) as f:
                cache = json.load(f)

        from describer import _call_claude, _save_cache

        client = anthropic.Anthropic(api_key=api_key)
        descriptions = []
        total = len(keyframes)

        for i, kf in enumerate(keyframes):
            key = str(kf["timestamp_ms"])
            if key in cache:
                desc = cache[key]
            else:
                desc = _call_claude(client, kf["image_base64"], kf)
                cache[key] = desc
                _save_cache(desc_path, cache)

            descriptions.append({"timestamp_ms": kf["timestamp_ms"], "description": desc})
            pct = 27 + int(((i + 1) / total) * 46)
            update_job(job_id, progress=pct,
                       message=f"Describing frames {i+1}/{total}...")

        # ── 3. Embed subtitles ────────────────────────────────────────────────
        update_job(job_id, status="embedding", progress=75,
                   message="Embedding subtitle track...")

        srt_path = storage.get_srt_target_path(video_id)
        output_path = storage.get_output_target_path(video_id, original_filename)

        from embedder import embed_subtitles, save_srt
        save_srt(descriptions, srt_path)
        embed_subtitles(video_path, descriptions, output_path=output_path)

        # ── 4. Build index ────────────────────────────────────────────────────
        update_job(job_id, status="indexing", progress=88,
                   message="Building search index...")

        from indexer import build_index
        build_index(
            descriptions,
            storage.index_path(video_id),
            storage.meta_path(video_id),
            keyframes=keyframes,  # passes vision data through to metadata
        )

        # ── 5. Register & cleanup ─────────────────────────────────────────────
        storage.register_video(video_id, {
            "filename": original_filename,
            "frame_count": len(keyframes),
            "description_count": len(descriptions),
            "face_count": face_count,
            "plate_count": plate_count,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "output_path": output_path,
            "srt_path": srt_path,
        })
        storage.delete_upload(video_id)

        update_job(job_id, status="done", progress=100,
                   message="Done!", video_id=video_id)

    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        update_job(job_id, status="failed", error=str(e))


@router.post("/api/process")
async def process_video(
    request: Request,
    file: UploadFile = File(...),
    interval: float = Form(FRAME_INTERVAL_DEFAULT),
):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.0f}MB > {MAX_UPLOAD_MB}MB limit)")

    video_id, video_path = storage.save_upload(content, file.filename)
    job_id = create_job(video_id, file.filename)
    await enqueue(job_id, _pipeline_worker, video_id, video_path, file.filename, interval)
    return {"job_id": job_id, "video_id": video_id}


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    from jobs import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job
