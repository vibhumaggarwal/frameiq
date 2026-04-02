"""
extractor.py — Keyframe extraction from video using OpenCV.

Extracts one frame every N seconds, deduplicates visually similar
consecutive frames using mean absolute difference (MAD).

Optionally runs vision modules (faces, plates, objects) on each frame.
Saves extracted frames as JPEGs to cache/frames/{video_id}/ permanently.
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image
import io

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"


def _frames_dir(video_id: str) -> Path:
    d = CACHE_DIR / "frames" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_keyframes(
    video_path: str,
    interval_seconds: float = 5.0,
    diff_threshold: float = 10.0,
    video_id: str | None = None,
    run_vision: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract keyframes from a video file.

    Args:
        video_path: Path to the video file.
        interval_seconds: Extract one frame every N seconds.
        diff_threshold: MAD threshold for deduplication.
        video_id: If provided, saves JPEGs to cache/frames/{video_id}/.
        run_vision: If True, runs face/plate/object detection on each frame.

    Returns:
        List of dicts: {timestamp_ms, frame_index, image_base64, frame_path,
                        faces, plates, objects}
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Invalid FPS detected; cannot process video.")

    interval_frames = max(1, int(fps * interval_seconds))
    output_dir = _frames_dir(video_id) if video_id else None

    # Lazy-import vision modules (graceful if not installed)
    if run_vision:
        try:
            from vision.faces import detect_faces, load_known_faces
            load_known_faces()
            _faces_ok = True
        except Exception:
            _faces_ok = False

        try:
            from vision.plates import detect_plates
            _plates_ok = True
        except Exception:
            _plates_ok = False

        try:
            from vision.objects import detect_objects
            _objects_ok = True
        except Exception:
            _objects_ok = False
    else:
        _faces_ok = _plates_ok = _objects_ok = False

    keyframes: List[Dict[str, Any]] = []
    prev_gray: np.ndarray | None = None
    frame_index = 0
    skipped = 0

    duration_s = total_frames / fps
    print(f"  → Video: {duration_s:.1f}s, {fps:.1f} fps, sampling every {interval_seconds}s")

    sample_positions = list(range(0, total_frames, interval_frames))

    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Dedup check
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            if diff < diff_threshold:
                skipped += 1
                continue

        prev_gray = gray
        timestamp_ms = int((pos / fps) * 1000)

        # Save JPEG
        frame_filename = f"frame_{frame_index:06d}_{timestamp_ms}ms.jpg"
        if output_dir:
            frame_path = str(output_dir / frame_filename)
        else:
            import tempfile
            frame_path = os.path.join(tempfile.mkdtemp(prefix="frameiq_"), frame_filename)

        cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Base64 for API
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ── Vision analysis ───────────────────────────────────────────────
        faces = []
        plates = []
        objects: Dict[str, Any] = {"objects": {}, "tracked_ids": []}

        if _faces_ok:
            try:
                from vision.faces import detect_faces
                faces = detect_faces(frame_rgb)
            except Exception as e:
                pass

        if _plates_ok:
            try:
                from vision.plates import detect_plates
                plates = detect_plates(frame_bgr)
            except Exception as e:
                pass

        if _objects_ok:
            try:
                from vision.objects import detect_objects
                objects = detect_objects(frame_bgr)
            except Exception as e:
                pass

        keyframes.append({
            "timestamp_ms": timestamp_ms,
            "frame_index": frame_index,
            "image_base64": image_b64,
            "frame_path": frame_path,
            "faces": faces,
            "plates": plates,
            "objects": objects,
        })
        frame_index += 1

    cap.release()

    print(f"  → {len(sample_positions)} frames sampled, {skipped} duplicates skipped")
    print(f"  → {len(keyframes)} unique frames to describe")

    return keyframes
