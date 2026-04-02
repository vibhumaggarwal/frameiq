"""
describer.py — AI frame description using Claude claude-haiku-3-5-20241022.

Injects vision metadata (faces, plates, objects) into the prompt for richer descriptions.
Caches results to avoid redundant API calls.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any

import anthropic
from tqdm import tqdm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a video scene describer. Given a frame from a video, describe:
1. What is happening in the scene (1-2 sentences)
2. Key objects, people, or text visible
3. Setting/location
4. Any text or captions visible on screen
Be concise. Max 100 words.\
"""

MODEL = "claude-haiku-3-5-20241022"
COST_PER_IMAGE = 0.0004  # approx USD


def _build_vision_context(kf: Dict[str, Any]) -> str:
    """Format vision metadata as a prompt prefix for Claude."""
    lines = []

    faces = kf.get("faces", [])
    if faces:
        face_strs = []
        for f in faces:
            if f["name"] != "Unknown":
                face_strs.append(f"{f['name']} ({f['confidence']:.2f})")
            else:
                face_strs.append("Unknown")
        # Merge Unknown counts
        unknowns = face_strs.count("Unknown")
        named = [s for s in face_strs if s != "Unknown"]
        parts = named + ([f"Unknown x{unknowns}"] if unknowns > 1 else (["Unknown"] if unknowns == 1 else []))
        lines.append(f"- Faces: {', '.join(parts)}")

    plates = kf.get("plates", [])
    if plates:
        plate_strs = [p["plate_text"] for p in plates]
        lines.append(f"- License plates: {', '.join(plate_strs)}")

    objs = kf.get("objects", {})
    obj_counts = objs.get("objects", {}) if isinstance(objs, dict) else {}
    if obj_counts:
        obj_strs = [f"{cls} x{cnt}" for cls, cnt in sorted(obj_counts.items(), key=lambda x: -x[1])]
        lines.append(f"- Objects: {', '.join(obj_strs)}")

    if not lines:
        return ""

    return "Detected in this frame:\n" + "\n".join(lines) + "\n\n"


def describe_frames(
    keyframes: List[Dict[str, Any]],
    cache_path: str,
    api_key: str | None = None,
    no_cache: bool = False,
) -> List[Dict[str, Any]]:
    """
    Describe each keyframe using Claude vision.

    Args:
        keyframes: List of {timestamp_ms, image_base64, faces, plates, objects, ...}
        cache_path: Path to JSON cache file.
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        no_cache: If True, ignore and overwrite existing cache.

    Returns:
        List of {timestamp_ms, description}
    """
    cache: Dict[str, str] = {}
    if not no_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        print(f"  → Loaded {len(cache)} cached descriptions")

    client = anthropic.Anthropic(api_key=api_key)
    results: List[Dict[str, Any]] = []
    new_calls = 0

    for kf in tqdm(keyframes, desc="[FRAMEIQ] Describing frames", unit="frame"):
        key = str(kf["timestamp_ms"])

        if key in cache:
            results.append({"timestamp_ms": kf["timestamp_ms"], "description": cache[key]})
            continue

        description = _call_claude(client, kf["image_base64"], kf)
        cache[key] = description
        new_calls += 1
        results.append({"timestamp_ms": kf["timestamp_ms"], "description": description})
        _save_cache(cache_path, cache)

    print(f"  → {new_calls} new API calls made, {len(keyframes) - new_calls} served from cache")
    return results


def _call_claude(client: anthropic.Anthropic, image_b64: str,
                 kf: Dict[str, Any] | None = None, retries: int = 3) -> str:
    """Send a base64 image + vision metadata to Claude and return description."""
    vision_context = _build_vision_context(kf) if kf else ""
    user_text = f"{vision_context}Describe this video frame."

    for attempt in range(retries):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            return message.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limited. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Claude API error (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return "[Description unavailable due to API error]"
            time.sleep(1)

    return "[Description unavailable due to API error]"


def _save_cache(cache_path: str, cache: Dict[str, str]) -> None:
    """Persist cache to disk safely."""
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp_path, cache_path)


def estimate_cost(frame_count: int) -> float:
    """Return estimated API cost in USD."""
    return frame_count * COST_PER_IMAGE
