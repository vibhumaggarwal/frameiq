"""
vision/faces.py — Face recognition using face_recognition (dlib-based).

Loads known faces from known_faces/<name>/*.jpg at startup.
Detects and identifies faces in frames, returns bounding boxes + names.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

FACE_CONF_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.6"))
KNOWN_FACES_DIR = Path(__file__).parent.parent / "known_faces"

# Module-level cache: loaded once
_known_encodings: List[np.ndarray] = []
_known_names: List[str] = []
_loaded = False


def _try_import():
    """Lazy import — face_recognition / dlib are heavy."""
    try:
        import face_recognition
        return face_recognition
    except ImportError:
        logger.warning("face_recognition not installed. Face detection disabled.")
        return None


def load_known_faces(force: bool = False):
    """Load all known faces from known_faces/ directory."""
    global _known_encodings, _known_names, _loaded
    if _loaded and not force:
        return

    fr = _try_import()
    if fr is None:
        _loaded = True
        return

    _known_encodings = []
    _known_names = []

    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        _loaded = True
        return

    for person_dir in KNOWN_FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in person_dir.glob("*.jpg"):
            try:
                img = fr.load_image_file(str(img_path))
                encs = fr.face_encodings(img)
                if encs:
                    _known_encodings.append(encs[0])
                    _known_names.append(name)
            except Exception as e:
                logger.warning(f"Could not encode {img_path}: {e}")

        # Also handle .png, .jpeg
        for ext in ("*.png", "*.jpeg", "*.webp"):
            for img_path in person_dir.glob(ext):
                try:
                    img = fr.load_image_file(str(img_path))
                    encs = fr.face_encodings(img)
                    if encs:
                        _known_encodings.append(encs[0])
                        _known_names.append(name)
                except Exception as e:
                    logger.warning(f"Could not encode {img_path}: {e}")

    logger.info(f"Loaded {len(_known_encodings)} known face encodings for {len(set(_known_names))} people")
    _loaded = True


def detect_faces(frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect and identify faces in an RGB frame.

    Args:
        frame_rgb: HxWx3 numpy array in RGB format.

    Returns:
        List of {name, confidence, bounding_box: [top, right, bottom, left]}
    """
    fr = _try_import()
    if fr is None:
        return []

    load_known_faces()

    results = []
    try:
        # Downscale for speed (process at 1/4 size)
        small = frame_rgb[::2, ::2]
        locations = fr.face_locations(small, model="hog")
        if not locations:
            return []

        encodings = fr.face_encodings(small, locations)

        for enc, loc in zip(encodings, locations):
            top, right, bottom, left = [v * 2 for v in loc]  # scale back up

            if _known_encodings:
                distances = fr.face_distance(_known_encodings, enc)
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])
                confidence = max(0.0, 1.0 - best_dist)

                if confidence >= FACE_CONF_THRESHOLD:
                    name = _known_names[best_idx]
                else:
                    name = "Unknown"
                    confidence = 0.0
            else:
                name = "Unknown"
                confidence = 0.0

            results.append({
                "name": name,
                "confidence": round(confidence, 3),
                "bounding_box": [top, right, bottom, left],
            })
    except Exception as e:
        logger.error(f"Face detection error: {e}")

    return results


def list_known_faces() -> List[str]:
    """Return list of known person names."""
    if not KNOWN_FACES_DIR.exists():
        return []
    return [d.name for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()]


def add_known_face(name: str, image_bytes: bytes, filename: str) -> bool:
    """Add a photo to the known_faces directory and reload."""
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    dest = person_dir / filename
    dest.write_bytes(image_bytes)
    load_known_faces(force=True)
    return True


def remove_known_face(name: str) -> bool:
    """Remove all photos for a person."""
    import shutil
    person_dir = KNOWN_FACES_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)
        load_known_faces(force=True)
        return True
    return False
