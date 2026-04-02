"""
vision/plates.py — License plate detection and OCR using OpenCV + EasyOCR.

Detects plate-shaped regions via contour analysis, then runs EasyOCR on each.
Fully local — no API calls.
"""

import os
import re
import logging
from typing import List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PLATE_CONF_THRESHOLD = float(os.getenv("PLATE_CONFIDENCE_THRESHOLD", "0.5"))

# Lazy-loaded EasyOCR reader
_reader = None


def _get_reader():
    """Lazy-load EasyOCR (downloads ~100MB model on first use)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Loading EasyOCR model (may download ~100MB on first run)...")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR ready.")
        except ImportError:
            logger.warning("easyocr not installed. Plate detection disabled.")
    return _reader


def _looks_like_plate(text: str) -> bool:
    """Heuristic filter: alphanumeric, 4–10 chars, mixed letters+digits."""
    text = re.sub(r"\s+", "", text).upper()
    if not (4 <= len(text) <= 10):
        return False
    if not re.search(r"[A-Z]", text):
        return False
    if not re.search(r"[0-9]", text):
        return False
    # Must be mostly alphanumeric
    alnum = sum(c.isalnum() for c in text)
    return alnum / len(text) >= 0.8


def _detect_plate_regions(gray: np.ndarray) -> List[tuple]:
    """
    Find plate-like rectangular regions using morphological ops + contours.
    Returns list of (x, y, w, h) tuples.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    h_img, w_img = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / (h + 1e-6)
        area_ratio = (w * h) / (w_img * h_img)

        # Plates are wide (2:1 to 6:1 aspect) and reasonably sized
        if 2.0 <= aspect <= 6.0 and 0.005 <= area_ratio <= 0.15:
            regions.append((x, y, w, h))

    return regions


def detect_plates(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect license plates in a BGR frame.

    Args:
        frame_bgr: HxWx3 numpy array in BGR format (OpenCV native).

    Returns:
        List of {plate_text, confidence, bounding_box: [x, y, w, h]}
    """
    reader = _get_reader()
    if reader is None:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    regions = _detect_plate_regions(gray)

    results = []
    seen_texts = set()

    for (x, y, w, h) in regions:
        # Crop with small padding
        pad = 4
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        try:
            ocr_results = reader.readtext(crop, detail=1)
        except Exception as e:
            logger.debug(f"OCR error on crop: {e}")
            continue

        for (_, text, conf) in ocr_results:
            text_clean = re.sub(r"\s+", "", text).upper()
            if (
                conf >= PLATE_CONF_THRESHOLD
                and _looks_like_plate(text_clean)
                and text_clean not in seen_texts
            ):
                seen_texts.add(text_clean)
                results.append({
                    "plate_text": text_clean,
                    "confidence": round(float(conf), 3),
                    "bounding_box": [x, y, w, h],
                })

    return results
