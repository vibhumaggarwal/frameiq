"""
vision/objects.py — Object detection + tracking using YOLOv8n (ultralytics).

Auto-downloads YOLOv8n (~6MB) on first run.
Returns object counts per class and per-object tracked IDs.
"""

import os
import logging
from typing import Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)

OBJ_CONF_THRESHOLD = float(os.getenv("OBJECT_CONFIDENCE_THRESHOLD", "0.4"))

# Per-process model cache
_model = None


def _get_model():
    """Lazy-load YOLOv8n — downloads ~6MB on first use."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8n model (may download ~6MB on first run)...")
            _model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n ready.")
        except ImportError:
            logger.warning("ultralytics not installed. Object detection disabled.")
    return _model


def detect_objects(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Detect and count objects in a BGR frame using YOLOv8n.

    Args:
        frame_bgr: HxWx3 numpy array in BGR format.

    Returns:
        {
            "objects": {"person": 3, "car": 1, ...},
            "tracked_ids": [{"id": 1, "class": "person", "bbox": [x1,y1,x2,y2], "conf": 0.91}, ...]
        }
    """
    model = _get_model()
    if model is None:
        return {"objects": {}, "tracked_ids": []}

    try:
        results = model.track(
            frame_bgr,
            conf=OBJ_CONF_THRESHOLD,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        counts: Dict[str, int] = {}
        tracked: List[Dict[str, Any]] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else -1

                counts[cls_name] = counts.get(cls_name, 0) + 1
                tracked.append({
                    "id": track_id,
                    "class": cls_name,
                    "bbox": [round(v) for v in xyxy],
                    "conf": round(conf, 3),
                })

        return {"objects": counts, "tracked_ids": tracked}

    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return {"objects": {}, "tracked_ids": []}
