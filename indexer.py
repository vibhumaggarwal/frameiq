"""
indexer.py — Build and persist a FAISS semantic search index over scene descriptions.

Stores vision metadata (faces, plates, objects) in _meta.json per frame.
Uses sentence-transformers (all-MiniLM-L6-v2) — fully local.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def ms_to_human(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS string."""
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def build_index(
    descriptions: List[Dict[str, Any]],
    index_path: str,
    meta_path: str,
    keyframes: List[Dict[str, Any]] | None = None,
) -> None:
    """
    Build a FAISS flat index and persist it with full metadata.

    Args:
        descriptions: List of {timestamp_ms, description}.
        index_path: Path to save FAISS index (.faiss).
        meta_path: Path to save metadata JSON.
        keyframes: Optional list of keyframes with vision data {faces, plates, objects}.
    """
    print("  → Loading embedding model (all-MiniLM-L6-v2)...")
    model = _get_model()

    texts = [d["description"] for d in descriptions]
    print(f"  → Embedding {len(texts)} descriptions...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    print(f"  → FAISS index saved to {index_path}")

    # Build a lookup from timestamp_ms → keyframe vision data
    kf_lookup: Dict[int, Dict[str, Any]] = {}
    if keyframes:
        for kf in keyframes:
            kf_lookup[kf["timestamp_ms"]] = kf

    metadata = []
    for i, desc in enumerate(descriptions):
        ts_ms = desc["timestamp_ms"]
        kf = kf_lookup.get(ts_ms, {})
        metadata.append({
            "timestamp_ms": ts_ms,
            "timestamp_human": ms_to_human(ts_ms),
            "description": desc["description"],
            "frame_index": i,
            "faces": kf.get("faces", []),
            "plates": kf.get("plates", []),
            "objects": kf.get("objects", {"objects": {}, "tracked_ids": []}),
        })

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  → Metadata saved to {meta_path}")


def load_index(index_path: str, meta_path: str):
    """Load FAISS index and metadata from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}. Run `process` first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Run `process` first.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return index, metadata


def embed_query(query: str) -> np.ndarray:
    """Embed a text query into a normalized vector."""
    model = _get_model()
    vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)
    return vec
