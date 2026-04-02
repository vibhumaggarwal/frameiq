"""
search.py — Natural language scene search over FAISS + vision filter support.
"""

from typing import List, Dict, Any, Optional

from indexer import load_index, embed_query, ms_to_human


def _face_match(faces: List[Dict], name_filter: str) -> bool:
    """Check if a specific person name appears in faces list (case-insensitive)."""
    name_filter = name_filter.lower()
    return any(f["name"].lower() == name_filter for f in faces)


def _plate_match(plates: List[Dict], plate_filter: str) -> bool:
    """Partial match on plate text (case-insensitive)."""
    plate_filter = plate_filter.upper()
    return any(plate_filter in p["plate_text"].upper() for p in plates)


def _object_match(objects: Dict, obj_filter: str) -> bool:
    """Check if an object class appears in the detection results."""
    obj_filter = obj_filter.lower()
    counts = objects.get("objects", {}) if isinstance(objects, dict) else {}
    return any(obj_filter in cls.lower() for cls in counts)


def search_scenes(
    index_path: str,
    meta_path: str,
    query: str,
    top_k: int = 5,
    face_filter: Optional[str] = None,
    plate_filter: Optional[str] = None,
    object_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for scenes matching a natural language query.
    Optional AND-logic filters for face name, plate text, and object class.

    Returns:
        List of {timestamp_human, timestamp_ms, description, score, faces, plates, objects}
    """
    index, metadata = load_index(index_path, meta_path)
    query_vec = embed_query(query)

    # Fetch more candidates so we can filter down to top_k
    fetch_k = min(len(metadata), top_k * 10)
    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]

        # Apply AND filters
        if face_filter and not _face_match(meta.get("faces", []), face_filter):
            continue
        if plate_filter and not _plate_match(meta.get("plates", []), plate_filter):
            continue
        if object_filter and not _object_match(meta.get("objects", {}), object_filter):
            continue

        results.append({
            "timestamp_ms": meta["timestamp_ms"],
            "timestamp_human": meta["timestamp_human"],
            "description": meta["description"],
            "score": float(score),
            "faces": meta.get("faces", []),
            "plates": meta.get("plates", []),
            "objects": meta.get("objects", {}),
        })

        if len(results) >= top_k:
            break

    return results


def format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for CLI display."""
    if not results:
        return "No results found."

    lines = ["Results:"]
    for r in results:
        lines.append(f"  [{r['timestamp_human']}] ({r['score']:.2f}) {r['description']}")

    return "\n".join(lines)
