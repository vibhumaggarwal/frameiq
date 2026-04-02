"""
routes/search.py — POST /api/search (with optional vision filters)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from storage import storage

router = APIRouter()


class SearchRequest(BaseModel):
    video_id: str
    query: str
    top_k: int = 5
    face_filter: Optional[str] = None
    plate_filter: Optional[str] = None
    object_filter: Optional[str] = None


@router.post("/api/search")
async def search_scenes(req: SearchRequest):
    if not storage.index_exists(req.video_id):
        raise HTTPException(404, "Index not found. Process this video first.")

    from search import search_scenes as _search
    results = _search(
        storage.index_path(req.video_id),
        storage.meta_path(req.video_id),
        req.query,
        top_k=req.top_k,
        face_filter=req.face_filter,
        plate_filter=req.plate_filter,
        object_filter=req.object_filter,
    )
    return {"results": results}
