"""
routes/faces.py — Known faces management endpoints.
"""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from vision.faces import list_known_faces, add_known_face, remove_known_face

router = APIRouter()

KNOWN_FACES_DIR = Path(__file__).parent.parent / "known_faces"


@router.post("/api/known-faces")
async def upload_known_face(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a photo to add to (or expand) a known person."""
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    add_known_face(name.strip(), content, file.filename)
    return {"name": name, "file": file.filename, "status": "added"}


@router.get("/api/known-faces")
async def get_known_faces():
    """List all known person names."""
    names = list_known_faces()
    result = []
    for name in names:
        person_dir = KNOWN_FACES_DIR / name
        photos = [p.name for p in person_dir.iterdir() if p.is_file()]
        result.append({"name": name, "photo_count": len(photos), "photos": photos})
    return {"faces": result}


@router.delete("/api/known-faces/{name}")
async def delete_known_face(name: str):
    """Remove a person from known faces."""
    removed = remove_known_face(name)
    if not removed:
        raise HTTPException(404, f"No known face found for '{name}'")
    return {"deleted": name}


@router.get("/api/known-faces/{name}/photo/{filename}")
async def serve_known_face_photo(name: str, filename: str):
    """Serve a known face photo."""
    path = KNOWN_FACES_DIR / name / filename
    if not path.exists():
        raise HTTPException(404, "Photo not found")
    return FileResponse(str(path))
