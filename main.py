"""
main.py — FRAMEIQ FastAPI application
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from routes.process import router as process_router
from routes.search import router as search_router
from routes.ask import router as ask_router
from routes.files import router as files_router
from routes.faces import router as faces_router
from routes.vision import router as vision_router

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="FRAMEIQ", version="1.0.0", docs_url="/api/docs")

# ── CORS ──────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth middleware ───────────────────────────────────────────────────────────

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
UNPROTECTED = {"/", "/api/health", "/api/auth"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not APP_PASSWORD:
        return await call_next(request)

    path = request.url.path
    if path in UNPROTECTED or path.startswith("/static"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if auth_header == f"Bearer {APP_PASSWORD}":
        return await call_next(request)

    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


# ── Auth endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/auth")
async def authenticate(request: Request):
    body = await request.json()
    password = body.get("password", "")
    if not APP_PASSWORD or password == APP_PASSWORD:
        return {"authenticated": True}
    raise HTTPException(401, "Wrong password")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(process_router)
app.include_router(search_router)
app.include_router(ask_router)
app.include_router(files_router)
app.include_router(faces_router)
app.include_router(vision_router)

# ── Static frontend ───────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(index.read_text())
