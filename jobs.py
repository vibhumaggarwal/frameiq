"""
jobs.py — In-memory async job queue with JSON persistence.

Job states: pending → extracting → describing → embedding → indexing → done → failed
Persisted to cache/jobs.json so state survives restarts.
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

JOBS_FILE = Path(__file__).parent / "cache" / "jobs.json"
_executor = ThreadPoolExecutor(max_workers=2)

# In-memory store
_jobs: Dict[str, Dict[str, Any]] = {}


def _load_persisted():
    """Load jobs from disk on startup."""
    if JOBS_FILE.exists():
        try:
            saved = json.loads(JOBS_FILE.read_text())
            for job_id, job in saved.items():
                # Mark any non-terminal in-flight jobs as failed on restart
                if job["status"] not in ("done", "failed"):
                    job["status"] = "failed"
                    job["error"] = "Server restarted during processing"
                _jobs[job_id] = job
        except Exception as e:
            logger.warning(f"Could not load jobs.json: {e}")


def _persist():
    """Save current job state to disk."""
    JOBS_FILE.parent.mkdir(exist_ok=True)
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jobs, indent=2, default=str))
    tmp.replace(JOBS_FILE)


def create_job(video_id: str, filename: str) -> str:
    """Create a new job and return its job_id."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "filename": filename,
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    _persist()
    return job_id


def get_job(job_id: str) -> Dict[str, Any] | None:
    return _jobs.get(job_id)


def list_jobs() -> list:
    return list(_jobs.values())


def update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)
        _persist()


def _run_sync(fn: Callable, *args):
    """Run a sync function in the thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, fn, *args)


async def enqueue(job_id: str, fn: Callable, *args):
    """
    Run fn(*args) in a background thread. fn receives a progress_cb(pct, msg).
    fn signature: fn(job_id, *args) — it calls update_job internally.
    """
    update_job(job_id, status="pending", message="Starting...")

    loop = asyncio.get_event_loop()

    def _worker():
        try:
            fn(job_id, *args)
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            update_job(job_id, status="failed", error=str(e))

    loop.run_in_executor(_executor, _worker)


# Load persisted jobs at import time
_load_persisted()
