"""
routes/ask.py — POST /api/ask  (SSE streaming)
"""

import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from storage import storage

router = APIRouter()


class AskRequest(BaseModel):
    video_id: str
    question: str


@router.post("/api/ask")
async def ask_question(req: AskRequest):
    if not storage.desc_exists(req.video_id):
        raise HTTPException(404, "Descriptions not found. Process this video first.")

    desc_path = storage.desc_path(req.video_id)
    with open(desc_path) as f:
        raw = json.load(f)

    descriptions = sorted(
        [{"timestamp_ms": int(k), "description": v} for k, v in raw.items()],
        key=lambda x: x["timestamp_ms"],
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")

    async def event_stream():
        import anthropic
        from indexer import ms_to_human

        context_lines = [
            f"[{ms_to_human(d['timestamp_ms'])}] {d['description']}"
            for d in descriptions
        ]
        context = "\n".join(context_lines)

        user_message = (
            f"Scene descriptions from the video:\n\n{context}\n\n"
            f"Question: {req.question}"
        )

        client = anthropic.Anthropic(api_key=api_key)
        full_response = ""

        try:
            with client.messages.stream(
                model="claude-haiku-3-5-20241022",
                max_tokens=1024,
                system=(
                    "You are an assistant that answers questions about a video "
                    "based on scene descriptions with timestamps. Be specific and "
                    "cite timestamps (HH:MM:SS) when relevant."
                ),
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for chunk in stream.text_stream:
                    full_response += chunk
                    yield f"data: {json.dumps({'token': chunk})}\n\n"

            # Find timestamps mentioned in response
            relevant = [
                ms_to_human(d["timestamp_ms"])
                for d in descriptions
                if ms_to_human(d["timestamp_ms"]) in full_response
            ]
            yield f"data: {json.dumps({'done': True, 'timestamps': relevant})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
