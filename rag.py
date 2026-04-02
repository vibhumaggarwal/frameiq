"""
rag.py — RAG pipeline: answer questions about a video using scene descriptions.

Uses Claude with streaming for natural-feeling responses.
"""

import json
from typing import List, Dict, Any

import anthropic

from indexer import ms_to_human

MODEL = "claude-haiku-3-5-20241022"

RAG_SYSTEM = """\
You are an assistant that answers questions about a video based on scene descriptions \
with timestamps. Be specific and cite timestamps (HH:MM:SS) when relevant. \
If the answer isn't clear from the descriptions, say so honestly.\
"""


def _build_context(descriptions: List[Dict[str, Any]]) -> str:
    """Format all scene descriptions into a context block."""
    lines = []
    for d in descriptions:
        ts = ms_to_human(d["timestamp_ms"])
        lines.append(f"[{ts}] {d['description']}")
    return "\n".join(lines)


def ask_question(
    descriptions: List[Dict[str, Any]],
    question: str,
    api_key: str | None = None,
    stream: bool = True,
) -> List[str]:
    """
    Answer a question about a video using RAG.

    Args:
        descriptions: List of {timestamp_ms, description}.
        question: User's natural language question.
        api_key: Anthropic API key (falls back to env var).
        stream: If True, stream response to stdout.

    Returns:
        List of relevant timestamp strings mentioned in the answer.
    """
    client = anthropic.Anthropic(api_key=api_key)
    context = _build_context(descriptions)

    user_message = (
        f"Scene descriptions from the video:\n\n{context}\n\n"
        f"Question: {question}"
    )

    print("\nBased on the scene descriptions...\n")

    full_response = ""

    if stream:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=RAG_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        ) as s:
            for chunk in s.text_stream:
                print(chunk, end="", flush=True)
                full_response += chunk
        print()  # newline after streaming
    else:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=RAG_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        full_response = msg.content[0].text
        print(full_response)

    # Extract relevant timestamps from descriptions that seem mentioned
    relevant_ts = _find_relevant_timestamps(full_response, descriptions)
    if relevant_ts:
        print(f"\n→ Relevant timestamps: {', '.join(relevant_ts)}")

    return relevant_ts


def _find_relevant_timestamps(response: str, descriptions: List[Dict[str, Any]]) -> List[str]:
    """
    Heuristic: find timestamps from descriptions that appear in the response text.
    """
    found = []
    for d in descriptions:
        ts = ms_to_human(d["timestamp_ms"])
        if ts in response:
            found.append(ts)
    return found


def load_descriptions_from_cache(cache_path: str) -> List[Dict[str, Any]]:
    """Load cached descriptions from disk."""
    with open(cache_path, "r") as f:
        raw: Dict[str, str] = json.load(f)

    # Cache format: {timestamp_ms_str: description}
    descriptions = []
    for ts_str, desc in raw.items():
        descriptions.append({"timestamp_ms": int(ts_str), "description": desc})

    # Sort by timestamp
    descriptions.sort(key=lambda x: x["timestamp_ms"])
    return descriptions
