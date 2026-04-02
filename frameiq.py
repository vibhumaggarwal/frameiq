"""
frameiq.py — Main CLI entry point for FRAMEIQ.

Usage:
  python frameiq.py process <video>   [--interval N] [--no-cache]
  python frameiq.py search  <video>   <query> [--top-k N]
  python frameiq.py ask     <video>   <question>
  python frameiq.py embed   <video>
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _video_hash(video_path: str) -> str:
    """SHA256 of the first 4 MB + file size — fast, reliable fingerprint."""
    h = hashlib.sha256()
    size = os.path.getsize(video_path)
    h.update(str(size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def _cache_paths(video_hash: str):
    desc_path  = os.path.join(CACHE_DIR, f"{video_hash}_descriptions.json")
    index_path = os.path.join(CACHE_DIR, f"{video_hash}_index.faiss")
    meta_path  = os.path.join(CACHE_DIR, f"{video_hash}_meta.json")
    return desc_path, index_path, meta_path


def _load_descriptions(desc_path: str):
    """Load cached descriptions → sorted list of {timestamp_ms, description}."""
    with open(desc_path, "r") as f:
        raw = json.load(f)
    items = [{"timestamp_ms": int(k), "description": v} for k, v in raw.items()]
    items.sort(key=lambda x: x["timestamp_ms"])
    return items


def _require_processed(video_path: str, desc_path: str):
    if not os.path.exists(desc_path):
        print(
            f"[FRAMEIQ] ✗ No cached descriptions for '{video_path}'.\n"
            f"          Run `python frameiq.py process {video_path}` first."
        )
        sys.exit(1)


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_process(args):
    """Full pipeline: extract → describe → embed subtitles → build index."""
    from extractor import extract_keyframes
    from describer import describe_frames, estimate_cost
    from embedder import embed_subtitles
    from indexer import build_index

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[FRAMEIQ] ✗ File not found: {video_path}")
        sys.exit(1)

    _ensure_cache_dir()
    video_hash = _video_hash(video_path)
    desc_path, index_path, meta_path = _cache_paths(video_hash)

    # ── 1. Extract keyframes ──────────────────────────────────────────────────
    print(f"\n[FRAMEIQ] Extracting keyframes from {video_path}...")
    keyframes = extract_keyframes(video_path, interval_seconds=args.interval)

    if not keyframes:
        print("[FRAMEIQ] ✗ No keyframes extracted. Is this a valid video?")
        sys.exit(1)

    # ── 2. Cost estimate ──────────────────────────────────────────────────────
    existing_cache: dict = {}
    if not args.no_cache and os.path.exists(desc_path):
        with open(desc_path) as f:
            existing_cache = json.load(f)

    frames_to_call = [
        kf for kf in keyframes
        if str(kf["timestamp_ms"]) not in existing_cache
    ]

    if frames_to_call:
        cost = estimate_cost(len(frames_to_call))
        print(
            f"\n[FRAMEIQ] Estimated API cost: ~${cost:.4f} "
            f"({len(frames_to_call)} frames × $0.0004)"
        )
        try:
            answer = input("Continue? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "y"
        if answer not in ("y", "yes", ""):
            print("[FRAMEIQ] Aborted.")
            sys.exit(0)
    else:
        print("\n[FRAMEIQ] All frames already cached — no API calls needed.")

    # ── 3. Describe frames ────────────────────────────────────────────────────
    print("\n[FRAMEIQ] Describing frames...")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    descriptions = describe_frames(
        keyframes,
        cache_path=desc_path,
        api_key=api_key,
        no_cache=args.no_cache,
    )

    # ── 4. Embed subtitle track ────────────────────────────────────────────────
    print("\n[FRAMEIQ] Embedding subtitle track into video...")
    output_path = embed_subtitles(video_path, descriptions)

    # ── 5. Build semantic search index ────────────────────────────────────────
    print("\n[FRAMEIQ] Building semantic search index...")
    build_index(descriptions, index_path, meta_path)

    print(f"\n✅ Done! {output_path} is ready.")
    print(f"   Cache: {CACHE_DIR}/")


def cmd_search(args):
    """Search for a scene using natural language."""
    from search import search_scenes, format_results

    _ensure_cache_dir()
    video_hash = _video_hash(args.video)
    desc_path, index_path, meta_path = _cache_paths(video_hash)
    _require_processed(args.video, desc_path)

    if not os.path.exists(index_path):
        print("[FRAMEIQ] Index not found. Run `process` first (or it may have failed).")
        sys.exit(1)

    results = search_scenes(index_path, meta_path, args.query, top_k=args.top_k)
    print(format_results(results))


def cmd_ask(args):
    """Answer a question about the video via RAG."""
    from rag import ask_question

    _ensure_cache_dir()
    video_hash = _video_hash(args.video)
    desc_path, index_path, meta_path = _cache_paths(video_hash)
    _require_processed(args.video, desc_path)

    descriptions = _load_descriptions(desc_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    ask_question(descriptions, args.question, api_key=api_key, stream=True)


def cmd_embed(args):
    """(Re-)embed subtitles from existing cached descriptions."""
    from embedder import embed_subtitles

    _ensure_cache_dir()
    video_hash = _video_hash(args.video)
    desc_path, index_path, meta_path = _cache_paths(video_hash)
    _require_processed(args.video, desc_path)

    descriptions = _load_descriptions(desc_path)
    output_path = embed_subtitles(args.video, descriptions)
    print(f"\n✅ Done! {output_path} is ready.")


# ── Argument Parsing ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frameiq",
        description="FRAMEIQ — AI-powered video scene analysis and semantic search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frameiq.py process movie.mp4 --interval 5
  python frameiq.py search  movie.mp4 "car chase at night"
  python frameiq.py ask     movie.mp4 "does anyone die in this movie?"
  python frameiq.py embed   movie.mp4
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # process
    p_process = sub.add_parser("process", help="Full pipeline: extract → describe → embed → index")
    p_process.add_argument("video", help="Path to input video file")
    p_process.add_argument(
        "--interval", type=float, default=5.0,
        metavar="N", help="Seconds between keyframes (default: 5)"
    )
    p_process.add_argument(
        "--no-cache", action="store_true",
        help="Force re-process even if cache exists"
    )
    p_process.set_defaults(func=cmd_process)

    # search
    p_search = sub.add_parser("search", help="Natural language scene search")
    p_search.add_argument("video", help="Path to (original) video file")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument(
        "--top-k", type=int, default=5,
        metavar="K", help="Number of results (default: 5)"
    )
    p_search.set_defaults(func=cmd_search)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about the video (RAG)")
    p_ask.add_argument("video", help="Path to (original) video file")
    p_ask.add_argument("question", help="Question to answer")
    p_ask.set_defaults(func=cmd_ask)

    # embed
    p_embed = sub.add_parser("embed", help="(Re-)embed subtitle track from cached descriptions")
    p_embed.add_argument("video", help="Path to input video file")
    p_embed.set_defaults(func=cmd_embed)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
