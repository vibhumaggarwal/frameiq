"""
embedder.py — Convert scene descriptions to SRT and mux into video via ffmpeg.

Output is always MKV to support SRT subtitle tracks without re-encoding.
"""

import os
import tempfile

# Ensure Homebrew ffmpeg is on PATH (macOS)
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta

import srt
import ffmpeg


def _ms_to_timedelta(ms: int) -> timedelta:
    return timedelta(milliseconds=ms)


def descriptions_to_srt(descriptions: List[Dict[str, Any]]) -> str:
    """
    Convert a list of {timestamp_ms, description} dicts to an SRT string.

    Each subtitle is displayed from its timestamp until the next one.
    The last subtitle is shown for 5 seconds.
    """
    subtitles = []
    for i, desc in enumerate(descriptions):
        start = _ms_to_timedelta(desc["timestamp_ms"])

        if i + 1 < len(descriptions):
            end = _ms_to_timedelta(descriptions[i + 1]["timestamp_ms"])
        else:
            end = start + timedelta(seconds=5)

        sub = srt.Subtitle(
            index=i + 1,
            start=start,
            end=end,
            content=desc["description"],
        )
        subtitles.append(sub)

    return srt.compose(subtitles)


def embed_subtitles(
    video_path: str,
    descriptions: List[Dict[str, Any]],
    output_path: str | None = None,
) -> str:
    """
    Mux SRT subtitle track into a video file using subprocess directly.

    - Uses -c copy (zero re-encoding, 100% quality preserved)
    - Outputs .mkv (native SRT support)
    """
    import subprocess
    import shutil

    video_path = str(video_path)
    video_stem = Path(video_path).stem

    if output_path is None:
        output_dir = str(Path(video_path).parent)
        output_path = os.path.join(output_dir, f"{video_stem}_frameiq.mkv")

    # Write SRT to a temp file
    srt_content = descriptions_to_srt(descriptions)
    tmp_srt = tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    )
    tmp_srt.write(srt_content)
    tmp_srt.flush()
    tmp_srt.close()
    srt_path = tmp_srt.name

    # Locate ffmpeg binary (Homebrew on macOS, or system)
    ffmpeg_bin = (
        shutil.which("ffmpeg")
        or "/opt/homebrew/bin/ffmpeg"
        or "/usr/local/bin/ffmpeg"
    )

    cmd = [
        ffmpeg_bin,
        "-y",                          # overwrite output
        "-i", video_path,              # input video
        "-i", srt_path,                # input SRT
        "-c", "copy",                  # copy all streams (no re-encoding)
        "-c:s", "srt",                 # subtitle codec
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=FRAMEIQ Scene Descriptions",
        output_path,
    ]

    try:
        print(f"  → Muxing subtitle track into {output_path}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )
        print(f"  → Output: {output_path} (identical quality, -c copy)")
    finally:
        os.unlink(srt_path)

    return output_path


def save_srt(descriptions: List[Dict[str, Any]], output_path: str) -> None:
    """Save SRT file to disk (useful for inspection or external players)."""
    srt_content = descriptions_to_srt(descriptions)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"  → SRT saved to {output_path}")
