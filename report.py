"""builds the text stats report from pipeline outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2


def video_metadata(video_path: str) -> dict[str, Any]:
    """read basic metadata (fps, frame count, duration) from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "duration_s": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = (frame_count / fps) if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frame_count, "duration_s": duration}


