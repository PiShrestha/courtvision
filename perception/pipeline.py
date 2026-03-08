"""reads a video, runs the tracker per frame, yields per-frame results."""

from __future__ import annotations
from typing import Any, Generator

import cv2

from logic.homography import Homography
from perception.tracker import Tracker


class PerceptionPipeline:
    """iterate frames, call the tracker, optionally project foot points to court."""

    def __init__(
        self,
        video_path: str,
        weights_path: str = "yolov8m.pt",
        homography: Homography | None = None,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
