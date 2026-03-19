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
        stride: int = 1,
        confidence_threshold: float = 0.35,
        imgsz: int = 640,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.video_path = video_path
        self.tracker = Tracker(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            tracker_config=tracker_config,
            imgsz=imgsz,
        )
        self.homography = homography
        self.start_seconds = float(start_seconds)
        self.end_seconds = float(end_seconds) if end_seconds is not None else None
        self.stride = max(1, int(stride))

    def run(self) -> Generator[dict[str, Any], None, None]:
        """yield {frame_id, frame_height, frame_width, tracks} per kept frame."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
