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
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

        start_frame = int(round(self.start_seconds * fps))
        end_frame = int(round(self.end_seconds * fps)) if self.end_seconds is not None else None

        # seek snaps to the nearest keyframe, so the first yielded frame can
        # land a few frames before start_frame on heavily compressed video.
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_id = start_frame
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if end_frame is not None and frame_id >= end_frame:
                    break

                if (frame_id - start_frame) % self.stride == 0:
                    tracks = self.tracker.update(frame, frame_id=frame_id)
                    if self.homography is not None:
                        for t in tracks:
                            x1, _, x2, y2 = t["bbox"]
                            # bottom-center: where the player meets the floor.
                            t["court_xy"] = self.homography.pixel_to_court(
                                (x1 + x2) / 2.0, y2
                            )
                    yield {
                        "frame_id": frame_id,
                        "frame_height": frame_h,
                        "frame_width": frame_w,
                        "tracks": tracks,
                    }
                frame_id += 1
        finally:
            cap.release()
