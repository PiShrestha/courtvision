"""yolov8 + ultralytics bytetrack wrapper. owns the model because bytetrack
needs the model's internal state across frames (persist=True)."""

from __future__ import annotations
from typing import Any

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional at import
    YOLO = None


# coco class ids -> our domain vocabulary. coco has no hoop class.
CLASS_MAP = {
    0: "player",   # person
    32: "ball",    # sports ball
}


class Tracker:
    """run detection + tracking on one frame at a time.

    each call returns a list of dicts:
        {"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float, "class_name": str}
    """

    def __init__(
        self,
        weights_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.35,
        tracker_config: str = "bytetrack.yaml",
        imgsz: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.imgsz = int(imgsz)
        self.model = YOLO(weights_path) if YOLO is not None else None

    def update(self, frame, frame_id: int) -> list[dict[str, Any]]:
        """detect + track a single bgr frame. drops boxes without a track id."""
        if self.model is None:
            return []

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        tracks: list[dict[str, Any]] = []
        for track_id, conf, cls_id, xyxy in zip(
            boxes.id.int().tolist(),
            boxes.conf.tolist(),
            boxes.cls.int().tolist(),
            boxes.xyxy.tolist(),
        ):
            class_name = CLASS_MAP.get(int(cls_id))
            if class_name is None:
                continue
            if conf < self.confidence_threshold:
                continue
            tracks.append(
                {
