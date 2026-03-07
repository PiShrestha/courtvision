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
