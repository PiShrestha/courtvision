"""deterministic rule engine: per-frame tracks -> possession and shot_attempt events.

shot_made is intentionally not detected. it needs a hoop track, which the
current yolov8 coco model cannot produce (coco has no basketball hoop class).
"""

from __future__ import annotations
from typing import Any


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _track_position(track: dict[str, Any]) -> tuple[tuple[float, float], bool]:
    """return (position, is_court_space). prefers court_xy when the pipeline set it."""
    court_xy = track.get("court_xy")
    if court_xy is not None:
        return (float(court_xy[0]), float(court_xy[1])), True
    return _bbox_center(track["bbox"]), False


