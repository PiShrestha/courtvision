"""small geometry helpers shared across the demo pipeline.

duplicated copies used to live in shot_detector.py, shot_attempt_v2.py,
shot_made_v2.py, and duo_tracker.py. this module owns them once.
"""

from __future__ import annotations

from typing import Sequence


Point = tuple[float, float]
Bbox = Sequence[float]        # [x1, y1, x2, y2]


def bbox_center(bbox: Bbox) -> Point:
    """return the (x, y) center of an [x1,y1,x2,y2] bbox."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def distance(a: Point, b: Point) -> float:
    """euclidean distance between two 2d points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def bbox_iou(a: Bbox, b: Bbox) -> float:
    """intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
