"""find the hoop position in a static-camera clip.

strategy:
- sample N evenly-spaced frames across the input range.
- run cv2.HoughCircles looking for rims (orange/white, ~40-120 px radius at 1080p).
- restrict to the upper 55% of the frame (rims live high).
- cluster the candidates across frames by 2d proximity.
- return the largest cluster's median center as the hoop position.

a manual json config overrides the auto-detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Hoop:
    center: tuple[int, int]          # (x, y) pixels
    radius: int                      # rim radius in pixels
    source: str                      # "auto" or "manual"

    def contains(self, point: tuple[float, float]) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return (dx * dx + dy * dy) <= self.radius * self.radius

    def to_dict(self) -> dict:
        return {"center": list(self.center), "radius": int(self.radius), "source": self.source}

    @classmethod
    def from_json(cls, path: str | Path) -> "Hoop":
        data = json.loads(Path(path).read_text())
        return cls(center=tuple(data["center"]), radius=int(data["radius"]),
                    source=data.get("source", "manual"))


def detect_hoop(
    video_path: str,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    n_samples: int = 24,
    upper_fraction: float = 0.55,
    min_radius: int = 20,
    max_radius: int = 90,
) -> Hoop | None:
    """sample frames, run hough on each, cluster candidate centers."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps) if end_seconds else total
    if end_frame <= start_frame:
        cap.release(); return None

    frame_ids = np.linspace(start_frame, end_frame - 1, n_samples).astype(int)
    candidates: list[tuple[int, int, int]] = []  # (x, y, r)

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        crop = frame[: int(h * upper_fraction), :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=80,
            param1=120, param2=32,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is None:
            continue
        for x, y, r in circles[0]:
            candidates.append((int(x), int(y), int(r)))

    cap.release()
    if not candidates:
        return None

    # cluster candidates: pick the center whose neighborhood (within 60 px)
    # contains the most other candidates. median of that cluster becomes the hoop.
    pts = np.array(candidates, dtype=np.int32)
    best_center: tuple[int, int] | None = None
    best_radius: int = 0
    best_count = 0
    for cx, cy, _ in candidates:
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        mask = dist <= 60
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            median = np.median(pts[mask], axis=0)
            best_center = (int(median[0]), int(median[1]))
            best_radius = int(median[2])
    if best_center is None:
        return None
    return Hoop(center=best_center, radius=best_radius, source="auto")


def load_or_detect(
    video_path: str,
    manual_config: str | None,
    **detect_kwargs,
) -> Hoop | None:
    """load a manual hoop json if provided, else auto-detect. returns None on failure."""
    if manual_config:
        path = Path(manual_config)
        if path.exists():
            return Hoop.from_json(path)
    return detect_hoop(video_path, **detect_kwargs)
