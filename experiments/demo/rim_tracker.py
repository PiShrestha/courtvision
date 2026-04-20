"""per-frame rim position with hybrid static-config + CSRT/KCF tracking.

strategy (approach (d) in SHOT_MADE_PROMPT.md ambiguity #6):
- bootstrap from the manual / auto-detected Hoop as the anchor.
- build an opencv tracker (CSRT by default, KCF fallback) seeded with a
  bbox around the rim center at init time.
- every `reseed_every` frames, reset the tracker to the static anchor so
  drift on moving-camera clips is bounded. on static-camera clips this is a
  no-op (the anchor IS the truth).
- if the tracker returns nonsense (bbox outside the frame, area collapses,
  or jumps > max_jump_px from the anchor) fall back to the static anchor.

output per frame: a TrackedHoop with center, radius, and `source`
("static" | "tracker" | "reseed"). downstream shot_made_v2 only cares
about center/radius; the source tag is for debugging.

this module has no GPU / ML dependencies. safe to import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from hoop import Hoop


@dataclass
class TrackedHoop:
    center: tuple[int, int]
    radius: int
    source: str            # "static" | "tracker" | "reseed"


_FALLBACK_CHAIN = ("csrt", "kcf", "mil")


def _build_tracker(kind: str) -> Any:
    """opencv tracker factory with graceful fallback.

    csrt is ideal for a rigid region like a rim but isn't always compiled
    into the installed opencv. we try the caller's choice first, then
    fall through the chain csrt → kcf → mil. only MIL is guaranteed on
    stock `opencv-python`; CSRT/KCF require opencv-contrib-python.
    """
    kind = kind.lower()
    if kind == "static":
        return None
    preferred = kind if kind in _FALLBACK_CHAIN else "csrt"
    chain = [preferred, *(k for k in _FALLBACK_CHAIN if k != preferred)]
    for choice in chain:
        for ctor in (f"Tracker{choice.upper()}_create",
                      f"legacy_Tracker{choice.upper()}_create"):
            fn = getattr(cv2, ctor, None)
            if fn is not None:
                return fn()
    return None


class RimTracker:
    """hybrid rim localisation: static anchor + re-seeded cv2 tracker.

    parameters:
        tracker_kind: "csrt" (default) | "kcf" | "static".
        reseed_every: how often to force-reset the tracker to the anchor.
        max_jump_px: reject a tracker update whose center jumps further
            than this from the anchor — the tracker has lost the rim.
        box_scale: size of the bbox seeded around the rim center relative
            to `radius` (wider boxes give the tracker more texture but
            start absorbing the backboard / net).
    """

    def __init__(
        self,
        anchor: Hoop,
        tracker_kind: str = "static",
        reseed_every: int = 90,
        max_jump_px: int = 30,
        box_scale: float = 2.4,
    ) -> None:
        # default changed from "csrt" to "static" after the live matrix
        # showed MIL (our opencv fallback for CSRT) drifts ~160px on a
        # static-camera clip. the tracker path is still available but
        # opt-in; see FINDINGS_v2.md "rim tracker drift" section.
        self.anchor = anchor
        self.kind = tracker_kind
        self.reseed_every = max(1, int(reseed_every))
        self.max_jump = int(max_jump_px)
        self.box_scale = float(box_scale)
        self._tracker: Any = None
        self._last_seeded_frame: int = -10 ** 9
        self._last_center: tuple[int, int] = anchor.center
        self._last_radius: int = anchor.radius

    def update(self, frame: np.ndarray, frame_id: int) -> TrackedHoop:
        """return the best rim position for this frame."""
        # pure static mode: every frame resolves to the anchor, no tracker work.
        if self.kind == "static":
            return self._static_hoop()

        # (re)seed on the first call and every reseed_every frames.
        if (self._tracker is None
                or frame_id - self._last_seeded_frame >= self.reseed_every):
            self._seed(frame)
            self._last_seeded_frame = frame_id
            # first frame after seeding returns the anchor so state is clean.
            return TrackedHoop(center=self.anchor.center,
                               radius=self.anchor.radius, source="reseed")

        # tracker ran but couldn't be built on this opencv — fall back.
        if self._tracker is None:
            return self._static_hoop()

        ok, bbox = self._tracker.update(frame)
        if not ok:
            self._tracker = None
            return self._static_hoop()

        x, y, w, h = (int(v) for v in bbox)
        H, W = frame.shape[:2]
        cx, cy = x + w // 2, y + h // 2
        off_frame = w <= 4 or h <= 4 or x < 0 or y < 0 or x + w > W or y + h > H
        jumped = (abs(cx - self.anchor.center[0]) > self.max_jump
                  or abs(cy - self.anchor.center[1]) > self.max_jump)
        if off_frame or jumped:
            self._tracker = None
            return self._static_hoop()

        # radius: half the min side of the tracked box divided by box_scale.
        r = max(6, int(min(w, h) / (2.0 * self.box_scale) + 0.5))
        self._last_center = (cx, cy)
        self._last_radius = r
        return TrackedHoop(center=(cx, cy), radius=r, source="tracker")

    def _static_hoop(self) -> TrackedHoop:
        self._last_center = self.anchor.center
        self._last_radius = self.anchor.radius
        return TrackedHoop(center=self.anchor.center,
                           radius=self.anchor.radius, source="static")

    # ---- internals ----------------------------------------------------------

    def _seed(self, frame: np.ndarray) -> None:
        cx, cy = self.anchor.center
        r = self.anchor.radius
        side = max(8, int(2 * r * self.box_scale))
        x = int(max(0, cx - side // 2))
        y = int(max(0, cy - side // 2))
        H, W = frame.shape[:2]
        w = int(min(side, W - x))
        h = int(min(side, H - y))
        tracker = _build_tracker(self.kind)
        if tracker is None:
            self._tracker = None
            return
        try:
            tracker.init(frame, (x, y, w, h))
            self._tracker = tracker
        except Exception:
            self._tracker = None
