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
    source: str            # "static" | "tracker" | "reseed" | "detection" | "flow"


_FALLBACK_CHAIN = ("csrt", "kcf", "mil")


def _compose_affine(a_t: np.ndarray, a_prev: np.ndarray) -> np.ndarray:
    """compose two 2x3 affine matrices: a_t(a_prev(x)) for 3x1 points."""
    # lift to 3x3 homogeneous form, multiply, drop bottom row.
    row = np.array([[0, 0, 1]], dtype=np.float32)
    A = np.vstack([a_t, row])
    B = np.vstack([a_prev, row])
    return (A @ B)[:2]


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
        flow_max_per_frame_jump_px: int = 25,
    ) -> None:
        # flow mode bounds *per-frame* displacement. 25 px/frame is a very
        # fast pan at 30 fps (~750 px/s); anything faster is almost certainly
        # a cut or a lost track.
        # default changed from "csrt" to "static" after the live matrix
        # showed MIL (our opencv fallback for CSRT) drifts ~160px on a
        # static-camera clip. the tracker path is still available but
        # opt-in; see FINDINGS_v2.md "rim tracker drift" section.
        self.anchor = anchor
        self.kind = tracker_kind
        self.reseed_every = max(1, int(reseed_every))
        self.max_jump = int(max_jump_px)
        self.box_scale = float(box_scale)
        self.flow_max_per_frame_jump = int(flow_max_per_frame_jump_px)
        self._tracker: Any = None
        self._last_seeded_frame: int = -10 ** 9
        self._last_center: tuple[int, int] = anchor.center
        self._last_radius: int = anchor.radius

    def update(
        self,
        frame: np.ndarray,
        frame_id: int,
        detected_bbox: list[float] | None = None,
    ) -> TrackedHoop:
        """return the best rim position for this frame.

        precedence:
          1. `detected_bbox` from a per-frame hoop detector (most reliable).
             when given, the CSRT/MIL tracker is bypassed entirely and the
             anchor is snapped to the detection.
          2. the cv2 tracker if `kind` is not "static".
          3. the static JSON anchor as a fallback.
        """
        if detected_bbox is not None:
            return self._hoop_from_detection(detected_bbox)

        # pure static mode: every frame resolves to the anchor, no tracker work.
        if self.kind == "static":
            return self._static_hoop()

        # optical-flow mode: track feature points in the backboard/rim
        # neighbourhood and translate the anchor by their median flow. works
        # on any handheld footage without ML; defeated by large zooms (the
        # flow field stops being a pure translation).
        if self.kind == "flow":
            return self._flow_update(frame, frame_id)

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

    # ---- optical-flow path ---------------------------------------------------

    def _flow_update(self, frame: np.ndarray, frame_id: int) -> TrackedHoop:
        """RANSAC-based camera-motion estimation, anchored on the JSON hoop.

        Strategy (fixed after the first flow attempt drifted 800px on an
        effectively-still clip because players crossing the backboard
        dominated the median flow):

        1. Track feature points between consecutive frames with pyramidal
           LK optical flow.
        2. Run cv2.estimateAffinePartial2D (RANSAC) on the point pairs to
           find the dominant 2D similarity transform — this treats the
           camera as a rigid mover and explicitly rejects moving objects
           (players, ball) as outliers.
        3. Apply the transform to the JSON anchor center each frame so the
           rim position is always anchor + cumulative camera motion, never
           the drifting feature centroid.
        4. Require >=10 inliers in the RANSAC fit; otherwise fall back to
           static.

        Sampling the feature pool across the WHOLE upper frame (not just
        near the rim) gives RANSAC more signal from static pixels — court
        lines, ceiling, logos, spectators who aren't the players.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        needs_seed = (
            not hasattr(self, "_flow_prev_gray") or self._flow_prev_gray is None
            or not hasattr(self, "_flow_points") or self._flow_points is None
            or len(self._flow_points) < 10
            or frame_id - self._last_seeded_frame >= self.reseed_every
        )
        if needs_seed:
            self._seed_flow(gray)
            self._last_seeded_frame = frame_id
            # cumulative 2x3 affine from anchor frame -> current frame.
            self._cum_affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            self._flow_prev_gray = gray
            return TrackedHoop(center=self.anchor.center,
                                radius=self.anchor.radius, source="reseed")

        p0 = self._flow_points
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self._flow_prev_gray, gray, p0, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if p1 is None or status is None:
            self._flow_prev_gray = gray
            self._seed_flow(gray)
            return self._static_hoop()

        st = status.flatten() == 1
        good_new = p1[st]
        good_old = p0[st]
        if len(good_new) < 10:
            self._flow_prev_gray = gray
            self._seed_flow(gray)
            return self._static_hoop()

        # RANSAC-based camera motion: 2D similarity (translation + rotation
        # + uniform scale). Players and the ball end up as outliers.
        M, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=200,
        )
        if M is None or inliers is None or int(inliers.sum()) < 10:
            self._flow_prev_gray = gray
            self._seed_flow(gray)
            return self._static_hoop()

        # per-frame translation magnitude: reject implausible jumps.
        tx, ty = float(M[0, 2]), float(M[1, 2])
        if abs(tx) > self.flow_max_per_frame_jump or abs(ty) > self.flow_max_per_frame_jump:
            self._flow_prev_gray = gray
            self._seed_flow(gray)
            return self._static_hoop()

        # compose with the cumulative affine: anchor -> current frame.
        self._cum_affine = _compose_affine(M, self._cum_affine)
        ax, ay = self.anchor.center
        pt = np.array([[ax, ay, 1.0]], dtype=np.float32).T
        new_xy = self._cum_affine @ pt
        new_cx = int(round(float(new_xy[0, 0])))
        new_cy = int(round(float(new_xy[1, 0])))

        H, W = gray.shape[:2]
        if not (0 <= new_cx < W and 0 <= new_cy < H):
            self._flow_prev_gray = gray
            self._seed_flow(gray)
            return self._static_hoop()

        # radius scales with the similarity transform's uniform-scale factor.
        scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
        self._flow_radius = max(6, int(round(self._flow_radius * scale)))

        # keep only the RANSAC-inlier points for the next frame so outliers
        # (players, ball) don't linger in the sample.
        inlier_mask = inliers.flatten().astype(bool)
        self._flow_points = good_new[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
        self._flow_prev_gray = gray
        self._flow_center = (new_cx, new_cy)
        self._last_center = (new_cx, new_cy)
        self._last_radius = self._flow_radius
        return TrackedHoop(center=(new_cx, new_cy),
                           radius=self._flow_radius, source="flow")

    def _seed_flow(self, gray: np.ndarray) -> None:
        """seed feature points across the upper half of the frame.

        the WHOLE upper half — not just near the rim — so RANSAC has more
        static pixels (logos, court lines, backboard edges, ceiling) to
        outvote the moving players/ball. the rim region itself is still
        included, just no longer dominant.
        """
        H, W = gray.shape[:2]
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[: int(H * 0.6), :] = 255   # upper 60% of the frame
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.02, minDistance=8,
            blockSize=7, mask=mask,
        )
        self._flow_points = pts if pts is not None else None
        self._flow_radius = self.anchor.radius
        self._flow_center = self.anchor.center

    def _static_hoop(self) -> TrackedHoop:
        self._last_center = self.anchor.center
        self._last_radius = self.anchor.radius
        return TrackedHoop(center=self.anchor.center,
                           radius=self.anchor.radius, source="static")

    def _hoop_from_detection(self, bbox: list[float]) -> TrackedHoop:
        """snap the rim to a per-frame hoop bbox from a custom YOLO model."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2 + 0.5)
        cy = int((y1 + y2) / 2 + 0.5)
        # the rim radius is approximately half the bbox width for a hoop seen
        # head-on. we take half the min side to avoid distortion when the
        # detector box covers backboard + rim together.
        r = max(6, int(min(x2 - x1, y2 - y1) / 2 + 0.5))
        self._last_center = (cx, cy)
        self._last_radius = r
        # invalidate the cv2 tracker so we reseed next time it's needed.
        self._tracker = None
        return TrackedHoop(center=(cx, cy), radius=r, source="detection")

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
