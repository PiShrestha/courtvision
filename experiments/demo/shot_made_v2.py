"""shot_made_v2: rim-plane crossing with occlusion-gap tolerance.

improvements over v1 (shot_detector.py:_ball_passes_hoop):
- requires the ball to cross the horizontal rim plane going downward
  (y increasing past rim center y) rather than merely overlapping a disc,
  which filtered out lateral air-balls.
- treats the ball as "present" for up to `occlusion_gap_frames` missing
  frames, extrapolating position from the last known velocity. real makes
  hide the ball in the net for 0.1-0.5s at 30 fps.
- binds a shot_made to the most recent shot_attempt within
  `attempt_to_made_window` frames; unresolved attempts expire and emit a
  synthetic `shot_miss`.
- enforces a small horizontal tolerance (ball must be within
  `horizontal_pad_factor * rim_radius` of rim_center_x at plane crossing)
  rather than the disc radius, killing wide lateral fly-bys.

state machine per in-flight attempt:
    WAITING        : attempt registered, ball not yet in "enter zone".
    ABOVE_RIM      : ball seen within enter-zone and y < rim_y.
    CROSSING       : y has just passed rim_y (downward crossing detected).
                      emit shot_made and terminate.
    EXPIRED        : attempt_to_made_window elapsed; emit shot_miss.

the module consumes the same per-frame inputs as the existing
`ShotDetector.update` (ball_bbox, player_tracks) plus a `TrackedHoop`
supplied by `rim_tracker.RimTracker` — and emits events in the same dict
schema so callers can drop it in without changes downstream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from _geom import bbox_center as _center
from rim_tracker import TrackedHoop


@dataclass
class _InFlight:
    start_frame: int
    player: int
    state: str = "waiting"         # waiting | above_rim | done
    last_ball: tuple[float, float] | None = None


@dataclass
class _BallSample:
    frame_id: int
    center: tuple[float, float]
    interpolated: bool = False     # True when filled from velocity


class ShotMadeV2:
    """rim-plane crossing shot-made detector with occlusion gap handling.

    tunables:
        occlusion_gap_frames: max gap of missing ball detections we tolerate
            via velocity extrapolation. real makes occlude the ball 3-15 frames.
        attempt_to_made_window: max frames between a shot_attempt and a
            shot_made emission. unresolved attempts expire to shot_miss.
        enter_zone_radius_factor: multiplier on rim_radius for the vertical
            enter zone above the rim. we demand the ball be seen inside this
            zone before we start looking for a downward plane crossing.
        horizontal_pad_factor: multiplier on rim_radius for the horizontal
            tolerance at plane crossing. a clean make passes within 1.0x.
        min_downward_velocity: ball y-velocity (px/frame) required at the
            crossing frame to count it as a descent (rejects near-static balls
            that drift through the zone).
        ball_history: size of the rolling ball-position buffer used for
            velocity estimates during occlusion.
    """

    def __init__(
        self,
        occlusion_gap_frames: int = 10,
        attempt_to_made_window: int = 90,
        enter_zone_radius_factor: float = 2.0,
        horizontal_pad_factor: float = 1.5,
        min_downward_velocity: float = 1.5,
        ball_history: int = 8,
    ) -> None:
        self.occlusion_gap = int(occlusion_gap_frames)
        self.made_window = int(attempt_to_made_window)
        self.enter_factor = float(enter_zone_radius_factor)
        self.horiz_factor = float(horizontal_pad_factor)
        self.min_dy = float(min_downward_velocity)
        self.history: deque[_BallSample] = deque(maxlen=int(ball_history))
        self.in_flight: _InFlight | None = None

    # ---- public api ---------------------------------------------------------

    def on_shot_attempt(self, frame_id: int, player: int) -> None:
        """called by the orchestrator when a shot_attempt fires. if a prior
        attempt is still in-flight it's overwritten (the shooter re-gathered).
        """
        self.in_flight = _InFlight(start_frame=frame_id, player=int(player))

    def update(
        self,
        frame_id: int,
        ball_bbox: list[float] | None,
        hoop: TrackedHoop,
    ) -> list[dict[str, Any]]:
        """advance by one frame. returns 0 or 1 event (shot_made or shot_miss)."""
        events: list[dict[str, Any]] = []

        ball_center = _center(ball_bbox) if ball_bbox else None
        # record ball position. if we have no detection but a recent history,
        # extrapolate from velocity (used for occlusion-gap tolerance).
        if ball_center is not None:
            self.history.append(_BallSample(frame_id, ball_center, False))
        elif self.history:
            latest = self.history[-1]
            gap = frame_id - latest.frame_id
            if gap <= self.occlusion_gap:
                vx, vy = self._current_velocity()
                extrap = (latest.center[0] + vx * gap,
                          latest.center[1] + vy * gap)
                self.history.append(_BallSample(frame_id, extrap, True))

        # drive the state machine if an attempt is in flight.
        if self.in_flight is not None:
            elapsed = frame_id - self.in_flight.start_frame
            if elapsed > self.made_window:
                # expiration: fire a synthetic shot_miss and clear the attempt.
                events.append({
                    "frame_id": frame_id,
                    "event": "shot_miss",
                    "player": self.in_flight.player,
                })
                self.in_flight = None
            else:
                if self._check_made(hoop):
                    events.append({
                        "frame_id": frame_id,
                        "event": "shot_made",
                        "player": self.in_flight.player,
                    })
                    self.in_flight = None

        return events

    # ---- internals ----------------------------------------------------------

    def _current_velocity(self) -> tuple[float, float]:
        if len(self.history) < 2:
            return 0.0, 0.0
        a, b = self.history[0], self.history[-1]
        span = max(1, b.frame_id - a.frame_id)
        return ((b.center[0] - a.center[0]) / span,
                (b.center[1] - a.center[1]) / span)

    def _check_made(self, hoop: TrackedHoop) -> bool:
        """did the ball cross the rim's horizontal plane going downward?

        conditions (all required):
          - prev sample is above the rim by at least a sanity margin
            (enter_factor * rim_radius) — filters balls hovering at rim.
          - curr sample is at or below rim, strictly lower than prev.
          - curr is within `horiz_factor * rim_radius` of rim_x.
          - descent speed is non-trivial — rejects zero-velocity extrapolated
            samples that would otherwise glide through the plane.
        """
        if len(self.history) < 2:
            return False
        prev, curr = self.history[-2], self.history[-1]
        rim_x, rim_y = hoop.center
        rim_r = max(6, hoop.radius)

        prev_above = prev.center[1] <= rim_y - self.enter_factor * rim_r * 0.1
        curr_below_or_at = curr.center[1] >= rim_y
        descending = curr.center[1] > prev.center[1]
        if not (prev_above and curr_below_or_at and descending):
            return False

        if abs(curr.center[0] - rim_x) > self.horiz_factor * rim_r:
            return False

        _, vy = self._current_velocity()
        return vy >= self.min_dy


# _center lives in _geom.py; imported at module top.
