"""shot_attempt and shot_made events from ball trajectory + hoop position.

- shot_attempt: ball velocity flips from downward/static to strongly upward
  while someone has possession. sticky: one attempt per "flight".
- shot_made: ball descends through the hoop's bounding region within a few
  frames of an attempt. consumes the in-flight attempt.

we use a short ball-position history so single-frame detection dropouts
don't break velocity estimates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from _geom import bbox_center as _center, distance as _distance
from hoop import Hoop


@dataclass
class _Possession:
    player_id: int
    last_frame: int


class ShotDetector:
    """trajectory-based shot attempt and shot made detection.

    tunables:
        upward_velocity_trigger: negative dy (pixels/frame) to flag attempt.
        made_window_frames: max frames between attempt and made.
        hoop_pad: extra pixels added to hoop radius for "made" region.
    """

    def __init__(
        self,
        hoop: Hoop,
        upward_velocity_trigger: float = -10.0,
        made_window_frames: int = 45,
        hoop_pad: int = 12,
        ball_history: int = 6,
        possession_dist_px: float = 120.0,
    ):
        self.hoop = hoop
        self.upward_trigger = upward_velocity_trigger
        self.made_window = made_window_frames
        self.hoop_pad = hoop_pad
        self.ball_history = deque(maxlen=ball_history)
        self.possession_dist_px = possession_dist_px
        self.current_possession: _Possession | None = None
        self.shot_in_flight: dict[str, Any] | None = None

    def update(self, frame_id: int, ball_bbox: list[float] | None,
               player_tracks: list[dict]) -> list[dict]:
        events: list[dict] = []

        # update possession from nearest-player distance, emit event on change.
        ball_center = _center(ball_bbox) if ball_bbox else None
        if ball_center is not None and player_tracks:
            closest = min(player_tracks,
                          key=lambda t: _distance(ball_center, _center(t["bbox"])))
            if _distance(ball_center, _center(closest["bbox"])) <= self.possession_dist_px:
                new_player = int(closest["track_id"])
                if (self.current_possession is None
                        or self.current_possession.player_id != new_player):
                    self.current_possession = _Possession(new_player, frame_id)
                    events.append({
                        "frame_id": frame_id,
                        "event": "possession",
                        "player": new_player,
                    })
                else:
                    self.current_possession = _Possession(new_player, frame_id)

        # trajectory: maintain ball position history.
        if ball_center is not None:
            self.ball_history.append((frame_id, ball_center))

        # shot_attempt: average upward velocity crosses the trigger.
        if self.current_possession is not None and len(self.ball_history) >= 3:
            _, p_now = self.ball_history[-1]
            _, p_past = self.ball_history[0]
            span = max(1, self.ball_history[-1][0] - self.ball_history[0][0])
            dy_per_frame = (p_now[1] - p_past[1]) / span
            if self.shot_in_flight is None and dy_per_frame <= self.upward_trigger:
                shooter = self.current_possession.player_id
                self.shot_in_flight = {
                    "start_frame": frame_id, "player": shooter, "resolved": False,
                }
                events.append({
                    "frame_id": frame_id,
                    "event": "shot_attempt",
                    "player": shooter,
                })

        # shot_made: ball descends through hoop within the window.
        if self.shot_in_flight and ball_center is not None:
            frames_since = frame_id - self.shot_in_flight["start_frame"]
            if frames_since > self.made_window:
                # expire in-flight without a made event.
                self.shot_in_flight = None
            else:
                if self._ball_passes_hoop(ball_center):
                    events.append({
                        "frame_id": frame_id,
                        "event": "shot_made",
                        "player": self.shot_in_flight["player"],
                    })
                    self.shot_in_flight = None

        return events

    # ---- helpers ------------------------------------------------------------

    def _ball_passes_hoop(self, ball_center: tuple[float, float]) -> bool:
        """ball is inside the (padded) hoop disc and moving downward."""
        if len(self.ball_history) < 2:
            return False
        (_, prev), (_, curr) = self.ball_history[-2], self.ball_history[-1]
        descending = (curr[1] - prev[1]) > 0
        dx = ball_center[0] - self.hoop.center[0]
        dy = ball_center[1] - self.hoop.center[1]
        inside = (dx * dx + dy * dy) <= (self.hoop.radius + self.hoop_pad) ** 2
        return descending and inside


# _center and _distance live in _geom.py; imported at module top.
