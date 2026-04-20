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


class RuleEngine:
    """possession + shot_attempt rules driven by simple geometric thresholds.

    pixel thresholds are specified at a 360p reference and auto-scaled to the
    actual frame height, so the same engine works at 360p, 720p, 1080p, 4k.
    """

    def __init__(
        self,
        court_possession_dist: float = 6.0,   # feet
        pixel_possession_dist: float = 110.0, # pixels at 360p
    ):
        self.possession: int | None = None
        self.shot_in_flight: bool = False
        self.last_ball_center_pixel: tuple[float, float] | None = None
        self.last_shooter: int | None = None

        self.court_possession_dist = court_possession_dist
        self.pixel_possession_dist = pixel_possession_dist

    def process_frame(self, frame_data: dict[str, Any]) -> list[dict[str, Any]]:
        """analyse one frame, return any events that fired."""
        frame_id = int(frame_data.get("frame_id", -1))
        tracks = frame_data.get("tracks", [])
        events: list[dict[str, Any]] = []

        # auto-scale pixel thresholds from a 360p reference.
        frame_h = int(frame_data.get("frame_height", 360) or 360)
        pixel_possession_dist = self.pixel_possession_dist * (frame_h / 360.0)

        players = [t for t in tracks if t.get("class_name") == "player"]
        balls = [t for t in tracks if t.get("class_name") == "ball"]
        if not balls:
            return events

        ball = balls[0]
        ball_pos, ball_in_court = _track_position(ball)
        ball_pixel_center = _bbox_center(ball["bbox"])

        # possession: nearest player within threshold. uses court space when
        # homography is configured, otherwise falls back to pixel space.
        if players:
            def player_distance(p: dict[str, Any]) -> float:
                p_pos, p_in_court = _track_position(p)
                if p_in_court and ball_in_court:
                    return _distance(p_pos, ball_pos)
                return _distance(_bbox_center(p["bbox"]), ball_pixel_center)

            nearest = min(players, key=player_distance)
            _, nearest_in_court = _track_position(nearest)
            threshold = (
                self.court_possession_dist
                if (nearest_in_court and ball_in_court)
                else pixel_possession_dist
            )
            if player_distance(nearest) <= threshold:
                new_possession = int(nearest["track_id"])
                if new_possession != self.possession:
                    self.possession = new_possession
                    events.append({
                        "frame_id": frame_id,
                        "event": "possession",
                        "player": self.possession,
                    })

        # shot_attempt: abrupt upward ball motion in the image while someone
        # had possession. stays in pixel space: the floor homography would
        # flatten the ball's arc to a point and destroy the cue.
        if self.last_ball_center_pixel is not None and self.possession is not None:
            dy = ball_pixel_center[1] - self.last_ball_center_pixel[1]
            if not self.shot_in_flight and dy <= -12:
                self.shot_in_flight = True
                self.last_shooter = self.possession
                events.append({
                    "frame_id": frame_id,
                    "event": "shot_attempt",
                    "player": self.last_shooter,
                })

        self.last_ball_center_pixel = ball_pixel_center
        return events
