"""compose shot_attempt_v2 + shot_made_v2 into a single object that mirrors
the ShotDetector.update() contract.

usage:
    pipe = ShotPipelineV2(anchor=hoop, **tunables)
    for frame_id, (frame, ball_bbox, player_tracks) in enumerate(stream):
        events = pipe.update(frame, frame_id, ball_bbox, player_tracks)

the orchestrator owns the rim tracker so it can consume video frames when
available, and falls back to the static hoop when given tracks-only input
(used by offline sweeps that read tracks.jsonl without the source video).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hoop import Hoop
from rim_tracker import RimTracker, TrackedHoop
from shot_attempt_v2 import ShotAttemptV2
from shot_made_v2 import ShotMadeV2


class ShotPipelineV2:
    def __init__(
        self,
        anchor: Hoop,
        *,
        # rim tracker
        tracker_kind: str = "static",
        reseed_every: int = 90,
        # attempt tunables
        upward_trigger: float = -10.0,
        history_frames: int = 6,
        release_dist_px: float = 60.0,
        approach_dist_px: float = 260.0,
        cooldown_frames: int = 20,
        possession_dist_px: float = 140.0,
        # made tunables
        occlusion_gap_frames: int = 10,
        attempt_to_made_window: int = 90,
        enter_zone_radius_factor: float = 2.0,
        horizontal_pad_factor: float = 1.5,
        min_downward_velocity: float = 1.5,
        # frame-rate scaling. defaults tuned on 30 fps footage; 60 fps
        # clips need the frame-based windows doubled so the *time* they
        # represent stays constant.
        fps: float = 30.0,
    ) -> None:
        scale = max(1.0, float(fps) / 30.0)
        occlusion_gap_frames = int(round(occlusion_gap_frames * scale))
        attempt_to_made_window = int(round(attempt_to_made_window * scale))
        cooldown_frames = int(round(cooldown_frames * scale))
        history_frames = max(3, int(round(history_frames * scale)))
        reseed_every = max(1, int(round(reseed_every * scale)))
        # velocity thresholds scale inversely: at 60 fps, the frame-to-frame
        # dy is half what it is at 30 fps for the same real-world velocity.
        upward_trigger = float(upward_trigger) / scale
        min_downward_velocity = float(min_downward_velocity) / scale
        self._fps_scale = scale
        self.anchor = anchor
        self.rim = RimTracker(anchor=anchor, tracker_kind=tracker_kind,
                              reseed_every=reseed_every)
        self.att = ShotAttemptV2(
            upward_trigger=upward_trigger,
            history_frames=history_frames,
            release_dist_px=release_dist_px,
            approach_dist_px=approach_dist_px,
            cooldown_frames=cooldown_frames,
            possession_dist_px=possession_dist_px,
        )
        self.made = ShotMadeV2(
            occlusion_gap_frames=occlusion_gap_frames,
            attempt_to_made_window=attempt_to_made_window,
            enter_zone_radius_factor=enter_zone_radius_factor,
            horizontal_pad_factor=horizontal_pad_factor,
            min_downward_velocity=min_downward_velocity,
        )
        self._last_hoop: TrackedHoop = TrackedHoop(
            center=anchor.center, radius=anchor.radius, source="static")

    def current_hoop(self) -> TrackedHoop:
        return self._last_hoop

    def update(
        self,
        frame: np.ndarray | None,
        frame_id: int,
        ball_bbox: list[float] | None,
        player_tracks: list[dict],
        hoop_bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """step one frame. hoop_bbox (when provided by a custom YOLO model)
        snaps the RimTracker to a per-frame detection, bypassing CSRT/MIL.
        """
        # step the rim tracker. if no frame, reuse the last (static) hoop.
        if frame is not None:
            self._last_hoop = self.rim.update(frame, frame_id, detected_bbox=hoop_bbox)
        elif hoop_bbox is not None:
            # offline mode with a recorded hoop detection (rare).
            self._last_hoop = self.rim._hoop_from_detection(hoop_bbox)
        # step attempt detector — emits possession + shot_attempt.
        attempt_events = self.att.update(frame_id, ball_bbox, player_tracks,
                                          hoop=self._last_hoop)
        # notify made detector of any attempts that fired this frame.
        for e in attempt_events:
            if e["event"] == "shot_attempt":
                self.made.on_shot_attempt(e["frame_id"], e["player"])
        # step made detector — may emit shot_made or shot_miss.
        made_events = self.made.update(frame_id, ball_bbox, self._last_hoop)
        for e in made_events:
            if e["event"] in ("shot_made", "shot_miss"):
                self.att.resolve_attempt()
        return attempt_events + made_events
