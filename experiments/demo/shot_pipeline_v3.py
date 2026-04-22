"""shot_pipeline_v3: consensus-driven orchestrator composing v2 rules +
optional YOLOE / RTMPose / SAM 3 (or SAM 3.1) backends.

what's new vs v2:

  pluggable backends
    - use_yoloe:       open-vocab YOLOE producing rim/ball/player signals
    - use_pose:        RTMPose producing shooting-pose score per player
    - sam3_cache_path: pre-computed SAM 3 signals from disk (JSONL)
    - custom_model:    legacy path from custom_tracker.py (hoop model)
  all optional; when none are enabled v3 behaves exactly like v2.

  fused perception
    every enabled backend emits PerceptionSignal records. the
    ConsensusFuser combines them per-target per-frame. downstream
    rules (ShotAttemptV2, ShotMadeV2) consume the fused result as if
    it were a single detector's output, but with confidence that
    reflects inter-detector agreement.

  shooting-pose predicate
    when --use-pose is on, shot_attempt requires pose-based evidence
    in addition to its existing conjunction. reduces false positives
    on dribbles that happen to produce upward ball motion but no
    actual shooting posture.

the pipeline writes the same event schema as v2. downstream viz /
evaluation / aggregator do not need to change.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from _fg_stats import field_goal_stats
from fusion import ConsensusFuser, FusedResult, PerceptionSignal
from hoop import Hoop
from rim_tracker import RimTracker, TrackedHoop
from shot_attempt_v2 import ShotAttemptV2
from shot_made_v2 import ShotMadeV2


@dataclass
class V3Config:
    """declarative config for one v3 run. mirrors run_demo_v2 CLI flags."""
    # rim tracker (handcrafted)
    tracker_kind: str = "flow"
    reseed_every: int = 90
    # shot attempt
    upward_trigger: float = -10.0
    history_frames: int = 6
    release_dist_px: float = 60.0
    approach_dist_px: float = 260.0
    cooldown_frames: int = 20
    possession_dist_px: float = 140.0
    possession_switch_evidence: int = 4
    possession_switch_ratio: float = 1.25
    # shot made
    occlusion_gap_frames: int = 15
    attempt_to_made_window: int = 120
    enter_zone_radius_factor: float = 2.5
    horizontal_pad_factor: float = 2.0
    min_downward_velocity: float = 1.0
    fps: float = 30.0
    # new: backends
    use_yoloe: bool = False
    use_pose: bool = False
    sam3_cache_path: str | None = None
    # fusion knobs
    min_signal_conf: float = 0.15
    center_agreement_px: float = 40.0
    # shooting-pose predicate
    pose_window_frames: int = 10
    pose_score_threshold: float = 0.5


@dataclass
class _PoseHistory:
    """rolling per-player shooting-pose scores from RTMPose."""
    per_player: dict[int, deque] = field(default_factory=dict)
    window: int = 10

    def push(self, player_id: int, score: float) -> None:
        d = self.per_player.setdefault(player_id, deque(maxlen=self.window))
        d.append(float(score))

    def peak(self, player_id: int) -> float:
        d = self.per_player.get(player_id)
        if not d:
            return 0.0
        return max(d)


class ShotPipelineV3:
    """orchestrator for consensus-driven shot detection."""

    def __init__(
        self,
        anchor: Hoop,
        config: V3Config | None = None,
    ) -> None:
        self.anchor = anchor
        self.config = config or V3Config()
        self.rim = RimTracker(anchor=anchor, tracker_kind=self.config.tracker_kind,
                                reseed_every=self.config.reseed_every)

        # fps-scaled rule windows (reuse v2 semantics).
        scale = max(1.0, self.config.fps / 30.0)
        self.att = ShotAttemptV2(
            upward_trigger=self.config.upward_trigger / scale,
            history_frames=max(3, int(round(self.config.history_frames * scale))),
            release_dist_px=self.config.release_dist_px,
            approach_dist_px=self.config.approach_dist_px,
            cooldown_frames=int(round(self.config.cooldown_frames * scale)),
            possession_dist_px=self.config.possession_dist_px,
            possession_switch_evidence=self.config.possession_switch_evidence,
            possession_switch_ratio=self.config.possession_switch_ratio,
        )
        self.made = ShotMadeV2(
            occlusion_gap_frames=int(round(self.config.occlusion_gap_frames * scale)),
            attempt_to_made_window=int(round(self.config.attempt_to_made_window * scale)),
            enter_zone_radius_factor=self.config.enter_zone_radius_factor,
            horizontal_pad_factor=self.config.horizontal_pad_factor,
            min_downward_velocity=self.config.min_downward_velocity / scale,
        )
        self.fuser = ConsensusFuser(
            min_signal_conf=self.config.min_signal_conf,
            center_agreement_px=self.config.center_agreement_px,
        )
        self.pose_history = _PoseHistory(window=self.config.pose_window_frames)
        self._last_hoop: TrackedHoop = TrackedHoop(
            center=anchor.center, radius=anchor.radius, source="static")
        # preloaded SAM 3 cache, keyed by frame_id -> list[PerceptionSignal].
        self._sam3_by_frame: dict[int, list[PerceptionSignal]] = {}
        if self.config.sam3_cache_path:
            self._load_sam3_cache(self.config.sam3_cache_path)

    # ---- public api ---------------------------------------------------------

    def current_hoop(self) -> TrackedHoop:
        return self._last_hoop

    def update(
        self,
        frame: np.ndarray | None,
        frame_id: int,
        ball_bbox: list[float] | None,
        player_tracks: list[dict],
        *,
        yoloe_signals: list[PerceptionSignal] | None = None,
        pose_signals: list[PerceptionSignal] | None = None,
    ) -> list[dict[str, Any]]:
        """consume one frame's signals, emit events.

        this is a superset of v2's update(): v2 callers pass no optional
        signal kwargs and get v2 behaviour unchanged (baseline config).
        v3 callers pass open-vocab / pose / SAM 3 signals for consensus.
        """
        # ---- gather signals per target ---------------------------------------
        rim_signals: list[PerceptionSignal] = []
        ball_signals: list[PerceptionSignal] = []
        player_signals: list[PerceptionSignal] = []

        # handcrafted rim tracker → one rim signal.
        if frame is not None:
            self._last_hoop = self.rim.update(frame, frame_id)
            rim_signals.append(PerceptionSignal(
                source=f"flow_{self._last_hoop.source}",
                target="rim", frame_id=frame_id,
                confidence=0.6,
                center=tuple(self._last_hoop.center),
                radius=int(self._last_hoop.radius),
            ))

        # COCO YOLO ball → ball signal.
        if ball_bbox is not None:
            ball_signals.append(PerceptionSignal(
                source="coco_yolo", target="ball", frame_id=frame_id,
                confidence=0.5, bbox=list(ball_bbox),
            ))

        # COCO YOLO players (from DuoTracker output upstream).
        for t in player_tracks:
            player_signals.append(PerceptionSignal(
                source="coco_yolo_duotracker", target="player", frame_id=frame_id,
                confidence=float(t.get("confidence", 0.8)),
                bbox=list(t["bbox"]),
                track_id=int(t.get("track_id", -1)),
            ))

        # optional: YOLOE open-vocab signals (pre-separated by caller).
        for s in (yoloe_signals or []):
            if s.target == "rim":   rim_signals.append(s)
            if s.target == "ball":  ball_signals.append(s)
            if s.target == "player": player_signals.append(s)

        # optional: SAM 3 cache signals for this frame.
        for s in self._sam3_by_frame.get(frame_id, []):
            if s.target == "rim":   rim_signals.append(s)
            if s.target == "ball":  ball_signals.append(s)
            if s.target == "player": player_signals.append(s)

        # ---- fuse ------------------------------------------------------------
        fused_rim = self.fuser.fuse_rim(rim_signals)
        fused_ball = self.fuser.fuse_ball(ball_signals)

        # if fusion produced a better rim position than our handcrafted tracker,
        # update _last_hoop so downstream rules use the fused rim.
        if fused_rim is not None and fused_rim.confidence > 0.5:
            self._last_hoop = TrackedHoop(
                center=(int(fused_rim.center[0]), int(fused_rim.center[1])),
                radius=fused_rim.radius or self._last_hoop.radius,
                source=f"fused({len(fused_rim.contributing_sources)})",
            )

        # ---- pose history update ---------------------------------------------
        for s in (pose_signals or []):
            if s.track_id is not None:
                extra = s.extras.get("pose", {}) if s.extras else {}
                score = float(extra.get("shooting_pose_score", 0.0))
                self.pose_history.push(s.track_id, score)

        # ---- drive the rules -------------------------------------------------
        # shot_attempt + possession (v2 logic unchanged).
        fused_ball_bbox = fused_ball.bbox if fused_ball else None
        attempt_events = self.att.update(
            frame_id, fused_ball_bbox, player_tracks, hoop=self._last_hoop,
        )

        # optional pose gate on shot_attempt — drop attempts that lack a
        # plausible shooting pose within the recent window.
        if self.config.use_pose:
            attempt_events = [e for e in attempt_events
                              if self._pose_gate_passes(e)]

        for e in attempt_events:
            if e["event"] == "shot_attempt":
                self.made.on_shot_attempt(e["frame_id"], e["player"])

        made_events = self.made.update(frame_id, fused_ball_bbox, self._last_hoop)
        for e in made_events:
            if e["event"] in ("shot_made", "shot_miss"):
                self.att.resolve_attempt()

        return attempt_events + made_events

    def stats(self, events: list[dict]) -> dict:
        """per-player + overall FG stats, same schema as v2."""
        return field_goal_stats(events)

    # ---- internals ----------------------------------------------------------

    def _pose_gate_passes(self, event: dict) -> bool:
        """for shot_attempt, require peak shooting_pose in recent window."""
        if event.get("event") != "shot_attempt":
            return True
        pid = int(event.get("player", -1))
        peak = self.pose_history.peak(pid)
        return peak >= self.config.pose_score_threshold

    def _load_sam3_cache(self, path: str) -> None:
        """load pre-computed SAM 3 PerceptionSignals from JSONL."""
        import json
        self._sam3_by_frame = {}
        p = Path(path)
        if not p.exists():
            return
        with p.open() as f:
            for line in f:
                d = json.loads(line)
                sig = PerceptionSignal(
                    source=d["source"], target=d["target"],
                    frame_id=int(d["frame_id"]),
                    confidence=float(d["confidence"]),
                    bbox=d.get("bbox"),
                    center=tuple(d["center"]) if d.get("center") else None,
                    radius=d.get("radius"),
                    track_id=d.get("track_id"),
                    extras=d.get("extras") or {},
                )
                self._sam3_by_frame.setdefault(sig.frame_id, []).append(sig)
