"""shot_attempt_v2: multi-predicate rule to reduce false positives.

v1 fires on a single test: avg dy <= upward_trigger. this catches every
dribble bounce, crossover, and lob pass.

v2 demands all of:
  1. sustained upward velocity: avg(dy) over last K frames <= upward_trigger
  2. ball leaves possessor: distance to current-possession player is
     monotonically increasing over last K frames and has exceeded
     `release_dist_px`.
  3. trajectory is toward the rim: |dx to rim_x| is not growing, AND the
     horizontal distance to the rim at current frame < `approach_dist_px`,
     OR the ball is already high enough (y < rim_y + 3*rim_radius).
  4. cooldown: no second attempt within `cooldown_frames` of the last one
     (prevents dribble re-triggering).

state: short rolling ball history + current possession (player_id +
last-known player position). emits possession changes like v1.

this module is a drop-in replacement for the attempt logic inside
`ShotDetector`. it coordinates with `shot_made_v2.ShotMadeV2` via
`on_shot_attempt` so the two share in-flight state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from _geom import bbox_center as _center, distance as _distance
from rim_tracker import TrackedHoop


@dataclass
class _Possession:
    player_id: int
    last_frame: int
    last_player_center: tuple[float, float] | None = None


class ShotAttemptV2:
    """trajectory + context-aware shot_attempt detector.

    tunables:
        upward_trigger: avg dy threshold for "going up". same sign as v1.
        history_frames: window for velocity averaging.
        release_dist_px: min distance between ball and possessor for an
            attempt to be plausible (the ball has actually left the hand).
        approach_dist_px: if the ball is this close horizontally to the rim
            center, the trajectory-toward-rim test is considered satisfied
            on its own.
        cooldown_frames: frames to suppress a new attempt after one fires.
        possession_dist_px: radius for "is this player holding the ball".
    """

    def __init__(
        self,
        upward_trigger: float = -10.0,
        history_frames: int = 6,
        release_dist_px: float = 60.0,
        approach_dist_px: float = 260.0,
        cooldown_frames: int = 20,
        possession_dist_px: float = 140.0,
        # possession hysteresis: avoid flipping the current possession on a
        # single noisy frame where the ball is almost equidistant between
        # P1 and P2. we require the "new" candidate to be closer for at
        # least `possession_switch_evidence` consecutive ball-seen frames,
        # AND meaningfully closer (ratio >= `possession_switch_ratio`) in
        # those frames, before the possession flips.
        possession_switch_evidence: int = 4,
        possession_switch_ratio: float = 1.25,
    ) -> None:
        self.upward_trigger = float(upward_trigger)
        self.history_len = max(3, int(history_frames))
        self.release_dist = float(release_dist_px)
        self.approach_dist = float(approach_dist_px)
        self.cooldown = max(0, int(cooldown_frames))
        self.possession_dist_px = float(possession_dist_px)
        self.switch_evidence = max(1, int(possession_switch_evidence))
        self.switch_ratio = max(1.0, float(possession_switch_ratio))
        self.ball_history: deque[tuple[int, tuple[float, float]]] = deque(
            maxlen=self.history_len)
        self.player_dist_history: deque[float] = deque(maxlen=self.history_len)
        self.current_possession: _Possession | None = None
        # running count of consecutive frames where a specific "new"
        # player beat the incumbent by the switch_ratio margin.
        self._switch_candidate: int | None = None
        self._switch_streak: int = 0
        self.last_attempt_frame: int = -10 ** 9
        self.attempt_in_flight: bool = False

    # ---- public api ---------------------------------------------------------

    def update(
        self,
        frame_id: int,
        ball_bbox: list[float] | None,
        player_tracks: list[dict],
        hoop: TrackedHoop | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        ball_center = _center(ball_bbox) if ball_bbox else None

        # (1) possession tracking: nearest player within threshold.
        possession_event = self._update_possession(frame_id, ball_center,
                                                    player_tracks)
        if possession_event:
            events.append(possession_event)

        # (2) record ball history + ball-to-possessor distance.
        if ball_center is not None:
            self.ball_history.append((frame_id, ball_center))
            if (self.current_possession is not None
                    and self.current_possession.last_player_center is not None):
                d = _distance(ball_center,
                              self.current_possession.last_player_center)
                self.player_dist_history.append(d)
            else:
                self.player_dist_history.append(0.0)

        # (3) shot_attempt gate: all predicates must agree.
        if self._all_predicates_satisfied(frame_id, ball_center, hoop):
            shooter = (self.current_possession.player_id
                       if self.current_possession is not None else -1)
            events.append({
                "frame_id": frame_id,
                "event": "shot_attempt",
                "player": int(shooter),
            })
            self.last_attempt_frame = frame_id
            self.attempt_in_flight = True

        return events

    def resolve_attempt(self) -> None:
        """called when shot_made_v2 resolves (made or miss) the in-flight
        attempt so a new one can fire without waiting out the cooldown.
        """
        self.attempt_in_flight = False

    # ---- internals ----------------------------------------------------------

    def _update_possession(
        self,
        frame_id: int,
        ball_center: tuple[float, float] | None,
        player_tracks: list[dict],
    ) -> dict[str, Any] | None:
        """possession rule with hysteresis.

        Why: when the ball sits roughly between two players, the bare
        "nearest-player" rule flips repeatedly from frame to frame on
        detection noise — flooding the event log with spurious possession
        events. We now require a candidate player to be *meaningfully*
        closer (ratio >= switch_ratio) for a run of `switch_evidence`
        consecutive ball-seen frames before swapping possession.

        The incumbent keeps possession through:
          - frames with no ball detection,
          - frames where the ball is present but nobody is within
            possession_dist_px,
          - frames where the candidate's lead over the incumbent is less
            than switch_ratio (i.e. ambiguous proximity).
        """
        # keep the incumbent's last known center fresh even when the ball
        # isn't visible this frame, so release_distance stays current.
        if ball_center is None or not player_tracks:
            if self.current_possession is not None and player_tracks:
                for t in player_tracks:
                    if int(t.get("track_id", -1)) == self.current_possession.player_id:
                        self.current_possession.last_player_center = _center(t["bbox"])
                        break
            return None

        # distance from ball to each tracked player.
        dists = [(int(t["track_id"]), _center(t["bbox"]),
                  _distance(ball_center, _center(t["bbox"])))
                 for t in player_tracks]
        if not dists:
            return None
        dists.sort(key=lambda d: d[2])
        best_id, best_center, best_d = dists[0]

        # nobody close enough: hold whatever possession we already had.
        if best_d > self.possession_dist_px:
            self._switch_candidate = None
            self._switch_streak = 0
            return None

        # bootstrap: first possession assignment.
        if self.current_possession is None:
            self.current_possession = _Possession(best_id, frame_id, best_center)
            self._switch_candidate = None
            self._switch_streak = 0
            return {"frame_id": frame_id, "event": "possession", "player": best_id}

        incumbent_id = self.current_possession.player_id

        # incumbent still closest: stay. update state, clear switch counter.
        if best_id == incumbent_id:
            self.current_possession.last_frame = frame_id
            self.current_possession.last_player_center = best_center
            self._switch_candidate = None
            self._switch_streak = 0
            return None

        # candidate is closer. require it to beat the incumbent by the
        # switch_ratio margin, for switch_evidence consecutive ball-seen
        # frames, before we flip.
        incumbent_d = next((d for (tid, _, d) in dists if tid == incumbent_id), None)
        if incumbent_d is None:
            # incumbent not even in this frame's detections — likely a short
            # occlusion. hold possession; don't start a switch streak yet.
            return None
        if incumbent_d < best_d * self.switch_ratio:
            # ratio too close to call; reset streak.
            self._switch_candidate = None
            self._switch_streak = 0
            return None

        if self._switch_candidate == best_id:
            self._switch_streak += 1
        else:
            self._switch_candidate = best_id
            self._switch_streak = 1

        if self._switch_streak < self.switch_evidence:
            return None

        # evidence accumulated: flip possession.
        self.current_possession = _Possession(best_id, frame_id, best_center)
        self._switch_candidate = None
        self._switch_streak = 0
        return {"frame_id": frame_id, "event": "possession", "player": best_id}

    def _all_predicates_satisfied(
        self,
        frame_id: int,
        ball_center: tuple[float, float] | None,
        hoop: TrackedHoop | None,
    ) -> bool:
        if self.attempt_in_flight:
            return False
        if frame_id - self.last_attempt_frame < self.cooldown:
            return False
        if self.current_possession is None:
            return False
        if len(self.ball_history) < 3:
            return False

        # (a) sustained upward velocity over the history window.
        first_fid, first_pt = self.ball_history[0]
        last_fid, last_pt = self.ball_history[-1]
        span = max(1, last_fid - first_fid)
        avg_dy = (last_pt[1] - first_pt[1]) / span
        if avg_dy > self.upward_trigger:
            return False

        # (b) ball has left the possessor's hand.
        if ball_center is None or self.current_possession.last_player_center is None:
            return False
        ball_to_player = _distance(ball_center,
                                   self.current_possession.last_player_center)
        if ball_to_player < self.release_dist:
            return False

        # (c) trajectory approaches the rim (if hoop provided).
        if hoop is not None:
            rim_x, rim_y = hoop.center
            horiz = abs(last_pt[0] - rim_x)
            if horiz > self.approach_dist:
                # last chance: ball is already near rim height (high arc that
                # will soon come down), accept.
                if last_pt[1] > rim_y + 3 * max(6, hoop.radius):
                    return False
        return True


# _center and _distance live in _geom.py; imported at module top.
