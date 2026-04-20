"""keep exactly two persistent player tracks for a 1v1 clip.

strategy:
- each frame: consider all detected "person" boxes, keep the top-k by confidence.
- assign them to the two persistent slots (A, B) using hungarian matching
  on a cost = (1 - iou) + alpha * (1 - hist_similarity) + beta * motion_distance.
- during short detection gaps, predict via kalman and keep the slot alive.
- colors/ids never change over the clip, so downstream viz is stable.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Any

from _geom import bbox_iou as _bbox_iou

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore


@dataclass
class PlayerSlot:
    """one of the two persistent slots in a 1v1 match."""
    slot_id: int                           # 1 or 2
    bbox: list[float]                      # last known [x1,y1,x2,y2]
    hist: np.ndarray                       # hsv color histogram of torso region
    velocity: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    last_seen: int = -1
    frames_alive: int = 0
    frames_missed: int = 0


class DuoTracker:
    """tracks two players with persistent identity.

    parameters:
        max_missed: drop a slot after this many consecutive missed frames.
        iou_weight / color_weight / motion_weight: matching cost contributions.
    """

    def __init__(
        self,
        max_missed: int = 30,
        iou_weight: float = 1.0,
        color_weight: float = 1.5,
        motion_weight: float = 0.001,
        min_bbox_height_frac: float = 0.12,
        min_bbox_width_frac: float = 0.03,
    ):
        self.max_missed = max_missed
        self.iou_w = iou_weight
        self.color_w = color_weight
        self.motion_w = motion_weight
        # filter out tiny phantom boxes (corner logos, spectators).
        # sizes are fractions of the frame height / width.
        self.min_bbox_h_frac = min_bbox_height_frac
        self.min_bbox_w_frac = min_bbox_width_frac
        self.slots: list[PlayerSlot] = []
        self._next_id = 1

    # ---- public api ---------------------------------------------------------

    def update(self, frame: np.ndarray, detections: list[dict[str, Any]],
               frame_id: int) -> list[dict[str, Any]]:
        """consume this frame's detections, return two tracked player dicts."""
        h, w = frame.shape[:2]
        min_h = self.min_bbox_h_frac * h
        min_w = self.min_bbox_w_frac * w
        # filter tiny boxes (phantoms) before picking top candidates.
        big_enough = [
            d for d in detections
            if d.get("class_name") == "player"
            and (d["bbox"][2] - d["bbox"][0]) >= min_w
            and (d["bbox"][3] - d["bbox"][1]) >= min_h
        ]
        players = sorted(big_enough, key=lambda d: d.get("confidence", 0.0),
                           reverse=True)[:4]

        # bootstrap the two slots if we don't have them yet.
        if len(self.slots) < 2 and len(players) >= 1:
            self._bootstrap(frame, players, frame_id)

        if not self.slots:
            return []

        # match detections to existing slots via hungarian.
        matches, unmatched_dets = self._match(frame, players)

        # apply matches: update bbox / hist / velocity on each matched slot.
        matched_slot_ids = set()
        for slot_idx, det_idx in matches:
            slot = self.slots[slot_idx]
            det = players[det_idx]
            self._update_slot(slot, frame, det, frame_id)
            matched_slot_ids.add(slot.slot_id)

        # unmatched slots: increment missed counter, predict via velocity.
        for slot in self.slots:
            if slot.slot_id in matched_slot_ids:
                continue
            slot.frames_missed += 1
            bx1, by1, bx2, by2 = slot.bbox
            vx, vy = slot.velocity
            slot.bbox = [bx1 + vx, by1 + vy, bx2 + vx, by2 + vy]

        # drop slots that have been missing too long, let them re-acquire
        # from a new detection next frame.
        self.slots = [s for s in self.slots if s.frames_missed <= self.max_missed]

        # re-acquire: if we have free slots and unmatched detections, bind them.
        while len(self.slots) < 2 and unmatched_dets:
            det = unmatched_dets.pop(0)
            self._add_slot(frame, det, frame_id)

        return self._as_tracks()

    # ---- internals ----------------------------------------------------------

    def _bootstrap(self, frame: np.ndarray, players: list[dict], frame_id: int) -> None:
        """pick the first one or two player boxes and seed the slots."""
        for det in players[: 2 - len(self.slots)]:
            self._add_slot(frame, det, frame_id)

    def _add_slot(self, frame: np.ndarray, det: dict, frame_id: int) -> None:
        slot = PlayerSlot(
            slot_id=self._next_id,
            bbox=list(det["bbox"]),
            hist=self._torso_hist(frame, det["bbox"]),
            confidence=float(det.get("confidence", 0.0)),
            last_seen=frame_id,
            frames_alive=1,
        )
        self._next_id += 1
        self.slots.append(slot)

    def _update_slot(self, slot: PlayerSlot, frame: np.ndarray,
                      det: dict, frame_id: int) -> None:
        old_cx = (slot.bbox[0] + slot.bbox[2]) / 2
        old_cy = (slot.bbox[1] + slot.bbox[3]) / 2
        new_bbox = list(det["bbox"])
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        slot.velocity = (new_cx - old_cx, new_cy - old_cy)
        slot.bbox = new_bbox
        # blend histograms (exponential moving average) for appearance drift.
        new_hist = self._torso_hist(frame, new_bbox)
        slot.hist = 0.7 * slot.hist + 0.3 * new_hist
        slot.confidence = float(det.get("confidence", slot.confidence))
        slot.last_seen = frame_id
        slot.frames_alive += 1
        slot.frames_missed = 0

    def _match(self, frame: np.ndarray, players: list[dict]) -> tuple[list[tuple[int, int]], list[dict]]:
        if not players or not self.slots:
            return [], players[:]

        n_slots, n_dets = len(self.slots), len(players)
        cost = np.zeros((n_slots, n_dets), dtype=np.float32)
        for i, slot in enumerate(self.slots):
            for j, det in enumerate(players):
                cost[i, j] = self._pair_cost(slot, det, frame)

        # hungarian with scipy, or naive greedy fallback.
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            matches_raw = list(zip(rows.tolist(), cols.tolist()))
        else:
            matches_raw = _greedy_match(cost)

        # accept only matches below a cost cap so phantom detections don't
        # steal an identity. cap is loose; tight enough to reject obvious junk.
        COST_CAP = 2.5
        matches = [(s, d) for s, d in matches_raw if cost[s, d] < COST_CAP]
        matched_dets = {d for _, d in matches}
        unmatched_dets = [players[j] for j in range(n_dets) if j not in matched_dets]
        return matches, unmatched_dets

    def _pair_cost(self, slot: PlayerSlot, det: dict, frame: np.ndarray) -> float:
        iou = _bbox_iou(slot.bbox, det["bbox"])
        hist_sim = _hist_similarity(slot.hist, self._torso_hist(frame, det["bbox"]))
        dx = (slot.bbox[0] + slot.bbox[2]) / 2 - (det["bbox"][0] + det["bbox"][2]) / 2
        dy = (slot.bbox[1] + slot.bbox[3]) / 2 - (det["bbox"][1] + det["bbox"][3]) / 2
        motion = (dx * dx + dy * dy) ** 0.5
        return self.iou_w * (1 - iou) + self.color_w * (1 - hist_sim) + self.motion_w * motion

    def _torso_hist(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """hsv histogram of the torso region (middle 60% of the bbox)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2 = min(w, x2); y2 = min(h, y2)
        bh = max(1, y2 - y1)
        bw = max(1, x2 - x1)
        # torso: middle third vertically, central 80% horizontally.
        ty1 = y1 + int(0.20 * bh); ty2 = y1 + int(0.70 * bh)
        tx1 = x1 + int(0.10 * bw); tx2 = x2 - int(0.10 * bw)
        if ty2 <= ty1 or tx2 <= tx1:
            return np.zeros((16 * 16,), dtype=np.float32)
        region = frame[ty1:ty2, tx1:tx2]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def _as_tracks(self) -> list[dict[str, Any]]:
        return [
            {
                "track_id": slot.slot_id,
                "bbox": slot.bbox,
                "class_name": "player",
                "confidence": slot.confidence,
                "frames_alive": slot.frames_alive,
            }
            for slot in self.slots
        ]


# ---- helpers --------------------------------------------------------------


def _hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """bhattacharyya-inspired similarity in [0, 1]."""
    if h1.size != h2.size:
        return 0.0
    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                                   cv2.HISTCMP_CORREL))


def _greedy_match(cost: np.ndarray) -> list[tuple[int, int]]:
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort()
    out: list[tuple[int, int]] = []
    for _, i, j in flat:
        if i in used_rows or j in used_cols:
            continue
        out.append((i, j)); used_rows.add(i); used_cols.add(j)
    return out
