"""RTMPose skeleton keypoints for shooting-motion detection.

wraps rtmlib (CPU / ONNX-runtime, no mmpose/pytorch dep for inference).
emits PerceptionSignal extras containing 17 COCO keypoints per player
bbox + derived features that the shot_attempt rule can gate on:

  shooting_pose_score  [0, 1]  combined score from three cues:
      wrist_above_head  wrist y < head y (ball is typically above)
      elbow_extension    elbow angle in last frame > 160 deg (release)
      torso_forward      torso pitched forward (shot prep)

the scorer returns a float per player. the shot_attempt rule uses it
as one more predicate in the conjunction (requiring e.g. >= 0.5 in
the 10 frames preceding the velocity trigger).

keypoint indices follow COCO-WholeBody (first 17 are body):
  0 nose, 1/2 eye, 3/4 ear, 5/6 shoulder, 7/8 elbow, 9/10 wrist,
  11/12 hip, 13/14 knee, 15/16 ankle.

runtime: rtmlib runs ONNX CPU at ~30-50 ms per player bbox. acceptable
for 2 players at 30 fps; we run it on-demand only around windows
where velocity+release are already plausible, not every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from rtmlib import Body
except Exception:  # pragma: no cover
    Body = None

from fusion import PerceptionSignal


# COCO body keypoint indices we care about.
IDX = {
    "nose": 0,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7,    "r_elbow": 8,
    "l_wrist": 9,    "r_wrist": 10,
    "l_hip": 11,     "r_hip": 12,
}


@dataclass
class PoseFeatures:
    """derived shooting-pose cues on one player on one frame."""
    wrist_above_head: bool
    elbow_extended: bool
    torso_forward: bool
    shooting_pose_score: float      # 0-1 weighted sum of the three cues
    keypoints: np.ndarray           # 17x3 (x, y, conf)

    def to_dict(self) -> dict:
        return {
            "wrist_above_head": bool(self.wrist_above_head),
            "elbow_extended": bool(self.elbow_extended),
            "torso_forward": bool(self.torso_forward),
            "shooting_pose_score": float(self.shooting_pose_score),
        }


class PoseEstimator:
    """RTMPose + shooting-motion derivation.

    args:
        device:            "cpu" (default, onnx-runtime) or "cuda".
        backend:           "onnxruntime" (default) or "opencv".
        mode:              "lightweight" | "balanced" | "performance"
                           (rtmlib preset names; bigger = slower + more accurate)
        min_kpt_conf:      per-keypoint confidence floor for the derived
                           features. below this we treat the keypoint as
                           missing.
    """

    def __init__(
        self,
        device: str = "cuda",             # gpu by default; "cpu" for login node
        backend: str = "onnxruntime",
        mode: str = "balanced",
        min_kpt_conf: float = 0.3,
    ) -> None:
        self.min_kpt_conf = float(min_kpt_conf)
        self.model = Body(mode=mode, backend=backend, device=device) if Body else None

    def available(self) -> bool:
        return self.model is not None

    def describe(self) -> dict:
        return {"available": self.available(),
                "min_kpt_conf": self.min_kpt_conf}

    def infer(self, frame: np.ndarray, player_bboxes: list[list[float]]
               ) -> list[PoseFeatures | None]:
        """one PoseFeatures per input bbox (None if keypoints are too sparse)."""
        if self.model is None or not player_bboxes:
            return [None] * len(player_bboxes)
        # rtmlib takes the whole frame and runs its own detector unless you
        # feed precomputed bboxes. Body(mode=...).__call__(img) returns
        # (keypoints, scores) for all detected bodies. we match each returned
        # body back to our bboxes by IoU so the output order matches input.
        kpts, scores = self.model(frame)
        if kpts is None or len(kpts) == 0:
            return [None] * len(player_bboxes)
        out: list[PoseFeatures | None] = []
        for bbox in player_bboxes:
            best_i, best_iou = -1, 0.0
            for i, k in enumerate(kpts):
                b = _kpts_to_bbox(k)
                iou = _iou(bbox, b)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i < 0 or best_iou < 0.1:
                out.append(None); continue
            k = kpts[best_i]
            s = scores[best_i]
            out.append(self._derive(k, s))
        return out

    def emit_signals(
        self,
        frame: np.ndarray,
        player_tracks: list[dict],
        frame_id: int,
    ) -> list[PerceptionSignal]:
        """produce PerceptionSignal records with pose extras for each player."""
        if not player_tracks:
            return []
        bboxes = [t["bbox"] for t in player_tracks]
        feats = self.infer(frame, bboxes)
        out: list[PerceptionSignal] = []
        for t, f in zip(player_tracks, feats):
            if f is None:
                continue
            out.append(PerceptionSignal(
                source="rtmpose",
                target="player",
                frame_id=int(frame_id),
                confidence=float(np.clip(f.shooting_pose_score, 0.0, 1.0)),
                bbox=[float(v) for v in t["bbox"]],
                track_id=int(t.get("track_id", -1)),
                extras={"pose": f.to_dict()},
            ))
        return out

    # ---- internals ----------------------------------------------------------

    def _derive(self, kpts: np.ndarray, scores: np.ndarray) -> PoseFeatures:
        """compute the three shooting cues + a combined score."""
        kxy = np.asarray(kpts, dtype=float).reshape(-1, 2)
        ks = np.asarray(scores, dtype=float)
        full = np.concatenate([kxy, ks[:, None]], axis=1)   # 17x3

        # wrist above head: use the higher-confidence wrist.
        nose = self._pt(full, IDX["nose"])
        lw = self._pt(full, IDX["l_wrist"])
        rw = self._pt(full, IDX["r_wrist"])
        shooting_wrist = None
        for w in (lw, rw):
            if w is None: continue
            if shooting_wrist is None or w[1] < shooting_wrist[1]:
                shooting_wrist = w
        wrist_above_head = (nose is not None and shooting_wrist is not None
                             and shooting_wrist[1] < nose[1])

        # elbow extension: inner angle at the shooting elbow.
        # same-side of the shooting wrist.
        if shooting_wrist is lw:
            sh, el, wr = (self._pt(full, IDX["l_shoulder"]),
                          self._pt(full, IDX["l_elbow"]),
                          lw)
        elif shooting_wrist is rw:
            sh, el, wr = (self._pt(full, IDX["r_shoulder"]),
                          self._pt(full, IDX["r_elbow"]),
                          rw)
        else:
            sh = el = wr = None
        elbow_angle = _angle(sh, el, wr) if all(p is not None for p in (sh, el, wr)) else None
        elbow_extended = (elbow_angle is not None and elbow_angle >= 150.0)

        # torso forward: vertical line shoulder -> hip pitched forward.
        l_sh = self._pt(full, IDX["l_shoulder"])
        r_sh = self._pt(full, IDX["r_shoulder"])
        l_hip = self._pt(full, IDX["l_hip"])
        r_hip = self._pt(full, IDX["r_hip"])
        mid_sh = _midpoint(l_sh, r_sh)
        mid_hip = _midpoint(l_hip, r_hip)
        torso_pitch = _vertical_pitch(mid_sh, mid_hip)
        torso_forward = torso_pitch is not None and 5 <= torso_pitch <= 45

        score = (0.5 * float(wrist_above_head)
                 + 0.3 * float(elbow_extended)
                 + 0.2 * float(torso_forward))

        return PoseFeatures(
            wrist_above_head=wrist_above_head,
            elbow_extended=elbow_extended,
            torso_forward=torso_forward,
            shooting_pose_score=score,
            keypoints=full.astype(np.float32),
        )

    def _pt(self, full: np.ndarray, idx: int) -> np.ndarray | None:
        if full[idx, 2] < self.min_kpt_conf:
            return None
        return full[idx, :2]


# ---- geometry helpers -----------------------------------------------------


def _kpts_to_bbox(kpts_xy: np.ndarray) -> list[float]:
    xy = np.asarray(kpts_xy, dtype=float).reshape(-1, 2)
    x1, y1 = xy.min(axis=0).tolist()
    x2, y2 = xy.max(axis=0).tolist()
    return [x1, y1, x2, y2]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _angle(a, b, c) -> float:
    """degree angle at vertex b for triangle a-b-c."""
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _midpoint(a, b) -> np.ndarray | None:
    if a is None or b is None: return None
    return (np.asarray(a) + np.asarray(b)) / 2.0


def _vertical_pitch(top: np.ndarray | None, bottom: np.ndarray | None
                     ) -> float | None:
    """pitch (deg) of the line top→bottom from true vertical."""
    if top is None or bottom is None: return None
    dx = float(bottom[0] - top[0])
    dy = float(bottom[1] - top[1])
    if abs(dy) < 1e-3: return None
    return float(abs(np.degrees(np.arctan2(dx, dy))))


def iter_all_signals(estimators: Iterable[PoseEstimator],
                      frame: np.ndarray, player_tracks: list[dict],
                      frame_id: int) -> list[PerceptionSignal]:
    """run one or more pose estimators on a frame; flatten signals."""
    out: list[PerceptionSignal] = []
    for e in estimators:
        out.extend(e.emit_signals(frame, player_tracks, frame_id))
    return out
