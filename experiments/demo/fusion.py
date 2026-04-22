"""confidence-weighted fusion of detector signals per target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Target = Literal["rim", "ball", "player"]


@dataclass
class PerceptionSignal:
    """one detector's output for one target on one frame."""
    source: str
    target: Target
    frame_id: int
    confidence: float
    bbox: list[float] | None = None
    center: tuple[float, float] | None = None
    radius: int | None = None
    track_id: int | None = None
    extras: dict = field(default_factory=dict)

    def xy(self) -> tuple[float, float] | None:
        if self.center is not None:
            return self.center
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return None


@dataclass
class FusedResult:
    """consensus across contributing signals for one target on one frame."""
    target: Target
    frame_id: int
    center: tuple[float, float] | None = None
    bbox: list[float] | None = None
    radius: int | None = None
    confidence: float = 0.0
    contributing_sources: list[str] = field(default_factory=list)
    signal_count: int = 0
    agreement_score: float = 0.0


class ConsensusFuser:
    """fuse PerceptionSignal lists into FusedResult per target."""

    def __init__(
        self,
        *,
        min_signal_conf: float = 0.15,
        center_agreement_px: float = 40.0,
        bbox_iou_merge: float = 0.4,
        disagreement_penalty: float = 0.5,
    ) -> None:
        self.min_signal_conf = float(min_signal_conf)
        self.center_agreement_px = float(center_agreement_px)
        self.bbox_iou_merge = float(bbox_iou_merge)
        self.disagreement_penalty = float(disagreement_penalty)

    def fuse_rim(self, signals: list[PerceptionSignal]) -> FusedResult | None:
        # rim is one point + optional radius.
        return self._fuse_single_point(signals, target="rim",
                                        include_radius=True)

    def fuse_ball(self, signals: list[PerceptionSignal]) -> FusedResult | None:
        # ball is one bbox.
        return self._fuse_single_bbox(signals, target="ball")

    def fuse_players(self, signals: list[PerceptionSignal]
                     ) -> list[FusedResult]:
        # players are multiple bboxes; cluster by iou, one result per cluster.
        if not signals:
            return []
        signals = [s for s in signals if s.confidence >= self.min_signal_conf
                    and s.bbox is not None]
        if not signals:
            return []
        # greedy iou clustering.
        clusters: list[list[PerceptionSignal]] = []
        for s in sorted(signals, key=lambda x: x.confidence, reverse=True):
            placed = False
            for cluster in clusters:
                if any(_bbox_iou(s.bbox, other.bbox) >= self.bbox_iou_merge
                       for other in cluster):
                    cluster.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])
        return [self._reduce_bbox_cluster(c, target="player") for c in clusters]

    # ---- internals ----------------------------------------------------------

    def _fuse_single_point(
        self,
        signals: list[PerceptionSignal],
        target: Target,
        *,
        include_radius: bool,
    ) -> FusedResult | None:
        signals = [s for s in signals
                    if s.confidence >= self.min_signal_conf
                    and s.xy() is not None]
        if not signals:
            return None
        if len(signals) == 1:
            s = signals[0]
            return FusedResult(
                target=target, frame_id=s.frame_id,
                center=s.xy(),
                radius=s.radius if include_radius else None,
                confidence=s.confidence,
                contributing_sources=[s.source],
                signal_count=1, agreement_score=1.0,
            )
        frame_id = signals[0].frame_id
        xs = np.array([s.xy()[0] for s in signals], dtype=float)
        ys = np.array([s.xy()[1] for s in signals], dtype=float)
        ws = np.array([max(s.confidence, 1e-3) for s in signals], dtype=float)
        cx = float(_weighted_median(xs, ws))
        cy = float(_weighted_median(ys, ws))

        d = np.hypot(xs - cx, ys - cy)
        agreeing = (d <= self.center_agreement_px).sum() / len(d)
        base_conf = float((ws * np.array([s.confidence for s in signals])).sum()
                          / ws.sum())
        # penalty applied once per outlier beyond agreement radius.
        outliers = int((d > self.center_agreement_px).sum())
        fused_conf = float(np.clip(base_conf * (self.disagreement_penalty ** outliers),
                                     0.0, 1.0))

        radius = None
        if include_radius:
            rs = [s.radius for s in signals if s.radius is not None]
            if rs:
                radius = int(round(float(np.median(rs))))

        return FusedResult(
            target=target, frame_id=frame_id,
            center=(cx, cy), radius=radius,
            confidence=fused_conf,
            contributing_sources=[s.source for s in signals],
            signal_count=len(signals),
            agreement_score=float(agreeing),
        )

    def _fuse_single_bbox(
        self,
        signals: list[PerceptionSignal],
        target: Target,
    ) -> FusedResult | None:
        signals = [s for s in signals
                    if s.confidence >= self.min_signal_conf
                    and s.bbox is not None]
        if not signals:
            return None
        if len(signals) == 1:
            s = signals[0]
            return FusedResult(
                target=target, frame_id=s.frame_id,
                center=s.xy(), bbox=list(s.bbox),
                confidence=s.confidence,
                contributing_sources=[s.source],
                signal_count=1, agreement_score=1.0,
            )
        # seed = highest-conf; merge any others that overlap the seed.
        signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
        seed = signals[0]
        merged = [seed]
        for s in signals[1:]:
            if _bbox_iou(seed.bbox, s.bbox) >= self.bbox_iou_merge:
                merged.append(s)
        return self._reduce_bbox_cluster(merged, target=target)

    def _reduce_bbox_cluster(self, cluster: list[PerceptionSignal],
                              target: Target) -> FusedResult:
        ws = np.array([max(s.confidence, 1e-3) for s in cluster], dtype=float)
        boxes = np.array([s.bbox for s in cluster], dtype=float)
        fused_bbox = (boxes * ws[:, None]).sum(axis=0) / ws.sum()
        cx = float((fused_bbox[0] + fused_bbox[2]) / 2)
        cy = float((fused_bbox[1] + fused_bbox[3]) / 2)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        d = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
        agreement = float((d <= self.center_agreement_px).mean())
        base = float((ws * np.array([s.confidence for s in cluster])).sum()
                      / ws.sum())
        outliers = int((d > self.center_agreement_px).sum())
        fused_conf = float(np.clip(base * (self.disagreement_penalty ** outliers),
                                     0.0, 1.0))
        return FusedResult(
            target=target, frame_id=cluster[0].frame_id,
            center=(cx, cy), bbox=fused_bbox.tolist(),
            confidence=fused_conf,
            contributing_sources=[s.source for s in cluster],
            signal_count=len(cluster),
            agreement_score=agreement,
        )


# ---- helpers --------------------------------------------------------------


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumw = np.cumsum(w)
    half = cumw[-1] / 2.0
    idx = int(np.searchsorted(cumw, half))
    idx = min(idx, len(v) - 1)
    return float(v[idx])
