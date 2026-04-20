"""annotate frames with persistent player colors, ball, hoop, and event banners."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from hoop import Hoop


# bgr palette for persistent player slots. indexed by slot_id-1.
PLAYER_COLORS = [
    (255, 140, 0),     # orange-blue (player 1)
    (0, 220, 80),      # green (player 2)
]
BALL_COLOR = (0, 200, 255)         # amber
HOOP_COLOR = (255, 120, 255)       # magenta
EVENT_COLOR = (0, 255, 255)        # yellow

BANNER_HOLD_FRAMES = 45
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def annotate_video(
    video_path: str,
    tracks_by_frame: Iterable[dict[str, Any]],
    events: list[dict[str, Any]],
    hoop: Hoop,
    output_path: str,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    rim_trace: Iterable[dict[str, Any]] | None = None,
) -> None:
    """second-pass annotator: read the video, draw, write mp4.

    rim_trace (optional): per-frame dicts with {frame_id, center, radius,
    source} produced by RimTracker. when provided, the magenta hoop dot
    tracks the per-frame rim position instead of staying pinned to the
    static JSON anchor.
    """
    tracks_index = {f["frame_id"]: f for f in tracks_by_frame}
    events_by_frame: dict[int, list[dict]] = {}
    for e in events:
        events_by_frame.setdefault(int(e["frame_id"]), []).append(e)
    rim_by_frame: dict[int, dict] = {}
    for h in (rim_trace or []):
        rim_by_frame[int(h["frame_id"])] = h

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps)) if end_seconds is not None else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), FOURCC, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer at {out}")

    recent_events: deque[tuple[int, dict]] = deque()
    frame_id = start_frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if end_frame is not None and frame_id >= end_frame: break

            # decay old banner entries.
            while recent_events and frame_id - recent_events[0][0] > BANNER_HOLD_FRAMES:
                recent_events.popleft()
            for e in events_by_frame.get(frame_id, []):
                recent_events.append((frame_id, e))

            # prefer the per-frame rim trace; fall back to the static anchor.
            live = rim_by_frame.get(frame_id)
            if live is not None:
                _draw_live_hoop(frame, live)
            else:
                _draw_hoop(frame, hoop)
            frame_data = tracks_index.get(frame_id, {})
            _draw_players(frame, [t for t in frame_data.get("tracks", [])
                                    if t.get("class_name") == "player"])
            _draw_ball(frame, [t for t in frame_data.get("tracks", [])
                                if t.get("class_name") == "ball"])
            _draw_events_banner(frame, list(recent_events))
            _draw_header(frame, frame_id, fps)
            writer.write(frame)
            frame_id += 1
    finally:
        writer.release(); cap.release()


# ---- drawing primitives --------------------------------------------------


def _color_for_slot(slot_id: int) -> tuple[int, int, int]:
    return PLAYER_COLORS[(slot_id - 1) % len(PLAYER_COLORS)]


def _draw_players(frame: np.ndarray, players: list[dict[str, Any]]) -> None:
    for t in players:
        slot = int(t.get("track_id", 0))
        color = _color_for_slot(slot)
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"P{slot}"
        _text_with_shadow(frame, label, (x1, max(14, y1 - 6)), color, 0.8, 2)


def _draw_ball(frame: np.ndarray, balls: list[dict[str, Any]]) -> None:
    for b in balls:
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max(4, min(x2 - x1, y2 - y1) // 2)
        cv2.circle(frame, (cx, cy), r, BALL_COLOR, 2)
        cv2.circle(frame, (cx, cy), 2, BALL_COLOR, -1)


def _draw_hoop(frame: np.ndarray, hoop: Hoop) -> None:
    cv2.circle(frame, hoop.center, hoop.radius, HOOP_COLOR, 2)
    cv2.drawMarker(frame, hoop.center, HOOP_COLOR,
                    cv2.MARKER_CROSS, 14, 2)


def _draw_live_hoop(frame: np.ndarray, live: dict[str, Any]) -> None:
    """rim from the per-frame trace. tags the source in the corner so it's
    obvious whether the pipeline is using static anchor vs tracker vs
    detection."""
    cx, cy = [int(v) for v in live["center"]]
    r = int(live.get("radius", 30))
    src = str(live.get("source", "?"))
    cv2.circle(frame, (cx, cy), r, HOOP_COLOR, 2)
    cv2.drawMarker(frame, (cx, cy), HOOP_COLOR, cv2.MARKER_CROSS, 14, 2)
    # small label near the rim circle.
    _text_with_shadow(frame, f"hoop:{src}", (cx + r + 4, cy - 4),
                        HOOP_COLOR, 0.5, 1)


def _draw_events_banner(frame: np.ndarray, recent: list[tuple[int, dict]]) -> None:
    if not recent: return
    h, w = frame.shape[:2]
    banner_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # take the most recent 3 events.
    latest = recent[-3:]
    text = "  ".join(f"{e['event']} -> P{e.get('player', '?')}" for _, e in latest)
    _text_with_shadow(frame, text, (10, 26), EVENT_COLOR, 0.8, 2)


def _draw_header(frame: np.ndarray, frame_id: int, fps: float) -> None:
    h, w = frame.shape[:2]
    t_s = frame_id / fps
    mins, secs = int(t_s // 60), int(t_s % 60)
    label = f"frame {frame_id}  t={mins:02d}:{secs:02d}"
    _text_with_shadow(frame, label, (10, h - 12), (255, 255, 255), 0.6, 1)


def _text_with_shadow(frame, text, org, color, scale, thickness):
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                 scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                 scale, color, thickness, cv2.LINE_AA)
