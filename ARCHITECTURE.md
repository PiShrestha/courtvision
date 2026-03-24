# CourtVision Architecture

> **Goal:** turn raw 1v1 basketball footage into a timestamped event log and a natural-language scouting summary, with auditable intermediate artifacts at every stage.

This document is the engineering companion to the proposal. It describes the pipeline data flow, the contract between stages, and the reasoning behind the major design choices.

---

## High-level data flow

```
┌─────────────┐     ┌────────────────────┐     ┌──────────────────────┐     ┌───────────────┐
│    VIDEO    │ ──▶ │    PERCEPTION      │ ──▶ │   SYMBOLIC LOGIC     │ ──▶ │   NARRATIVE   │
│  (.mp4/.mov)│     │ YOLOv8 + ByteTrack │     │ RuleEngine + Homo    │     │ Gemini / local│
└─────────────┘     └────────────────────┘     └──────────────────────┘     └───────────────┘
                            │                            │                           │
                   per-frame tracks              event log (list of dicts)     scouting text
                   [track_id, bbox,              [{frame_id, event, player}]  (free-form string)
                    class, court_xy?]
```

The arrows are not abstractions — they are plain Python lists of dicts. Any stage can be run in isolation and the output serialised for inspection.

---

## Stage 1 — Perception (`perception/`)

| File | Responsibility |
|---|---|
| `pipeline.py` | opens the video, iterates frames, applies `start/end/stride`, and projects bottom-center bboxes to court coordinates when a homography is configured. |
| `tracker.py` | wraps Ultralytics `model.track(persist=True)` (ByteTrack by default, BoT-SORT optional). Owns the YOLO model because Ultralytics' tracker requires model state across calls. |

**Input:** a video file.

**Output per kept frame:**
```python
{
    "frame_id": int,
    "frame_height": int,
    "frame_width": int,
    "tracks": [
        {"track_id": int, "bbox": [x1, y1, x2, y2],
         "confidence": float, "class_name": "player"|"ball",
         "court_xy": (x, y)?}   # present only when homography is configured
    ]
}
```

**Why this split.** Detection and tracking are not separable in Ultralytics' API (the tracker reuses the model's internal state via `persist=True`). We had a separate `Detector` class at an earlier stage; it was removed once we switched to the real Ultralytics ByteTrack because it became dead code.

**Class mapping.** We only emit `player` (COCO class 0, "person") and `ball` (COCO class 32, "sports ball"). COCO has no `hoop`/`rim` class. This is the single biggest reason the project cannot currently detect `shot_made` — see [CONSTRAINTS.md](CONSTRAINTS.md).

---

## Stage 2 — Symbolic reasoning (`logic/`)

| File | Responsibility |
|---|---|
| `homography.py` | solves a 3×3 planar homography via `cv2.findHomography` (RANSAC, 5 px), projects pixels to court coordinates on the floor plane. Loads correspondences from JSON. |
| `rules.py` | deterministic rule engine: possession = nearest player within a distance threshold; shot_attempt = abrupt upward ball motion between consecutive frames while someone has possession. |
| `event_log.py` | append-only ordered container. Can export to plain list or a Pandas DataFrame. |

**Input:** the per-frame tracks from Stage 1.

**Output:**
```python
[
    {"frame_id": 10, "event": "possession",   "player": 1},
    {"frame_id": 24, "event": "shot_attempt", "player": 1},
    {"frame_id": 37, "event": "possession",   "player": 2},
    ...
]
```

**Thresholds and units.** The rule engine keeps thresholds in two flavours so the same engine works with or without a calibrated homography:

| Setting | Applies when | Default |
|---|---|---|
| `court_possession_dist` | tracks carry `court_xy` (homography configured) | 6.0 (court feet) |
| `pixel_possession_dist` | fall-back to bbox-center distances in pixel space | 110 px at 360p, auto-scaled by `frame_height / 360` |

**Why the shot-attempt rule stays in pixel space.** A floor-plane homography flattens the ball's arc to a single point on the court, destroying the vertical cue we depend on. The upward-motion heuristic is intentionally expressed in image-vertical pixels.

**What is *not* implemented and why.** `shot_made` would require a tracked hoop, which the COCO-trained YOLOv8 cannot produce. The code path was removed rather than left as dead text in the rule engine.

---

