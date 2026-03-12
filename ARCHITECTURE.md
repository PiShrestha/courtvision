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
