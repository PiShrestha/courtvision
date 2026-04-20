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

## Stage 3 — Narrative (`narrative/`)

| File | Responsibility |
|---|---|
| `generator.py` | either calls Gemini (`google-generativeai`) with a text prompt built from the event log, or falls back to a deterministic Python summariser. |

**Input:** the event log.

**Output:** a multi-line scouting summary (free-form text).

**Why the VLM never sees raw frames in this design.** The SportR and xVARs benchmarks both show that end-to-end video-to-text models hallucinate temporal ordering — they describe actions out of sequence because they have no verified spatial reference. By restricting the narrative stage to the pre-validated event stream, we remove that entire failure mode. Llama 3.2-Vision is the planned upgrade; it is a drop-in for `NarrativeGenerator.generate`.

---

## Stage contracts (what gets passed between modules)

| From → To | Data shape | Transport |
|---|---|---|
| Perception → Logic | `list[dict]` (`frame_id`, `frame_height`, `frame_width`, `tracks`) | in-memory Python list |
| Logic → Narrative | `list[dict]` (`frame_id`, `event`, `player`) | in-memory Python list |
| Narrative → Report | `str` | in-memory string |

Everything is plain Python — no custom serialisers, no ORM, no message bus. This is deliberate: the pipeline is meant to be trivially debuggable with `print()` and `json.dumps()`.

---

## Configuration and entry points

- **`config.yaml`** — single source of truth for every tunable knob. Loaded by `config.py`.
- **`main.py`** — CLI entry point. Flags override YAML values; missing flags fall back to the YAML defaults.
- **`scripts/run_courtvision.sh`** — single Slurm / bash runner. Environment variables override CLI flags in turn. The runner encodes the perception config into the output filename so a parameter sweep produces distinguishable artifacts.
- **`scripts/sweep_90s.sh` + `scripts/compare_sweep.sh`** — submit a 6-config sweep, then later summarise the results in one table.

The layering — **YAML → CLI → env** — means any particular field can be set at whichever level is most convenient without editing code.

---

## Data flow diagram (expanded)

```
                                                                     (optional .env)
                                                                           │
                                                                     GEMINI_API_KEY
                                                                           │
                                                                           ▼
   video.mov                                                     ┌──────────────────┐
       │                                                         │   Gemini 1.5     │
       ▼                                                         │  (text prompt)   │
┌────────────┐   per-frame    ┌────────────┐   event list    ┌──────────────────────┐
│ Perception │ ─────────────▶ │ RuleEngine │ ──────────────▶ │ NarrativeGenerator   │
│  pipeline  │   (tracks)     │            │                 │  - gemini path       │
│            │                │            │                 │  - local fallback    │
└────────────┘                └────────────┘                 └──────────────────────┘
       ▲                             ▲                                │
       │                             │                                ▼
  cv2.VideoCapture              Homography?                 ┌──────────────────┐
  + Ultralytics                 (optional)                  │  report.py       │
  ByteTrack                                                 │  - stats table   │
                                                            │  - save to .txt  │
                                                            └──────────────────┘
```

---

## Extension points (not implemented yet; see [CONSTRAINTS.md](CONSTRAINTS.md))

1. **Per-frame court keypoint detection** — would make the homography path usable on moving-camera clips. The current `Homography` class expects one fixed calibration for an entire video.
2. **Custom basketball YOLO checkpoint** — unlocks a `hoop` class, which unblocks shot_made and enables ROI-based filtering to kill spectator detections.
3. **Game-mode prior** (`--game-mode 1v1 | 2v2 | ...`) — cap the rule engine's player input at the top-N most confident tracks per frame.
4. **SAM 2 + DINOv2** — the proposal's final-stage upgrade for occlusion-robust tracking and re-identification.
5. **Llama 3.2-Vision** — replaces the Gemini API call with a local multimodal model conditioned on event log + keyframes.
