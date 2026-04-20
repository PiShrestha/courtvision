# Demo Pipeline Architecture

An isolated, uncommitted experimental pipeline under `experiments/demo/` that
targets the three concrete failure modes of the main repo:

1. **Player over-detection** on moving-camera clips (2× to 2.4× phantom tracks).
2. **ID fragmentation** (38–48 distinct track IDs in a 90 s 1v1 window — ground
   truth: 2).
3. **No `shot_made` events** because COCO has no `basketball hoop` class.

The demo keeps the main repo's three-stage split (perception → symbolic →
narrative) but replaces stages with components designed to address the above.

---

## Data flow

```
┌──────────┐   frame   ┌───────────────┐   top-2   ┌────────────┐   events  ┌────────┐
│  VIDEO   │ ────────▶ │   YOLOv8 +    │ ────────▶ │   DuoTracker   │ ────────▶ │ VIZ +  │
│  (mp4/   │           │   ByteTrack   │  players  │  (2 slots)     │           │ EVENTS │
│   mov)   │           │   (Ultralytics)│  + ball   │  + HSV id      │           │ JSON   │
└──────────┘           └───────────────┘           └────────────┘           └────────┘
                                                        │
                                                  hoop_config.json
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │ ShotDetector │
                                                │  attempt     │
                                                │  made        │
                                                └──────────────┘
```

Each stage is independently importable and operates on plain Python
dicts/lists. Outputs are serialisable (JSON / JSONL) for downstream analysis.

---

## Components

### `hoop.py` — hoop localisation
Two modes:
- **Manual** (preferred): a JSON file with `{center: [x,y], radius: int}`.
- **Auto**: `detect_hoop(video_path, ...)` runs Hough circles on sampled frames
  restricted to the upper 55 % of the frame, then clusters candidate centers
  and returns the median of the largest cluster.

Failure modes observed:
- Hough alone picks up logo circles in frame corners. We switched to a
  **color-persistence + connected-components** approach during development
  (see `RUN_LOG.md`) which worked on `1v1-mk.mov` but failed on `1v1-ddg.mp4`
  and `1v1-jason.mp4` (rim not orange or too thin).
- Default mitigation: a placeholder config `{960, 200, 40}` with a clear
  `note` field telling the reviewer to verify the sample JPG and correct.

### `duo_tracker.py` — exactly two persistent player tracks
State: at most two `PlayerSlot`s, each with:
- `bbox` (latest), `hist` (HSV torso histogram, EMA-blended), `velocity`,
  `frames_alive`, `frames_missed`.

Per-frame loop:
1. Filter player detections by a **minimum bbox size** (`12 %` of frame height,
   `3 %` of width) — kills most corner-phantom boxes that the main repo's
   raw `Tracker` class produced.
2. Take the top 4 survivors by detection confidence.
3. Hungarian match existing slots to these detections using a cost:
   ```
   cost = iou_w · (1 - iou) + color_w · (1 - hist_similarity) + motion_w · distance
   ```
   Matches above a cost cap are rejected (phantom stealing an identity is
   worse than temporarily missing a frame).
4. Unmatched slots drift with their last velocity for up to `max_missed`
   frames, then get dropped so the next detection can re-seed the slot.
5. Bootstrap the two slots from the first frames that have size-qualifying
   detections.

Trade-offs:
- Appearance histogram is HSV of the torso region only (middle 60 % of bbox),
  which is more invariant to pose than the full bbox.
- No Kalman filter; a linear velocity prediction is sufficient for short gaps
  and keeps the implementation transparent.

### `shot_detector.py` — `shot_attempt` + `shot_made`
Two stateful rules driven by ball history + hoop position:

- **shot_attempt**: average `dy` across the last ~6 ball positions crosses a
  negative threshold (`upward_velocity_trigger`) while a player has possession
  (nearest player within `possession_dist_px`). Sticky: one attempt per flight.
- **shot_made**: within `made_window_frames` of an attempt, the ball's center
  enters the padded hoop disc **and** the most recent velocity is downward.
  Consumes the in-flight attempt.

Every threshold is exposed as a constructor arg so the matrix sweep can vary
them independently.

### `viz.py` — persistent-color annotation
- Player colors fixed by slot (P1=orange, P2=green) and never change, because
  DuoTracker guarantees two stable slot IDs.
- Ball drawn as an amber circle; hoop as a magenta circle.
- Event banner across the top holds the last ~45 frames of events so you can
  see a `shot_attempt → shot_made` sequence at a glance.
- Frame header shows `frame_id` and `mm:ss`.

### `run_demo.py` — orchestration + logging
One entry point. Writes four artefacts per run under `experiments/demo/outputs/`:

| File | Contents |
|---|---|
| `*.mp4` | annotated video (optional — sampled for 1/6 of matrix runs) |
| `*.tracks.jsonl` | per-frame tracks, one JSON object per line |
| `*.events.json` | complete event list |
| `*.meta.json` | run parameters + timings + hoop used + event counts |

All inputs (video, window, all detector/shot hyperparameters, hoop config path)
are captured in `meta.json`, so a single file is enough to reproduce a run.

---

## Experiment matrix

Generated by `build_matrix.py`. Axes:

**Detector (full cross product = 12 configs):**
| Axis | Values |
|---|---|
| model | yolov8m / yolov8l / yolov8x |
| imgsz | 640 / 1280 |
| confidence | 0.25 / 0.35 |

**Shot detector (3 presets):**
| Preset | upward_trigger | made_window | hoop_pad |
|---|---|---|---|
| loose | −8.0 | 60 | 20 |
| center | −12.0 | 45 | 14 |
| strict | −16.0 | 30 | 8 |

**Video × window:** 3 videos × 3 randomly-sampled 60–90 s windows from the
middle 60 % of each (seed=7).

**Total:** 3 × 3 × 12 × 3 = **324 runs.**

Submitted as a single Slurm job array with throttle `%8` (8 concurrent GPUs
maximum) and an aggregation job that fires `afterany` the array finishes.

---

## Aggregation

`aggregate_matrix.py` parses every `*.meta.json` + `*.events.json` into a
single long-format DataFrame and emits:

- `results_raw.csv` — one row per run, all parameters + all metrics
- `summary.md` — event totals by detector config, shot sensitivity tables,
  per-video totals, runtime heatmap
- `chart_events_per_detector.png` — bars of possession / attempt / made by
  detector config
- `chart_shot_sensitivity.png` — line charts of attempt/made vs each shot
  threshold
- `chart_per_video.png` — per-video event totals
- `chart_runtime_heatmap.png` — mean perception seconds per (model × imgsz)
- `chart_attempts_distribution.png` — histogram of shot_attempt counts by model

The aggregator is also written to run standalone from the shell, so you can
spot-check results at any intermediate point while the array is still in
flight.
