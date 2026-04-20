# Interim Findings — Demo Matrix (90 of 324 runs)

Generated from the first 90 tasks that completed before the cuDNN failure
triggered on nodes `udc-an33-37/38`. Retry of the 234 failed tasks is
in flight on `gpu-a6000`; this document will be refreshed once those land.

**Important caveat:** these 90 runs were missing possession events (bug in
`shot_detector.py` fixed after the first submission). `possessions=0` across
the table is a pipeline artefact, not a real measurement. The retry run will
produce proper possession counts.

---

## Coverage

| Video | runs so far | expected |
|---|---:|---:|
| 1v1-mk.mov | 12 | 108 |
| 1v1-jason.mp4 | 78 | 108 |
| 1v1-ddg.mp4 | 0 | 108 |

`1v1-ddg.mp4` mostly hit the failed nodes; all retries will cover it.
`1v1-friga.mp4` excluded — corrupt transfer (moov atom missing).

---

## Headline findings from partial data

### 1. `shot_attempt` is sensitive to `upward_trigger`, not to the detector
Across 30 runs each:
| upward_trigger | attempts/run (mean) | attempts/run (std) |
|---:|---:|---:|
| −8  | **13.67** | 3.46 |
| −12 | 10.27 | 3.67 |
| −16 |  8.37 | 3.27 |

A factor-of-2 change in attempts from `−8` to `−16`. Standard deviation is
roughly constant — the detector's upward-motion threshold shifts the mean
without changing the variance. **Recommendation: pick `-12` (center) as the
default; it's the knee of the curve.**

### 2. `shot_made` is dominated by the detector, not the threshold
Across 12 detector configs:
| config | made/run |
|---|---:|
| yolov8x + imgsz=1280 | **0.78** |
| yolov8m + imgsz=1280 | 0.22 |
| yolov8l + imgsz=1280 | 0.17 (conf=0.25) / 0.0 (conf=0.35) |
| any model + imgsz=640 | 0.00 – 0.17 |

At imgsz=640 the ball is too small to detect reliably as it passes through
the hoop, so `shot_made` almost never fires regardless of the model.
**imgsz=1280 is effectively required for the shot_made rule to work.**

### 3. Runtime scales with model × imgsz roughly as expected
| model | 640 | 1280 |
|---|---:|---:|
| yolov8m | 42 s | 76 s |
| yolov8l | 57 s | 126 s |
| yolov8x | 74 s | 165 s |

Per 60–90 s window, A6000 GPU, with full second-pass MP4 annotation included
for every sixth run. The kitchen-sink config (yolov8x + imgsz=1280) costs
~3× baseline; if you only care about `shot_made`, **yolov8m @ 1280 is the
pragmatic sweet spot** (76 s / run, ~0.22 made / run — a third of the kitchen
sink quality at half the cost).

### 4. The `hoop_pad × made_window` diagonal is coincidental
The partial table shows values only on the diagonal because the three shot
presets are linked (loose/center/strict all vary together); this is by
design but easy to misread. The full factorial would require 3³ = 27 shot
configs × 108 runs = large. The diagonal tells us relative ranking; we do
not have independent sensitivity per shot axis.

---

## Open questions

1. **Is `shot_made` overcounting or undercounting?** 0.78 mades per 90 s
   window on yolov8x @ 1280 is plausible — a typical 1v1 has ~1–3 makes
   per 90 s. Need manual verification on the annotated MP4s.
2. **Why did yolov8x @ 640 produce `shot_made=0`?** Possibly the ball
   detection is fine but the ball position never crosses the unpadded
   hoop radius at 640 (smaller input = less precise localisation).
3. **Duo tracker on 1v1-jason.mp4:** no ground truth yet; the annotated
   MP4s in that video will tell us if the two slots are being held stably.

---

## What changes when the retry lands

The 234 retry tasks will add:
- Complete `1v1-ddg.mp4` coverage (108 runs).
- Missing windows / configs on `1v1-mk.mov` and `1v1-jason.mp4`.
- Possession events (bug fix #1 in `shot_detector.py`).
- Second-try data for the tasks that failed the first time.

Once the retry aggregation runs, this file is replaced by `FINDINGS.md`
with the full picture.
