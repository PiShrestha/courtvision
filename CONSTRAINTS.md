# Constraints, tradeoffs, and honest limitations

This document is the place where CourtVision's scope is described without marketing. Everything here is grounded in the actual sweep results under `outputs/` (all numbers below come from real runs of [`1v1-mk.mov`](1v1-mk.mov) on UVA's A6000 nodes).

---

## What currently works

| Capability | Evidence |
|---|---|
| End-to-end pipeline runs on real handheld 1v1 footage | 90 s of 1080p processed in ~46 s wall time (yolov8m, A6000). |
| Real Ultralytics ByteTrack in place (not our earlier greedy IoU matcher) | Persistent IDs, Kalman prediction, short-occlusion recovery. |
| Threshold auto-scaling | Same rule engine works at 360p / 720p / 1080p / 4K without editing code. |
| Deterministic symbolic stage | Possession and shot_attempt events have auditable triggers in `logic/rules.py`. |
| Parameter sweep as infrastructure | 6-config sweep submits via one bash script; results named by config so they never collide. |
| Larger model + higher imgsz helps, as expected | yolov8x @ imgsz=1280 cut player over-detection from ~2.40× → ~1.96× and lifted ball recall from ~33 % → ~46 % on the same 90 s window. |

---

## What does not work (and why)

### 1. COCO has no `basketball hoop` class

YOLOv8 ships trained on COCO, which contains `person` and `sports ball` but no `basketball hoop` or `rim`. Every shot-made rule needs to reason about ball position relative to the hoop. Until we either (a) retrain on a basketball-labeled dataset (e.g. from Roboflow Universe) or (b) swap in a community checkpoint with a hoop class, `shot_made` cannot fire. We deleted the dead code rather than leaving it as a false affordance.

### 2. Player over-detection dominates error on moving-camera clips

Ground truth on a 1v1 frame is 2 players. Even the best detector we tested (yolov8x, imgsz=1280) produces on average **1.96 players per ground-truth player** across a 90 s moving-camera window — i.e., phantom tracks on spectators, motion-blurred half-bodies, and partial occlusions. ByteTrack cannot fix this; it is a *detection* problem.

The same sweep showed:

| Config | Player det / 90 s | Ball det / 90 s | Distinct track IDs |
|---|---:|---:|---:|
| yolov8m / conf=0.35 / 640 / bytetrack | 12,950 | 892 | 38 |
| yolov8m / conf=0.45 / 640 / bytetrack | 12,640 | 842 | 36 |
| yolov8m / conf=0.25 / 1280 / bytetrack | 12,801 | 1,066 | 44 |
| yolov8l / conf=0.35 / 640 / bytetrack | 12,502 | 1,023 | 48 |
| yolov8x / conf=0.35 / 1280 / bytetrack | **10,598** | **1,242** | 42 |
| yolov8m / conf=0.35 / 640 / botsort | 12,952 | 964 | 40 |

**Reading:** ground truth is ~5,400 player detections (2 × 2,697 frames). Actual: **2.0–2.4× too many**. 38–48 distinct track IDs in a single 90 s 1v1 window (ground truth: 2).

### 3. Shot-attempt heuristic is uncalibrated

The rule fires when the ball moves ≥12 pixels upward between consecutive frames while a player has possession. On 90 s of gameplay we see 21–29 `shot_attempt` events, when a human scout counts 2–5. Calibrating this properly requires either (a) a two-stage detector (vertical-dominance + release-and-fall trajectory check over N frames) or (b) a per-frame hoop class to anchor "toward the basket".

### 4. Static homography is unusable on moving-camera footage

The `Homography` class works and passes unit tests. It requires ≥4 fixed pixel↔court correspondences. The instant the camera pans, zooms, or the operator reframes, every correspondence is wrong and the projection is meaningless. Our dataset (TNC) is ~59 % moving-camera. The homography code path is ready for per-frame court-keypoint detection, which is the next real piece of work.

### 5. Ball recall drops over time in the same clip

Running the same config on six disjoint 90 s windows of `1v1-mk.mov`:

| Window | Ball detections |
|---|---:|
| 0–90 s | 1,242 |
| 180–270 s | 1,441 |
| 360–450 s | 1,273 |
| 540–630 s | 817 |
| 720–810 s | 682 |
| 870–960 s | 467 |

Same model, same input, same tracker — **3× drop from start to end**. The camera behavior changes (zoom, shakier handholding late game), and the detector degrades with it. This is a dataset / capture-protocol issue, not a pipeline bug, but it means any aggregate metric over a full video hides major variation across its segments.

---

## Design tradeoffs (what we chose and what we gave up)
