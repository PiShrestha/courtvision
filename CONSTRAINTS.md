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

