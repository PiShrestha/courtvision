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

### Modular neuro-symbolic vs. end-to-end VLM

**Chosen:** explicit three-stage split with plain-Python data contracts.
**Given up:** no learned representation of "a basketball possession" beyond hard-coded rules; all error modes that humans can identify, rule tweaks can address; none that are subtle enough to need learning.
**Won:** every failure is traceable to a specific stage and often a specific rule. The narrative stage cannot hallucinate an event order because it does not see raw frames.

### COCO-pretrained YOLO vs. custom basketball training

**Chosen:** generic Ultralytics COCO checkpoint; swap via `--model` flag.
**Given up:** hoop class, improved recall on small/blurry basketballs, and domain-specific prior on body pose during a shot.
**Won:** zero training time; immediate out-of-the-box detection; fair baseline before claiming "custom model improves X by Y".

### Single static homography vs. per-frame court keypoints

**Chosen:** single JSON of correspondences, loaded once.
**Given up:** ability to use homography on any moving-camera clip (most of the dataset).
**Won:** trivially simple calibration workflow for a tripod setup; no need for a keypoint-detection model; the module is complete enough to plug into when per-frame calibration lands.

### Ultralytics ByteTrack vs. hand-rolled tracker

**Chosen:** Ultralytics' real ByteTrack via `model.track(persist=True)`.
**Given up:** ability to cleanly separate `Detector` and `Tracker` classes (Ultralytics ties them together).
**Won:** Kalman prediction, short-occlusion recovery, and a real implementation instead of our previous 40-line greedy IoU matcher.

### Gemini API vs. local multimodal model

**Chosen:** `google-generativeai` with a local deterministic fallback.
**Given up:** offline inference; no dependency on external API.
**Won:** near-zero setup for narrative generation; we can swap in Llama 3.2-Vision at the `NarrativeGenerator.generate` boundary without touching the rest of the system.

---

## Known operational constraints

| Constraint | Detail |
|---|---|
| Python ≥ 3.9 required | Ultralytics + torch 2.x drop Python 3.6 / 3.7. The repo targets Python 3.11 (tested). |
| GPU strongly recommended | On A6000: ~60 fps for yolov8m @ 640. On CPU (login node): 3–8 fps — a 15 min clip takes hours. |
| `--start` seeks to keyframe | `cv2.VideoCapture.set(POS_FRAMES, N)` snaps to the nearest preceding keyframe, so the first yielded frame can land a few frames before the requested start time. Acceptable for analysis; not for exact-timestamp alignment. |
| Stride interacts with shot rule | `dy ≤ -12 px` fires per kept frame. At `stride=5` the ball moves ~5× further between samples, so the shot rule triggers more easily. Recalibrate the threshold when raising stride. |
| HEVC `.mov` files may fail to decode | patent-encumbered codec; workaround is a one-time `ffmpeg -c:v libx264` transcode to mp4. |
| Gemini deprecation warning | `google-generativeai` has been put into maintenance mode. Migration to `google-genai` is a follow-up. |

---

## Future work, ranked by expected impact

1. **Custom basketball YOLO checkpoint** — unlocks `hoop`, enables `shot_made`, and cuts spectator over-detection because the model is trained on basketball data. Roboflow Universe datasets are a reasonable starting point.
2. **Per-frame court keypoint detection** — enables a moving homography, which lets `RuleEngine` work in court-space on moving-camera footage and opens the door to court-polygon-based detection filtering (reject any player outside the court).
3. **Game-mode prior (`--game-mode 1v1 | 2v2 | 3v3 | 5v5`)** — cap the rule engine's player input at the top-N most confident tracks per frame, where N is the known number of players. Cheap; deterministic; directly attacks the dominant failure mode.
4. **Calibrated shot-attempt heuristic** — replace `dy ≤ -12 px` with a two-stage trigger: vertical-dominance + release-and-fall trajectory check over N frames, ideally anchored by a `hoop` track.
5. **SAM 2 + DINOv2 for re-identification** — occlusion-robust pixel-level tracking; per-player embedding for identity continuity across camera-angle changes.
6. **Llama 3.2-Vision narrative upgrade** — multimodal, local, conditioned on event log + keyframes.
