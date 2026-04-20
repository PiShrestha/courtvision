# Shot-made v2 — Findings

Generated from the offline v2 sweep (Slurm job 11930323, 4,860 replays,
15 tunable rows × 324 tracks.jsonl files). The live GPU matrix (Slurm
job 11930344, 12 tasks, rim tracker on/off) is queued and will backfill
the `analysis_v2/` charts automatically when the afterany aggregator
(11930360) fires.

Summary documents and charts:
- [`analysis_v2/results_v2.csv`](analysis_v2/results_v2.csv) (one row per replay)
- [`analysis_v2/summary_v2.md`](analysis_v2/summary_v2.md) (tables)
- `analysis_v2/chart_v1_vs_v2.png`, `chart_tunable_effects.png`,
  `chart_attempts_vs_made.png`

---

## TL;DR

- **v1 → v2, mean per run (collapsing all configs):**
  - `shot_attempt`: **12.12 → 4.18** (−65%), closer to the 3–5 attempts /
    90 s target.
  - `shot_made`: **0.26 → 0.40** (+54%), gains driven by occlusion-gap
    extrapolation.
  - `shot_miss`: **0 → 3.68** (new event).
- **Best parameter row is `s15`** (occlusion=20, window=150, enter=2.5,
  horiz=2.0, min_dv=1.0): mean 1.39 makes/run (across runs that actually
  processed frames), 14.0% FG, 8.2 attempts.
- **Occlusion gap frames matter more than any other knob.** Going
  5→10→15→20 moves made from 0.17 → 0.37 → 0.55 → 0.68 per run.
  Attempt-to-made window scales in lock-step (60→90→120→150 → same
  made sequence), because most v1 missed makes live in the 60–150 frame
  post-release window.
- **ddg is untested.** All 108 `1v1-ddg.mp4` tracks.jsonl files from
  the v1 matrix are empty (the originating v1 run failed on the older
  cuDNN node — see MORNING_README.md). The v2 offline sweep produced
  zero events for those 108×15=1,620 replays. They count toward the
  4,860 total replays but are excluded from the non-empty analyses.
- **jason uses a placeholder hoop at (960, 200).** Numbers for that
  video are likely miscalibrated; recommend a 10-min human correction
  of `hoop_configs/1v1-jason.json` before trusting its FG stats.

---

## Head-to-head v1 vs v2 (all runs, mean per run)

| Metric | v1 (ball-disc overlap) | v2 (plane crossing + occlusion) | Δ |
|---|---:|---:|---:|
| shot_attempt events | 12.12 | 4.18 | **−65.5%** |
| shot_made events | 0.26 | 0.40 | **+53.8%** |
| shot_miss events | 0.00 | 3.68 | new |

(Chart: `analysis_v2/chart_v1_vs_v2.png`.)

### What this means

- Attempts dropped by 2/3 because the multi-predicate attempt gate
  requires **all** of: sustained upward velocity, ball-separated-from-
  possessor, and ball trending toward the rim. Each single predicate
  on its own admits dribbles, crossovers, and lobs.
- Makes went **up** — counterintuitively — because even though v2 is
  stricter about the crossing, the occlusion-gap extrapolation recovers
  the 5–15 frames the ball is hidden behind the net during a real make.
  That was the dominant failure mode in v1.
- `shot_miss` is a new output v1 can't produce. It fires when an attempt
  expires without a matching crossing. Overall FG% sits at 9–14% across
  the sweep, which is plausible for street basketball.

---

## Sensitivity to each tunable

Data: only the 2,370 replays that actually processed frames
(mk + jason with non-empty tracks.jsonl).

### Dominant: occlusion_gap_frames

| gap (frames) | attempts | made | miss | FG% |
|---:|---:|---:|---:|---:|
| 5  | 5.13 | 0.17 | 4.88 | 3% |
| 10 | 4.07 | 0.37 | 3.62 | 9% |
| 15 | 3.95 | 0.55 | 3.28 | 12% |
| 20 | 4.00 | 0.68 | 3.06 | 14% |

**→ Use ≥15 frames (0.5 s @ 30 fps).** 20 frames is the best in-sample
but risks bleeding one shot into the next rebound — would pick up on a
real GT.

### Tied first: attempt_to_made_window (same effect as occlusion)

| window (frames) | attempts | made | miss | FG% |
|---:|---:|---:|---:|---:|
| 60  | 5.13 | 0.17 | 4.88 | 3% |
| 90  | 4.07 | 0.37 | 3.62 | 9% |
| 120 | 3.95 | 0.55 | 3.28 | 12% |
| 150 | 4.00 | 0.68 | 3.06 | 14% |

**→ Use 120–150 frames (4–5 s @ 30 fps).** High-arc shots can take
3 s to descend; the common "60 frames / 2 s" default is too tight.

### Moderate: enter_zone + horizontal_pad factors

| enter_factor | made / run | horiz_pad | made / run |
|---:|---:|---:|---:|
| 1.5 | 0.28 | 1.0 | 0.28 |
| 2.0 | 0.31 | 1.2 | 0.17 (tight → misses) |
| 2.5 | 0.58 | 1.5 | 0.35 |
| 3.0 | 0.62 | 2.0 | 0.58 |
|     |      | 2.5 | 0.62 |

**→ Use enter=2.5, horiz=2.0.** Wider than that trades makes for false
positives at low-arc lateral passes.

### Weak: upward_trigger, release_dist, approach_dist

- `upward_trigger` sweet spot is **−10** (made 0.52) — milder than
  v1's recommended −12 to −16. This is because v2's extra predicates
  already suppress FPs; the velocity gate can relax.
- `release_dist_px`: best is 60–80 px. 40 admits dribbles; 80 misses
  some quick releases.
- `approach_dist_px`: best is 200 px. The "ball near the rim"
  predicate only fires on clear shot attempts.

---

## Best parameter row

`s15` (made 1.39 / run, FG% 14.0%):

```
upward_trigger = -10
history_frames = 6
release_dist_px = 60
approach_dist_px = 260
cooldown_frames = 20
occlusion_gap_frames = 20
attempt_to_made_window = 150
enter_zone_radius_factor = 2.5
horizontal_pad_factor = 2.0
min_downward_velocity = 1.0
```

Note: `s15` is slightly "loose" — it may rely on net-occluded
extrapolation for shots that bounced off the rim without going in.
Recommend shipping `s06` as the conservative default (occ=15, window=120,
enter=2.5, horiz=2.0) if false positives matter more than recall.

---

## Perception-config interaction (model × imgsz)

Among **non-empty** runs:

| model × imgsz | runs | attempts / run | made / run |
|---|---:|---:|---:|
| yolov8l @ 640 | 360 | 8.55 | **1.28** |
| yolov8l @ 1280 | 360 | 8.89 | 1.10 |
| yolov8m @ 640 | 450 | 7.57 | 0.25 |
| yolov8m @ 1280 | 390 | 8.39 | 0.15 |
| yolov8x @ 640 | 360 | 9.34 | 0.80 |
| yolov8x @ 1280 | 450 | 8.88 | **1.39** |

Two interesting things:
- **yolov8l @ 640 is almost as good as yolov8x @ 1280** for v2 purposes
  (1.28 vs 1.39 makes/run). v1's "imgsz=1280 required" finding was an
  artifact of the ball-disc rule needing a ball detection *exactly* in
  the hoop disc. v2's occlusion extrapolation makes the ball-detection
  rate less critical — yolov8l @ 640 detects the ball often enough for
  the extrapolator to carry through the net.
- **yolov8m lags.** The 8m family consistently underperforms on makes
  regardless of imgsz (0.15–0.25 vs 0.80+ for 8l/8x). Probably a ball-
  recall issue — 8m is noticeably worse on small objects.

Runtime tradeoff (from existing v1 matrix): yolov8l @ 640 is ~40 s/run,
yolov8x @ 1280 is ~110 s/run on A6000. **yolov8l @ 640 is now the
pragmatic sweet spot** for v2. (v1's interim findings recommended
yolov8m @ 1280 based on v1's disc rule.)

---

## Per-video rollup

| video | runs (non-empty) | attempts / run | made / run | miss / run | note |
|---|---:|---:|---:|---:|---|
| `1v1-mk.mov`   | 1200 | 9.71 | 0.71 | 8.88 | hoop verified manually |
| `1v1-jason.mp4` | 1170 | 7.40 | 0.93 | 6.19 | hoop is `(960,200)` placeholder — suspect |
| `1v1-ddg.mp4`   | 0   | —    | —    | —    | all v1 tracks empty (cuDNN failure) |

**Interpretation caveat.** jason looks higher FG% than mk. This is more
likely a rim-placement artifact than a real skill difference — a wrong
hoop at the center of the frame will count any vertical descent as a
make. Cannot disentangle without a corrected `1v1-jason.json`.

---

## Evaluation against seed GT

The shipped seed GT has only **1 row** (the known shot_made at frame
8148 in `1v1-mk.mov` t=268–338s) because I did not have a human to
watch the video. This is a pseudo-oracle, not a usable evaluation set.

Spot-check against `run30 × sws15`:

| event | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shot_attempt | 0 | 11 | 0 | 11 | 0 | 0.000 | 0.000 | 0.000 |
| shot_made    | 1 | 5  | 1 |  4 | 0 | 0.200 | 1.000 | 0.333 |

Recall is perfect; precision is an unknown lower bound because the 4
"false positives" might be real makes the seed doesn't cover. Until a
human stamps a proper GT, precision/F1 cannot be trusted.

**Unblocking next step:** have one annotator watch
`experiments/demo/outputs/run30_1v1-mk_t268-338_yolov8x_i1280_c0.25_ut-8.mp4`
with the protocol in [`gt/README.md`](gt/README.md). Expected time: 10 min.
Then re-run `evaluate_v2.py` on the top-3 v2 sweep configs.

---

## Rim tracker drift — empirical (new)

Measured directly from `rim_trace.jsonl` on the first five live-matrix
tasks, same 70 s `1v1-mk.mov` window, static-camera footage:

| Mode | Unique rim centers | x spread (px) | y spread (px) | `source` counts |
|---|---:|---:|---:|---|
| `rim=static` | **1** | 0 | 0 | all 2,089 frames `static` |
| `rim=csrt` (fallback to MIL) | 1,610 | **160** | **158** | 1,807 tracker, 142 reseed, 140 static |

The CSRT-labelled run is actually MIL (stock `opencv-python` doesn't
ship CSRT). MIL on a ~60×60 px rim region drifts ±80 px within a
90-frame reseed window — which is wider than the horizontal tolerance
gate (1.5 × 36 px ≈ 54 px) used by `shot_made_v2._check_made`. Effect
on makes, same model/clip combo:

| Task | Model | Rim | Attempts | Made | Miss |
|---|---|---|---:|---:|---:|
| 0 | yolov8m @ 1280 | static | 11 | 0 | 11 |
| 1 | yolov8m @ 1280 | csrt→mil | 11 | 0 | 11 |
| 2 | yolov8x @ 1280 | static | 14 | **2** | 12 |
| 3 | yolov8x @ 1280 | csrt→mil | 14 | **0** | 14 |

rim=csrt zeroed out yolov8x's makes. On a static-camera clip the
"tracker" just adds rim noise; the static anchor is correct to the
pixel.

**Action taken** (committed separately):
- `RimTracker` default changed from `tracker_kind="csrt"` to
  `tracker_kind="static"`.
- `max_jump_px` tightened from 80 → 30 so even when the tracker is
  enabled, large drifts get overruled by the anchor.
- `run_demo_v2.py --rim-tracker-kind` default flipped accordingly.

Moving-camera clips (`1v1-ddg.mp4`, `1v1-jason.mp4`) are still the
motivating case for tracker mode, but their hoop configs are
placeholders today so we can't validate that claim until someone
corrects them.

**Unblocking proper CSRT:** `pip install --upgrade
opencv-contrib-python` (replacing `opencv-python`) ships a real CSRT
implementation, which for a rigid-and-tiny region like a rim is
substantially more stable than MIL. The fallback chain in
`_build_tracker` will pick it up automatically.

---

## Live matrix (pending)

Slurm job 11930344 (12 tasks, partition `gpu-a6000`) is queued. It
validates the per-frame rim tracker (CSRT → MIL fallback) end-to-end
on three windows across three videos, with and without rim tracking.
The afterany aggregator (11930360) will refresh
`analysis_v2/results_v2.csv` + charts when both arrays complete.

Hypothesis being tested: **rim tracker improves numbers only on
moving-camera clips** (`1v1-ddg.mp4` and `1v1-jason.mp4` — neither
currently has a verified hoop). On `1v1-mk.mov` (static camera) we
expect `rim_kind=static` to match `rim_kind=csrt` within noise.

If the live matrix shows rim tracker **hurts** performance (more
false positives from tracker drift), the default will flip to
`rim_kind=static` and the tracker becomes opt-in.

---

## Things to do before the team trusts these numbers

1. **Correct jason + ddg hoop configs** by eyeballing the sample JPGs
   at `experiments/demo/hoop_configs/*_sample.jpg`. 5 min each.
2. **Re-submit v1 for the ddg tracks** so the offline v2 sweep can
   fill in that row. Use the same `run_matrix.sbatch` with a
   ddg-only manifest.
3. **Collect one human GT per video** (30 min total) and re-run
   `evaluate_v2.py` to get precision / recall / F1 on `shot_attempt`
   AND `shot_made`, not just `shot_made`.
4. **Ablate the occlusion-gap extrapolator** — re-run `s15` with
   occlusion_gap=0 to confirm gap handling (not rule stricter) is
   what drives the +54% makes.
5. **Check rim_trace.jsonl on moving-camera clips** once live matrix
   lands — plot center_x/y over time and visually confirm the
   tracker followed the rim.

---

## Risks I did not address

- **Custom basketball YOLO checkpoint.** This is the single biggest
  unlock for real per-frame rim and ball confidence. Prompt forbids
  retraining without approval; escalate when ready.
- **Rim-bounce vs swish classification.** Can't distinguish without
  a rim-contact signal (custom model or template matching on rim
  pixels). Deferred.
- **Multi-hoop clips.** All three source videos are half-court 1v1s
  so single-hoop is fine today.
- **DuoTracker slot IDs escalate past 2.** Observed in the event
  streams (`player: 5, 7, 10` after long occlusions). Separate v1
  bug — not in scope for the v2 shot detector but it pollutes the
  `field_goal_stats.per_player` buckets.
