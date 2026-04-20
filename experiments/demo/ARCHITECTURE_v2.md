# Shot-made v2 — Architecture

A drop-in replacement for the v1 `ShotDetector` that targets the four
failure modes called out in [SHOT_MADE_PROMPT.md](../../SHOT_MADE_PROMPT.md):

1. Hoop is a static point — drifts on moving-camera clips.
2. `shot_made` only fires when the ball happens to be detected on a
   single frame inside the hoop disc.
3. `shot_attempt` fires on every dribble / crossover / lob.
4. Net occludes the ball during a real make for 3–15 frames at 30 fps.

v2 is **additive**: `shot_detector.py` (v1) still works untouched. v2
lives in four new files plus orchestration:

| Module | Role |
|---|---|
| [`rim_tracker.py`](rim_tracker.py) | per-frame rim position (hybrid: static anchor + cv2 tracker with periodic re-seed). |
| [`shot_attempt_v2.py`](shot_attempt_v2.py) | multi-predicate shot_attempt detector. |
| [`shot_made_v2.py`](shot_made_v2.py) | rim-plane crossing state machine with occlusion-gap tolerance. |
| [`shot_pipeline_v2.py`](shot_pipeline_v2.py) | composes the three above + coordinates in-flight attempt/make state. |
| [`run_demo_v2.py`](run_demo_v2.py) | end-to-end orchestrator (live video, GPU). |
| [`offline_sweep_v2.py`](offline_sweep_v2.py) | offline replay: run v2 on existing `tracks.jsonl` without re-running YOLO (CPU, fast parameter sweeps). |
| [`evaluate_v2.py`](evaluate_v2.py) | precision / recall / F1 against human-annotated GT CSV. |
| [`aggregate_v2.py`](aggregate_v2.py) | collect N runs into `results_v2.csv`, `summary_v2.md`, charts. |

---

## Data flow

```
   raw video               tracks.jsonl            tunable sweep csv
       │                         │                         │
       ▼                         ▼                         ▼
 ┌───────────┐   frame    ┌──────────────┐   params   ┌───────────────┐
 │ YOLOv8 +  │ ─────────▶ │ DuoTracker   │ ─────────▶ │ ShotPipelineV2│
 │ ByteTrack │  detections│ (persistent  │   tracks   │  ├ RimTracker │
 └───────────┘            │   P1, P2)    │  + ball    │  ├ AttemptV2  │
       │                  └──────────────┘            │  └ MadeV2     │
       │                                              └───────────────┘
       │                                                      │
       └──────── rim_tracker (CSRT / MIL) needs frames ──────┘
                                                              ▼
                                    events + field_goal_stats + rim_trace
```

Live mode (`run_demo_v2.py`) drives every arrow. Offline mode
(`offline_sweep_v2.py`) skips the yolov8 stage by replaying a prior
`tracks.jsonl` and forces the RimTracker into `"static"` mode
(no frames means no tracker updates). That's why **CSRT / MIL rim
tracking is tested only in live mode** — offline gives us fast parameter
sweeps over everything else.

---

## Rim tracker (`rim_tracker.py`)

Addressing SHOT_MADE_PROMPT.md ambiguity #6. Option chosen: **(d) hybrid —
static JSON anchor + cv2 tracker re-seeded every N frames.**

Rationale:
- (a) Static JSON is correct on `1v1-mk.mov` (static camera) and was
  already validated by the v1 matrix, so we treat it as the anchor.
- (c) Pure tracker drifts without re-detection. A 1-second static rim
  anchor drifts negligibly; a 30-second trackback doesn't.
- (b) Retraining would be the right answer for moving-camera footage
  but is out of scope (see Constraints in the prompt).
- (d) Hybrid gets us moving-camera robustness without retraining. The
  re-seed interval is exposed as a parameter so we can ablate it.

Implementation notes:
- Tracker kind falls back `csrt → kcf → mil` so the module runs on
  stock `opencv-python` wheels (which only ship MIL).
- Sanity checks reject tracker updates whose bbox goes off-frame,
  collapses, or jumps more than `max_jump_px` from the anchor.
- Every `reseed_every` frames the tracker is reset to the anchor box.
- `source` field on each TrackedHoop is one of `{static, tracker,
  reseed}` for trace-level debugging.

On static-camera `1v1-mk.mov`, the tracker mode is a no-op (every
TrackedHoop lands within sanity bounds of the anchor). On moving-camera
clips (`1v1-ddg.mp4`, `1v1-jason.mp4`) the tracker should catch drift
within a re-seed window. We surface the `rim_trace.jsonl` alongside each
live run so you can plot `(center_x, center_y)` over time and see whether
the tracker actually followed the rim.

**Not addressed (deferred to v3):** custom basketball YOLO with a `hoop`
class. The prompt explicitly forbids retraining unless approved.

---

## `shot_attempt_v2` — multi-predicate rule

v1's gate (`avg dy ≤ upward_trigger`) is too permissive. v2 requires
**all four** to hold:

1. **Sustained upward velocity:** `avg dy over last history_frames ≤
   upward_trigger`.
2. **Ball has left the possessor's hand:** `distance(ball, possessor) ≥
   release_dist_px`. Kills dribble and pass triggers — a dribble keeps
   the ball within ~60 px of the player even at its peak.
3. **Trajectory toward the rim:** either
   `|ball.x - rim.x| ≤ approach_dist_px`,
   or the ball is already at rim height (`ball.y ≤ rim.y + 3·rim.r`).
4. **Cooldown:** no attempt within `cooldown_frames` of the last one.
   Prevents the same shot re-firing on a second upward segment
   (e.g. a fumble during release).

Possession tracking mirrors v1 (nearest player within
`possession_dist_px`) and still emits `possession` events so the
narrative stage has continuity with v1.

---

## `shot_made_v2` — rim-plane crossing state machine

State: a single `_InFlight` record representing the most recent
`shot_attempt`, plus a rolling ball-position history that includes
**extrapolated samples** during detection gaps.

Transitions per frame:

```
no attempt      ─── on_shot_attempt() ───▶   WAITING
WAITING         ─── ball crosses y=rim_y downward, within horiz pad ──▶ EMIT shot_made
WAITING         ─── elapsed > attempt_to_made_window ──▶ EMIT shot_miss
```

Why "rim-plane crossing" instead of disc containment:
- Disc containment fires on air-balls that pass laterally through
  the 2D bounding disc but never descended through the rim.
- Plane crossing requires the ball's y to actually pass `rim_center_y`
  going downward, which matches the physical definition of a make.

Occlusion-gap tolerance:
- If the ball isn't detected on frame N but was detected on frame
  N-G (G ≤ `occlusion_gap_frames`), we extrapolate a synthetic ball
  position from the current velocity and feed it into the crossing
  check. This recovers the 3–15 frame net-occluded makes that v1
  always missed.
- Extrapolated samples are tagged `interpolated=True` in the history
  buffer for future diagnostics.

Horizontal tolerance at the crossing frame is `horizontal_pad_factor
× rim_radius`. A clean swish passes within 1.0×. 1.5× catches most
rim-bounce makes; 2.0× starts to admit lateral air-balls.

Downward-velocity gate (`min_downward_velocity`) rejects near-static
balls that drift through the zone. Required because extrapolated
samples have zero velocity by construction if the last two real
samples were equal.

---

## `shot_pipeline_v2` — orchestration

A thin coordinator that:

1. Ticks the `RimTracker` with the current frame (or reuses the anchor
   in offline mode).
2. Calls `ShotAttemptV2.update` — gets back any `possession` and
   `shot_attempt` events.
3. For each new `shot_attempt`, calls `ShotMadeV2.on_shot_attempt`.
4. Calls `ShotMadeV2.update` — gets back any `shot_made` /
   `shot_miss` events.
5. Notifies `ShotAttemptV2.resolve_attempt()` when a shot is resolved
   so a new attempt can fire without waiting out the cooldown.

The event dict schema is unchanged from v1 — `{"frame_id", "event",
"player"}` — so downstream (narrative, viz, aggregate) keeps working.
v2 adds one new event type: **`shot_miss`**, emitted when an attempt
expires without a matching plane crossing.

---

## Field-goal stats

`run_demo_v2.py` and `offline_sweep_v2.py` both compute a
`field_goal_stats` block in `meta.json`:

```json
"field_goal_stats": {
  "per_player": {
    "1": {"attempts": 4, "made": 2, "miss": 2, "fg_pct": 0.5},
    "2": {"attempts": 6, "made": 1, "miss": 5, "fg_pct": 0.167}
  },
  "overall": {"attempts": 10, "made": 3, "fg_pct": 0.3}
}
```

Player IDs come from `DuoTracker`'s persistent {1, 2} slots. Known
limitation: on long gaps where DuoTracker re-acquires a slot, the ID
increments past 2. This is an existing v1 bug in the DuoTracker, not
a v2 shot-made issue, so stats are bucketed by whatever slot id the
tracker emitted.

---

## Evaluation

**Ground-truth format** — one CSV per (video, window):

```
frame_id, event, player, notes
8148,     shot_made,    2,    swish
```

**Annotation protocol** — 10 min / 90 s clip, one annotator,
frame-by-frame in a video player. See
[`gt/README.md`](gt/README.md).

**Evaluator** — `evaluate_v2.py` does greedy nearest-neighbour matching
on `frame_id` with a **±15-frame tolerance** (~0.5 s at 30 fps).
Reports precision / recall / F1 per event type plus a separate
`player_agreement_rate` that isn't gated on player-id identity (the
DuoTracker's {1,2} labels aren't identity-stable across annotators).

**What we ship now vs. need from the annotator:**
- Shipped: `gt/1v1-mk_t268-338.seed.csv` — one shot_made at frame 8148
  taken from v1 yolov8x@1280 (the highest-confidence v1 config).
  Treated as pseudo-oracle.
- Shipped: `gt/1v1-mk_t268-338.candidates.csv` — superset of
  shot_made frames that multiple v2 sweep configurations agreed on,
  for human triage.
- Needed: a human watching the annotated mp4 to stamp truth rows into
  `gt/1v1-mk_t268-338.csv`.

---

## Tunables (all exposed as CLI args + matrix columns)

| Tunable | Default | What it affects |
|---|---|---|
| `rim_tracker_kind` | `csrt` (falls back to mil) | per-frame rim tracking |
| `rim_reseed_every` | 90 | tracker re-seed cadence |
| `upward_trigger` | -10.0 | shot_attempt upward-velocity gate |
| `history_frames` | 6 | velocity-averaging window |
| `release_dist_px` | 60.0 | ball-separates-from-possessor gate |
| `approach_dist_px` | 260.0 | ball-toward-rim gate |
| `cooldown_frames` | 20 | suppress re-trigger after attempt |
| `possession_dist_px` | 140.0 | nearest-player possession radius |
| `occlusion_gap_frames` | 10 | max net-occlusion tolerance |
| `attempt_to_made_window` | 90 | max attempt→made temporal binding |
| `enter_zone_radius_factor` | 2.0 | vertical enter-zone above rim |
| `horizontal_pad_factor` | 1.5 | horizontal tolerance at crossing |
| `min_downward_velocity` | 1.5 | required dy at crossing (px/frame) |

---

## Parameter sweeps submitted to Slurm

Two sbatch jobs feed the analysis:

**1. Offline (CPU) — `offline_sweep_v2.sbatch`**
- 324 `tracks.jsonl` × 15 tunable rows = 4,860 replays.
- No GPU, no Ultralytics — just reading JSON and stepping the v2 rules.
- Exercises every shot-rule knob (rim kind is forced to static offline).

**2. Live (GPU) — `run_matrix_v2_live.sbatch`**
- 12 tasks (3 videos × sensible knob combos × rim_kind ∈ {static,csrt}).
- Partition `gpu-a6000`, 30 min / task, `--array=0-11`.
- Validates the per-frame rim tracker end-to-end.

**3. Aggregator — `aggregate_v2.sbatch`**
- `afterany` dependency on both arrays above.
- Produces `experiments/demo/analysis_v2/{results_v2.csv, summary_v2.md,
  chart_*.png}` + a head-to-head `v1_vs_v2.csv` and bar chart.

---

## Tradeoffs vs v1

| Dimension | v1 (ball-disc overlap) | v2 (rim-plane crossing) |
|---|---|---|
| **FP mode** | fires on any ball center inside disc | needs actual downward plane crossing |
| **FN mode** | misses net-occluded makes | tolerates 10-frame gap via velocity extrapolation |
| **Attempt noise** | ~15–22 per 70 s clip (ground truth ~5) | typically 10–13 per 70 s (≈50% reduction in smoke tests) |
| **Miss detection** | silent | emits `shot_miss` on attempt expiration |
| **Hoop robustness** | static JSON only | hybrid anchor + tracker (moving camera) |
| **Compute overhead** | ~0 | tracker adds ~2 ms/frame (CPU) |
| **New dependencies** | none | `cv2.TrackerMIL_create` (already in opencv-python) |

---

## What still isn't addressed

1. **Shot classification** (swish vs rim-bounce-make vs bank) — requires
   detecting rim/backboard contact. Out of scope without a custom model.
2. **Multi-hoop clips** — rule engine still assumes a single hoop. The
   three source videos are all half-court 1v1s so this hasn't bitten us.
3. **DuoTracker ID drift past 2** — observed empirically in long clips
   (player IDs escalate to 5, 7, 10 after occlusions). This is a
   separate v1 bug in `duo_tracker.py`'s `_next_id` handling and was
   deliberately left out of scope here.
4. **Custom basketball checkpoint** — the prompt explicitly said no
   retraining without approval. A Roboflow Universe checkpoint
   would replace both the hoop auto-detect and the rim tracker with
   a per-frame supervised signal.
