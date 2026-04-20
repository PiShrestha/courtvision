# Ground truth annotation — 90 s clip protocol

Total time budget per clip: **~10 minutes** for one annotator.

## Format

One CSV per video × window, named `<video-stem>_t<start>-<end>.csv`, with
the header:

```
frame_id,event,player,notes
```

- **frame_id**: absolute frame index in the original video (not
  window-relative). At 30 fps, `frame_id = round(t_seconds * 30)`.
- **event**: one of `shot_attempt`, `shot_made`, `shot_miss`. Optional but
  useful: `possession` (omit for speed).
- **player**: 1 or 2 (whichever side of the court from the camera's POV
  the shooter is on). If unclear, leave empty.
- **notes**: free text. Example: "swish", "bank shot", "air-ball".

## Protocol

1. Open the source video in a player that shows the frame counter (VLC:
   set frame-by-frame step, then note the timestamp × fps).
2. Scrub to the window start (e.g. t=268s). Watch at 0.5× speed.
3. For each visible shooting motion:
    - When the shooter's hand releases the ball → log one row
      `frame_id, shot_attempt, <player>`.
    - If the ball passes through the rim/net → log another row
      `frame_id, shot_made, <player>` at the moment it crosses the rim.
    - If it misses → log `frame_id, shot_miss, <player>` when the rebound
      is secured.
4. Err on the side of marking borderline cases; the matcher uses a
   ±15-frame tolerance.

## Seeding the first GT file

For `1v1-mk.mov` window 268–338s we have a pseudo-oracle seed in
`1v1-mk_t268-338.seed.csv` generated from the best v1 run (yolov8x,
imgsz=1280, ut=-12). **Open the matching annotated mp4** under
`experiments/demo/outputs/run*.mp4` with the same parameters, validate
each row visually, then save the cleaned copy as
`1v1-mk_t268-338.csv`. Budget: 10–15 min.

## Using the evaluator

```
.venv/bin/python experiments/demo/evaluate_v2.py \
    --pred experiments/demo/outputs_v2/<run>.events.json \
    --gt experiments/demo/gt/1v1-mk_t268-338.csv \
    --tolerance 15
```
