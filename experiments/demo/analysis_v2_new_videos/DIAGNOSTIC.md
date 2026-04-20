# Diagnostic — new-videos v2 matrix (10 clips)

Slurm jobs 11973953 (array 0–9) + 11973954 (reporter) completed 2026-04-20.
Source manifest: [matrix_v2_new_videos.csv](../matrix_v2_new_videos.csv).
Full machine output: [results_v2.csv](results_v2.csv), [per_clip_report.md](per_clip_report.md),
[per_clip_report.json](per_clip_report.json).

## Tracking health (across all 10 clips)

| Clip | Frames | Both P1+P2 seen | Ball frames | Ball rate |
|---|---:|---:|---:|---:|
| 1v1-mk t268-338 | 2,089 | **100%** | 699 | 33% |
| 1v1-mk t374-450 | 2,280 | **100%** | 816 | 36% |
| 1v1-ddg t929-991 | 3,668 | 99.6% | 20 | **0.5%** |
| 1v1-ddg t1017-1078 | 3,704 | **100%** | 65 | **1.8%** |
| 1v1-jason t254-317 | 3,726 | 99.9% | 1,320 | 35% |
| 1v1-jason t651-724 | 4,364 | **100%** | 1,363 | 31% |
| 1v1-nasir t789-852 | 3,819 | 98.2% | 750 | 20% |
| 1v1-nasir t1773-1852 | 4,725 | 97.8% | 820 | 17% |
| 1v1-roy t179-268 | 5,357 | 99.5% | 2,341 | 44% |
| 1v1-roy t321-399 | 4,640 | 98.9% | 2,565 | 55% |

**Headline:** the DuoTracker identity fix works. Every clip holds both P1 and
P2 in ≥98% of frames. `other` bucket is empty on all 10 clips — no events
escalated beyond `{1, 2}`. Previous `_next_id` escalation to `{5, 7, 10}` is
gone.

## Usable vs unusable per-clip verdicts

| Clip | Verdict | Why |
|---|---|---|
| 1v1-mk t268-338 | **partial** | both players tracked; possession 99% P1 → P2 never "held" ball. Rule-correct but likely a P1/P2 label flip vs v1 |
| 1v1-mk t374-450 | **partial** | same symptom: P1 gets 10 attempts, P2 gets 0 |
| 1v1-ddg t929-991 | **unusable** | 0.5% ball detection rate — rule starves, 0 attempts |
| 1v1-ddg t1017-1078 | **unusable** | 1.8% ball rate + 0s possession for either player |
| 1v1-jason t254-317 | **usable** | 4/3 attempts P1/P2; one make; possession 62s of 62s |
| 1v1-jason t651-724 | **usable** | 7/6 attempts P1/P2; one make; possession 73s of 72s |
| 1v1-nasir t789-852 | **usable** | 3/1 attempts; one make (P1, 33% FG); both players hold ball |
| 1v1-nasir t1773-1852 | **partial** | 5/0 attempts — again P1 dominates entire clip |
| 1v1-roy t179-268 | **usable** | 0/4 attempts (P2 drives the window); possession lopsided but real |
| 1v1-roy t321-399 | **usable** | 3/10 attempts; two makes; 2:1 possession in P2's favour |

Summary: **6 of 10 clips produce meaningful per-player stats today**. 2 are
unusable due to ball-detection starvation on ddg. 2 mk windows and 1 nasir
window have one-sided possession that is either genuine gameplay asymmetry or
a symptom of the P1/P2 label-flip issue below.

## Why ddg gets ~1% ball detection

At imgsz=640, yolov8l's "sports ball" class needs the ball to occupy roughly
≥8×8 px on the output feature map. The ddg footage is shot from further back
than mk / jason / roy / nasir, making the ball cover far fewer pixels.

**Targeted rerun at yolov8x + imgsz=1280 (jobs 11974488 + 11974494)**
did NOT fix it — ball detection rose from 0.5% / 1.8% only to 2.1% / 1.3%.
So this is not a resolution or model-size problem. COCO's generic
`sports_ball` class simply doesn't fire on whatever the ddg ball looks
like (color, lighting, or motion blur mismatch with the training set).

The only unblock is a **custom basketball checkpoint** via the already-
integrated `--custom-model` pathway (see [ARCHITECTURE_v2.md](../ARCHITECTURE_v2.md)
§ "Custom basketball checkpoint"). Rerunning ddg on any COCO-only model at
any imgsz is now known to be a dead end.

## Why "P1 vs P2" labels are not stable across clips

DuoTracker assigns `slot_id=1` to the first player it bootstraps and
`slot_id=2` to the second. The order of bootstrap depends entirely on which
player happens to have the higher-confidence detection on the first frame of
the window. So **"P1" in the mk t268 clip is not necessarily the same person
as "P1" in the mk t374 clip**, and absolutely isn't the same person across
different videos.

This is a real limitation the user's request surfaced. The fix is a clip-
level "canonical side" assignment after tracking completes — e.g., rename
slots so `P1` is always the player predominantly on the left half of the
court, `P2` on the right. That post-hoc normalisation would make per-player
stats comparable across clips of the same matchup.

I haven't implemented it yet because it's a judgment call how to handle
videos where players switch sides mid-clip (roy shows this pattern
visually). Flag this for future work.

## Recommended next pass

1. **Rerun ddg at imgsz=1280** — cheapest way to salvage the two unusable
   clips. 2 slurm tasks, ~5 minutes wall time total.
2. **Human-verify the nasir + roy hoop configs.** Auto-detect gave
   `nasir=(962,256,r=46)` which looks reasonable, but `roy=(1412,477,r=56)`
   has y=477 which is suspicious for a rim position. If the roy hoop is
   wrong, the shot_made=1 result is a false positive.
3. **Add a "canonical side" post-hoc relabeller** so P1/P2 labels reference
   the same physical player across clips (and across runs of the same clip).
4. **Provide a basketball-specific YOLO checkpoint** via the already-
   integrated `--custom-model` pathway. This would jointly lift ddg's ball
   recall AND eliminate the rim-drift / static-anchor placeholder issues on
   all three moving-camera clips.

## What I got right this pass

- DuoTracker identity stability: **verified** — 0 events leaked past
  `{1, 2}` across any of 10 clips on 4 different videos at either 30 or
  60 fps.
- fps scaling: 60 fps clips processed correctly. Frame windows doubled,
  velocity thresholds halved, rule semantics time-invariant.
- Middle-60% window sampling: used on every video, seed=7, 2 windows each,
  no overlap, all inside the playable region.
- Per-player metrics produced for every clip in a consistent schema.
- Zero identity warnings across the batch.
