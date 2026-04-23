# Fast annotation protocol (target: 5–8 min per clip)

## Why
We need real F1 numbers for the final report. The worksheets below pre-seed
each clip with the model's predicted events so an annotator mostly marks
**agree / disagree / wrong type**, rather than typing events from scratch.

## Step 1 — Generate worksheets (one-time)

```bash
# pick 5 random clips from the ceiling variant, seeded for reproducibility
python experiments/demo/make_annotation_worksheet.py \
    --variant-dir experiments/demo/outputs_v3_ablation/full_ensemble_26 \
    --out-dir     experiments/demo/annotation_worksheets \
    --n-clips     5 --seed 42
```

Output: five `*.worksheet.csv` files, one per clip.

## Step 2 — For each worksheet

Copy the matching annotated `.mp4` to your laptop:

```bash
rsync -av --progress \
  jpj8rf@login.hpc.virginia.edu:/home/jpj8rf/Documents/github/courtvision/experiments/demo/outputs_v3_ablation/full_ensemble_26/<CLIP_STEM>.mp4 \
  jpj8rf@login.hpc.virginia.edu:/home/jpj8rf/Documents/github/courtvision/experiments/demo/annotation_worksheets/<CLIP_STEM>.worksheet.csv \
  ~/annotation/
```

Open the `.mp4` in VLC (show frame counter: View → Media Information → Codec
tab shows fps; press `e` to advance one frame). Open the `.worksheet.csv`
alongside in any spreadsheet or text editor.

### For each row in the `predicted` section
Scrub to that `frame_id` (seconds ≈ frame_id / 30). Watch 1 s before and 1 s
after. Then set `annotator_verdict` to one of:

- **`ok`** — the model was right: event happened, labelled correctly
- **`wrong_type`** — event happened but labelled wrong (e.g. model said
  `shot_made` but ball missed). Fill `annotator_correct_event` with the right
  label.
- **`false_positive`** — no event there (e.g. ball bounce the model flagged
  as `shot_attempt`). Leave the correct-event column blank.

Leaving `annotator_verdict` **blank** means "skip, I couldn't tell" — those
rows are dropped from the F1 computation, not counted against the model.

### Missed events
At the end of the worksheet there are blank `missed` rows. If you saw a shot
event the model failed to flag, fill:
- `frame_id` (from VLC scrubbing)
- `annotator_correct_event` (shot_attempt / shot_made / shot_miss)
- `model_player` (1 or 2)
- `annotator_verdict` is already pre-filled as `missed`

Add rows as needed — any row with `section=missed` and a non-empty
`annotator_correct_event` counts as one False Negative.

## Step 3 — Score

Once at least one worksheet is filled in:

```bash
python experiments/demo/score_annotations.py \
    --worksheet-dir experiments/demo/annotation_worksheets \
    --variant-dir   experiments/demo/outputs_v3_ablation/full_ensemble_26 \
    --out           experiments/demo/annotation_worksheets/f1_summary.md
```

This prints + writes a markdown table: precision / recall / F1 per event type,
a micro-averaged row, and per-clip breakdowns. That table drops directly into
the presentation's Results slide.

## Team split suggestion
5 clips × ~8 min = 40 min total. Split 2/2/1 across team members and finish in
~15 min of wall-clock each. The scorer handles partial completion — run it
against whatever is filled in at any time.

## Tips for speed
- Watch at 0.5× speed (VLC `[` and `]` keys)
- Don't second-guess: if unclear, leave verdict blank. Skipped rows are not
  counted against the model.
- `ok` is the most common verdict. If model looks right, mark and move on.
- For `shot_made` vs `shot_miss`: watch the ball one second *after* the event
  frame. If it goes through the rim, `ok` the `shot_made`; if it bounces off,
  `wrong_type` with `shot_miss` as correct.
