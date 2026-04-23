"""generate per-clip annotation worksheets to collect fast human ground truth.

for each selected clip we write a csv with rows pre-seeded from the model's
events.json, so the annotator mostly marks agree / disagree / wrong-type
rather than typing event rows from scratch. a trailing section lets them
add events the model missed.

usage:
    python experiments/demo/make_annotation_worksheet.py \\
        --variant-dir experiments/demo/outputs_v3_ablation/full_ensemble_26 \\
        --out-dir  experiments/demo/annotation_worksheets \\
        --n-clips  5 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


SHOT_EVENTS = {"shot_attempt", "shot_made", "shot_miss"}


def pick_clips(variant_dir: Path, n: int, seed: int,
               require_mp4: bool = True) -> list[Path]:
    """prefer clips with a saved annotated mp4 so annotators can scrub visually.
    the sbatch saves mp4 every 5th task only, so without this filter most
    random picks give worksheets with a MISSING video."""
    events_files = sorted(variant_dir.glob("*.events.json"))
    if not events_files:
        raise SystemExit(f"no events.json in {variant_dir}")
    if require_mp4:
        with_mp4 = [p for p in events_files
                    if p.with_suffix("").with_suffix(".mp4").exists()]
        if with_mp4:
            events_files = with_mp4
        else:
            print("warn: no clips have saved mp4s; picking anyway")
    rng = random.Random(seed)
    return rng.sample(events_files, min(n, len(events_files)))


def write_worksheet(events_path: Path, out_dir: Path) -> Path:
    meta_path = events_path.with_suffix("").with_suffix(".meta.json")
    mp4_path  = events_path.with_suffix("").with_suffix(".mp4")
    with events_path.open() as f:
        events = json.load(f)

    stem = events_path.name.replace(".events.json", "")
    out = out_dir / f"{stem}.worksheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "frame_id", "model_event", "model_player",
                    "annotator_verdict", "annotator_correct_event", "notes"])
        w.writerow(["# video",       mp4_path.name if mp4_path.exists() else "MISSING", "", "", "", "", ""])
        w.writerow(["# meta",        meta_path.name if meta_path.exists() else "MISSING", "", "", "", "", ""])
        w.writerow(["# verdict values",
                    "ok | wrong_type | false_positive", "", "", "", "", ""])
        w.writerow([])

        # predicted events the annotator will rate
        w.writerow(["# --- model-predicted events (rate each) ---", "", "", "", "", "", ""])
        for e in events:
            if e.get("event") not in SHOT_EVENTS:
                continue   # skip possession-only spam; annotator can add them manually below if needed
            w.writerow(["predicted", e.get("frame_id"), e.get("event"),
                        e.get("player", ""), "", "", ""])

        # room for events the model missed
        w.writerow([])
        w.writerow(["# --- missed events (add rows as needed) ---", "", "", "", "", "", ""])
        w.writerow(["# set verdict=missed and fill annotator_correct_event + player",
                    "", "", "", "", "", ""])
        for _ in range(5):
            w.writerow(["missed", "", "", "", "missed", "", ""])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", required=True, type=Path)
    ap.add_argument("--out-dir",     required=True, type=Path)
    ap.add_argument("--n-clips",     type=int, default=5)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    picks = pick_clips(args.variant_dir, args.n_clips, args.seed)
    print(f"selected {len(picks)} clips:")
    for p in picks:
        out = write_worksheet(p, args.out_dir)
        print(f"  {out}")
    print(f"\nopen each .worksheet.csv alongside the matching "
          f"{args.variant_dir.name}/*.mp4 in a video player with a frame "
          f"counter (VLC: Tools -> Preferences -> Show settings=All -> "
          f"Input/Codecs, or use View -> Media Information while playing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
