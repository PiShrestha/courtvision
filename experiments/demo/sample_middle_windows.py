"""pick 60-90s windows from the middle 60% of each video.

skips the first 20% and last 20% of each clip (warm-up / post-game
footage). samples `windows_per_video` non-overlapping windows uniformly
at random with a fixed seed for reproducibility.

emits a csv compatible with run_matrix_v2_live.sbatch — columns match
matrix_v2_live.csv.

usage:
    .venv/bin/python experiments/demo/sample_middle_windows.py \\
        --videos 1v1-mk.mov 1v1-ddg.mp4 1v1-jason.mp4 \\
                  1v1-nasir.mp4 1v1-roy.mp4 \\
        --hoop-dir experiments/demo/hoop_configs \\
        --windows-per-video 2 \\
        --min-seconds 60 --max-seconds 90 \\
        --seed 7 \\
        --out experiments/demo/matrix_v2_new_videos.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path


DEFAULT_COLUMNS = [
    "run_id", "video", "start", "end", "hoop_config",
    "model", "imgsz", "confidence",
    "rim_tracker_kind", "rim_reseed_every",
    "upward_trigger", "history_frames", "release_dist_px",
    "approach_dist_px", "cooldown_frames", "possession_dist_px",
    "occlusion_gap_frames", "attempt_to_made_window",
    "enter_zone_radius_factor", "horizontal_pad_factor",
    "min_downward_velocity",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--hoop-dir", default="experiments/demo/hoop_configs")
    ap.add_argument("--windows-per-video", type=int, default=2)
    ap.add_argument("--min-seconds", type=float, default=60.0)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--middle-fraction", type=float, default=0.6,
                    help="fraction of clip duration to sample from "
                         "(default 0.6 = middle 60%%, trims first/last 20%%)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    # default rule + perception config applied to every row.
    ap.add_argument("--model", default="yolov8l.pt",
                    help="per v2 findings: yolov8l @ 640 is the sweet spot")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--confidence", type=float, default=0.30)
    ap.add_argument("--rim-tracker-kind", default="static")
    ap.add_argument("--rim-reseed-every", type=int, default=90)
    ap.add_argument("--upward-trigger", type=float, default=-10.0)
    ap.add_argument("--history-frames", type=int, default=6)
    ap.add_argument("--release-dist-px", type=float, default=60.0)
    ap.add_argument("--approach-dist-px", type=float, default=260.0)
    ap.add_argument("--cooldown-frames", type=int, default=20)
    ap.add_argument("--possession-dist-px", type=float, default=140.0)
    ap.add_argument("--occlusion-gap-frames", type=int, default=15)
    ap.add_argument("--attempt-to-made-window", type=int, default=120)
    ap.add_argument("--enter-zone-radius-factor", type=float, default=2.5)
    ap.add_argument("--horizontal-pad-factor", type=float, default=2.0)
    ap.add_argument("--min-downward-velocity", type=float, default=1.0)
    return ap.parse_args()


def video_duration(path: str) -> float | None:
    """ffprobe duration in seconds, or None if the file is unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
             ValueError, FileNotFoundError):
        return None


def sample_windows(duration: float, rng: random.Random,
                    n: int, min_s: float, max_s: float,
                    middle_frac: float) -> list[tuple[float, float]]:
    """return n non-overlapping (start, end) windows inside the middle
    `middle_frac` of the clip. falls back to fewer windows if the middle
    is too short to fit them."""
    margin = (1 - middle_frac) / 2
    lo = duration * margin
    hi = duration * (1 - margin)
    span = hi - lo
    if span < min_s:
        return []
    picks: list[tuple[float, float]] = []
    attempts = 0
    while len(picks) < n and attempts < 1000:
        attempts += 1
        length = rng.uniform(min_s, min(max_s, span))
        start = rng.uniform(lo, hi - length)
        end = start + length
        # check non-overlap.
        if all(end <= s0 or start >= e0 for s0, e0 in picks):
            picks.append((round(start, 2), round(end, 2)))
    picks.sort()
    return picks


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    hoop_dir = Path(args.hoop_dir)
    rows: list[dict] = []
    run_id = 0

    for video in args.videos:
        path = Path(video)
        if not path.exists():
            print(f"skipping missing video: {path}")
            continue
        dur = video_duration(str(path))
        if dur is None:
            print(f"skipping unreadable video: {path}")
            continue
        stem = path.stem
        hoop_path = hoop_dir / f"{stem}.json"
        if not hoop_path.exists():
            print(f"skipping {stem}: no hoop config at {hoop_path}")
            continue
        windows = sample_windows(dur, rng, args.windows_per_video,
                                  args.min_seconds, args.max_seconds,
                                  args.middle_fraction)
        print(f"{stem}  dur={dur:.1f}s  picked {len(windows)} windows: {windows}")
        for start, end in windows:
            rows.append({
                "run_id": run_id,
                "video": str(path),
                "start": start, "end": end,
                "hoop_config": str(hoop_path),
                "model": args.model, "imgsz": args.imgsz,
                "confidence": args.confidence,
                "rim_tracker_kind": args.rim_tracker_kind,
                "rim_reseed_every": args.rim_reseed_every,
                "upward_trigger": args.upward_trigger,
                "history_frames": args.history_frames,
                "release_dist_px": args.release_dist_px,
                "approach_dist_px": args.approach_dist_px,
                "cooldown_frames": args.cooldown_frames,
                "possession_dist_px": args.possession_dist_px,
                "occlusion_gap_frames": args.occlusion_gap_frames,
                "attempt_to_made_window": args.attempt_to_made_window,
                "enter_zone_radius_factor": args.enter_zone_radius_factor,
                "horizontal_pad_factor": args.horizontal_pad_factor,
                "min_downward_velocity": args.min_downward_velocity,
            })
            run_id += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS,
                            lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
