"""produce a broad experiment matrix for the demo pipeline.

columns per row (one slurm array task each):
    run_id, video, start, end, model, imgsz, confidence, tracker,
    upward_trigger, made_window, hoop_pad, possession_dist_px, hoop_config

writes:
    experiments/demo/matrix.csv
    experiments/demo/matrix_summary.md  (human-friendly design-of-experiments summary)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Axis:
    name: str
    values: list


def video_duration_s(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n / fps if fps > 0 else 0.0


def sample_windows(duration: float, n: int, rng: random.Random,
                     min_len: float = 60.0, max_len: float = 90.0,
                     mid_lo: float = 0.2, mid_hi: float = 0.8) -> list[tuple[float, float]]:
    lo, hi = duration * mid_lo, duration * mid_hi
    windows: list[tuple[float, float]] = []
    attempts = 0
    while len(windows) < n and attempts < 200:
        attempts += 1
        length = rng.uniform(min_len, max_len)
        latest_start = hi - length
        if latest_start <= lo: continue
        start = rng.uniform(lo, latest_start)
        end = start + length
        if any(start < e and end > s for s, e in windows):
            continue
        windows.append((round(start, 2), round(end, 2)))
    return sorted(windows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="+", default=["1v1-mk.mov"])
    ap.add_argument("--windows-per-video", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="experiments/demo/matrix.csv")
    ap.add_argument("--summary", default="experiments/demo/matrix_summary.md")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # perception axes.
    detector_axes = {
        "model": ["yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        "imgsz": [640, 1280],
        "confidence": [0.25, 0.35],
    }
    # three shot-detector presets covering the trade-off space.
    shot_configs = [
        {"upward_trigger": -8.0,  "made_window": 60, "hoop_pad": 20},  # loose
        {"upward_trigger": -12.0, "made_window": 45, "hoop_pad": 14},  # center
        {"upward_trigger": -16.0, "made_window": 30, "hoop_pad": 8},   # strict
    ]

    # fix tracker and possession threshold to reduce combinatorial explosion.
    fixed = {
        "tracker": "bytetrack.yaml",
        "possession_dist_px": 200.0,
    }

    # cartesian product of detector axes: 3x2x2 = 12.
    det_configs = list(itertools.product(*detector_axes.values()))
    det_names = list(detector_axes.keys())

    rows = []
    run_id = 0
    hoop_lookup = {
        "1v1-mk.mov": "experiments/demo/hoop_configs/1v1-mk.json",
        "1v1-ddg.mp4": "experiments/demo/hoop_configs/1v1-ddg.json",
        "1v1-jason.mp4": "experiments/demo/hoop_configs/1v1-jason.json",
    }

    for video in args.videos:
        vpath = ROOT / video
        if not vpath.exists():
            print(f"skip missing: {vpath}", file=sys.stderr); continue
        dur = video_duration_s(str(vpath))
        wins = sample_windows(dur, args.windows_per_video, rng)
        for (ws, we) in wins:
            for det in det_configs:
                det_kwargs = dict(zip(det_names, det))
                for shot in shot_configs:
                    rows.append({
                        "run_id": run_id,
                        "video": str(vpath.relative_to(ROOT)),
                        "start": ws, "end": we,
                        "hoop_config": hoop_lookup.get(video, ""),
                        **det_kwargs,
                        **fixed,
                        **shot,
                    })
                    run_id += 1

    if not rows:
        print("no rows produced"); return 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n": downstream bash IFS parsing breaks on CRLF
    # because the trailing \r ends up inside the last field.
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    # human-friendly summary.
    videos_seen = sorted({r["video"] for r in rows})
    summary = [
        "# Demo Experiment Matrix",
        "",
        f"total runs: **{len(rows)}**",
        f"videos: {len(videos_seen)} ({', '.join(videos_seen)})",
        f"windows per video: {args.windows_per_video}",
        f"detector configs per window: {len(det_configs)}",
        f"shot-detector configs per detector: {len(shot_configs)}",
        "",
        "## axes",
        "",
        "### detector (full cross product)",
        *[f"- {k}: {v}" for k, v in detector_axes.items()],
        "",
        "### shot detector presets",
        *[f"- {sc}" for sc in shot_configs],
        "",
        "### fixed",
        *[f"- {k}: {v}" for k, v in fixed.items()],
    ]
    Path(args.summary).write_text("\n".join(summary))
    print(f"wrote {len(rows)} rows to {out}")
    print(f"wrote summary to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
