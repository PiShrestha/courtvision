"""render an annotated mp4 that visualises the per-frame rim tracker.

the default viz.py draws the static hoop from the run's config on every
frame. this tool reads the companion `rim_trace.jsonl` — which records
(frame_id, center, radius, source) for every processed frame — and
overlays it so you can watch the tracker drift (or stay put) in real
time.

usage:
    .venv/bin/python experiments/demo/overlay_rim_trace.py \
        --run experiments/demo/outputs_v2_live/v2live_run3_..._11930344_3 \
        --video 1v1-mk.mov \
        --out experiments/demo/outputs_v2_live/v2live_run3_rim_overlay.mp4

color legend (bgr):
    magenta circle   static anchor (never moves)
    cyan circle      tracker-predicted rim (per frame)
    thin yellow line from anchor to tracker showing drift vector
    top-left label:  "rim source: static | tracker | reseed"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ANCHOR_BGR = (255, 120, 255)         # magenta
TRACKER_BGR = (255, 255, 0)          # cyan
DRIFT_BGR = (0, 200, 255)            # amber


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True,
                    help="run prefix (no extension); we pick up "
                         ".rim_trace.jsonl and .meta.json beside it")
    ap.add_argument("--video", required=True,
                    help="source mp4/mov used by the run")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--start", type=float, default=None,
                    help="seconds to start (defaults to run's meta.start)")
    ap.add_argument("--end", type=float, default=None,
                    help="seconds to end (defaults to run's meta.end)")
    return ap.parse_args()


def load_rim_trace(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            h = json.loads(line)
            out[int(h["frame_id"])] = h
    return out


def main() -> int:
    args = parse_args()
    run_prefix = Path(args.run)
    trace_path = Path(str(run_prefix) + ".rim_trace.jsonl")
    meta_path = Path(str(run_prefix) + ".meta.json")
    if not trace_path.exists():
        print(f"rim trace not found: {trace_path}", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    anchor = meta.get("hoop", {})
    ax, ay = anchor.get("center", [0, 0])
    ar = int(anchor.get("radius", 30))

    trace = load_rim_trace(trace_path)
    if not trace:
        print("rim trace is empty", file=sys.stderr); return 1

    start = args.start if args.start is not None else float(meta.get("start", 0.0))
    end_val = args.end if args.end is not None else meta.get("end")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr); return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(round(start * fps))
    end_frame = int(round(end_val * fps)) if end_val is not None else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"could not open writer at {out_path}", file=sys.stderr); return 1

    n = 0
    source_counts: dict[str, int] = {}
    frame_id = start_frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if end_frame is not None and frame_id >= end_frame:
                break

            # always show the static anchor.
            cv2.circle(frame, (int(ax), int(ay)), ar, ANCHOR_BGR, 2)

            # show the per-frame tracker output (if present).
            info = trace.get(frame_id)
            if info is not None:
                tx, ty = info["center"]
                tr = int(info["radius"])
                src = info.get("source", "?")
                source_counts[src] = source_counts.get(src, 0) + 1

                # only draw the tracker circle when it actually differs.
                if (tx, ty) != (ax, ay):
                    cv2.circle(frame, (int(tx), int(ty)), tr, TRACKER_BGR, 2)
                    cv2.line(frame, (int(ax), int(ay)), (int(tx), int(ty)),
                             DRIFT_BGR, 1, cv2.LINE_AA)

                drift = ((tx - ax) ** 2 + (ty - ay) ** 2) ** 0.5
                label = f"rim {src:<7}  drift={drift:5.1f}px  f={frame_id}"
            else:
                label = f"rim (no trace)  f={frame_id}"

            # text header
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, 2)
            cv2.rectangle(frame, (8, 8), (18 + tw, 14 + th), (0, 0, 0), -1)
            cv2.putText(frame, label, (14, 8 + th),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                         cv2.LINE_AA)

            writer.write(frame)
            n += 1
            frame_id += 1
    finally:
        writer.release(); cap.release()

    print(f"wrote {n} frames to {out_path}")
    print(f"rim source breakdown: {source_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
